"""Inference Pipeline for FPL Point Predictions

Generates projections_horizon.parquet for the MIP Solver using
trained position-specific ML models.

Pipeline:
1. Load trained models (GKP, DEF, MID, FWD + Availability)
2. Build features for all active players
3. Generate xP predictions for each GW in horizon
4. Adjust by availability probability
5. Output solver-ready parquet file

Usage:
    python -m models.inference                 # Full inference
    python -m models.inference --horizon 8     # 8-week horizon
    python -m models.inference --fallback      # Use heuristics if models missing
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.train import PositionModel, AvailabilityPredictor, MODELS_DIR
from models.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PARQUET = PROJECT_ROOT / 'data' / 'parquet'


class FPLInferencePipeline:
    """Generates ML-based projections for the MIP Solver."""
    
    POSITIONS = ['GKP', 'DEF', 'MID', 'FWD']
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    def __init__(self, use_fallback: bool = True):
        """Initialize inference pipeline.
        
        Args:
            use_fallback: If True, use heuristics when models unavailable.
        """
        self.models = {}
        self.availability_predictor = None
        self.feature_engineer = None
        self.use_fallback = use_fallback
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models."""
        for position in self.POSITIONS:
            filepath = MODELS_DIR / f'model_{position}.pkl'
            if filepath.exists():
                try:
                    self.models[position] = PositionModel.load(filepath)
                    logger.info(f"Loaded {position} model")
                except Exception as e:
                    logger.warning(f"Failed to load {position} model: {e}")
            else:
                logger.warning(f"No model found for {position}")
        
        # Load availability predictor
        avail_path = MODELS_DIR / 'availability_model.pkl'
        if avail_path.exists():
            try:
                self.availability_predictor = AvailabilityPredictor.load(avail_path)
                logger.info("Loaded availability predictor")
            except Exception as e:
                logger.warning(f"Failed to load availability predictor: {e}")
    
    def _load_data(self):
        """Load data from warehouse."""
        from etl.transformers import load_parquet
        
        self.players_df = load_parquet('players.parquet')
        self.fixtures_df = load_parquet('fixtures.parquet')
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.players_df, self.fixtures_df)
        
        # Determine current GW
        finished = self.fixtures_df[self.fixtures_df['finished'] == True]
        self.current_gw = int(finished['gameweek'].max()) if not finished.empty else 17
        
        logger.info(f"Loaded {len(self.players_df)} players, current GW: {self.current_gw}")
    
    def generate_projections(self, horizon: int = 5,
                             min_minutes: int = 45) -> pd.DataFrame:
        """Generate xP projections for all players over the horizon.
        
        Args:
            horizon: Number of future gameweeks to project.
            min_minutes: Minimum minutes to include player.
            
        Returns:
            DataFrame in projections_horizon.parquet schema.
        """
        self._load_data()
        
        # Filter to active players with some minutes
        active_players = self.players_df[
            (self.players_df['minutes'] >= min_minutes) &
            (self.players_df['status'] == 'a')
        ]
        
        logger.info(f"Generating projections for {len(active_players)} active players...")
        
        records = []
        
        for _, player in active_players.iterrows():
            pid = int(player['player_id'])
            position_id = int(player['position_id'])
            position = self.POSITION_MAP.get(position_id, 'MID')
            team_id = int(player['team_id'])
            cost = int(player['cost'])
            
            # Load player history
            history_df = self.feature_engineer.load_player_history(pid)
            
            for gw_offset in range(1, horizon + 1):
                target_gw = self.current_gw + gw_offset
                
                # Build feature vector
                features = self.feature_engineer.build_feature_vector(
                    pid, target_gw, history_df
                )
                
                if features is None:
                    # No history - use simple fallback
                    xp = self._fallback_prediction(player, target_gw, position)
                else:
                    # Get ML prediction
                    xp = self._get_prediction(features, position)
                
                # Apply availability adjustment
                xp = self._adjust_for_availability(xp, features, player)
                
                # Estimate minutes
                minutes_proj = self._estimate_minutes(player, features)
                
                records.append({
                    'player_id': pid,
                    'gameweek': target_gw,
                    'xp': round(float(xp), 2),
                    'cost': cost,
                    'position_id': position_id,
                    'team_id': team_id,
                    'minutes_projected': minutes_proj,
                })
        
        df = pd.DataFrame(records)
        
        # Set dtypes per schema
        dtype_map = {
            'player_id': 'int32',
            'gameweek': 'int32',
            'xp': 'float32',
            'cost': 'int32',
            'position_id': 'int8',
            'team_id': 'int8',
            'minutes_projected': 'int16',
        }
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        # Sort by gameweek, then xp descending
        df = df.sort_values(['gameweek', 'xp'], ascending=[True, False]).reset_index(drop=True)
        
        return df
    
    def _get_prediction(self, features: Dict, position: str) -> float:
        """Get ML-based prediction for features."""
        if position not in self.models:
            if self.use_fallback:
                return self._heuristic_xp(features, position)
            return 0.0
        
        model = self.models[position]
        
        # Convert features to DataFrame row
        feature_df = pd.DataFrame([{
            k: v for k, v in features.items() 
            if not k.startswith('_')
        }])
        
        try:
            pred = model.predict(feature_df)[0]
            return max(0, pred)
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            if self.use_fallback:
                return self._heuristic_xp(features, position)
            return 0.0
    
    def _fallback_prediction(self, player: pd.Series, 
                            target_gw: int, position: str) -> float:
        """Generate fallback prediction when no features available."""
        # Use PPG or form as base
        ppg = float(player.get('points_per_game', 0) or 0)
        form = float(player.get('form', 0) or 0)
        
        base = max(ppg, form, 2.0)  # Minimum 2 points
        
        # Simple position adjustment
        pos_mult = {'GKP': 0.9, 'DEF': 0.95, 'MID': 1.0, 'FWD': 1.1}
        mult = pos_mult.get(position, 1.0)
        
        return base * mult
    
    def _heuristic_xp(self, features: Dict, position: str) -> float:
        """Heuristic-based xP when ML model unavailable."""
        # Base from rolling points
        base = features.get('points_rolling_3', 0) or features.get('points_rolling_6', 0) or 2.0
        
        # Fixture adjustment
        fdr = features.get('fixture_difficulty_elo', features.get('fixture_difficulty', 3))
        fdr_mult = {1: 1.20, 2: 1.10, 3: 1.0, 4: 0.90, 5: 0.80}.get(fdr, 1.0)
        
        # Home bonus
        home_mult = 1.05 if features.get('is_home', 0) else 1.0
        
        # Form trend
        trend = features.get('points_trend', 0)
        trend_mult = 1.0 + (trend * 0.05)  # +/-5% per point of trend
        
        xp = base * fdr_mult * home_mult * trend_mult
        
        # Position-specific adjustments
        if position in ['GKP', 'DEF']:
            cs_rate = features.get('cs_rolling_3', 0)
            win_prob = features.get('win_probability', 0.33)
            xp += cs_rate * 2 + (win_prob - 0.33) * 1.5
        
        if position in ['MID', 'FWD']:
            xgi = features.get('xgi_rolling_3', 0)
            xp += xgi * 3  # xGI is highly predictive
        
        return max(0, xp)
    
    def _adjust_for_availability(self, raw_xp: float, 
                                 features: Optional[Dict],
                                 player: pd.Series) -> float:
        """Adjust xP by availability probability."""
        # Get FPL's chance of playing
        chance = player.get('chance_playing', 100)
        if pd.isna(chance):
            chance = 100
        
        # If player flagged, use that
        if chance < 100:
            return raw_xp * (chance / 100.0)
        
        # Use ML availability predictor if available
        if self.availability_predictor and self.availability_predictor.is_trained and features:
            feature_df = pd.DataFrame([features])
            
            # Add chance_playing column
            feature_df['chance_playing'] = chance
            
            try:
                prob = self.availability_predictor.predict_probability(feature_df)[0]
                return raw_xp * prob
            except Exception:
                pass
        
        return raw_xp
    
    def _estimate_minutes(self, player: pd.Series, 
                         features: Optional[Dict]) -> int:
        """Estimate expected minutes for a player."""
        status = player.get('status', 'a')
        if status != 'a':
            return 0
        
        chance = player.get('chance_playing', 100)
        if pd.isna(chance):
            chance = 100
        
        # Base from rolling average
        if features:
            avg_mins = features.get('minutes_rolling_3', 0) or features.get('minutes_rolling_6', 0)
            if avg_mins > 0:
                return int(min(90, avg_mins) * (chance / 100.0))
        
        # Fallback: estimate from total minutes / games
        total_mins = player.get('minutes', 0)
        if total_mins > 0 and self.current_gw > 0:
            avg_mins = total_mins / self.current_gw
            return int(min(90, avg_mins) * (chance / 100.0))
        
        return 45  # Default for new players
    
    def save_projections(self, df: pd.DataFrame,
                         filename: str = 'projections_horizon.parquet') -> Path:
        """Save projections to parquet file."""
        filepath = DATA_PARQUET / filename
        df.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Saved {len(df)} projections to {filepath}")
        return filepath
    
    def run(self, horizon: int = 5, min_minutes: int = 45) -> pd.DataFrame:
        """Run full inference pipeline.
        
        Args:
            horizon: Number of future GWs to project.
            min_minutes: Minimum minutes for player inclusion.
            
        Returns:
            DataFrame of projections.
        """
        logger.info("=" * 60)
        logger.info("FPL Inference Pipeline")
        logger.info("=" * 60)
        
        # Check model availability
        available_models = [p for p in self.POSITIONS if p in self.models]
        logger.info(f"Models available: {available_models or 'None (using heuristics)'}")
        
        # Generate projections
        df = self.generate_projections(horizon, min_minutes)
        
        # Save
        self.save_projections(df)
        
        # Print summary
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print projection summary."""
        # Merge with player names
        merged = df.merge(
            self.players_df[['player_id', 'web_name', 'position', 'team_name']],
            on='player_id'
        )
        
        # Total xP per player
        totals = merged.groupby(['player_id', 'web_name', 'position', 'team_name'])['xp'].sum()
        totals = totals.reset_index().nlargest(15, 'xp')
        
        logger.info("\nTop 15 Players by Total xP:")
        for _, row in totals.iterrows():
            logger.info(f"  {row['web_name']:15} ({row['position']}) {row['team_name']:4} - {row['xp']:.1f} xP")
        
        # Per-GW summary
        gw_stats = df.groupby('gameweek')['xp'].agg(['mean', 'max', 'sum'])
        logger.info("\nPer-GW Summary:")
        for gw, row in gw_stats.iterrows():
            logger.info(f"  GW{gw}: avg={row['mean']:.2f}, max={row['max']:.2f}, total={row['sum']:.0f}")
        
        # Model vs Heuristic breakdown
        if self.models:
            logger.info(f"\nModels used: {list(self.models.keys())}")
        else:
            logger.info("\nAll predictions from heuristics (no trained models)")


def run_inference(horizon: int = 5, 
                  min_minutes: int = 45,
                  use_fallback: bool = True) -> pd.DataFrame:
    """Convenience function to run inference pipeline.
    
    Args:
        horizon: Number of future GWs.
        min_minutes: Minimum minutes for inclusion.
        use_fallback: Use heuristics if models missing.
        
    Returns:
        Projections DataFrame.
    """
    logging.basicConfig(level=logging.INFO)
    
    pipeline = FPLInferencePipeline(use_fallback=use_fallback)
    return pipeline.run(horizon, min_minutes)


def main():
    parser = argparse.ArgumentParser(
        description='Generate FPL point projections for the MIP Solver'
    )
    parser.add_argument('--horizon', type=int, default=5,
                       help='Number of future gameweeks to project')
    parser.add_argument('--min-minutes', type=int, default=45,
                       help='Minimum total minutes for player inclusion')
    parser.add_argument('--fallback', action='store_true', default=True,
                       help='Use heuristics when models unavailable')
    parser.add_argument('--no-fallback', action='store_true',
                       help='Fail if models unavailable')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    use_fallback = not args.no_fallback
    
    run_inference(
        horizon=args.horizon,
        min_minutes=args.min_minutes,
        use_fallback=use_fallback
    )


if __name__ == '__main__':
    main()

