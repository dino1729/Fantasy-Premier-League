"""ETL Pipeline Orchestrator

Coordinates the full Extract-Transform-Load flow from raw API data
to solver-ready parquet files.

Usage:
    python -m etl.pipeline                  # Full pipeline
    python -m etl.pipeline --fetch-only     # Just fetch raw data
    python -m etl.pipeline --transform-only # Transform existing raw data
    python -m etl.pipeline --adapter-only   # Just generate projections

The pipeline produces:
1. data/parquet/players.parquet - Master player table
2. data/parquet/fixtures.parquet - All fixtures with Elo difficulty
3. data/parquet/projections_horizon.parquet - Solver input (player xP per GW)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.fetchers import (
    FPLFetcher, 
    ClubEloFetcher, 
    FixtureDifficultyCalculator,
    fetch_all_data
)
from etl.transformers import (
    PlayerTransformer,
    FixtureTransformer,
    ProjectionTransformer,
    load_parquet,
    DATA_PARQUET,
    DATA_MAPPINGS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class HeuristicAdapter:
    """Generates expected points using heuristics until ML models are ready.
    
    This bridges the gap between the current heuristic approach and the
    future ML-based predictions. It produces projections_horizon.parquet
    that the solver can consume immediately.
    
    Heuristic formula:
    xP = base_points + fixture_bonus + form_adjustment
    
    Where:
    - base_points: PPG * availability_factor
    - fixture_bonus: adjustment based on Elo-derived difficulty
    - form_adjustment: recent form trend
    """
    
    # Position-specific base expectations (avg PPG for "good" player)
    POSITION_BASE = {
        1: 4.0,   # GKP: ~4 pts avg
        2: 4.0,   # DEF: ~4 pts avg
        3: 4.5,   # MID: ~4.5 pts avg
        4: 4.5,   # FWD: ~4.5 pts avg
    }
    
    # FDR adjustments (multiplier for base)
    FDR_MULTIPLIER = {
        1: 1.20,  # Easy fixture: +20%
        2: 1.10,  # Fairly easy: +10%
        3: 1.00,  # Medium: no change
        4: 0.90,  # Hard: -10%
        5: 0.80,  # Very hard: -20%
    }
    
    def __init__(self, players_df: pd.DataFrame, 
                 fixtures_df: pd.DataFrame,
                 current_gw: int,
                 horizon: int = 5):
        """Initialize the heuristic adapter.
        
        Args:
            players_df: Clean players DataFrame from PlayerTransformer.
            fixtures_df: Clean fixtures DataFrame from FixtureTransformer.
            current_gw: Current gameweek number.
            horizon: Number of future GWs to project.
        """
        self.players_df = players_df
        self.fixtures_df = fixtures_df
        self.current_gw = current_gw
        self.horizon = horizon
        
        # Build fixture lookup: (team_id, gw) -> fdr_elo
        self._build_fixture_lookup()
    
    def _build_fixture_lookup(self):
        """Build lookup table for fixture difficulty by team and GW."""
        self.fixture_lookup = {}  # (team_id, gw) -> fdr_elo
        
        for _, fix in self.fixtures_df.iterrows():
            gw = fix['gameweek']
            if gw <= self.current_gw:
                continue
            
            # Home team's difficulty is based on away team strength
            home_key = (fix['home_team_id'], gw)
            self.fixture_lookup[home_key] = {
                'fdr': fix['home_difficulty_elo'],
                'is_home': True,
                'win_prob': fix['home_win_prob']
            }
            
            # Away team's difficulty is based on home team strength
            away_key = (fix['away_team_id'], gw)
            self.fixture_lookup[away_key] = {
                'fdr': fix['away_difficulty_elo'],
                'is_home': False,
                'win_prob': fix['away_win_prob']
            }
    
    def generate_predictions(self) -> Dict[int, List[float]]:
        """Generate xP predictions for all players over the horizon.
        
        Returns:
            Dict mapping player_id -> list of xP for each GW.
        """
        predictions = {}
        
        for _, player in self.players_df.iterrows():
            pid = int(player['player_id'])
            team_id = int(player['team_id'])
            position_id = int(player['position_id'])
            
            xp_list = []
            for i in range(self.horizon):
                gw = self.current_gw + 1 + i
                xp = self._predict_single_gw(player, team_id, position_id, gw)
                xp_list.append(round(xp, 2))
            
            predictions[pid] = xp_list
        
        return predictions
    
    def _predict_single_gw(self, player: pd.Series, 
                           team_id: int, 
                           position_id: int, 
                           gw: int) -> float:
        """Predict xP for a single player in a single gameweek."""
        # Base: player's PPG or position average
        ppg = player.get('points_per_game', 0)
        if ppg <= 0:
            ppg = self.POSITION_BASE.get(position_id, 4.0)
        
        # Availability factor
        status = player.get('status', 'a')
        chance = player.get('chance_playing', 100)
        
        if status != 'a':
            return 0.0
        
        availability = chance / 100.0
        
        # Fixture difficulty adjustment
        fixture_info = self.fixture_lookup.get((team_id, gw), {'fdr': 3, 'is_home': True})
        fdr = fixture_info.get('fdr', 3)
        fdr_mult = self.FDR_MULTIPLIER.get(fdr, 1.0)
        
        # Home advantage bonus
        home_bonus = 1.05 if fixture_info.get('is_home', True) else 1.0
        
        # Form adjustment
        form = player.get('form', 0)
        form_adjustment = 1.0
        if ppg > 0:
            form_ratio = form / ppg
            # Cap form adjustment to +/- 15%
            form_adjustment = min(1.15, max(0.85, 0.95 + form_ratio * 0.05))
        
        # Calculate final xP
        xp = ppg * availability * fdr_mult * home_bonus * form_adjustment
        
        # Position-specific bonuses
        if position_id in [1, 2]:  # GKP, DEF
            # Clean sheet probability affects defensive players
            win_prob = fixture_info.get('win_prob', 0.33)
            # Higher win prob = higher clean sheet chance
            cs_bonus = (win_prob - 0.33) * 2.0  # Range: -0.66 to +1.34
            xp += cs_bonus * 0.5  # Small bonus
        
        return max(0, xp)


class ETLPipeline:
    """Main orchestrator for the ETL pipeline."""
    
    def __init__(self, fetch_fresh: bool = True):
        """Initialize the pipeline.
        
        Args:
            fetch_fresh: Whether to fetch fresh data from APIs.
        """
        self.fetch_fresh = fetch_fresh
        self.fpl = FPLFetcher()
        self.elo = ClubEloFetcher()
        self.calc = FixtureDifficultyCalculator(self.fpl, self.elo)
        
        # Data containers
        self.bootstrap = None
        self.fixtures_raw = None
        self.elo_ratings = None
        self.fixture_difficulties = None
        self.current_gw = None
        
        # Transformed data
        self.players_df = None
        self.fixtures_df = None
        self.projections_df = None
    
    def run(self, 
            fetch: bool = True, 
            transform: bool = True, 
            generate_projections: bool = True) -> Dict[str, Path]:
        """Run the full ETL pipeline.
        
        Args:
            fetch: Whether to fetch fresh data.
            transform: Whether to run transformations.
            generate_projections: Whether to generate projections.
            
        Returns:
            Dict mapping output names to file paths.
        """
        outputs = {}
        
        if fetch:
            logger.info("=" * 60)
            logger.info("PHASE 1: Extract (Fetching Data)")
            logger.info("=" * 60)
            self._fetch_all()
        
        if transform:
            logger.info("=" * 60)
            logger.info("PHASE 2: Transform (Cleaning & Normalizing)")
            logger.info("=" * 60)
            
            if self.bootstrap is None:
                self._load_latest_raw()
            
            outputs.update(self._transform_all())
        
        if generate_projections:
            logger.info("=" * 60)
            logger.info("PHASE 3: Generate Projections (Heuristic Adapter)")
            logger.info("=" * 60)
            
            if self.players_df is None or self.fixtures_df is None:
                self._load_parquet_data()
            
            outputs['projections'] = self._generate_projections()
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        for name, path in outputs.items():
            logger.info(f"  {name}: {path}")
        
        return outputs
    
    def _fetch_all(self):
        """Fetch all raw data from APIs."""
        logger.info("Fetching FPL bootstrap-static...")
        self.bootstrap = self.fpl.get_bootstrap_static(save_raw=True)
        logger.info(f"  -> {len(self.bootstrap.get('elements', []))} players")
        
        logger.info("Fetching FPL fixtures...")
        self.fixtures_raw = self.fpl.get_fixtures(save_raw=True)
        logger.info(f"  -> {len(self.fixtures_raw)} fixtures")
        
        logger.info("Fetching ClubElo ratings...")
        self.elo_ratings = self.elo.get_ratings(save_raw=True)
        logger.info(f"  -> {len(self.elo_ratings)} team ratings")
        
        logger.info("Calculating Elo-based fixture difficulties...")
        self.fixture_difficulties = self.calc.get_fixture_difficulties()
        logger.info(f"  -> Calculated for {len(self.fixture_difficulties)} teams")
        
        self.current_gw = self.fpl.get_current_gameweek()
        logger.info(f"Current gameweek: {self.current_gw}")
    
    def _load_latest_raw(self):
        """Load most recent raw data from disk."""
        from etl.fetchers import DATA_RAW
        
        # Find latest bootstrap file
        bootstrap_files = sorted(DATA_RAW.glob('bootstrap_static_*.json'), reverse=True)
        if bootstrap_files:
            with open(bootstrap_files[0], 'r') as f:
                self.bootstrap = json.load(f)
            logger.info(f"Loaded {bootstrap_files[0].name}")
        else:
            raise FileNotFoundError("No bootstrap data found. Run with --fetch first.")
        
        # Find latest fixtures file
        fixture_files = sorted(DATA_RAW.glob('fixtures_*.json'), reverse=True)
        if fixture_files:
            with open(fixture_files[0], 'r') as f:
                self.fixtures_raw = json.load(f)
            logger.info(f"Loaded {fixture_files[0].name}")
        
        # Elo ratings (need to refetch since CSV is simple)
        self.elo_ratings = self.elo.get_ratings(save_raw=False)
        self.current_gw = self.fpl.get_current_gameweek()
    
    def _transform_all(self) -> Dict[str, Path]:
        """Run all transformations."""
        outputs = {}
        
        # Build team map
        teams_map = {t['id']: t['short_name'] 
                     for t in self.bootstrap.get('teams', [])}
        
        # Transform players
        logger.info("Transforming players...")
        player_transformer = PlayerTransformer(self.bootstrap, teams_map)
        self.players_df = player_transformer.transform()
        outputs['players'] = player_transformer.save_parquet(self.players_df)
        logger.info(f"  -> {len(self.players_df)} players")
        
        # Recalculate fixture difficulties if not available
        if self.fixture_difficulties is None:
            self.fixture_difficulties = self.calc.get_fixture_difficulties()
        
        # Transform fixtures
        logger.info("Transforming fixtures...")
        fixture_transformer = FixtureTransformer(
            self.fixtures_raw, 
            teams_map,
            self.elo_ratings or {},
            self.fixture_difficulties
        )
        self.fixtures_df = fixture_transformer.transform()
        outputs['fixtures'] = fixture_transformer.save_parquet(self.fixtures_df)
        logger.info(f"  -> {len(self.fixtures_df)} fixtures")
        
        return outputs
    
    def _load_parquet_data(self):
        """Load existing parquet files."""
        self.players_df = load_parquet('players.parquet')
        self.fixtures_df = load_parquet('fixtures.parquet')
        self.current_gw = self.fpl.get_current_gameweek()
    
    def _generate_projections(self) -> Path:
        """Generate projections using heuristic adapter."""
        logger.info("Generating heuristic predictions...")
        
        adapter = HeuristicAdapter(
            self.players_df,
            self.fixtures_df,
            self.current_gw,
            horizon=5
        )
        
        predictions = adapter.generate_predictions()
        logger.info(f"  -> Generated predictions for {len(predictions)} players")
        
        # Transform to solver format
        logger.info("Creating projections_horizon.parquet...")
        projection_transformer = ProjectionTransformer(
            self.players_df,
            predictions,
            self.current_gw,
            horizon=5
        )
        
        self.projections_df = projection_transformer.transform()
        output_path = projection_transformer.save_parquet(self.projections_df)
        logger.info(f"  -> {len(self.projections_df)} projection rows")
        
        # Print summary stats
        self._print_projection_summary()
        
        return output_path
    
    def _print_projection_summary(self):
        """Print summary statistics for projections."""
        if self.projections_df is None:
            return
        
        df = self.projections_df
        
        # Top players by total xP
        player_totals = df.groupby('player_id')['xp'].sum().reset_index()
        player_totals = player_totals.merge(
            self.players_df[['player_id', 'web_name', 'position', 'team_name']],
            on='player_id'
        )
        top_players = player_totals.nlargest(10, 'xp')
        
        logger.info("\nTop 10 Players by 5-GW xP:")
        for _, row in top_players.iterrows():
            logger.info(f"  {row['web_name']:15} ({row['position']}) {row['team_name']:4} - {row['xp']:.1f} xP")
        
        # Per-GW averages
        gw_avgs = df.groupby('gameweek')['xp'].agg(['mean', 'max']).reset_index()
        logger.info("\nPer-GW xP Summary:")
        for _, row in gw_avgs.iterrows():
            logger.info(f"  GW{row['gameweek']}: avg={row['mean']:.2f}, max={row['max']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='FPL ETL Pipeline - Build the Data Warehouse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m etl.pipeline                  # Full pipeline
  python -m etl.pipeline --fetch-only     # Just fetch raw data
  python -m etl.pipeline --transform-only # Transform existing raw data
  python -m etl.pipeline --adapter-only   # Generate projections only
        """
    )
    
    parser.add_argument('--fetch-only', action='store_true',
                        help='Only fetch raw data, skip transforms')
    parser.add_argument('--transform-only', action='store_true',
                        help='Only run transforms on existing raw data')
    parser.add_argument('--adapter-only', action='store_true',
                        help='Only generate projections from existing parquet')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Number of gameweeks to project (default: 5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    pipeline = ETLPipeline()
    
    if args.fetch_only:
        pipeline.run(fetch=True, transform=False, generate_projections=False)
    elif args.transform_only:
        pipeline.run(fetch=False, transform=True, generate_projections=False)
    elif args.adapter_only:
        pipeline.run(fetch=False, transform=False, generate_projections=True)
    else:
        # Full pipeline
        pipeline.run(fetch=True, transform=True, generate_projections=True)


if __name__ == '__main__':
    main()

