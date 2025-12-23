"""FPL Core Insights Points Predictor Module

Enhanced prediction model using the rich FPL Core Insights dataset which includes
Opta-like detailed match statistics. This provides significantly more features
than the basic FPL API data.

Key Data Sources:
- playermatchstats: Detailed per-match Opta metrics (xG, xA, shots, passes, tackles, etc.)
- player_gameweek_stats: FPL-specific GW data (points, BPS, clean sheets)
- players: Player info (position, team)

Feature Categories:
1. Expected Stats: xG, xA, xGOT per 90
2. Shot Quality: shots on target %, big chances, touches in box
3. Passing Quality: accurate passes %, key passes, chances created
4. Defensive Actions: tackles won %, interceptions, clearances, blocks
5. Physical Stats: distance covered, sprints (if available)
6. Duel Stats: aerial duels won %, ground duels won %
7. Form Indicators: rolling averages over 3, 5 GW windows
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Optional, Tuple, Any
from scipy.stats import uniform, randint
import logging
import warnings
import time

# Model artifacts directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models' / 'artifacts' / 'fpl_core'

# XGBoost import with fallback
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available, ensemble will use 3 models instead of 4")

logger = logging.getLogger(__name__)


# Hyperparameter distributions for randomized search
HYPERPARAM_DISTRIBUTIONS = {
    'hgb': {
        'max_iter': randint(100, 300),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.02, 0.15),
        'min_samples_leaf': randint(5, 25),
        'l2_regularization': uniform(0.01, 0.5),
    },
    'rf': {
        'n_estimators': randint(80, 250),
        'max_depth': randint(4, 10),
        'min_samples_split': randint(5, 20),
        'min_samples_leaf': randint(3, 15),
    },
    'xgb': {
        'n_estimators': randint(80, 250),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.02, 0.15),
        'min_child_weight': randint(3, 15),
        'subsample': uniform(0.6, 0.35),
        'colsample_bytree': uniform(0.6, 0.35),
        'reg_alpha': uniform(0.01, 0.5),
        'reg_lambda': uniform(0.5, 2.0),
    },
    'ridge': {
        'alpha': uniform(0.1, 10.0),
    },
}


class FPLCorePredictor:
    """Enhanced FPL Points Predictor using FPL Core Insights data.
    
    Uses 40+ features from Opta-like match statistics for superior predictions.
    Trains on ALL players (700+) for maximum model accuracy.
    """

    # Feature columns grouped by category
    ATTACKING_FEATURES = [
        'avg_xg_per_90',
        'avg_xa_per_90',
        'avg_xgot_per_90',
        'avg_shots_per_90',
        'avg_shots_on_target_pct',
        'avg_big_chances_per_90',
        'avg_touches_box_per_90',
        'avg_chances_created_per_90',
    ]
    
    PASSING_FEATURES = [
        'avg_accurate_passes_pct',
        'avg_accurate_crosses_pct',
        'avg_accurate_long_balls_pct',
        'avg_final_third_passes_per_90',
    ]
    
    DEFENSIVE_FEATURES = [
        'avg_tackles_per_90',
        'avg_tackles_won_pct',
        'avg_interceptions_per_90',
        'avg_clearances_per_90',
        'avg_blocks_per_90',
        'avg_recoveries_per_90',
        'avg_def_contributions_per_90',
    ]
    
    DUEL_FEATURES = [
        'avg_aerial_duels_won_pct',
        'avg_ground_duels_won_pct',
        'avg_duels_won_per_90',
    ]
    
    DRIBBLING_FEATURES = [
        'avg_successful_dribbles_pct',
        'avg_dispossessed_per_90',
    ]
    
    GK_FEATURES = [
        'avg_saves_per_90',
        'avg_saves_inside_box_per_90',
        'avg_goals_prevented_per_90',
    ]
    
    # FPL features - removed avg_points_per_90 and avg_bonus_per_90 (target-derived leakage)
    FPL_FEATURES = [
        'avg_bps_per_90',  # BPS is NOT target leakage - it's a component that explains bonus
        'avg_ict_per_90',
        'avg_influence_per_90',
        'avg_creativity_per_90',
        'avg_threat_per_90',
    ]
    
    # Form features - replaced points-based with xGI-based (no target leakage)
    FORM_FEATURES = [
        'xgi_trend_3gw',      # xG+xA trend over last 3 GWs
        'xgi_trend_5gw',      # xG+xA trend over last 5 GWs
        'minutes_trend',
        'xgi_consistency',    # Inverse std dev of xGI (not points)
    ]
    
    # Context features - now includes real Elo-based fixture data and season indicator
    CONTEXT_FEATURES = [
        'value',
        'is_gkp',
        'is_def',
        'is_mid',
        'is_fwd',
        'fixture_difficulty',
        'is_home',
        'own_elo',
        'opponent_elo',
        'elo_diff',
        'win_prob',
        'team_attack_strength',
        'team_defense_strength',
        'opp_attack_strength',
        'opp_defense_strength',
        'season_start_year',  # For cross-season training (e.g., 2024 for 2024-25)
    ]

    # Minimum minutes threshold - only train on players with significant playing time
    MIN_MINUTES_THRESHOLD = 270  # ~3 full games minimum
    MAX_PLAYERS_TO_TRAIN = 300   # Cap players to speed up training
    
    # Position codes
    POSITIONS = ['GKP', 'DEF', 'MID', 'FWD']
    
    def __init__(self):
        """Initialize the FPL Core predictor with position-specific 4-model ensemble.
        
        Ensemble includes:
        - HistGradientBoosting: Native NaN handling, boosting for interactions
        - RandomForest: Bagging for variance reduction
        - XGBoost: Different boosting algorithm for diversity
        - Ridge: Linear model for stable baseline and diversity
        """
        # All feature columns (union for fallback model)
        self.feature_cols = (
            self.ATTACKING_FEATURES +
            self.PASSING_FEATURES +
            self.DEFENSIVE_FEATURES +
            self.DUEL_FEATURES +
            self.DRIBBLING_FEATURES +
            self.GK_FEATURES +
            self.FPL_FEATURES +
            self.FORM_FEATURES +
            self.CONTEXT_FEATURES
        )
        
        # Position-specific feature subsets (tailored to each position's role)
        self.position_feature_cols = {
            'GKP': (
                self.GK_FEATURES +           # Primary: saves, goals_prevented
                self.DEFENSIVE_FEATURES +    # Relevant: clearances, recoveries
                self.DUEL_FEATURES +         # Aerial duels matter for GKs
                self.FPL_FEATURES +
                self.FORM_FEATURES +
                self.CONTEXT_FEATURES
            ),
            'DEF': (
                self.DEFENSIVE_FEATURES +    # Primary: tackles, blocks, clearances
                self.DUEL_FEATURES +         # Aerial/ground duels crucial
                self.PASSING_FEATURES +      # Long balls, distribution
                [                            # Selective attacking (for attacking fullbacks)
                    'avg_xa_per_90',
                    'avg_chances_created_per_90',
                ] +
                self.FPL_FEATURES +
                self.FORM_FEATURES +
                self.CONTEXT_FEATURES
            ),
            'MID': (
                self.ATTACKING_FEATURES +    # Goals and assists
                self.PASSING_FEATURES +      # Chance creation, key passes
                self.DEFENSIVE_FEATURES +    # Pressing, recoveries
                self.DUEL_FEATURES +
                self.DRIBBLING_FEATURES +
                self.FPL_FEATURES +
                self.FORM_FEATURES +
                self.CONTEXT_FEATURES
            ),
            'FWD': (
                self.ATTACKING_FEATURES +    # Primary: xG, shots, big chances
                self.DRIBBLING_FEATURES +    # Taking on defenders
                [                            # Selective passing (link-up play)
                    'avg_accurate_passes_pct',
                    'avg_final_third_passes_per_90',
                ] +
                self.DUEL_FEATURES +         # Aerial duels for target men
                self.FPL_FEATURES +
                self.FORM_FEATURES +
                self.CONTEXT_FEATURES
            ),
        }
        
        # Position-specific models: each position gets its own 4-model ensemble
        # with position-tailored feature subsets
        self.position_models = {}
        for pos in self.POSITIONS:
            pos_features = self.position_feature_cols.get(pos, self.feature_cols)
            self.position_models[pos] = self._create_model_stack(feature_cols=pos_features)
        
        # Legacy single-model fallback (for when position data is unavailable)
        # Uses full feature set for maximum coverage
        self.fallback_models = self._create_model_stack(feature_cols=self.feature_cols)
        
        # Convenience aliases for backward compatibility
        self.gb_model = self.fallback_models['hgb']
        self.rf_model = self.fallback_models['rf']
        self.scaler = self.fallback_models['scaler']
        
        self.is_trained = False
        self.use_ensemble = True
        
        # Model metrics (aggregated across positions)
        self.metrics = {
            'mae': None,
            'rmse': None,
            'r2': None,
            'training_samples': 0,
            'validation_samples': 0,
            'num_features': len(self.feature_cols),
            'num_players_trained': 0,
            'by_position': {},  # Per-position metrics
            'ensemble_weights': {},  # Learned blend weights per position
        }
        
        # Cache for team strength (computed per target_gw to avoid leakage)
        self._team_strength = {}
        self._team_strength_by_gw = {}  # {gw: {team_id: {'attack': X, 'defense': Y}}}
        
        # Fixture lookup cache: {(team_id, gw): {...fixture info...}}
        self._fixture_cache = {}
        
        # Season start year used during training (for prediction context)
        self._season_start_year = 0
    
    def _reset_context_caches(self):
        """Reset all context caches between seasons.
        
        Call this between building datasets for different seasons to prevent
        cache key collisions (e.g., same (team_id, gw) across seasons).
        """
        self._team_strength = {}
        self._team_strength_by_gw = {}
        self._fixture_cache = {}
    
    def load_from_disk(self, suffix: str = None) -> bool:
        """Load pre-trained models from disk.
        
        Args:
            suffix: Model suffix to load. If None, loads the latest models.
            
        Returns:
            True if models were loaded successfully, False otherwise.
        """
        if not MODELS_DIR.exists():
            logger.info("No saved models directory found")
            return False
        
        # Find suffix to load
        if suffix is None:
            latest_path = MODELS_DIR / 'latest.json'
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    latest = json.load(f)
                    suffix = latest.get('suffix')
            else:
                logger.info("No latest.json found, cannot determine which models to load")
                return False
        
        logger.info(f"Loading pre-trained models with suffix: {suffix}")
        
        loaded_count = 0
        
        # Load position-specific models
        for pos in self.POSITIONS:
            filepath = MODELS_DIR / f'model_{pos}_{suffix}.pkl'
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    self.position_models[pos] = pickle.load(f)
                loaded_count += 1
                logger.info(f"  Loaded: {filepath.name}")
        
        # Load fallback model
        fallback_path = MODELS_DIR / f'model_fallback_{suffix}.pkl'
        if fallback_path.exists():
            with open(fallback_path, 'rb') as f:
                self.fallback_models = pickle.load(f)
            loaded_count += 1
            logger.info(f"  Loaded: {fallback_path.name}")
        
        # Load metadata
        metadata_path = MODELS_DIR / f'training_metadata_{suffix}.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self._season_start_year = metadata.get('season_start_year', 0)
                self.metrics = metadata.get('metrics', {})
                self.feature_cols = metadata.get('feature_cols', self.feature_cols)
            logger.info(f"  Loaded: {metadata_path.name}")
        
        # Mark as trained if we loaded all position models
        if loaded_count >= len(self.POSITIONS):
            self.is_trained = True
            logger.info(f"Successfully loaded {loaded_count} model files")
            
            # Log training metrics from loaded model
            if self.metrics:
                logger.info(f"  Pre-trained model metrics: MAE={self.metrics.get('mae', 'N/A')}, "
                          f"R²={self.metrics.get('r2', 'N/A')}, "
                          f"Training samples={self.metrics.get('training_samples', 'N/A')}")
                if 'cross_season' in self.metrics:
                    cs = self.metrics['cross_season']
                    logger.info(f"  Cross-season: prev={cs.get('prev_season_samples', 0)}, "
                              f"current={cs.get('current_season_samples', 0)}")
            return True
        else:
            logger.warning(f"Only loaded {loaded_count} model files, expected {len(self.POSITIONS) + 1}")
            return False
    
    def _create_model_stack(self, feature_cols: Optional[List[str]] = None) -> Dict:
        """Create a 4-model ensemble stack with per-model preprocessing.
        
        Args:
            feature_cols: Optional list of feature column names for this stack.
                         If None, will use self.feature_cols at training time.
        
        Returns:
            Dict with model objects and preprocessing pipelines.
        """
        stack = {
            # HistGradientBoosting: handles NaN natively, no scaling needed
            'hgb': HistGradientBoostingRegressor(
                max_iter=150,  # Increased for better convergence
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=10,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42
            ),
            # RandomForest: needs imputation for missing values
            'rf': RandomForestRegressor(
                n_estimators=150,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            # Ridge: linear model for diversity, needs scaling + imputation
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
            # Preprocessing components
            'imputer': SimpleImputer(strategy='median', add_indicator=True),
            'scaler': StandardScaler(),
            # Blend weights (learned on holdout set)
            'blend_weights': None,  # Will be set during training
            # Feature columns for this stack (position-specific or full)
            'feature_cols': feature_cols,
            # Status
            'is_trained': False,
            'samples': 0
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            stack['xgb'] = XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        return stack
        
    def _calculate_team_strength(self, all_gw_data: Dict, up_to_gw: Optional[int] = None) -> Dict:
        """Calculate team attack/defense strength from match results.
        
        Args:
            all_gw_data: Dict of gameweek data
            up_to_gw: Only use matches from GWs strictly before this (prevents leakage)
            
        Returns:
            Dict of team_id -> {'attack': X, 'defense': Y}
        """
        # If no GW limit, return cached global strength
        if up_to_gw is None:
            if self._team_strength:
                return self._team_strength
            up_to_gw = max(all_gw_data.keys()) + 1
        
        # Check if we already computed for this GW
        if up_to_gw in self._team_strength_by_gw:
            return self._team_strength_by_gw[up_to_gw]
            
        team_stats = {}
        
        for gw_num, gw_data in all_gw_data.items():
            if gw_num >= up_to_gw:
                continue  # Only use past matches
                
            matches = gw_data.get('matches')
            if matches is None or matches.empty:
                # Try fixtures with scores
                matches = gw_data.get('fixtures')
                if matches is None or matches.empty:
                    continue
                    
            for _, match in matches.iterrows():
                # Handle different column names
                home_id = match.get('team_h') or match.get('home_team')
                away_id = match.get('team_a') or match.get('away_team')
                home_score = match.get('team_h_score') or match.get('home_score') or 0
                away_score = match.get('team_a_score') or match.get('away_score') or 0
                
                # Skip matches with missing data
                if pd.isna(home_score) or pd.isna(away_score):
                    continue
                if pd.isna(home_id) or pd.isna(away_id):
                    continue
                    
                home_score = int(home_score)
                away_score = int(away_score)
                
                if home_id:
                    home_id = int(home_id)
                    if home_id not in team_stats:
                        team_stats[home_id] = {'scored': 0, 'conceded': 0, 'games': 0}
                    team_stats[home_id]['scored'] += home_score
                    team_stats[home_id]['conceded'] += away_score
                    team_stats[home_id]['games'] += 1
                    
                if away_id:
                    away_id = int(away_id)
                    if away_id not in team_stats:
                        team_stats[away_id] = {'scored': 0, 'conceded': 0, 'games': 0}
                    team_stats[away_id]['scored'] += away_score
                    team_stats[away_id]['conceded'] += home_score
                    team_stats[away_id]['games'] += 1
        
        result = {}
        for tid, stats in team_stats.items():
            games = max(stats['games'], 1)
            result[tid] = {
                'attack': stats['scored'] / games,
                'defense': stats['conceded'] / games
            }
        
        # Cache result
        self._team_strength_by_gw[up_to_gw] = result
        if up_to_gw == max(all_gw_data.keys()) + 1:
            self._team_strength = result
            
        return result
    
    def _build_fixture_cache(self, all_gw_data: Dict) -> None:
        """Build fixture lookup cache from all gameweek data.
        
        Populates self._fixture_cache with (team_id, gw) -> fixture_info mappings.
        """
        if self._fixture_cache:
            return
            
        for gw_num, gw_data in all_gw_data.items():
            fixtures = gw_data.get('fixtures')
            teams = gw_data.get('teams')
            
            if fixtures is None or fixtures.empty:
                continue
            
            # Build team Elo lookup for this GW
            elo_lookup = {}
            if teams is not None and not teams.empty:
                for _, team in teams.iterrows():
                    tid = team.get('id') or team.get('code')
                    if tid and 'elo' in team:
                        elo_lookup[int(tid)] = float(team['elo'])
            
            for _, fix in fixtures.iterrows():
                fix_gw = fix.get('gameweek') or gw_num
                home_id = fix.get('home_team')
                away_id = fix.get('away_team')
                
                if pd.isna(home_id) or pd.isna(away_id):
                    continue
                    
                home_id = int(home_id)
                away_id = int(away_id)
                
                # Get Elo ratings (from fixture or teams)
                home_elo = fix.get('home_team_elo') or elo_lookup.get(home_id, 1600.0)
                away_elo = fix.get('away_team_elo') or elo_lookup.get(away_id, 1600.0)
                
                if pd.isna(home_elo):
                    home_elo = 1600.0
                if pd.isna(away_elo):
                    away_elo = 1600.0
                    
                home_elo = float(home_elo)
                away_elo = float(away_elo)
                
                # Calculate win probability using Elo
                h_win, draw, a_win = self._calculate_win_probability(home_elo, away_elo)
                
                # Home team fixture
                self._fixture_cache[(home_id, int(fix_gw))] = {
                    'is_home': 1,
                    'own_elo': home_elo,
                    'opponent_elo': away_elo,
                    'elo_diff': home_elo - away_elo,
                    'win_prob': h_win,
                    'opponent_id': away_id,
                    'fixture_difficulty': self._elo_to_fdr(a_win)
                }
                
                # Away team fixture
                self._fixture_cache[(away_id, int(fix_gw))] = {
                    'is_home': 0,
                    'own_elo': away_elo,
                    'opponent_elo': home_elo,
                    'elo_diff': away_elo - home_elo,
                    'win_prob': a_win,
                    'opponent_id': home_id,
                    'fixture_difficulty': self._elo_to_fdr(h_win)
                }
    
    @staticmethod
    def _calculate_win_probability(elo_home: float, elo_away: float, 
                                   home_advantage: float = 65.0) -> Tuple[float, float, float]:
        """Calculate win/draw/loss probabilities from Elo ratings.
        
        Based on standard Elo expectation formula with home advantage.
        """
        elo_home_adj = elo_home + home_advantage
        dr = elo_away - elo_home_adj
        expected_home = 1 / (1 + 10 ** (dr / 400))
        expected_away = 1 - expected_home
        
        elo_diff = abs(elo_home_adj - elo_away)
        draw_base = 0.25 * (1 - elo_diff / 600)
        draw_prob = max(0.1, min(0.35, draw_base))
        
        home_win = expected_home * (1 - draw_prob)
        away_win = expected_away * (1 - draw_prob)
        
        total = home_win + draw_prob + away_win
        return home_win / total, draw_prob / total, away_win / total
    
    @staticmethod
    def _elo_to_fdr(loss_prob: float) -> int:
        """Convert loss probability to FDR 1-5 scale."""
        if loss_prob < 0.20:
            return 1
        elif loss_prob < 0.35:
            return 2
        elif loss_prob < 0.50:
            return 3
        elif loss_prob < 0.65:
            return 4
        else:
            return 5
    
    def _get_fixture_context(self, team_id: int, gw: int) -> Dict:
        """Get fixture context for a team in a specific gameweek.
        
        Returns:
            Dict with is_home, own_elo, opponent_elo, elo_diff, win_prob, 
            fixture_difficulty, opponent_id
        """
        key = (int(team_id), int(gw))
        if key in self._fixture_cache:
            return self._fixture_cache[key]
        
        # Return neutral defaults if fixture not found
        return {
            'is_home': 0,
            'own_elo': 1600.0,
            'opponent_elo': 1600.0,
            'elo_diff': 0.0,
            'win_prob': 0.33,
            'opponent_id': 0,
            'fixture_difficulty': 3
        }
    
    def _chronological_split(self, df: pd.DataFrame, 
                            validation_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset by gameweek with no overlap.
        
        The last N gameweeks (based on validation_split) go entirely to validation.
        Guarantees no GW appears in both train and val sets.
        """
        if 'gameweek' not in df.columns:
            # Fall back to row-based split
            split_idx = int(len(df) * (1 - validation_split))
            return df.iloc[:split_idx], df.iloc[split_idx:]
        
        unique_gws = sorted(df['gameweek'].unique())
        n_gws = len(unique_gws)
        
        if n_gws < 2:
            return df, pd.DataFrame(columns=df.columns)
        
        # Put at least 1 GW in validation, up to validation_split fraction
        n_val_gws = max(1, int(n_gws * validation_split))
        
        train_gws = set(unique_gws[:-n_val_gws])
        val_gws = set(unique_gws[-n_val_gws:])
        
        train_df = df[df['gameweek'].isin(train_gws)].copy()
        val_df = df[df['gameweek'].isin(val_gws)].copy()
        
        return train_df, val_df

    def _aggregate_match_stats(self, playermatchstats_list: List[pd.DataFrame], 
                                player_id: int, 
                                target_gw: int,
                                window: int = 4) -> Optional[Dict]:
        """Aggregate match-level stats for a player over a rolling window.
        
        Args:
            playermatchstats_list: List of (gw_num, playermatchstats_df) tuples
            player_id: Player ID to aggregate for
            target_gw: Target gameweek (stats from GWs before this)
            window: Number of gameweeks to look back
            
        Returns:
            Dictionary of aggregated features or None if insufficient data
        """
        # Collect match stats from the window
        player_matches = []
        
        for gw_num, pms_df in playermatchstats_list:
            if gw_num >= target_gw:
                continue
            if gw_num < target_gw - window:
                continue
            if pms_df is None or pms_df.empty:
                continue
                
            player_data = pms_df[pms_df['player_id'] == player_id]
            if not player_data.empty:
                row = player_data.iloc[0].to_dict()
                row['gw'] = gw_num
                player_matches.append(row)
        
        if len(player_matches) < 1:
            return None
            
        df = pd.DataFrame(player_matches)
        total_minutes = df['minutes_played'].sum()
        
        if total_minutes < 45:  # Need at least 45 mins of data
            return None
            
        minutes_90 = total_minutes / 90.0
        
        def safe_sum(col):
            if col in df.columns:
                return pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
            return 0
            
        def safe_mean_pct(col):
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                return vals.mean() if len(vals) > 0 else 0
            return 0
        
        stats = {
            # Attacking per 90
            'avg_xg_per_90': safe_sum('xg') / minutes_90,
            'avg_xa_per_90': safe_sum('xa') / minutes_90,
            'avg_xgot_per_90': safe_sum('xgot') / minutes_90,
            'avg_shots_per_90': safe_sum('total_shots') / minutes_90,
            'avg_shots_on_target_pct': safe_mean_pct('shots_on_target') / max(safe_sum('total_shots'), 1) * 100 if safe_sum('total_shots') > 0 else 0,
            'avg_big_chances_per_90': safe_sum('big_chances_missed') / minutes_90,  # Approximation
            'avg_touches_box_per_90': safe_sum('touches_opposition_box') / minutes_90,
            'avg_chances_created_per_90': safe_sum('chances_created') / minutes_90,
            
            # Passing
            'avg_accurate_passes_pct': safe_mean_pct('accurate_passes_percent'),
            'avg_accurate_crosses_pct': safe_mean_pct('accurate_crosses_percent'),
            'avg_accurate_long_balls_pct': safe_mean_pct('accurate_long_balls_percent'),
            'avg_final_third_passes_per_90': safe_sum('final_third_passes') / minutes_90,
            
            # Defensive
            'avg_tackles_per_90': safe_sum('tackles') / minutes_90,
            'avg_tackles_won_pct': safe_mean_pct('tackles_won_percent'),
            'avg_interceptions_per_90': safe_sum('interceptions') / minutes_90,
            'avg_clearances_per_90': safe_sum('clearances') / minutes_90,
            'avg_blocks_per_90': safe_sum('blocks') / minutes_90,
            'avg_recoveries_per_90': safe_sum('recoveries') / minutes_90,
            'avg_def_contributions_per_90': safe_sum('defensive_contributions') / minutes_90,
            
            # Duels
            'avg_aerial_duels_won_pct': safe_mean_pct('aerial_duels_won_percent'),
            'avg_ground_duels_won_pct': safe_mean_pct('ground_duels_won_percent'),
            'avg_duels_won_per_90': safe_sum('duels_won') / minutes_90,
            
            # Dribbling
            'avg_successful_dribbles_pct': safe_mean_pct('successful_dribbles_percent'),
            'avg_dispossessed_per_90': safe_sum('dispossessed') / minutes_90,
            
            # GK
            'avg_saves_per_90': safe_sum('saves') / minutes_90,
            'avg_saves_inside_box_per_90': safe_sum('saves_inside_box') / minutes_90,
            'avg_goals_prevented_per_90': safe_sum('goals_prevented') / minutes_90,
        }
        
        return stats

    def _aggregate_gw_stats(self, player_gw_stats_list: List[Tuple[int, pd.DataFrame]],
                            player_id: int,
                            target_gw: int,
                            window: int = 4,
                            playermatchstats_list: Optional[List[Tuple[int, pd.DataFrame]]] = None
                            ) -> Optional[Dict]:
        """Aggregate gameweek-level FPL stats for a player.
        
        Uses xGI-based form indicators instead of points-based (no target leakage).
        """
        player_gws = []
        
        for gw_num, pgs_df in player_gw_stats_list:
            if gw_num >= target_gw:
                continue
            if gw_num < target_gw - window:
                continue
            if pgs_df is None or pgs_df.empty:
                continue
                
            player_data = pgs_df[pgs_df['id'] == player_id]
            if not player_data.empty:
                row = player_data.iloc[0].to_dict()
                row['gw'] = gw_num
                player_gws.append(row)
        
        if len(player_gws) < 1:
            return None
            
        df = pd.DataFrame(player_gws)
        total_minutes = pd.to_numeric(df['minutes'], errors='coerce').fillna(0).sum()
        
        if total_minutes < 45:
            return None
            
        minutes_90 = total_minutes / 90.0
        
        def safe_sum(col):
            if col in df.columns:
                return pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
            return 0
            
        def safe_mean(col):
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                return vals.mean() if len(vals) > 0 else 0
            return 0
        
        # FPL stats - BPS is NOT target leakage (it explains bonus, doesn't derive from points)
        stats = {
            'avg_bps_per_90': safe_sum('bps') / minutes_90,
            'avg_ict_per_90': safe_mean('ict_index'),
            'avg_influence_per_90': safe_mean('influence'),
            'avg_creativity_per_90': safe_mean('creativity'),
            'avg_threat_per_90': safe_mean('threat'),
        }
        
        # Get xGI data from playermatchstats if available
        xgi_per_gw = []
        if playermatchstats_list:
            for gw_num, pms_df in playermatchstats_list:
                if gw_num >= target_gw:
                    continue
                if gw_num < target_gw - window:
                    continue
                if pms_df is None or pms_df.empty:
                    continue
                player_data = pms_df[pms_df['player_id'] == player_id]
                if not player_data.empty:
                    row = player_data.iloc[0]
                    xg = pd.to_numeric(row.get('xg', 0), errors='coerce') or 0
                    xa = pd.to_numeric(row.get('xa', 0), errors='coerce') or 0
                    xgi_per_gw.append({'gw': gw_num, 'xgi': xg + xa})
        
        # Fall back to expected_goal_involvements from player_gameweek_stats if available
        if not xgi_per_gw and 'expected_goal_involvements' in df.columns:
            for _, row in df.iterrows():
                xgi = pd.to_numeric(row.get('expected_goal_involvements', 0), errors='coerce') or 0
                xgi_per_gw.append({'gw': row['gw'], 'xgi': xgi})
        
        # xGI-based form trends (replaces points-based to avoid leakage)
        if len(xgi_per_gw) >= 3:
            xgi_df = pd.DataFrame(xgi_per_gw).sort_values('gw')
            recent = xgi_df.tail(2)['xgi'].mean()
            older = xgi_df.head(len(xgi_df)-2)['xgi'].mean() if len(xgi_df) > 2 else recent
            stats['xgi_trend_3gw'] = recent - older
        else:
            stats['xgi_trend_3gw'] = 0
            
        if len(xgi_per_gw) >= 5:
            xgi_df = pd.DataFrame(xgi_per_gw).sort_values('gw')
            recent = xgi_df.tail(3)['xgi'].mean()
            older = xgi_df.head(len(xgi_df)-3)['xgi'].mean() if len(xgi_df) > 3 else recent
            stats['xgi_trend_5gw'] = recent - older
        else:
            stats['xgi_trend_5gw'] = stats.get('xgi_trend_3gw', 0)
        
        # Minutes trend
        if len(df) >= 2:
            recent_mins = df.tail(1)['minutes'].astype(float).mean()
            older_mins = df.head(len(df)-1)['minutes'].astype(float).mean()
            stats['minutes_trend'] = (recent_mins - older_mins) / max(older_mins, 1)
        else:
            stats['minutes_trend'] = 0
            
        # xGI consistency (replaces points-based to avoid leakage)
        if len(xgi_per_gw) >= 2:
            xgi_vals = [x['xgi'] for x in xgi_per_gw]
            xgi_std = np.std(xgi_vals)
            stats['xgi_consistency'] = 1 / (1 + xgi_std) if xgi_std > 0 else 1
        else:
            stats['xgi_consistency'] = 0.5  # Unknown consistency
        
        return stats

    def build_training_dataset(self, 
                               all_gw_data: Dict,
                               fpl_core_season_data: Dict,
                               current_gw: int,
                               min_gw: int = 5,
                               max_players_to_train: Optional[int] = None,
                               season_start_year: Optional[int] = None) -> pd.DataFrame:
        """Build training dataset from FPL Core Insights data.
        
        Args:
            all_gw_data: Dict of {gw_num: {dataset_name: DataFrame}}
            fpl_core_season_data: Season-level data (players, playerstats)
            current_gw: Current gameweek
            min_gw: Minimum gameweek to start training from (need history)
            max_players_to_train: Cap on players to include. None = no cap,
                otherwise uses this value (defaults to class constant if not specified).
            season_start_year: Start year of the season (e.g., 2024 for 2024-25).
                Used as a feature to help the model distinguish between seasons.
            
        Returns:
            Training DataFrame with features and target
        """
        # Resolve max players cap
        effective_max_players = (
            max_players_to_train if max_players_to_train is not None 
            else self.MAX_PLAYERS_TO_TRAIN
        )
        
        logger.info(f"Building training data from GW{min_gw} to GW{current_gw}...")
        
        # Build fixture cache for Elo-based fixture context
        self._build_fixture_cache(all_gw_data)
        
        # Get player info
        players_df = fpl_core_season_data.get('players')
        if players_df is None or players_df.empty:
            logger.error("No players data available")
            return pd.DataFrame()
        
        # Build lists for aggregation
        playermatchstats_list = []
        player_gw_stats_list = []
        
        for gw_num in range(1, current_gw + 1):
            gw_data = all_gw_data.get(gw_num, {})
            pms = gw_data.get('playermatchstats')
            pgs = gw_data.get('player_gameweek_stats')
            
            if pms is not None and not pms.empty:
                playermatchstats_list.append((gw_num, pms))
            if pgs is not None and not pgs.empty:
                player_gw_stats_list.append((gw_num, pgs))
        
        # Calculate total minutes per player to filter low-minute players
        player_minutes = {}
        for gw_num, pgs in player_gw_stats_list:
            for _, row in pgs.iterrows():
                pid = row.get('id')
                mins = pd.to_numeric(row.get('minutes', 0), errors='coerce') or 0
                player_minutes[pid] = player_minutes.get(pid, 0) + mins
        
        # Filter to players with significant minutes only
        qualified_players = {
            pid for pid, mins in player_minutes.items() 
            if mins >= self.MIN_MINUTES_THRESHOLD
        }
        
        # Further limit to top players by minutes if cap is set
        if effective_max_players is not None and len(qualified_players) > effective_max_players:
            sorted_by_mins = sorted(player_minutes.items(), key=lambda x: x[1], reverse=True)
            qualified_players = {pid for pid, _ in sorted_by_mins[:effective_max_players]}
        
        logger.info(f"Training on {len(qualified_players)} players with {self.MIN_MINUTES_THRESHOLD}+ minutes")
        
        # Position mapping
        pos_map = {'Goalkeeper': 'GKP', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}
        
        data = []
        players_included = set()
        
        for target_gw in range(min_gw, current_gw + 1):
            # Calculate team strength only using matches before target GW (no leakage)
            team_strength = self._calculate_team_strength(all_gw_data, up_to_gw=target_gw)
            
            # Get target points for this GW
            target_gw_data = all_gw_data.get(target_gw, {})
            target_pgs = target_gw_data.get('player_gameweek_stats')
            
            if target_pgs is None or target_pgs.empty:
                continue
                
            for player_id in qualified_players:
                # Get target points
                player_target = target_pgs[target_pgs['id'] == player_id]
                if player_target.empty:
                    continue
                    
                target_points = player_target.iloc[0].get('event_points', 0)
                if pd.isna(target_points):
                    target_points = 0
                    
                # Get player info
                player_info = players_df[players_df['player_id'] == player_id]
                if player_info.empty:
                    # Try from GW stats
                    position = 'MID'
                    value = 5.0
                    team_id = 0
                else:
                    position = pos_map.get(player_info.iloc[0].get('position', 'Midfielder'), 'MID')
                    value = 5.0  # Will get from GW stats
                    team_id = player_info.iloc[0].get('team_code', 0)
                
                # Get value from GW stats
                gw_info = player_target.iloc[0]
                value = float(gw_info.get('now_cost', 50) or 50) / 10.0
                team_id = int(team_id) if team_id else 0
                
                # Aggregate features
                match_stats = self._aggregate_match_stats(
                    playermatchstats_list, player_id, target_gw, window=4
                )
                
                gw_stats = self._aggregate_gw_stats(
                    player_gw_stats_list, player_id, target_gw, window=4,
                    playermatchstats_list=playermatchstats_list  # For xGI-based form
                )
                
                if match_stats is None and gw_stats is None:
                    continue
                    
                # Combine features
                row = {}
                if match_stats:
                    row.update(match_stats)
                if gw_stats:
                    row.update(gw_stats)
                
                # Add context features
                row['value'] = value
                row['is_gkp'] = 1 if position == 'GKP' else 0
                row['is_def'] = 1 if position == 'DEF' else 0
                row['is_mid'] = 1 if position == 'MID' else 0
                row['is_fwd'] = 1 if position == 'FWD' else 0
                
                # Real fixture context from Elo data (not hardcoded)
                fixture_ctx = self._get_fixture_context(team_id, target_gw)
                row['fixture_difficulty'] = fixture_ctx['fixture_difficulty']
                row['is_home'] = fixture_ctx['is_home']
                row['own_elo'] = fixture_ctx['own_elo']
                row['opponent_elo'] = fixture_ctx['opponent_elo']
                row['elo_diff'] = fixture_ctx['elo_diff']
                row['win_prob'] = fixture_ctx['win_prob']
                
                # Team strength (only using past matches to avoid leakage)
                team_str = team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
                row['team_attack_strength'] = team_str['attack']
                row['team_defense_strength'] = team_str['defense']
                
                # Opponent strength
                opp_id = fixture_ctx['opponent_id']
                opp_str = team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
                row['opp_attack_strength'] = opp_str['attack']
                row['opp_defense_strength'] = opp_str['defense']
                
                # Season identifier (for cross-season training)
                row['season_start_year'] = season_start_year if season_start_year else 0
                
                # Target
                row['target_points'] = float(target_points)
                row['player_id'] = player_id
                row['gameweek'] = target_gw
                
                data.append(row)
                players_included.add(player_id)
        
        logger.info(f"Built {len(data)} training samples from {len(players_included)} players")
        self.metrics['num_players_trained'] = len(players_included)
        
        return pd.DataFrame(data)

    def build_cross_season_training_dataset(
        self,
        prev_all_gw_data: Dict,
        prev_fpl_core_season_data: Dict,
        prev_season_start_year: int,
        current_all_gw_data: Dict,
        current_fpl_core_season_data: Dict,
        current_season_start_year: int,
        current_gw: int,
        min_gw: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build combined training dataset from multiple seasons.
        
        Filters previous-season samples to players still active in the current season
        using stable player_code matching.
        
        Args:
            prev_all_gw_data: Previous season gameweek data (e.g., 2024-25).
            prev_fpl_core_season_data: Previous season-level data.
            prev_season_start_year: Start year of previous season (e.g., 2024).
            current_all_gw_data: Current season gameweek data (e.g., 2025-26).
            current_fpl_core_season_data: Current season-level data.
            current_season_start_year: Start year of current season (e.g., 2025).
            current_gw: Current gameweek in the current season.
            min_gw: Minimum gameweek to start from (need history).
            
        Returns:
            Tuple of (prev_season_df, current_season_df) where:
            - prev_season_df: Previous season samples filtered to active players.
            - current_season_df: Current season samples up to current_gw.
        """
        logger.info("Building cross-season training dataset...")
        
        # Get player_code mapping from both seasons
        prev_players = prev_fpl_core_season_data.get('players')
        current_players = current_fpl_core_season_data.get('players')
        
        if prev_players is None or current_players is None:
            logger.error("Players data missing from one or both seasons")
            return pd.DataFrame(), pd.DataFrame()
        
        # Build player_code -> player_id mapping for previous season
        prev_code_to_id = {}
        if 'player_code' in prev_players.columns and 'player_id' in prev_players.columns:
            for _, row in prev_players.iterrows():
                code = row.get('player_code')
                pid = row.get('player_id')
                if pd.notna(code) and pd.notna(pid):
                    prev_code_to_id[int(code)] = int(pid)
        
        # Get set of active player_codes in current season
        current_player_codes = set()
        if 'player_code' in current_players.columns:
            current_player_codes = set(
                current_players['player_code'].dropna().astype(int).tolist()
            )
        
        logger.info(f"Previous season: {len(prev_code_to_id)} players with codes")
        logger.info(f"Current season: {len(current_player_codes)} active player codes")
        
        # Reset caches before building previous season dataset
        self._reset_context_caches()
        
        # Build previous season dataset (full season, GW5 to GW38)
        prev_max_gw = max(prev_all_gw_data.keys()) if prev_all_gw_data else 38
        logger.info(f"Building previous season dataset (GW{min_gw} to GW{prev_max_gw})...")
        
        prev_df = self.build_training_dataset(
            prev_all_gw_data,
            prev_fpl_core_season_data,
            current_gw=prev_max_gw,
            min_gw=min_gw,
            max_players_to_train=None,  # No cap for cross-season
            season_start_year=prev_season_start_year
        )
        
        # Filter previous season to players active in current season
        if not prev_df.empty and current_player_codes:
            # Map previous season player_id to player_code
            prev_id_to_code = {v: k for k, v in prev_code_to_id.items()}
            prev_df['_player_code'] = prev_df['player_id'].map(prev_id_to_code)
            
            # Filter to active players
            active_mask = prev_df['_player_code'].isin(current_player_codes)
            filtered_prev_df = prev_df[active_mask].drop(columns=['_player_code']).copy()
            
            excluded_count = len(prev_df) - len(filtered_prev_df)
            logger.info(f"Filtered previous season: {len(filtered_prev_df)} samples "
                       f"(excluded {excluded_count} from inactive players)")
            prev_df = filtered_prev_df
        
        # Reset caches before building current season dataset
        self._reset_context_caches()
        
        # Build current season dataset
        logger.info(f"Building current season dataset (GW{min_gw} to GW{current_gw})...")
        
        current_df = self.build_training_dataset(
            current_all_gw_data,
            current_fpl_core_season_data,
            current_gw=current_gw,
            min_gw=min_gw,
            max_players_to_train=None,  # No cap for cross-season
            season_start_year=current_season_start_year
        )
        
        logger.info(f"Cross-season dataset: {len(prev_df)} prev + {len(current_df)} current samples")
        
        return prev_df, current_df

    def _get_position_from_row(self, row: Dict) -> str:
        """Determine position from row features."""
        if row.get('is_gkp', 0) == 1:
            return 'GKP'
        elif row.get('is_def', 0) == 1:
            return 'DEF'
        elif row.get('is_mid', 0) == 1:
            return 'MID'
        elif row.get('is_fwd', 0) == 1:
            return 'FWD'
        return 'MID'  # Default
    
    def train(self, all_gw_data: Dict, fpl_core_season_data: Dict, 
              current_gw: int, validation_split: float = 0.2,
              tune: bool = False, tune_n_iter: int = 20, tune_time_budget: float = 60.0,
              recency_weighting: bool = True, recency_half_life_gws: float = 6.0):
        """Train position-specific 4-model ensembles on FPL Core Insights data.
        
        Each position gets:
        - HistGradientBoosting (handles NaN natively)
        - RandomForest (imputed data)
        - XGBoost (imputed data, if available)
        - Ridge (imputed + scaled data)
        
        Blend weights are learned on the validation set.
        
        Args:
            all_gw_data: Dict of gameweek data
            fpl_core_season_data: Season-level data
            current_gw: Current gameweek
            validation_split: Fraction for validation
            tune: If True, perform hyperparameter tuning before training.
            tune_n_iter: Number of hyperparameter combinations to try per model.
            tune_time_budget: Maximum time in seconds for tuning per position.
            recency_weighting: If True, weight recent samples more heavily.
            recency_half_life_gws: Half-life in gameweeks for exponential decay.
                A sample from recency_half_life_gws ago gets weight 0.5.
        """
        df = self.build_training_dataset(all_gw_data, fpl_core_season_data, current_gw)
        
        if df.empty:
            logger.error("No training data available")
            return
        
        # Add position column for splitting
        df['position'] = df.apply(self._get_position_from_row, axis=1)
            
        # Ensure all feature columns exist (keep NaN for proper handling)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Chronological split by gameweek (no overlap)
        train_df, val_df = self._chronological_split(df, validation_split)
        
        logger.info(f"Training on {len(train_df)} samples, validating on {len(val_df)}...")
        if 'gameweek' in train_df.columns and len(train_df) > 0:
            train_gws = sorted(train_df['gameweek'].unique())
            val_gws = sorted(val_df['gameweek'].unique()) if len(val_df) > 0 else []
            logger.info(f"Train GWs: {train_gws[0]}-{train_gws[-1]}, Val GWs: {val_gws if val_gws else 'none'}")
        
        # Compute recency weights for training samples
        # weight = exp(-ln(2) * age / half_life) where age = max_gw - sample_gw
        if recency_weighting and 'gameweek' in train_df.columns and len(train_df) > 0:
            max_train_gw = train_df['gameweek'].max()
            age = max_train_gw - train_df['gameweek']
            decay_rate = np.log(2) / recency_half_life_gws
            train_df = train_df.copy()
            train_df['_sample_weight'] = np.exp(-decay_rate * age)
            logger.info(f"Recency weighting enabled: half-life={recency_half_life_gws} GWs")
        else:
            train_df = train_df.copy()
            train_df['_sample_weight'] = 1.0
        
        # Perform hyperparameter tuning if requested
        tuned_params_by_pos = {}
        if tune:
            logger.info("Performing hyperparameter tuning...")
            for pos in self.POSITIONS:
                tuned = self.tune_hyperparameters(
                    train_df, pos, 
                    n_iter=tune_n_iter, 
                    time_budget=tune_time_budget
                )
                if tuned.get('params'):
                    tuned_params_by_pos[pos] = tuned['params']
                    self.metrics['by_position'].setdefault(pos, {})['tuned_params'] = tuned['params']
        
        # Train position-specific models
        all_y_val = []
        all_y_pred = []
        total_train_samples = 0
        total_val_samples = 0
        
        for pos in self.POSITIONS:
            pos_train = train_df[train_df['position'] == pos]
            pos_val = val_df[val_df['position'] == pos] if len(val_df) > 0 else pd.DataFrame()
            
            if len(pos_train) < 5:
                logger.info(f"  {pos}: Skipped (only {len(pos_train)} samples)")
                continue
            
            model_data = self.position_models[pos]
            
            # Use position-specific feature columns
            pos_feature_cols = model_data.get('feature_cols') or self.feature_cols
            
            # Ensure all feature columns exist in training data
            for col in pos_feature_cols:
                if col not in pos_train.columns:
                    pos_train[col] = np.nan
            
            # Get feature matrices (keep NaN for HGB, will impute for others)
            X_train_raw = pos_train[pos_feature_cols].values
            y_train = pos_train['target_points'].values
            sample_weights = pos_train['_sample_weight'].values
            
            # Apply tuned parameters if available
            if pos in tuned_params_by_pos:
                self._apply_tuned_params(model_data, tuned_params_by_pos[pos])
            
            # Train the 4-model ensemble with recency-weighted samples
            self._train_model_stack(model_data, X_train_raw, y_train, sample_weight=sample_weights)
            model_data['samples'] = len(pos_train)
            
            total_train_samples += len(pos_train)
            
            # Validate and learn blend weights
            if len(pos_val) > 0:
                # Ensure all feature columns exist in validation data
                for col in pos_feature_cols:
                    if col not in pos_val.columns:
                        pos_val[col] = np.nan
                        
                X_val_raw = pos_val[pos_feature_cols].values
                y_val = pos_val['target_points'].values
                
                # Get individual model predictions and learn blend weights
                y_pred, blend_weights = self._validate_and_blend(model_data, X_val_raw, y_val)
                model_data['blend_weights'] = blend_weights
                
                all_y_val.extend(y_val.tolist())
                all_y_pred.extend(y_pred.tolist())
                total_val_samples += len(pos_val)
                
                # Per-position metrics
                pos_mae = round(mean_absolute_error(y_val, y_pred), 2)
                pos_r2 = round(r2_score(y_val, y_pred), 3) if len(y_val) > 1 else 0
                self.metrics['by_position'][pos] = {
                    'mae': pos_mae,
                    'r2': pos_r2,
                    'train_samples': len(pos_train),
                    'val_samples': len(pos_val),
                    'blend_weights': blend_weights
                }
                logger.info(f"  {pos}: {len(pos_train)} train, {len(pos_val)} val, MAE={pos_mae}, R²={pos_r2}")
            else:
                # Default equal weights if no validation data
                n_models = sum(1 for k in model_data if k in ['hgb', 'rf', 'xgb', 'ridge'])
                model_data['blend_weights'] = {k: 1.0/n_models for k in ['hgb', 'rf', 'xgb', 'ridge'] if k in model_data}
                logger.info(f"  {pos}: {len(pos_train)} train, 0 val")
        
        # Also train fallback global model for cases where position is unknown
        X_train_all = train_df[self.feature_cols].values
        y_train_all = train_df['target_points'].values
        sample_weights_all = train_df['_sample_weight'].values
        self._train_model_stack(self.fallback_models, X_train_all, y_train_all, sample_weight=sample_weights_all)
        
        # Learn blend weights for fallback using all validation data
        if len(val_df) > 0:
            X_val_all = val_df[self.feature_cols].values
            y_val_all = val_df['target_points'].values
            _, blend_weights = self._validate_and_blend(self.fallback_models, X_val_all, y_val_all)
            self.fallback_models['blend_weights'] = blend_weights
        else:
            n_models = sum(1 for k in self.fallback_models if k in ['hgb', 'rf', 'xgb', 'ridge'])
            self.fallback_models['blend_weights'] = {k: 1.0/n_models for k in ['hgb', 'rf', 'xgb', 'ridge'] if k in self.fallback_models}
        
        self.is_trained = True
        
        # Overall metrics
        if all_y_val:
            self.metrics['mae'] = round(mean_absolute_error(all_y_val, all_y_pred), 2)
            self.metrics['rmse'] = round(np.sqrt(mean_squared_error(all_y_val, all_y_pred)), 2)
            self.metrics['r2'] = round(r2_score(all_y_val, all_y_pred), 3)
        
        self.metrics['training_samples'] = total_train_samples
        self.metrics['validation_samples'] = total_val_samples
        
        logger.info(f"Model trained. MAE: {self.metrics['mae']}, "
                   f"RMSE: {self.metrics['rmse']}, R²: {self.metrics['r2']}")
        logger.info(f"Trained on {self.metrics['num_players_trained']} players, "
                   f"{self.metrics['num_features']} features")

    def train_cross_season(
        self,
        prev_all_gw_data: Dict,
        prev_fpl_core_season_data: Dict,
        prev_season_start_year: int,
        current_all_gw_data: Dict,
        current_fpl_core_season_data: Dict,
        current_season_start_year: int,
        current_gw: int,
        validation_split: float = 0.2,
        tune: bool = False,
        tune_n_iter: int = 20,
        tune_time_budget: float = 60.0,
        recency_weighting: bool = True,
        recency_half_life_gws: float = 6.0,
        current_season_weight: float = 2.0,
        prev_season_weight: float = 1.0,
    ):
        """Train on multiple seasons with current-season-only validation.
        
        Combines previous season data (filtered to active players) with current
        season data. Applies season weighting (current season weighted higher)
        on top of within-season recency weighting.
        
        Validation is taken ONLY from the current season to ensure we measure
        how well the model predicts current-season patterns.
        
        Args:
            prev_all_gw_data: Previous season GW data (e.g., 2024-25).
            prev_fpl_core_season_data: Previous season-level data.
            prev_season_start_year: Start year of previous season (e.g., 2024).
            current_all_gw_data: Current season GW data (e.g., 2025-26).
            current_fpl_core_season_data: Current season-level data.
            current_season_start_year: Start year of current season (e.g., 2025).
            current_gw: Current gameweek in the current season.
            validation_split: Fraction of current season GWs for validation.
            tune: If True, perform hyperparameter tuning.
            tune_n_iter: Hyperparameter combinations to try per model.
            tune_time_budget: Max time for tuning per position.
            recency_weighting: If True, apply exponential decay to older samples.
            recency_half_life_gws: Half-life for within-season recency decay.
            current_season_weight: Base weight multiplier for current season samples.
            prev_season_weight: Base weight multiplier for previous season samples.
        """
        logger.info("Training cross-season model...")
        logger.info(f"  Previous season: {prev_season_start_year}-{prev_season_start_year + 1}")
        logger.info(f"  Current season: {current_season_start_year}-{current_season_start_year + 1}")
        logger.info(f"  Season weights: prev={prev_season_weight}, current={current_season_weight}")
        
        # Store season start year for predictions
        self._season_start_year = current_season_start_year
        
        # Build cross-season dataset
        prev_df, current_df = self.build_cross_season_training_dataset(
            prev_all_gw_data=prev_all_gw_data,
            prev_fpl_core_season_data=prev_fpl_core_season_data,
            prev_season_start_year=prev_season_start_year,
            current_all_gw_data=current_all_gw_data,
            current_fpl_core_season_data=current_fpl_core_season_data,
            current_season_start_year=current_season_start_year,
            current_gw=current_gw,
            min_gw=5,
        )
        
        if prev_df.empty and current_df.empty:
            logger.error("No training data from either season")
            return
        
        # Split current season into train/val (chronological)
        if not current_df.empty:
            current_train_df, val_df = self._chronological_split(current_df, validation_split)
        else:
            current_train_df = pd.DataFrame()
            val_df = pd.DataFrame()
        
        # Combine previous season (all) with current season train portion
        train_dfs = []
        if not prev_df.empty:
            train_dfs.append(prev_df)
        if not current_train_df.empty:
            train_dfs.append(current_train_df)
        
        if not train_dfs:
            logger.error("No training data after split")
            return
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Add position column
        train_df['position'] = train_df.apply(self._get_position_from_row, axis=1)
        if not val_df.empty:
            val_df = val_df.copy()
            val_df['position'] = val_df.apply(self._get_position_from_row, axis=1)
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in train_df.columns:
                train_df[col] = np.nan
            if not val_df.empty and col not in val_df.columns:
                val_df[col] = np.nan
        
        # Compute sample weights: season_weight × within-season recency
        train_df = train_df.copy()
        
        # Determine max GW per season for recency calculation
        prev_max_gw = prev_df['gameweek'].max() if not prev_df.empty else 0
        current_train_max_gw = current_train_df['gameweek'].max() if not current_train_df.empty else 0
        
        decay_rate = np.log(2) / recency_half_life_gws if recency_weighting else 0
        
        def compute_weight(row):
            season_year = row.get('season_start_year', 0)
            gw = row.get('gameweek', 1)
            
            if season_year == current_season_start_year:
                season_mult = current_season_weight
                max_gw = current_train_max_gw
            else:
                season_mult = prev_season_weight
                max_gw = prev_max_gw
            
            if recency_weighting and max_gw > 0:
                age = max_gw - gw
                recency_mult = np.exp(-decay_rate * age)
            else:
                recency_mult = 1.0
            
            return season_mult * recency_mult
        
        train_df['_sample_weight'] = train_df.apply(compute_weight, axis=1)
        
        # Log train/val split info
        prev_train_count = len(train_df[train_df['season_start_year'] == prev_season_start_year])
        current_train_count = len(train_df[train_df['season_start_year'] == current_season_start_year])
        
        logger.info(f"Training samples: {len(train_df)} "
                   f"(prev={prev_train_count}, current={current_train_count})")
        logger.info(f"Validation samples: {len(val_df)} (current season only)")
        
        if 'gameweek' in val_df.columns and len(val_df) > 0:
            val_gws = sorted(val_df['gameweek'].unique())
            logger.info(f"Validation GWs: {val_gws}")
        
        # Perform hyperparameter tuning if requested
        tuned_params_by_pos = {}
        if tune:
            logger.info("Performing hyperparameter tuning...")
            for pos in self.POSITIONS:
                tuned = self.tune_hyperparameters(
                    train_df, pos,
                    n_iter=tune_n_iter,
                    time_budget=tune_time_budget
                )
                if tuned.get('params'):
                    tuned_params_by_pos[pos] = tuned['params']
                    self.metrics['by_position'].setdefault(pos, {})['tuned_params'] = tuned['params']
        
        # Train position-specific models
        all_y_val = []
        all_y_pred = []
        total_train_samples = 0
        total_val_samples = 0
        
        for pos in self.POSITIONS:
            pos_train = train_df[train_df['position'] == pos]
            pos_val = val_df[val_df['position'] == pos] if len(val_df) > 0 else pd.DataFrame()
            
            if len(pos_train) < 5:
                logger.info(f"  {pos}: Skipped (only {len(pos_train)} samples)")
                continue
            
            model_data = self.position_models[pos]
            pos_feature_cols = model_data.get('feature_cols') or self.feature_cols
            
            # Ensure feature columns exist
            for col in pos_feature_cols:
                if col not in pos_train.columns:
                    pos_train = pos_train.copy()
                    pos_train[col] = np.nan
            
            X_train_raw = pos_train[pos_feature_cols].values
            y_train = pos_train['target_points'].values
            sample_weights = pos_train['_sample_weight'].values
            
            # Apply tuned parameters if available
            if pos in tuned_params_by_pos:
                self._apply_tuned_params(model_data, tuned_params_by_pos[pos])
            
            # Train
            self._train_model_stack(model_data, X_train_raw, y_train, sample_weight=sample_weights)
            model_data['samples'] = len(pos_train)
            total_train_samples += len(pos_train)
            
            # Validate
            if len(pos_val) > 0:
                for col in pos_feature_cols:
                    if col not in pos_val.columns:
                        pos_val = pos_val.copy()
                        pos_val[col] = np.nan
                
                X_val_raw = pos_val[pos_feature_cols].values
                y_val = pos_val['target_points'].values
                
                y_pred, blend_weights = self._validate_and_blend(model_data, X_val_raw, y_val)
                model_data['blend_weights'] = blend_weights
                
                all_y_val.extend(y_val.tolist())
                all_y_pred.extend(y_pred.tolist())
                total_val_samples += len(pos_val)
                
                pos_mae = round(mean_absolute_error(y_val, y_pred), 2)
                pos_r2 = round(r2_score(y_val, y_pred), 3) if len(y_val) > 1 else 0
                self.metrics['by_position'][pos] = {
                    'mae': pos_mae,
                    'r2': pos_r2,
                    'train_samples': len(pos_train),
                    'val_samples': len(pos_val),
                    'blend_weights': blend_weights
                }
                logger.info(f"  {pos}: {len(pos_train)} train, {len(pos_val)} val, MAE={pos_mae}, R²={pos_r2}")
            else:
                n_models = sum(1 for k in model_data if k in ['hgb', 'rf', 'xgb', 'ridge'])
                model_data['blend_weights'] = {k: 1.0/n_models for k in ['hgb', 'rf', 'xgb', 'ridge'] if k in model_data}
                logger.info(f"  {pos}: {len(pos_train)} train, 0 val")
        
        # Train fallback model
        X_train_all = train_df[self.feature_cols].values
        y_train_all = train_df['target_points'].values
        sample_weights_all = train_df['_sample_weight'].values
        self._train_model_stack(self.fallback_models, X_train_all, y_train_all, sample_weight=sample_weights_all)
        
        if len(val_df) > 0:
            X_val_all = val_df[self.feature_cols].values
            y_val_all = val_df['target_points'].values
            _, blend_weights = self._validate_and_blend(self.fallback_models, X_val_all, y_val_all)
            self.fallback_models['blend_weights'] = blend_weights
        else:
            n_models = sum(1 for k in self.fallback_models if k in ['hgb', 'rf', 'xgb', 'ridge'])
            self.fallback_models['blend_weights'] = {k: 1.0/n_models for k in ['hgb', 'rf', 'xgb', 'ridge'] if k in self.fallback_models}
        
        self.is_trained = True
        
        # Overall metrics
        if all_y_val:
            self.metrics['mae'] = round(mean_absolute_error(all_y_val, all_y_pred), 2)
            self.metrics['rmse'] = round(np.sqrt(mean_squared_error(all_y_val, all_y_pred)), 2)
            self.metrics['r2'] = round(r2_score(all_y_val, all_y_pred), 3)
        
        self.metrics['training_samples'] = total_train_samples
        self.metrics['validation_samples'] = total_val_samples
        
        # Cross-season specific metrics
        self.metrics['cross_season'] = {
            'prev_season_start_year': prev_season_start_year,
            'current_season_start_year': current_season_start_year,
            'prev_season_samples': prev_train_count,
            'current_season_samples': current_train_count,
            'season_weights': {
                'prev': prev_season_weight,
                'current': current_season_weight,
            },
            'val_gws': sorted(val_df['gameweek'].unique().tolist()) if 'gameweek' in val_df.columns and len(val_df) > 0 else [],
        }
        
        logger.info(f"Cross-season model trained. MAE: {self.metrics['mae']}, "
                   f"RMSE: {self.metrics['rmse']}, R²: {self.metrics['r2']}")
        logger.info(f"Total training: {total_train_samples} samples, "
                   f"Validation: {total_val_samples} samples (current season only)")
    
    def _train_model_stack(self, model_data: Dict, X_train: np.ndarray, y_train: np.ndarray,
                           sample_weight: Optional[np.ndarray] = None):
        """Train all models in a model stack with appropriate preprocessing.
        
        Args:
            model_data: Dict containing models and preprocessing components.
            X_train: Raw feature matrix (may contain NaN).
            y_train: Target values.
            sample_weight: Optional sample weights for training (e.g., recency weights).
        """
        # Fit imputer for models that need it (RF, XGB, Ridge)
        X_imputed = model_data['imputer'].fit_transform(X_train)
        
        # Fit scaler for Ridge (on imputed data)
        X_scaled = model_data['scaler'].fit_transform(X_imputed)
        
        # Train HistGradientBoosting on raw data (handles NaN natively)
        # Replace infinity with NaN first
        X_hgb = np.where(np.isinf(X_train), np.nan, X_train)
        model_data['hgb'].fit(X_hgb, y_train, sample_weight=sample_weight)
        
        # Train RandomForest on imputed data
        model_data['rf'].fit(X_imputed, y_train, sample_weight=sample_weight)
        
        # Train XGBoost on imputed data (if available)
        if 'xgb' in model_data:
            model_data['xgb'].fit(X_imputed, y_train, sample_weight=sample_weight)
        
        # Train Ridge on scaled + imputed data
        model_data['ridge'].fit(X_scaled, y_train, sample_weight=sample_weight)
        
        model_data['is_trained'] = True
    
    def _validate_and_blend(self, model_data: Dict, X_val: np.ndarray, 
                            y_val: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get predictions from all models and learn optimal blend weights.
        
        Uses simple OLS on the holdout to learn weights (stacking).
        
        Args:
            model_data: Dict containing trained models.
            X_val: Raw validation features.
            y_val: Validation targets.
            
        Returns:
            Tuple of (blended predictions, blend weights dict).
        """
        # Preprocess validation data
        X_imputed = model_data['imputer'].transform(X_val)
        X_scaled = model_data['scaler'].transform(X_imputed)
        X_hgb = np.where(np.isinf(X_val), np.nan, X_val)
        
        # Get predictions from each model
        preds = {}
        preds['hgb'] = model_data['hgb'].predict(X_hgb)
        preds['rf'] = model_data['rf'].predict(X_imputed)
        preds['ridge'] = model_data['ridge'].predict(X_scaled)
        
        if 'xgb' in model_data:
            preds['xgb'] = model_data['xgb'].predict(X_imputed)
        
        # Stack predictions and learn weights using Ridge regression
        model_keys = list(preds.keys())
        pred_matrix = np.column_stack([preds[k] for k in model_keys])
        
        # Use Ridge to learn blend weights (constrained to be positive via clipping)
        blender = Ridge(alpha=0.1, fit_intercept=False, positive=True)
        blender.fit(pred_matrix, y_val)
        
        # Normalize weights to sum to 1
        raw_weights = np.maximum(blender.coef_, 0)
        weight_sum = raw_weights.sum()
        if weight_sum > 0:
            normalized_weights = raw_weights / weight_sum
        else:
            normalized_weights = np.ones(len(model_keys)) / len(model_keys)
        
        blend_weights = {k: float(w) for k, w in zip(model_keys, normalized_weights)}
        
        # Compute blended prediction
        blended_pred = sum(preds[k] * blend_weights[k] for k in model_keys)
        
        return blended_pred, blend_weights
    
    def tune_hyperparameters(self, df: pd.DataFrame, position: str,
                             n_iter: int = 20, n_cv_folds: int = 3,
                             time_budget: float = 60.0) -> Dict[str, Dict]:
        """Perform time-aware randomized hyperparameter search for a position.
        
        Uses gameweek-based time-series CV where each fold is a contiguous
        block of gameweeks, ensuring no temporal leakage.
        
        Args:
            df: Training data with 'gameweek' column.
            position: Position code ('GKP', 'DEF', 'MID', 'FWD').
            n_iter: Number of random hyperparameter combinations to try.
            n_cv_folds: Number of time-series CV folds.
            time_budget: Maximum time in seconds for tuning.
            
        Returns:
            Dict with best params for each model type and CV scores.
        """
        if df.empty or 'gameweek' not in df.columns:
            return {}
        
        pos_df = df[df['position'] == position].copy()
        if len(pos_df) < 20:
            logger.info(f"  {position}: Not enough samples for tuning ({len(pos_df)})")
            return {}
        
        # Get position-specific feature columns
        pos_feature_cols = self.position_feature_cols.get(position, self.feature_cols)
        
        # Ensure all feature columns exist
        for col in pos_feature_cols:
            if col not in pos_df.columns:
                pos_df[col] = np.nan
        
        unique_gws = sorted(pos_df['gameweek'].unique())
        n_gws = len(unique_gws)
        
        if n_gws < n_cv_folds + 1:
            logger.info(f"  {position}: Not enough GWs for CV ({n_gws} < {n_cv_folds + 1})")
            return {}
        
        # Build time-series CV folds (forward chaining)
        cv_splits = []
        fold_size = max(1, n_gws // (n_cv_folds + 1))
        
        for fold_idx in range(n_cv_folds):
            # Train on GWs 0 to (fold_idx + 1) * fold_size
            # Validate on GWs (fold_idx + 1) * fold_size to (fold_idx + 2) * fold_size
            train_end = (fold_idx + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n_gws)
            
            train_gws = set(unique_gws[:train_end])
            val_gws = set(unique_gws[val_start:val_end])
            
            if not train_gws or not val_gws:
                continue
            
            train_idx = pos_df['gameweek'].isin(train_gws)
            val_idx = pos_df['gameweek'].isin(val_gws)
            
            cv_splits.append((
                pos_df[train_idx][pos_feature_cols].values,
                pos_df[train_idx]['target_points'].values,
                pos_df[val_idx][pos_feature_cols].values,
                pos_df[val_idx]['target_points'].values
            ))
        
        if not cv_splits:
            return {}
        
        logger.info(f"  {position}: Tuning with {len(cv_splits)} CV folds, {n_iter} iterations")
        
        start_time = time.time()
        best_params = {}
        best_scores = {}
        
        # Tune each model type separately
        for model_type in ['hgb', 'rf', 'ridge'] + (['xgb'] if XGBOOST_AVAILABLE else []):
            if time.time() - start_time > time_budget:
                logger.info(f"  {position}: Time budget exceeded, stopping tuning")
                break
            
            param_dist = HYPERPARAM_DISTRIBUTIONS.get(model_type, {})
            if not param_dist:
                continue
            
            param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
            
            best_score = float('inf')
            best_param = None
            
            for params in param_list:
                if time.time() - start_time > time_budget:
                    break
                
                # Evaluate on CV folds
                fold_scores = []
                for X_train, y_train, X_val, y_val in cv_splits:
                    try:
                        model = self._create_model_with_params(model_type, params)
                        
                        # Preprocess
                        imputer = SimpleImputer(strategy='median')
                        X_train_imp = imputer.fit_transform(X_train)
                        X_val_imp = imputer.transform(X_val)
                        
                        if model_type == 'ridge':
                            scaler = StandardScaler()
                            X_train_imp = scaler.fit_transform(X_train_imp)
                            X_val_imp = scaler.transform(X_val_imp)
                        
                        if model_type == 'hgb':
                            # HGB handles NaN natively
                            X_train_use = np.where(np.isinf(X_train), np.nan, X_train)
                            X_val_use = np.where(np.isinf(X_val), np.nan, X_val)
                        else:
                            X_train_use = X_train_imp
                            X_val_use = X_val_imp
                        
                        model.fit(X_train_use, y_train)
                        preds = model.predict(X_val_use)
                        mae = mean_absolute_error(y_val, preds)
                        fold_scores.append(mae)
                        
                    except Exception as e:
                        logger.debug(f"  {model_type} failed with params {params}: {e}")
                        fold_scores.append(float('inf'))
                
                avg_mae = np.mean(fold_scores)
                if avg_mae < best_score:
                    best_score = avg_mae
                    best_param = params.copy()
            
            if best_param is not None:
                best_params[model_type] = best_param
                best_scores[model_type] = best_score
                logger.info(f"    {model_type}: Best MAE={best_score:.3f}")
        
        return {'params': best_params, 'scores': best_scores}
    
    def _create_model_with_params(self, model_type: str, params: Dict) -> Any:
        """Create a model instance with specified hyperparameters.
        
        Args:
            model_type: One of 'hgb', 'rf', 'xgb', 'ridge'.
            params: Dict of hyperparameters.
            
        Returns:
            Instantiated model.
        """
        if model_type == 'hgb':
            return HistGradientBoostingRegressor(
                max_iter=params.get('max_iter', 150),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.05),
                min_samples_leaf=params.get('min_samples_leaf', 10),
                l2_regularization=params.get('l2_regularization', 0.1),
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42
            )
        elif model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', 6),
                min_samples_split=params.get('min_samples_split', 10),
                min_samples_leaf=params.get('min_samples_leaf', 5),
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgb' and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.05),
                min_child_weight=params.get('min_child_weight', 5),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 1.0),
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        elif model_type == 'ridge':
            return Ridge(
                alpha=params.get('alpha', 1.0),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _apply_tuned_params(self, model_data: Dict, tuned_params: Dict):
        """Apply tuned hyperparameters to a model stack.
        
        Args:
            model_data: Model stack dict.
            tuned_params: Dict of {model_type: params}.
        """
        for model_type, params in tuned_params.items():
            if model_type in model_data:
                model_data[model_type] = self._create_model_with_params(model_type, params)
    
    def _predict_with_stack(self, model_data: Dict, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model stack with learned blend weights.
        
        Args:
            model_data: Dict containing trained models and blend weights.
            X: Raw feature matrix.
            
        Returns:
            Blended predictions.
        """
        if not model_data.get('is_trained', False):
            return np.zeros(len(X))
        
        # Preprocess
        X_imputed = model_data['imputer'].transform(X)
        X_scaled = model_data['scaler'].transform(X_imputed)
        X_hgb = np.where(np.isinf(X), np.nan, X)
        
        # Get predictions from each model
        preds = {}
        preds['hgb'] = model_data['hgb'].predict(X_hgb)
        preds['rf'] = model_data['rf'].predict(X_imputed)
        preds['ridge'] = model_data['ridge'].predict(X_scaled)
        
        if 'xgb' in model_data:
            preds['xgb'] = model_data['xgb'].predict(X_imputed)
        
        # Apply blend weights
        blend_weights = model_data.get('blend_weights', {})
        if not blend_weights:
            # Default to equal weights
            blend_weights = {k: 1.0/len(preds) for k in preds}
        
        blended = sum(preds[k] * blend_weights.get(k, 0) for k in preds)
        return blended

    def predict(self, all_gw_data: Dict, fpl_core_season_data: Dict,
                player_ids: List[int], current_gw: int) -> Dict[int, float]:
        """Predict points for next gameweek using position-specific models.
        
        Args:
            all_gw_data: Gameweek data
            fpl_core_season_data: Season data
            player_ids: Players to predict for
            current_gw: Current gameweek
            
        Returns:
            Dict of player_id -> predicted points
        """
        if not self.is_trained:
            logger.warning("Model not trained")
            return {}
        
        # Build fixture cache if needed (for next GW predictions) with API fallback
        self._build_fixture_cache_with_fallback(all_gw_data, fpl_core_season_data, current_gw, num_gws=1)
        
        # Calculate team strength up to current GW
        team_strength = self._calculate_team_strength(all_gw_data, up_to_gw=current_gw + 1)
        
        players_df = fpl_core_season_data.get('players')
        pos_map = {'Goalkeeper': 'GKP', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}
        
        # Build feature lists
        playermatchstats_list = []
        player_gw_stats_list = []
        
        for gw_num in range(1, current_gw + 1):
            gw_data = all_gw_data.get(gw_num, {})
            pms = gw_data.get('playermatchstats')
            pgs = gw_data.get('player_gameweek_stats')
            
            if pms is not None and not pms.empty:
                playermatchstats_list.append((gw_num, pms))
            if pgs is not None and not pgs.empty:
                player_gw_stats_list.append((gw_num, pgs))
        
        predictions = {}
        
        # Group players by position for batch prediction
        pos_groups = {pos: [] for pos in self.POSITIONS}
        pos_pids = {pos: [] for pos in self.POSITIONS}
        
        for player_id in player_ids:
            # Get player info
            if players_df is not None and not players_df.empty:
                player_info = players_df[players_df['player_id'] == player_id]
                if not player_info.empty:
                    position = pos_map.get(player_info.iloc[0].get('position', 'Midfielder'), 'MID')
                    team_id = int(player_info.iloc[0].get('team_code', 0) or 0)
                else:
                    position = 'MID'
                    team_id = 0
            else:
                position = 'MID'
                team_id = 0
            
            # Get value from latest GW stats
            value = 5.0
            for gw_num, pgs in reversed(player_gw_stats_list):
                player_data = pgs[pgs['id'] == player_id]
                if not player_data.empty:
                    value = float(player_data.iloc[0].get('now_cost', 50) or 50) / 10.0
                    break
            
            # Aggregate features for prediction (using current GW + 1 as target)
            match_stats = self._aggregate_match_stats(
                playermatchstats_list, player_id, current_gw + 1, window=4
            )
            
            gw_stats = self._aggregate_gw_stats(
                player_gw_stats_list, player_id, current_gw + 1, window=4,
                playermatchstats_list=playermatchstats_list
            )
            
            if match_stats is None and gw_stats is None:
                predictions[player_id] = 2.0  # Default prediction
                continue
            
            row = {}
            if match_stats:
                row.update(match_stats)
            if gw_stats:
                row.update(gw_stats)
            
            # Context features
            row['value'] = value
            row['is_gkp'] = 1 if position == 'GKP' else 0
            row['is_def'] = 1 if position == 'DEF' else 0
            row['is_mid'] = 1 if position == 'MID' else 0
            row['is_fwd'] = 1 if position == 'FWD' else 0
            
            # Real fixture context for next GW
            next_gw = current_gw + 1
            fixture_ctx = self._get_fixture_context(team_id, next_gw)
            row['fixture_difficulty'] = fixture_ctx['fixture_difficulty']
            row['is_home'] = fixture_ctx['is_home']
            row['own_elo'] = fixture_ctx['own_elo']
            row['opponent_elo'] = fixture_ctx['opponent_elo']
            row['elo_diff'] = fixture_ctx['elo_diff']
            row['win_prob'] = fixture_ctx['win_prob']
            
            # Team strength
            team_str = team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
            row['team_attack_strength'] = team_str['attack']
            row['team_defense_strength'] = team_str['defense']
            
            # Opponent strength
            opp_id = fixture_ctx['opponent_id']
            opp_str = team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
            row['opp_attack_strength'] = opp_str['attack']
            row['opp_defense_strength'] = opp_str['defense']
            
            # Season identifier (use stored value from training)
            row['season_start_year'] = self._season_start_year
            
            pos_groups[position].append(row)
            pos_pids[position].append(player_id)
        
        # Predict per position group using position-specific 4-model ensemble
        for pos in self.POSITIONS:
            if not pos_groups[pos]:
                continue
            
            X_pred = pd.DataFrame(pos_groups[pos])
            
            # Use position-specific model if trained, otherwise fallback
            model_data = self.position_models.get(pos)
            if model_data and model_data.get('is_trained', False):
                # Use position-specific feature columns
                pos_feature_cols = model_data.get('feature_cols') or self.feature_cols
            else:
                # Fallback to global model with full features
                model_data = self.fallback_models
                pos_feature_cols = self.feature_cols
            
            # Ensure all feature columns exist (keep NaN for proper handling)
            for col in pos_feature_cols:
                if col not in X_pred.columns:
                    X_pred[col] = np.nan
            
            X_raw = X_pred[pos_feature_cols].values
            preds = self._predict_with_stack(model_data, X_raw)
            
            for i, pid in enumerate(pos_pids[pos]):
                predictions[pid] = round(max(0, preds[i]), 2)
        
        return predictions

    def predict_multiple_gws(self, all_gw_data: Dict, fpl_core_season_data: Dict,
                             player_ids: List[int], current_gw: int,
                             num_gws: int = 5) -> Dict[int, Dict]:
        """Predict points for multiple upcoming gameweeks with fixture-awareness.
        
        Each GW prediction uses the actual fixture context (opponent, home/away,
        Elo ratings) for that specific gameweek, rather than applying a decay.
        
        Falls back to FPL API fixtures if Core Insights fixtures unavailable.
        
        Args:
            all_gw_data: Dict of gameweek data from Core Insights.
            fpl_core_season_data: Season-level data (may include 'fpl_api_fixtures' fallback).
            player_ids: Players to predict for.
            current_gw: Current gameweek.
            num_gws: Number of future GWs to predict.
            
        Returns:
            Dict of player_id -> {predictions: [gw1, gw2, ...], cumulative, confidence, avg_per_gw}
        """
        if not self.is_trained:
            return {}
        
        # Build fixture cache including future GWs
        self._build_fixture_cache_with_fallback(all_gw_data, fpl_core_season_data, current_gw, num_gws)
        
        # Get team strength (using historical data only)
        team_strength = self._calculate_team_strength(all_gw_data, up_to_gw=current_gw + 1)
        
        # Get player info and build feature lists
        players_df = fpl_core_season_data.get('players')
        pos_map = {'Goalkeeper': 'GKP', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}
        
        # Build historical feature lists (same for all future GWs)
        playermatchstats_list = []
        player_gw_stats_list = []
        
        for gw_num in range(1, current_gw + 1):
            gw_data = all_gw_data.get(gw_num, {})
            pms = gw_data.get('playermatchstats')
            pgs = gw_data.get('player_gameweek_stats')
            
            if pms is not None and not pms.empty:
                playermatchstats_list.append((gw_num, pms))
            if pgs is not None and not pgs.empty:
                player_gw_stats_list.append((gw_num, pgs))
        
        results = {}
        
        for player_id in player_ids:
            # Get player info
            if players_df is not None and not players_df.empty:
                player_info = players_df[players_df['player_id'] == player_id]
                if not player_info.empty:
                    position = pos_map.get(player_info.iloc[0].get('position', 'Midfielder'), 'MID')
                    team_id = int(player_info.iloc[0].get('team_code', 0) or 0)
                else:
                    position = 'MID'
                    team_id = 0
            else:
                position = 'MID'
                team_id = 0
            
            # Get value from latest GW stats
            value = 5.0
            for gw_num, pgs in reversed(player_gw_stats_list):
                player_data = pgs[pgs['id'] == player_id]
                if not player_data.empty:
                    value = float(player_data.iloc[0].get('now_cost', 50) or 50) / 10.0
                    break
            
            # Get rolling stats (same for all future GWs - represents current form)
            match_stats = self._aggregate_match_stats(
                playermatchstats_list, player_id, current_gw + 1, window=4
            )
            gw_stats = self._aggregate_gw_stats(
                player_gw_stats_list, player_id, current_gw + 1, window=4,
                playermatchstats_list=playermatchstats_list
            )
            
            gw_predictions = []
            model_disagreements = []
            
            # Make prediction for each future GW with specific fixture context
            for gw_offset in range(1, num_gws + 1):
                target_gw = current_gw + gw_offset
                
                if match_stats is None and gw_stats is None:
                    gw_predictions.append(2.0)  # Default prediction
                    model_disagreements.append(0.5)
                    continue
                
                # Build feature row with GW-specific fixture context
                row = {}
                if match_stats:
                    row.update(match_stats)
                if gw_stats:
                    row.update(gw_stats)
                
                # Context features
                row['value'] = value
                row['is_gkp'] = 1 if position == 'GKP' else 0
                row['is_def'] = 1 if position == 'DEF' else 0
                row['is_mid'] = 1 if position == 'MID' else 0
                row['is_fwd'] = 1 if position == 'FWD' else 0
                
                # GW-specific fixture context
                fixture_ctx = self._get_fixture_context(team_id, target_gw)
                row['fixture_difficulty'] = fixture_ctx['fixture_difficulty']
                row['is_home'] = fixture_ctx['is_home']
                row['own_elo'] = fixture_ctx['own_elo']
                row['opponent_elo'] = fixture_ctx['opponent_elo']
                row['elo_diff'] = fixture_ctx['elo_diff']
                row['win_prob'] = fixture_ctx['win_prob']
                
                # Team strength
                team_str = team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
                row['team_attack_strength'] = team_str['attack']
                row['team_defense_strength'] = team_str['defense']
                
                # Opponent strength
                opp_id = fixture_ctx['opponent_id']
                opp_str = team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
                row['opp_attack_strength'] = opp_str['attack']
                row['opp_defense_strength'] = opp_str['defense']
                
                # Season identifier (use stored value from training)
                row['season_start_year'] = self._season_start_year
                
                # Get appropriate model and its feature columns
                model_data = self.position_models.get(position)
                if model_data and model_data.get('is_trained', False):
                    pos_feature_cols = model_data.get('feature_cols') or self.feature_cols
                else:
                    model_data = self.fallback_models
                    pos_feature_cols = self.feature_cols
                
                # Build feature matrix with position-specific features
                X_pred = pd.DataFrame([row])
                for col in pos_feature_cols:
                    if col not in X_pred.columns:
                        X_pred[col] = np.nan
                
                X_raw = X_pred[pos_feature_cols].values
                
                # Get prediction
                pred = self._predict_with_stack(model_data, X_raw)[0]
                disagreement = self._get_model_disagreement(model_data, X_raw)
                
                gw_predictions.append(round(max(0, pred), 2))
                model_disagreements.append(disagreement)
            
            # Calculate confidence based on model disagreement and horizon
            avg_disagreement = np.mean(model_disagreements) if model_disagreements else 0.5
            if avg_disagreement < 0.5:
                confidence = 'high'
            elif avg_disagreement < 1.0:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Round per-GW disagreement values for std_dev_by_gw
            std_dev_by_gw = [round(d, 3) for d in model_disagreements]
            
            results[player_id] = {
                'predictions': gw_predictions,
                'cumulative': round(sum(gw_predictions), 2),
                'confidence': confidence,
                'avg_per_gw': round(sum(gw_predictions) / max(len(gw_predictions), 1), 2),
                'std_dev': round(avg_disagreement, 3),
                'std_dev_by_gw': std_dev_by_gw,
            }
        
        return results
    
    def _build_fixture_cache_with_fallback(self, all_gw_data: Dict, 
                                           fpl_core_season_data: Dict,
                                           current_gw: int, num_gws: int):
        """Build fixture cache including future GWs with FPL API fallback.
        
        First tries Core Insights fixtures, then falls back to FPL API fixtures.
        """
        # Build from Core Insights data (historical + any available future)
        self._build_fixture_cache(all_gw_data)
        
        # Check for FPL API fixtures fallback for future GWs
        fpl_fixtures = fpl_core_season_data.get('fpl_api_fixtures')
        if fpl_fixtures is None or fpl_fixtures.empty:
            return
        
        # Add future fixtures from FPL API
        for gw_offset in range(1, num_gws + 1):
            target_gw = current_gw + gw_offset
            
            # Check if we already have this GW in cache
            gw_fixtures = fpl_fixtures[fpl_fixtures['event'] == target_gw]
            if gw_fixtures.empty:
                continue
            
            for _, fix in gw_fixtures.iterrows():
                home_id = fix.get('team_h')
                away_id = fix.get('team_a')
                
                if pd.isna(home_id) or pd.isna(away_id):
                    continue
                
                home_id = int(home_id)
                away_id = int(away_id)
                
                # Skip if already in cache
                if (home_id, target_gw) in self._fixture_cache:
                    continue
                
                # Use FDR as proxy for Elo-based difficulty
                home_diff = fix.get('team_h_difficulty', 3)
                away_diff = fix.get('team_a_difficulty', 3)
                
                # Estimate Elo from FDR (rough mapping)
                home_elo = 1600 + (3 - away_diff) * 100  # Higher opponent diff = lower opponent Elo
                away_elo = 1600 + (3 - home_diff) * 100
                
                h_win, draw, a_win = self._calculate_win_probability(home_elo, away_elo)
                
                # Home team fixture
                self._fixture_cache[(home_id, target_gw)] = {
                    'is_home': 1,
                    'own_elo': home_elo,
                    'opponent_elo': away_elo,
                    'elo_diff': home_elo - away_elo,
                    'win_prob': h_win,
                    'opponent_id': away_id,
                    'fixture_difficulty': home_diff
                }
                
                # Away team fixture
                self._fixture_cache[(away_id, target_gw)] = {
                    'is_home': 0,
                    'own_elo': away_elo,
                    'opponent_elo': home_elo,
                    'elo_diff': away_elo - home_elo,
                    'win_prob': a_win,
                    'opponent_id': home_id,
                    'fixture_difficulty': away_diff
                }
    
    def _get_model_disagreement(self, model_data: Dict, X: np.ndarray) -> float:
        """Calculate disagreement between models in the ensemble.
        
        Higher disagreement = lower confidence in prediction.
        
        Returns:
            Standard deviation of predictions across models.
        """
        if not model_data.get('is_trained', False):
            return 1.0
        
        # Preprocess
        X_imputed = model_data['imputer'].transform(X)
        X_scaled = model_data['scaler'].transform(X_imputed)
        X_hgb = np.where(np.isinf(X), np.nan, X)
        
        # Get predictions from each model
        preds = []
        preds.append(model_data['hgb'].predict(X_hgb)[0])
        preds.append(model_data['rf'].predict(X_imputed)[0])
        preds.append(model_data['ridge'].predict(X_scaled)[0])
        
        if 'xgb' in model_data:
            preds.append(model_data['xgb'].predict(X_imputed)[0])
        
        return float(np.std(preds))

    def get_model_metrics(self) -> Dict:
        """Get model performance metrics."""
        return self.metrics.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        
        # Use RF feature importance (more interpretable)
        importances = self.rf_model.feature_importances_
        return {
            col: round(imp, 4)
            for col, imp in sorted(zip(self.feature_cols, importances),
                                   key=lambda x: x[1], reverse=True)
        }

    def get_confidence_level(self) -> str:
        """Determine overall model confidence level based on metrics."""
        if self.metrics['r2'] is None:
            return 'unknown'
        
        r2 = self.metrics['r2']
        mae = self.metrics['mae'] or 999
        
        if r2 > 0.3 and mae < 2.0:
            return 'high'
        elif r2 > 0.15 or mae < 2.5:
            return 'medium'
        else:
            return 'low'

