"""Feature Engineering Module for FPL ML Predictions

Transforms raw parquet data into training vectors with:
- Rolling window statistics (3, 6, 10 GW horizons)
- Position-specific features (CBIT for defenders, xG for attackers)
- Fixture difficulty from Elo ratings
- Form and momentum indicators

This is the "secret sauce" that determines model quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PARQUET = PROJECT_ROOT / 'data' / 'parquet'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'


class FeatureEngineer:
    """Creates ML-ready feature vectors from raw FPL data.
    
    Implements rolling window strategy with position-specific features
    for training position-specific models.
    """
    
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    # Features common to all positions
    COMMON_FEATURES = [
        # Form (lagged rolling averages)
        'points_rolling_3', 'points_rolling_6',
        'minutes_rolling_3', 'minutes_rolling_6',
        # ICT Index components
        'ict_rolling_3', 'influence_rolling_3', 'creativity_rolling_3', 'threat_rolling_3',
        # BPS (actual bonus points system score)
        'bps_rolling_3', 'bonus_rolling_3',
        # Fixture context
        'fixture_difficulty', 'fixture_difficulty_elo', 'is_home',
        'opponent_elo', 'own_elo', 'win_probability',
        # Player quality
        'value', 'selected_pct',
        # Form momentum
        'points_trend', 'minutes_trend', 'form_consistency',
        # Position encoding
        'is_gkp', 'is_def', 'is_mid', 'is_fwd',
    ]
    
    # Position-specific features
    POSITION_FEATURES = {
        'GKP': [
            'saves_rolling_3', 'saves_rolling_6',
            'cs_rolling_3', 'cs_rolling_6',  # Clean sheets
            'goals_conceded_rolling_3',
            'penalties_saved_total',
            'bonus_from_saves',  # Derived: saves contribute to BPS
            'team_defense_strength', 'opp_attack_strength',
        ],
        'DEF': [
            'cs_rolling_3', 'cs_rolling_6',
            'goals_conceded_rolling_3',
            'xgi_rolling_3', 'xgi_rolling_6',  # Goal involvement
            'xg_rolling_3', 'xa_rolling_3',
            'goals_rolling_3', 'assists_rolling_3',
            # CBIT features (2025/26 BPS rules)
            'cbit_rolling_3',  # Clearances, Blocks, Interceptions, Tackles
            'team_defense_strength', 'opp_attack_strength',
            'set_piece_threat',  # Corners, free kicks involvement
        ],
        'MID': [
            'xg_rolling_3', 'xg_rolling_6',
            'xa_rolling_3', 'xa_rolling_6',
            'xgi_rolling_3', 'xgi_rolling_6',
            'goals_rolling_3', 'assists_rolling_3',
            'key_passes_rolling_3',
            'touches_in_box_rolling_3',
            'cs_rolling_3',  # Mids get CS points too
            'team_attack_strength', 'opp_defense_strength',
        ],
        'FWD': [
            'xg_rolling_3', 'xg_rolling_6',
            'xa_rolling_3',
            'xgi_rolling_3',
            'goals_rolling_3', 'assists_rolling_3',
            'big_chances_rolling_3',
            'shots_on_target_rolling_3',
            'touches_in_box_rolling_3',
            'team_attack_strength', 'opp_defense_strength',
        ],
    }
    
    def __init__(self, players_df: pd.DataFrame, 
                 fixtures_df: pd.DataFrame,
                 gameweek_history: Optional[Dict[int, pd.DataFrame]] = None):
        """Initialize feature engineer.
        
        Args:
            players_df: Clean players DataFrame from data warehouse.
            fixtures_df: Clean fixtures DataFrame with Elo difficulty.
            gameweek_history: Optional dict of player_id -> GW history DataFrame.
        """
        self.players_df = players_df
        self.fixtures_df = fixtures_df
        self.gw_history = gameweek_history or {}
        
        # Build fixture lookup for fast access
        self._build_fixture_lookup()
        
        # Load team strength data
        self._load_team_strength()
    
    def _build_fixture_lookup(self):
        """Build lookup tables for fixture data."""
        self.fixture_lookup = {}  # (team_id, gw) -> fixture info
        
        for _, fix in self.fixtures_df.iterrows():
            gw = fix['gameweek']
            
            # Home team fixture
            home_key = (fix['home_team_id'], gw)
            self.fixture_lookup[home_key] = {
                'fdr': fix['home_difficulty'],
                'fdr_elo': fix['home_difficulty_elo'],
                'is_home': True,
                'opponent_id': fix['away_team_id'],
                'opponent_elo': fix['away_elo'],
                'own_elo': fix['home_elo'],
                'win_prob': fix['home_win_prob'],
            }
            
            # Away team fixture
            away_key = (fix['away_team_id'], gw)
            self.fixture_lookup[away_key] = {
                'fdr': fix['away_difficulty'],
                'fdr_elo': fix['away_difficulty_elo'],
                'is_home': False,
                'opponent_id': fix['home_team_id'],
                'opponent_elo': fix['home_elo'],
                'own_elo': fix['away_elo'],
                'win_prob': fix['away_win_prob'],
            }
    
    def _load_team_strength(self):
        """Load or calculate team attack/defense strength."""
        self.team_strength = {}
        
        # Calculate from finished fixtures
        finished = self.fixtures_df[self.fixtures_df['finished'] == True]
        
        for team_id in range(1, 21):
            # Home goals
            home = finished[finished['home_team_id'] == team_id]
            home_scored = home['home_score'].sum() if 'home_score' in home.columns else 0
            home_conceded = home['away_score'].sum() if 'away_score' in home.columns else 0
            
            # Away goals
            away = finished[finished['away_team_id'] == team_id]
            away_scored = away['away_score'].sum() if 'away_score' in away.columns else 0
            away_conceded = away['home_score'].sum() if 'home_score' in away.columns else 0
            
            total_games = len(home) + len(away)
            
            if total_games > 0:
                attack = (home_scored + away_scored) / total_games
                defense = (home_conceded + away_conceded) / total_games
            else:
                attack = 1.5
                defense = 1.5
            
            self.team_strength[team_id] = {
                'attack': attack,
                'defense': defense,
                'games': total_games
            }
    
    def load_player_history(self, player_id: int) -> pd.DataFrame:
        """Load gameweek-by-gameweek history for a player.
        
        Attempts to load from data/2025-26/players/ CSV files.
        """
        if player_id in self.gw_history:
            return self.gw_history[player_id]
        
        # Try loading from season data
        season_path = PROJECT_ROOT / 'data' / '2025-26' / 'players'
        
        # Find player file by ID
        for player_dir in season_path.glob('*'):
            if player_dir.is_dir():
                gw_file = player_dir / 'gw.csv'
                if gw_file.exists():
                    # Check if this is the right player
                    try:
                        gw_df = pd.read_csv(gw_file)
                        # The player ID should match the directory name pattern
                        dir_name = player_dir.name
                        # Format is usually "PlayerName_ID"
                        if f'_{player_id}' in dir_name or dir_name.endswith(str(player_id)):
                            self.gw_history[player_id] = gw_df
                            return gw_df
                    except Exception:
                        continue
        
        # Fallback: try direct ID lookup
        for player_dir in season_path.glob(f'*_{player_id}'):
            gw_file = player_dir / 'gw.csv'
            if gw_file.exists():
                try:
                    gw_df = pd.read_csv(gw_file)
                    self.gw_history[player_id] = gw_df
                    return gw_df
                except Exception:
                    pass
        
        return pd.DataFrame()
    
    def compute_rolling_features(self, history_df: pd.DataFrame, 
                                 target_gw: int,
                                 position: str = 'MID') -> Optional[Dict[str, float]]:
        """Compute rolling window features for a player up to target_gw.
        
        Features are computed using data BEFORE target_gw (no lookahead).
        
        Args:
            history_df: Player's GW-by-GW history.
            target_gw: The gameweek we're predicting for.
            position: Player position for position-specific features.
            
        Returns:
            Dict of feature name -> value, or None if insufficient history.
        """
        if history_df.empty:
            return None
        
        # Filter to games BEFORE target_gw
        gw_col = 'round' if 'round' in history_df.columns else 'GW'
        if gw_col not in history_df.columns:
            return None
        
        past_data = history_df[history_df[gw_col] < target_gw].copy()
        
        # Need at least 3 games for rolling stats
        if len(past_data) < 3:
            return None
        
        # Sort by gameweek
        past_data = past_data.sort_values(gw_col)
        
        features = {}
        
        # Helper for safe rolling mean
        def safe_rolling(series, window, default=0.0):
            if len(series) < window:
                return series.mean() if len(series) > 0 else default
            return series.tail(window).mean()
        
        # Points rolling
        pts_col = 'total_points' if 'total_points' in past_data.columns else 'points'
        if pts_col in past_data.columns:
            pts = past_data[pts_col].astype(float)
            features['points_rolling_3'] = safe_rolling(pts, 3)
            features['points_rolling_6'] = safe_rolling(pts, 6)
            
            # Points trend: compare last 2 vs previous 2
            if len(pts) >= 4:
                recent = pts.tail(2).mean()
                older = pts.iloc[-4:-2].mean() if len(pts) >= 4 else pts.head(2).mean()
                features['points_trend'] = recent - older
            else:
                features['points_trend'] = 0.0
            
            # Consistency (lower std = more consistent)
            features['form_consistency'] = pts.tail(6).std() if len(pts) >= 3 else 5.0
        
        # Minutes rolling
        if 'minutes' in past_data.columns:
            mins = past_data['minutes'].astype(float)
            features['minutes_rolling_3'] = safe_rolling(mins, 3)
            features['minutes_rolling_6'] = safe_rolling(mins, 6)
            
            # Minutes trend
            if len(mins) >= 4:
                recent_mins = mins.tail(2).mean()
                older_mins = mins.iloc[-4:-2].mean() if len(mins) >= 4 else mins.head(2).mean()
                features['minutes_trend'] = (recent_mins - older_mins) / max(older_mins, 1)
            else:
                features['minutes_trend'] = 0.0
        
        # ICT Index components
        for col, feature_name in [
            ('ict_index', 'ict_rolling_3'),
            ('influence', 'influence_rolling_3'),
            ('creativity', 'creativity_rolling_3'),
            ('threat', 'threat_rolling_3'),
        ]:
            if col in past_data.columns:
                features[feature_name] = safe_rolling(past_data[col].astype(float), 3)
            else:
                features[feature_name] = 0.0
        
        # BPS and Bonus
        if 'bps' in past_data.columns:
            features['bps_rolling_3'] = safe_rolling(past_data['bps'].astype(float), 3)
        if 'bonus' in past_data.columns:
            features['bonus_rolling_3'] = safe_rolling(past_data['bonus'].astype(float), 3)
        
        # Expected stats (xG, xA, xGI)
        for col, prefix in [
            ('expected_goals', 'xg'),
            ('expected_assists', 'xa'),
            ('expected_goal_involvements', 'xgi'),
        ]:
            if col in past_data.columns:
                vals = past_data[col].astype(float)
                features[f'{prefix}_rolling_3'] = safe_rolling(vals, 3)
                features[f'{prefix}_rolling_6'] = safe_rolling(vals, 6)
            else:
                features[f'{prefix}_rolling_3'] = 0.0
                features[f'{prefix}_rolling_6'] = 0.0
        
        # Goals and Assists
        if 'goals_scored' in past_data.columns:
            features['goals_rolling_3'] = safe_rolling(past_data['goals_scored'].astype(float), 3)
        if 'assists' in past_data.columns:
            features['assists_rolling_3'] = safe_rolling(past_data['assists'].astype(float), 3)
        
        # Position-specific features
        if position in ['GKP', 'DEF']:
            # Clean sheets
            if 'clean_sheets' in past_data.columns:
                cs = past_data['clean_sheets'].astype(float)
                features['cs_rolling_3'] = safe_rolling(cs, 3)
                features['cs_rolling_6'] = safe_rolling(cs, 6)
            
            # Goals conceded
            if 'goals_conceded' in past_data.columns:
                features['goals_conceded_rolling_3'] = safe_rolling(
                    past_data['goals_conceded'].astype(float), 3
                )
        
        if position == 'GKP':
            # Saves
            if 'saves' in past_data.columns:
                saves = past_data['saves'].astype(float)
                features['saves_rolling_3'] = safe_rolling(saves, 3)
                features['saves_rolling_6'] = safe_rolling(saves, 6)
                # Bonus from saves (3 saves = 1 BPS point)
                features['bonus_from_saves'] = features.get('saves_rolling_3', 0) / 3.0
            
            if 'penalties_saved' in past_data.columns:
                features['penalties_saved_total'] = past_data['penalties_saved'].sum()
        
        if position == 'DEF':
            # CBIT: Clearances, Blocks, Interceptions, Tackles
            # These might be in different columns depending on data source
            cbit_cols = ['clearances_blocks_interceptions', 'tackles']
            cbit_sum = 0
            for col in cbit_cols:
                if col in past_data.columns:
                    cbit_sum += safe_rolling(past_data[col].astype(float), 3)
            features['cbit_rolling_3'] = cbit_sum
            
            # Set piece threat (approximate from corners/free kicks if available)
            # For now, use xG as proxy for set piece threat
            features['set_piece_threat'] = features.get('xg_rolling_3', 0) * 0.3
        
        if position in ['MID', 'FWD']:
            # Key passes (if available)
            if 'key_passes' in past_data.columns:
                features['key_passes_rolling_3'] = safe_rolling(
                    past_data['key_passes'].astype(float), 3
                )
            
            # Shots on target
            if 'shots_on_target' in past_data.columns:
                features['shots_on_target_rolling_3'] = safe_rolling(
                    past_data['shots_on_target'].astype(float), 3
                )
            
            # Touches in box (if available) - use threat as proxy
            features['touches_in_box_rolling_3'] = features.get('threat_rolling_3', 0) / 10.0
        
        if position == 'FWD':
            # Big chances (if available)
            if 'big_chances_missed' in past_data.columns and 'goals_scored' in past_data.columns:
                # Big chances = goals + big chances missed
                bc = past_data['goals_scored'].astype(float) + past_data['big_chances_missed'].astype(float)
                features['big_chances_rolling_3'] = safe_rolling(bc, 3)
            else:
                # Estimate from xG (big chance ~= 0.35 xG)
                features['big_chances_rolling_3'] = features.get('xg_rolling_3', 0) / 0.35
        
        return features
    
    def add_fixture_features(self, features: Dict, player_id: int, 
                            target_gw: int) -> Dict:
        """Add fixture-related features for the target gameweek."""
        # Get player's team
        player_row = self.players_df[self.players_df['player_id'] == player_id]
        if player_row.empty:
            return features
        
        team_id = int(player_row.iloc[0]['team_id'])
        
        # Get fixture info
        fixture_key = (team_id, target_gw)
        fix_info = self.fixture_lookup.get(fixture_key, {})
        
        features['fixture_difficulty'] = fix_info.get('fdr', 3)
        features['fixture_difficulty_elo'] = fix_info.get('fdr_elo', 3)
        features['is_home'] = 1 if fix_info.get('is_home', True) else 0
        features['opponent_elo'] = fix_info.get('opponent_elo', 1600)
        features['own_elo'] = fix_info.get('own_elo', 1600)
        features['win_probability'] = fix_info.get('win_prob', 0.33)
        
        # Team strength features
        opp_id = fix_info.get('opponent_id', 0)
        team_str = self.team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
        opp_str = self.team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
        
        features['team_attack_strength'] = team_str['attack']
        features['team_defense_strength'] = team_str['defense']
        features['opp_attack_strength'] = opp_str['attack']
        features['opp_defense_strength'] = opp_str['defense']
        
        return features
    
    def add_player_features(self, features: Dict, player_id: int) -> Dict:
        """Add player-level features (value, ownership, position)."""
        player_row = self.players_df[self.players_df['player_id'] == player_id]
        if player_row.empty:
            return features
        
        player = player_row.iloc[0]
        
        features['value'] = player.get('cost_m', player.get('cost', 50) / 10.0)
        features['selected_pct'] = player.get('selected_by_percent', 0)
        
        # Position encoding
        position_id = player.get('position_id', 3)
        position = self.POSITION_MAP.get(position_id, 'MID')
        
        features['is_gkp'] = 1 if position == 'GKP' else 0
        features['is_def'] = 1 if position == 'DEF' else 0
        features['is_mid'] = 1 if position == 'MID' else 0
        features['is_fwd'] = 1 if position == 'FWD' else 0
        
        return features
    
    def build_feature_vector(self, player_id: int, target_gw: int,
                            history_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """Build complete feature vector for a player/gameweek combination.
        
        Args:
            player_id: FPL player ID.
            target_gw: Gameweek to predict for.
            history_df: Optional pre-loaded history (performance optimization).
            
        Returns:
            Dict of features, or None if insufficient data.
        """
        # Get player position
        player_row = self.players_df[self.players_df['player_id'] == player_id]
        if player_row.empty:
            return None
        
        position_id = int(player_row.iloc[0].get('position_id', 3))
        position = self.POSITION_MAP.get(position_id, 'MID')
        
        # Load history if not provided
        if history_df is None:
            history_df = self.load_player_history(player_id)
        
        # Compute rolling features
        features = self.compute_rolling_features(history_df, target_gw, position)
        if features is None:
            return None
        
        # Add fixture features
        features = self.add_fixture_features(features, player_id, target_gw)
        
        # Add player features
        features = self.add_player_features(features, player_id)
        
        # Add metadata (not used as features, but useful for tracking)
        features['_player_id'] = player_id
        features['_target_gw'] = target_gw
        features['_position'] = position
        
        return features
    
    def build_training_dataset(self, start_gw: int = 5, 
                               end_gw: Optional[int] = None,
                               min_minutes: int = 90) -> pd.DataFrame:
        """Build complete training dataset across all players and gameweeks.
        
        Args:
            start_gw: First GW to include as target (needs prior data).
            end_gw: Last GW to include (defaults to latest in fixtures).
            min_minutes: Minimum total minutes to include player.
            
        Returns:
            DataFrame with features and target (actual_points).
        """
        if end_gw is None:
            finished = self.fixtures_df[self.fixtures_df['finished'] == True]
            end_gw = int(finished['gameweek'].max()) if not finished.empty else 17
        
        logger.info(f"Building training dataset for GW{start_gw}-{end_gw}...")
        
        # Filter players with minimum minutes
        eligible_players = self.players_df[
            self.players_df['minutes'] >= min_minutes
        ]['player_id'].tolist()
        
        logger.info(f"Processing {len(eligible_players)} eligible players...")
        
        records = []
        
        for pid in eligible_players:
            history_df = self.load_player_history(pid)
            if history_df.empty:
                continue
            
            gw_col = 'round' if 'round' in history_df.columns else 'GW'
            pts_col = 'total_points' if 'total_points' in history_df.columns else 'points'
            
            for gw in range(start_gw, end_gw + 1):
                # Get actual points for this GW (target)
                gw_row = history_df[history_df[gw_col] == gw]
                if gw_row.empty:
                    continue
                
                actual_points = float(gw_row.iloc[0][pts_col])
                actual_minutes = float(gw_row.iloc[0].get('minutes', 0))
                
                # Build features using data BEFORE this GW
                features = self.build_feature_vector(pid, gw, history_df)
                if features is None:
                    continue
                
                # Add target
                features['actual_points'] = actual_points
                features['actual_minutes'] = actual_minutes
                
                records.append(features)
        
        df = pd.DataFrame(records)
        logger.info(f"Built dataset with {len(df)} samples")
        
        return df
    
    def get_feature_columns(self, position: Optional[str] = None) -> List[str]:
        """Get list of feature columns for training.
        
        Args:
            position: If provided, returns position-specific features.
                     If None, returns common features only.
        """
        features = self.COMMON_FEATURES.copy()
        
        if position and position in self.POSITION_FEATURES:
            features.extend(self.POSITION_FEATURES[position])
        
        return features
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'training_data.parquet'):
        """Save training dataset to parquet."""
        filepath = DATA_PARQUET / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved training data to {filepath}")
        return filepath


def build_training_data_from_warehouse() -> pd.DataFrame:
    """Convenience function to build training data from existing warehouse.
    
    Loads parquet files and builds the full training dataset.
    """
    from etl.transformers import load_parquet
    
    players_df = load_parquet('players.parquet')
    fixtures_df = load_parquet('fixtures.parquet')
    
    engineer = FeatureEngineer(players_df, fixtures_df)
    return engineer.build_training_dataset()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Build training data
    df = build_training_data_from_warehouse()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFeature columns:")
    for col in sorted(df.columns):
        if not col.startswith('_') and col not in ['actual_points', 'actual_minutes']:
            print(f"  - {col}")
    
    print(f"\nSample statistics:")
    print(df[['actual_points', 'points_rolling_3', 'fixture_difficulty', 'xg_rolling_3']].describe())

