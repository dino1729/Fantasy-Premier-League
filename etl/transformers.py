"""ETL Transformers Module

Cleans, normalizes, and transforms raw data into the warehouse schema.

Target Schemas:
- players.parquet: Master player table with current stats
- fixtures.parquet: All fixtures with Elo-based difficulty
- projections_horizon.parquet: Per-player per-GW expected points for solver

Also handles ID mapping between FPL, Understat, and FBref player names.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PARQUET = PROJECT_ROOT / 'data' / 'parquet'
DATA_MAPPINGS = PROJECT_ROOT / 'data' / 'mappings'

# Ensure directories exist
DATA_PARQUET.mkdir(parents=True, exist_ok=True)
DATA_MAPPINGS.mkdir(parents=True, exist_ok=True)


@dataclass
class PlayerSchema:
    """Schema definition for players.parquet.
    
    This is the "Single Source of Truth" for player attributes.
    """
    player_id: int          # FPL element ID (unique)
    web_name: str           # Display name (e.g., "Salah")
    full_name: str          # Full name (e.g., "Mohamed Salah")
    position_id: int        # 1=GKP, 2=DEF, 3=MID, 4=FWD
    position: str           # Position label
    team_id: int            # FPL team ID (1-20)
    team_name: str          # Team short name (e.g., "LIV")
    cost: int               # Current price in 0.1m units (e.g., 125 = £12.5m)
    cost_m: float           # Current price in millions
    total_points: int       # Season total points
    points_per_game: float  # Average PPG
    minutes: int            # Total minutes played
    form: float             # Recent form (avg last 4 GW)
    selected_by_percent: float  # Ownership %
    status: str             # 'a'=available, 'i'=injured, 's'=suspended, etc.
    chance_playing: int     # 0, 25, 50, 75, 100
    # Expected stats
    expected_goals: float       # xG
    expected_assists: float     # xA
    expected_goal_involvements: float  # xGI
    # Underlying stats
    goals_scored: int
    assists: int
    clean_sheets: int
    bonus: int
    bps: int                    # Total BPS
    ict_index: float            # Influence + Creativity + Threat
    # Defensive stats (CBIT for 2025/26)
    saves: int
    penalties_saved: int


@dataclass 
class FixtureSchema:
    """Schema definition for fixtures.parquet."""
    fixture_id: int         # FPL fixture ID
    gameweek: int           # Event/GW number (1-38)
    kickoff_time: str       # ISO datetime
    finished: bool          # Whether match is complete
    # Teams
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    # Scores (None if not played)
    home_score: Optional[int]
    away_score: Optional[int]
    # FDR (original FPL 1-5)
    home_difficulty: int
    away_difficulty: int
    # Elo-based difficulty
    home_difficulty_elo: int
    away_difficulty_elo: int
    # Elo probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_elo: float
    away_elo: float


@dataclass
class ProjectionSchema:
    """Schema definition for projections_horizon.parquet.
    
    This is the input format the MIP Solver expects.
    One row per player per gameweek in the horizon.
    """
    player_id: int          # FPL element ID
    gameweek: int           # Future GW number
    xp: float               # Expected points (from predictor)
    cost: int               # Price at projection time (0.1m units)
    position_id: int        # 1=GKP, 2=DEF, 3=MID, 4=FWD
    team_id: int            # FPL team ID
    minutes_projected: int  # Expected minutes (0-90+)


class PlayerTransformer:
    """Transforms raw FPL bootstrap data into clean player table."""
    
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    def __init__(self, bootstrap_data: Dict, teams_map: Optional[Dict[int, str]] = None):
        """Initialize with raw bootstrap data.
        
        Args:
            bootstrap_data: Dict from FPL /bootstrap-static/ endpoint.
            teams_map: Optional dict of team_id -> short_name.
        """
        self.elements = bootstrap_data.get('elements', [])
        self.teams = bootstrap_data.get('teams', [])
        
        # Build team map
        if teams_map:
            self.teams_map = teams_map
        else:
            self.teams_map = {t['id']: t['short_name'] for t in self.teams}
    
    def transform(self) -> pd.DataFrame:
        """Transform raw elements into clean players DataFrame.
        
        Returns:
            DataFrame matching PlayerSchema with proper dtypes.
        """
        if not self.elements:
            return pd.DataFrame()
        
        records = []
        for elem in self.elements:
            record = {
                'player_id': int(elem['id']),
                'web_name': str(elem.get('web_name', '')),
                'full_name': f"{elem.get('first_name', '')} {elem.get('second_name', '')}".strip(),
                'position_id': int(elem.get('element_type', 0)),
                'position': self.POSITION_MAP.get(elem.get('element_type', 0), 'UNK'),
                'team_id': int(elem.get('team', 0)),
                'team_name': self.teams_map.get(elem.get('team', 0), 'UNK'),
                'cost': int(elem.get('now_cost', 0)),
                'cost_m': round(float(elem.get('now_cost', 0)) / 10.0, 1),
                'total_points': int(elem.get('total_points', 0) or 0),
                'points_per_game': float(elem.get('points_per_game', 0) or 0),
                'minutes': int(elem.get('minutes', 0) or 0),
                'form': float(elem.get('form', 0) or 0),
                'selected_by_percent': float(elem.get('selected_by_percent', 0) or 0),
                'status': str(elem.get('status', 'a')),
                'chance_playing': int(elem.get('chance_of_playing_next_round') or 100),
                # Expected stats
                'expected_goals': float(elem.get('expected_goals', 0) or 0),
                'expected_assists': float(elem.get('expected_assists', 0) or 0),
                'expected_goal_involvements': float(elem.get('expected_goal_involvements', 0) or 0),
                # Underlying stats
                'goals_scored': int(elem.get('goals_scored', 0) or 0),
                'assists': int(elem.get('assists', 0) or 0),
                'clean_sheets': int(elem.get('clean_sheets', 0) or 0),
                'bonus': int(elem.get('bonus', 0) or 0),
                'bps': int(elem.get('bps', 0) or 0),
                'ict_index': float(elem.get('ict_index', 0) or 0),
                # Defensive stats
                'saves': int(elem.get('saves', 0) or 0),
                'penalties_saved': int(elem.get('penalties_saved', 0) or 0),
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Set dtypes explicitly
        dtype_map = {
            'player_id': 'int32',
            'position_id': 'int8',
            'team_id': 'int8',
            'cost': 'int32',
            'total_points': 'int32',
            'minutes': 'int32',
            'chance_playing': 'int16',
            'goals_scored': 'int16',
            'assists': 'int16',
            'clean_sheets': 'int16',
            'bonus': 'int16',
            'bps': 'int32',
            'saves': 'int16',
            'penalties_saved': 'int8',
        }
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        return df
    
    def save_parquet(self, df: pd.DataFrame, filename: str = 'players.parquet') -> Path:
        """Save players DataFrame to parquet."""
        filepath = DATA_PARQUET / filename
        df.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Saved {len(df)} players to {filepath}")
        return filepath


class FixtureTransformer:
    """Transforms raw fixtures with Elo-based difficulty."""
    
    def __init__(self, fixtures_data: List[Dict], 
                 teams_map: Dict[int, str],
                 elo_ratings: Dict[int, float],
                 fixture_difficulties: Optional[Dict[int, List[Dict]]] = None):
        """Initialize with raw fixtures and Elo data.
        
        Args:
            fixtures_data: List of fixture dicts from FPL API.
            teams_map: Dict of team_id -> short_name.
            elo_ratings: Dict of team_id -> Elo rating.
            fixture_difficulties: Pre-calculated difficulties from fetcher.
        """
        self.fixtures = fixtures_data
        self.teams_map = teams_map
        self.elo_ratings = elo_ratings
        self.difficulties = fixture_difficulties or {}
    
    def transform(self) -> pd.DataFrame:
        """Transform raw fixtures into clean DataFrame with Elo difficulty."""
        if not self.fixtures:
            return pd.DataFrame()
        
        # Build lookup for Elo-based difficulties
        elo_lookup = {}  # (team_id, gw) -> difficulty_elo
        prob_lookup = {}  # (team_id, gw) -> (win, draw, loss)
        
        for tid, team_fixtures in self.difficulties.items():
            for fix in team_fixtures:
                key = (tid, fix['gameweek'])
                elo_lookup[key] = fix.get('fdr_elo', fix.get('fdr_original', 3))
                prob_lookup[key] = (
                    fix.get('win_prob', 0.33),
                    fix.get('draw_prob', 0.33),
                    fix.get('loss_prob', 0.33)
                )
        
        records = []
        for fix in self.fixtures:
            gw = fix.get('event')
            if gw is None:
                continue
            
            home_id = fix['team_h']
            away_id = fix['team_a']
            
            # Get Elo-based difficulty
            home_diff_elo = elo_lookup.get((home_id, gw), fix.get('team_h_difficulty', 3))
            away_diff_elo = elo_lookup.get((away_id, gw), fix.get('team_a_difficulty', 3))
            
            # Get probabilities
            home_probs = prob_lookup.get((home_id, gw), (0.33, 0.33, 0.33))
            
            record = {
                'fixture_id': int(fix.get('id', 0)),
                'gameweek': int(gw),
                'kickoff_time': fix.get('kickoff_time', ''),
                'finished': bool(fix.get('finished', False)),
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_team_name': self.teams_map.get(home_id, 'UNK'),
                'away_team_name': self.teams_map.get(away_id, 'UNK'),
                'home_score': fix.get('team_h_score'),
                'away_score': fix.get('team_a_score'),
                'home_difficulty': fix.get('team_h_difficulty', 3),
                'away_difficulty': fix.get('team_a_difficulty', 3),
                'home_difficulty_elo': home_diff_elo,
                'away_difficulty_elo': away_diff_elo,
                'home_win_prob': home_probs[0],
                'draw_prob': home_probs[1],
                'away_win_prob': home_probs[2],
                'home_elo': self.elo_ratings.get(home_id, 1600),
                'away_elo': self.elo_ratings.get(away_id, 1600),
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Set dtypes
        dtype_map = {
            'fixture_id': 'int32',
            'gameweek': 'int8',
            'home_team_id': 'int8',
            'away_team_id': 'int8',
            'home_difficulty': 'int8',
            'away_difficulty': 'int8',
            'home_difficulty_elo': 'int8',
            'away_difficulty_elo': 'int8',
        }
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        # Sort by gameweek, fixture_id
        df = df.sort_values(['gameweek', 'fixture_id']).reset_index(drop=True)
        
        return df
    
    def save_parquet(self, df: pd.DataFrame, filename: str = 'fixtures.parquet') -> Path:
        """Save fixtures DataFrame to parquet."""
        filepath = DATA_PARQUET / filename
        df.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Saved {len(df)} fixtures to {filepath}")
        return filepath


class ProjectionTransformer:
    """Generates projections_horizon.parquet from predictions.
    
    This transformer takes expected points predictions and formats them
    for the MIP Solver. It acts as the bridge between the Prediction Layer
    and the Decision Layer.
    """
    
    def __init__(self, players_df: pd.DataFrame, 
                 predictions: Dict[int, List[float]],
                 current_gw: int,
                 horizon: int = 5):
        """Initialize with players data and predictions.
        
        Args:
            players_df: Clean players DataFrame (from PlayerTransformer).
            predictions: Dict of player_id -> list of xP per gameweek.
            current_gw: Current gameweek number.
            horizon: Number of future GWs to project (default 5).
        """
        self.players_df = players_df
        self.predictions = predictions
        self.current_gw = current_gw
        self.horizon = horizon
    
    def transform(self) -> pd.DataFrame:
        """Create projections for solver consumption.
        
        Returns:
            DataFrame with one row per (player_id, gameweek) combination.
        """
        records = []
        
        for pid, xp_list in self.predictions.items():
            # Get player info
            player_row = self.players_df[self.players_df['player_id'] == pid]
            if player_row.empty:
                continue
            
            player = player_row.iloc[0]
            
            # Estimate minutes based on status and recent minutes
            base_minutes = self._estimate_minutes(player)
            
            for i, xp in enumerate(xp_list[:self.horizon]):
                gw = self.current_gw + 1 + i
                
                # Adjust xP by availability chance
                chance = player.get('chance_playing', 100) / 100.0
                adjusted_xp = xp * chance
                
                record = {
                    'player_id': int(pid),
                    'gameweek': int(gw),
                    'xp': round(float(adjusted_xp), 2),
                    'cost': int(player['cost']),
                    'position_id': int(player['position_id']),
                    'team_id': int(player['team_id']),
                    'minutes_projected': int(base_minutes * chance),
                }
                records.append(record)
        
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
        
        # Sort by gameweek, then by xp descending
        df = df.sort_values(['gameweek', 'xp'], ascending=[True, False]).reset_index(drop=True)
        
        return df
    
    def _estimate_minutes(self, player: pd.Series) -> int:
        """Estimate expected minutes for a player."""
        status = player.get('status', 'a')
        if status != 'a':
            return 0
        
        total_minutes = player.get('minutes', 0)
        # Rough estimate based on total minutes divided by likely GWs played
        # This is a heuristic until we have per-GW history
        avg_minutes = min(90, max(0, total_minutes / max(1, self.current_gw)))
        
        return int(avg_minutes)
    
    def save_parquet(self, df: pd.DataFrame, 
                     filename: str = 'projections_horizon.parquet') -> Path:
        """Save projections DataFrame to parquet."""
        filepath = DATA_PARQUET / filename
        df.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Saved {len(df)} projections to {filepath}")
        return filepath


class PlayerIDMapper:
    """Fuzzy matches player names across different data sources.
    
    Maps between:
    - FPL: player_id, web_name, full_name
    - Understat: understat_id, player_name
    - FBref: fbref_id, player_name
    
    Uses fuzzy string matching to handle naming inconsistencies like:
    - "Bruno Fernandes" (FPL) <-> "B. Borges Fernandes" (Understat)
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """Initialize with clean players DataFrame."""
        self.players_df = players_df
        self._mapping_cache = {}
    
    def find_match(self, name: str, team_name: Optional[str] = None,
                   threshold: float = 0.7) -> Optional[int]:
        """Find FPL player_id for a given name.
        
        Args:
            name: Player name to match.
            team_name: Optional team short name to filter candidates.
            threshold: Minimum similarity score (0-1).
            
        Returns:
            FPL player_id if match found, None otherwise.
        """
        name_lower = name.lower().strip()
        
        # Check cache
        cache_key = (name_lower, team_name)
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]
        
        # Filter candidates by team if provided
        candidates = self.players_df.copy()
        if team_name:
            candidates = candidates[candidates['team_name'] == team_name]
        
        best_match = None
        best_score = 0
        
        for _, player in candidates.iterrows():
            # Try matching against web_name and full_name
            for player_name in [player['web_name'], player['full_name']]:
                score = SequenceMatcher(None, name_lower, 
                                        player_name.lower()).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = int(player['player_id'])
        
        # Cache result
        self._mapping_cache[cache_key] = best_match
        return best_match
    
    def create_mapping_csv(self, external_names: List[Tuple[str, str]],
                           source: str = 'understat') -> pd.DataFrame:
        """Create a mapping CSV from external source names to FPL IDs.
        
        Args:
            external_names: List of (player_name, team_name) tuples.
            source: Name of external source for column naming.
            
        Returns:
            DataFrame with fpl_id, fpl_name, {source}_name, match_score.
        """
        records = []
        
        for name, team in external_names:
            fpl_id = self.find_match(name, team)
            
            if fpl_id:
                fpl_player = self.players_df[
                    self.players_df['player_id'] == fpl_id
                ].iloc[0]
                fpl_name = fpl_player['web_name']
                score = SequenceMatcher(None, name.lower(), 
                                        fpl_name.lower()).ratio()
            else:
                fpl_name = None
                score = 0
            
            records.append({
                'fpl_id': fpl_id,
                'fpl_name': fpl_name,
                f'{source}_name': name,
                'team': team,
                'match_score': round(score, 2)
            })
        
        df = pd.DataFrame(records)
        return df
    
    def save_mapping(self, df: pd.DataFrame, source: str = 'understat') -> Path:
        """Save mapping CSV to data/mappings/."""
        filepath = DATA_MAPPINGS / f'player_id_map_{source}.csv'
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} mappings to {filepath}")
        return filepath


def load_parquet(filename: str) -> pd.DataFrame:
    """Load a parquet file from data/parquet/."""
    filepath = DATA_PARQUET / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    return pd.read_parquet(filepath)


def verify_schema(df: pd.DataFrame, schema_class) -> List[str]:
    """Verify DataFrame matches expected schema.
    
    Args:
        df: DataFrame to verify.
        schema_class: Dataclass defining expected schema.
        
    Returns:
        List of missing or mistyped columns (empty if valid).
    """
    from dataclasses import fields
    
    expected_cols = {f.name for f in fields(schema_class)}
    actual_cols = set(df.columns)
    
    issues = []
    
    # Check missing columns
    missing = expected_cols - actual_cols
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check extra columns (warning only)
    extra = actual_cols - expected_cols
    if extra:
        logger.warning(f"Extra columns not in schema: {extra}")
    
    return issues

