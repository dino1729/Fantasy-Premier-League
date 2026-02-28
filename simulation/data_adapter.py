"""Historical data adapter for FPL simulation.

This module provides access to historical CSV data for backtesting,
loading all gameweek data into memory for fast access during simulation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np


class HistoricalDataAdapter:
    """Loads and provides access to historical FPL data.

    This adapter loads all gameweek CSV files into memory and provides
    methods to query player stats, prices, and results for any GW.

    Attributes:
        season_path: Path to season data directory
        gw_data: Dict mapping GW number to DataFrame
        fixtures_df: DataFrame of all fixtures
        player_lookup: Dict mapping element_id to player info
        team_lookup: Dict mapping team_id to team name
    """

    POSITION_MAP = {
        'GK': 'GKP',
        'GKP': 'GKP',
        'DEF': 'DEF',
        'MID': 'MID',
        'FWD': 'FWD',
    }

    def __init__(self, season_path: Path = None):
        """Initialize the adapter.

        Args:
            season_path: Path to season data directory.
                        Defaults to data/2024-25/
        """
        if season_path is None:
            season_path = Path('data/2024-25')
        self.season_path = Path(season_path)

        self.gw_data: Dict[int, pd.DataFrame] = {}
        self.fixtures_df: pd.DataFrame = pd.DataFrame()
        self.player_lookup: Dict[int, Dict[str, Any]] = {}
        self.team_lookup: Dict[int, str] = {}
        self._dgw_bgw_cache: Optional[Dict] = None
        self._cumulative_stats_cache: Dict[int, pd.DataFrame] = {}

        self._load_all_data()

    def _load_all_data(self):
        """Load all gameweek CSVs and fixtures into memory."""
        gws_path = self.season_path / 'gws'

        # Load each gameweek file
        for gw in range(1, 39):
            gw_file = gws_path / f'gw{gw}.csv'
            if gw_file.exists():
                df = pd.read_csv(gw_file)
                # Normalize position codes
                if 'position' in df.columns:
                    df['position'] = df['position'].map(
                        lambda x: self.POSITION_MAP.get(x, x)
                    )
                self.gw_data[gw] = df

        # Load fixtures
        fixtures_file = self.season_path / 'fixtures.csv'
        if fixtures_file.exists():
            self.fixtures_df = pd.read_csv(fixtures_file)

        # Build player lookup from GW1 data (or first available)
        self._build_player_lookup()

        # Build team lookup
        self._build_team_lookup()

    def _build_player_lookup(self):
        """Build player info lookup from available data."""
        # Try to use GW1 data first (most reliable), fall back to merged
        df = None
        if 1 in self.gw_data:
            df = self.gw_data[1]
        else:
            # Try merged_gw.csv as fallback
            merged_path = self.season_path / 'gws' / 'merged_gw.csv'
            if merged_path.exists():
                try:
                    df = pd.read_csv(merged_path, on_bad_lines='skip')
                except Exception:
                    pass

        if df is None or df.empty:
            return

        for _, row in df.iterrows():
            if 'element' in row:
                player_id = int(row['element'])
                self.player_lookup[player_id] = {
                    'id': player_id,
                    'name': row.get('name', ''),
                    'position': self.POSITION_MAP.get(
                        row.get('position', ''), row.get('position', '')
                    ),
                    'team': row.get('team', ''),
                }

    def _build_team_lookup(self):
        """Build team ID to name lookup."""
        teams_file = self.season_path / 'teams.csv'
        if teams_file.exists():
            teams_df = pd.read_csv(teams_file)
            for _, row in teams_df.iterrows():
                self.team_lookup[row['id']] = row['name']
        else:
            # Fallback: extract from fixtures
            if not self.fixtures_df.empty:
                # Extract unique team IDs from home/away columns
                for col in ['team_h', 'team_a']:
                    if col in self.fixtures_df.columns:
                        for team_id in self.fixtures_df[col].unique():
                            if team_id not in self.team_lookup:
                                self.team_lookup[int(team_id)] = f"Team_{team_id}"

    def get_available_gameweeks(self) -> List[int]:
        """Get list of gameweeks with available data."""
        return sorted(self.gw_data.keys())

    def get_gw_dataframe(self, gw: int) -> pd.DataFrame:
        """Get full DataFrame for a gameweek.

        Args:
            gw: Gameweek number

        Returns:
            DataFrame with all player data for that GW
        """
        return self.gw_data.get(gw, pd.DataFrame())

    def get_player_actual_points(self, player_id: int, gw: int) -> int:
        """Get actual points scored by player in specific GW.

        Args:
            player_id: FPL element ID
            gw: Gameweek number

        Returns:
            Points scored (0 if player not found)
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return 0

        row = df[df['element'] == player_id]
        if row.empty:
            return 0

        return int(row['total_points'].iloc[0])

    def get_player_minutes(self, player_id: int, gw: int) -> int:
        """Get minutes played by player in specific GW.

        Args:
            player_id: FPL element ID
            gw: Gameweek number

        Returns:
            Minutes played (0 if not found)
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return 0

        row = df[df['element'] == player_id]
        if row.empty:
            return 0

        return int(row['minutes'].iloc[0])

    def get_player_price(self, player_id: int, gw: int) -> float:
        """Get player price at specific GW.

        Args:
            player_id: FPL element ID
            gw: Gameweek number

        Returns:
            Price in millions (0.0 if not found)
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return 0.0

        row = df[df['element'] == player_id]
        if row.empty:
            return 0.0

        # Value stored in tenths (e.g., 105 = 10.5m)
        return float(row['value'].iloc[0]) / 10

    def get_player_xp(self, player_id: int, gw: int) -> float:
        """Get expected points (xP) for player in specific GW.

        Args:
            player_id: FPL element ID
            gw: Gameweek number

        Returns:
            Expected points (0.0 if not found)
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return 0.0

        row = df[df['element'] == player_id]
        if row.empty:
            return 0.0

        return float(row.get('xP', row.get('xp', 0)).iloc[0] if not row.empty else 0)

    def get_player_info(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get static player info.

        Args:
            player_id: FPL element ID

        Returns:
            Dict with player info or None
        """
        return self.player_lookup.get(player_id)

    def get_player_gw_data(self, player_id: int, gw: int) -> Optional[Dict[str, Any]]:
        """Get all data for a player in a specific GW.

        Args:
            player_id: FPL element ID
            gw: Gameweek number

        Returns:
            Dict with all columns for player or None
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return None

        row = df[df['element'] == player_id]
        if row.empty:
            return None

        return row.iloc[0].to_dict()

    def get_team_name(self, team_id: int) -> str:
        """Get team name from ID."""
        return self.team_lookup.get(team_id, f"Team_{team_id}")

    def get_all_players_in_gw(self, gw: int) -> List[Dict[str, Any]]:
        """Get list of all players with data in a GW.

        Args:
            gw: Gameweek number

        Returns:
            List of player dicts with essential info
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return []

        players = []
        for _, row in df.iterrows():
            players.append({
                'id': int(row['element']),
                'name': row.get('name', ''),
                'position': row.get('position', ''),
                'team': row.get('team', ''),
                'price': float(row.get('value', 0)) / 10,
                'total_points': int(row.get('total_points', 0)),
                'minutes': int(row.get('minutes', 0)),
                'xp': float(row.get('xP', row.get('xp', 0))),
            })
        return players

    def get_available_players_in_gw(
        self, gw: int, min_minutes: int = 0
    ) -> List[Dict[str, Any]]:
        """Get players who played at least min_minutes in a GW.

        Args:
            gw: Gameweek number
            min_minutes: Minimum minutes filter

        Returns:
            List of available player dicts
        """
        all_players = self.get_all_players_in_gw(gw)
        return [p for p in all_players if p['minutes'] >= min_minutes]

    def get_dgw_bgw_info(self) -> Dict[str, List[Dict]]:
        """Detect double and blank gameweeks from fixtures.

        Returns:
            Dict with 'dgw' and 'bgw' lists containing GW info
        """
        if self._dgw_bgw_cache is not None:
            return self._dgw_bgw_cache

        if self.fixtures_df.empty:
            return {'dgw': [], 'bgw': []}

        # Count fixtures per team per GW
        team_gw_fixtures: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for _, row in self.fixtures_df.iterrows():
            gw = int(row['event']) if pd.notna(row.get('event')) else 0
            if gw == 0:
                continue

            team_h = int(row['team_h'])
            team_a = int(row['team_a'])

            team_gw_fixtures[team_h][gw] += 1
            team_gw_fixtures[team_a][gw] += 1

        dgws = []
        bgws = []

        # Get all teams
        all_teams = set(team_gw_fixtures.keys())

        # Check each GW
        for gw in range(1, 39):
            doubling_teams = []
            blanking_teams = []

            for team_id in all_teams:
                fixture_count = team_gw_fixtures[team_id].get(gw, 0)
                if fixture_count >= 2:
                    doubling_teams.append(team_id)
                elif fixture_count == 0:
                    blanking_teams.append(team_id)

            if doubling_teams:
                dgws.append({
                    'gw': gw,
                    'team_ids': doubling_teams,
                    'team_names': [self.get_team_name(t) for t in doubling_teams],
                    'teams_doubling': len(doubling_teams),
                })

            if blanking_teams:
                bgws.append({
                    'gw': gw,
                    'team_ids': blanking_teams,
                    'team_names': [self.get_team_name(t) for t in blanking_teams],
                    'teams_missing': len(blanking_teams),
                })

        self._dgw_bgw_cache = {'dgw': dgws, 'bgw': bgws}
        return self._dgw_bgw_cache

    def is_dgw(self, team_id: int, gw: int) -> bool:
        """Check if a team has a double gameweek.

        Args:
            team_id: FPL team ID
            gw: Gameweek number

        Returns:
            True if team has 2+ fixtures in GW
        """
        dgw_bgw = self.get_dgw_bgw_info()
        for dgw in dgw_bgw['dgw']:
            if dgw['gw'] == gw and team_id in dgw['team_ids']:
                return True
        return False

    def is_bgw(self, team_id: int, gw: int) -> bool:
        """Check if a team has a blank gameweek.

        Args:
            team_id: FPL team ID
            gw: Gameweek number

        Returns:
            True if team has no fixtures in GW
        """
        dgw_bgw = self.get_dgw_bgw_info()
        for bgw in dgw_bgw['bgw']:
            if bgw['gw'] == gw and team_id in bgw['team_ids']:
                return True
        return False

    def get_player_team_id(self, player_id: int, gw: int) -> Optional[int]:
        """Get team ID for a player in a specific GW.

        Args:
            player_id: FPL element ID
            gw: Gameweek number

        Returns:
            Team ID or None
        """
        df = self.gw_data.get(gw)
        if df is None or df.empty:
            return None

        row = df[df['element'] == player_id]
        if row.empty:
            return None

        # Team might be stored as name, need to reverse lookup
        team_name = row['team'].iloc[0]
        for tid, tname in self.team_lookup.items():
            if tname == team_name:
                return tid

        return None

    def get_fixtures_for_gw(self, gw: int) -> List[Dict[str, Any]]:
        """Get all fixtures for a gameweek.

        Args:
            gw: Gameweek number

        Returns:
            List of fixture dicts
        """
        if self.fixtures_df.empty:
            return []

        gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == gw]
        fixtures = []

        for _, row in gw_fixtures.iterrows():
            fixtures.append({
                'id': row.get('id'),
                'gw': gw,
                'team_h': int(row['team_h']),
                'team_a': int(row['team_a']),
                'team_h_name': self.get_team_name(int(row['team_h'])),
                'team_a_name': self.get_team_name(int(row['team_a'])),
                'team_h_score': row.get('team_h_score'),
                'team_a_score': row.get('team_a_score'),
                'team_h_difficulty': row.get('team_h_difficulty', 3),
                'team_a_difficulty': row.get('team_a_difficulty', 3),
                'finished': row.get('finished', False),
            })

        return fixtures

    def get_fixture_difficulty(self, team_id: int, gw: int) -> float:
        """Get average fixture difficulty for a team in a GW.

        Args:
            team_id: FPL team ID
            gw: Gameweek number

        Returns:
            Average FDR (1-5 scale), or 3.0 if not found
        """
        fixtures = self.get_fixtures_for_gw(gw)
        difficulties = []

        for fixture in fixtures:
            if fixture['team_h'] == team_id:
                difficulties.append(fixture['team_h_difficulty'])
            elif fixture['team_a'] == team_id:
                difficulties.append(fixture['team_a_difficulty'])

        if not difficulties:
            return 3.0  # Default medium difficulty

        return sum(difficulties) / len(difficulties)

    def get_top_players_by_total_points(
        self, gw: int, position: Optional[str] = None, n: int = 30
    ) -> List[Dict[str, Any]]:
        """Get top players by cumulative total points up to a GW.

        This helps identify the best performers to consider for transfers.

        Args:
            gw: Gameweek number
            position: Filter by position (optional)
            n: Number of players to return

        Returns:
            List of top player dicts
        """
        # Aggregate points across all GWs BEFORE current (exclude current to prevent leakage)
        player_totals: Dict[int, int] = defaultdict(int)
        player_latest_data: Dict[int, Dict] = {}

        # CRITICAL: Use range(1, gw) to exclude current GW's actual results
        # This prevents data leakage where we'd know GW N results before GW N decisions
        for g in range(1, gw):
            df = self.gw_data.get(g)
            if df is None:
                continue

            for _, row in df.iterrows():
                pid = int(row['element'])
                player_totals[pid] += int(row.get('total_points', 0))
                player_latest_data[pid] = row.to_dict()

        # For GW1, we have no prior data - use current GW for metadata only (xP-based ranking)
        # For later GWs, get current GW metadata for any players not seen before
        current_df = self.gw_data.get(gw)
        if current_df is not None:
            for _, row in current_df.iterrows():
                pid = int(row['element'])
                # Only add metadata, NOT points (to prevent leakage)
                if pid not in player_latest_data:
                    player_latest_data[pid] = row.to_dict()
                    # For GW1, use xP as proxy for ranking since no historical points
                    if gw == 1:
                        player_totals[pid] = int(row.get('xP', row.get('xp', 0)) * 10)

        # Sort by total points (or xP proxy for GW1)
        sorted_players = sorted(
            player_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = []
        for pid, total_pts in sorted_players:
            data = player_latest_data.get(pid, {})
            pos = self.POSITION_MAP.get(data.get('position', ''), data.get('position', ''))

            if position and pos != position:
                continue

            result.append({
                'id': pid,
                'name': data.get('name', ''),
                'position': pos,
                'team': data.get('team', ''),
                'price': float(data.get('value', 0)) / 10,
                'total_season_points': total_pts,
            })

            if len(result) >= n:
                break

        return result

    def get_player_points_history(
        self, player_id: int, start_gw: int = 1, end_gw: int = 38
    ) -> List[int]:
        """Get points history for a player across GWs.

        Args:
            player_id: FPL element ID
            start_gw: Starting gameweek
            end_gw: Ending gameweek

        Returns:
            List of points per GW (0 for missing GWs)
        """
        points = []
        for gw in range(start_gw, end_gw + 1):
            points.append(self.get_player_actual_points(player_id, gw))
        return points

    def calculate_player_consistency(
        self, player_id: int, through_gw: int, window: int = 5
    ) -> float:
        """Calculate points consistency (inverse of variance).

        Higher values indicate more consistent performers.
        Uses data from GWs BEFORE through_gw to prevent leakage.

        Args:
            player_id: FPL element ID
            through_gw: Calculate using data before this GW (exclusive)
            window: Number of recent GWs to consider

        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        # Exclude current GW to prevent data leakage
        end_gw = through_gw - 1
        if end_gw < 1:
            return 0.5  # No historical data available (GW1)

        start_gw = max(1, end_gw - window + 1)
        points = self.get_player_points_history(player_id, start_gw, end_gw)

        # Only consider GWs where player played
        points_played = [p for p in points if p > 0]

        if len(points_played) < 2:
            return 0.5  # Not enough data

        variance = np.var(points_played)
        # Convert to 0-1 scale (lower variance = higher consistency)
        # Max reasonable variance ~100, so divide by that
        consistency = max(0, 1 - (variance / 100))
        return consistency

    def _get_team_fixture_count(self, team_id: int, gw: int) -> int:
        """Count fixtures for a team in a given GW (0=blank, 2+=double)."""
        if self.fixtures_df.empty or not team_id:
            return 0
        gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == gw]
        if gw_fixtures.empty:
            return 0
        return int(((gw_fixtures['team_h'] == team_id) | (gw_fixtures['team_a'] == team_id)).sum())

    @staticmethod
    def _fdr_multiplier(fdr: float) -> float:
        """Convert fixture difficulty (1-5) into a multiplicative xP factor."""
        try:
            fdr_f = float(fdr)
        except Exception:
            fdr_f = 3.0
        fdr_f = max(1.0, min(5.0, fdr_f))

        # Piecewise-linear between anchor points.
        anchors = [
            (1.0, 1.15),
            (2.0, 1.05),
            (3.0, 1.00),
            (4.0, 0.95),
            (5.0, 0.85),
        ]
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= fdr_f <= x1:
                t = (fdr_f - x0) / (x1 - x0) if x1 != x0 else 0.0
                return y0 + t * (y1 - y0)
        return 1.0

    def get_player_team_id_as_of(self, player_id: int, as_of_gw: int) -> Optional[int]:
        """Get player's team ID as-of a GW (never looks forward).

        This is used for forward-looking projections to prevent leakage from future GW files.
        """
        for gw in range(as_of_gw, 0, -1):
            team_id = self.get_player_team_id(player_id, gw)
            if team_id:
                return team_id
        return None

    def get_player_xp_forecast(self, player_id: int, as_of_gw: int, target_gw: int) -> float:
        """Forecast xP for a future GW using only information available at as_of_gw.

        IMPORTANT: This method will never read xP from future GW CSVs. It uses:
        - Baseline xP from the as_of_gw snapshot
        - Future fixture difficulty + fixture count (blank/double)
        """
        if target_gw < as_of_gw:
            # This function is intended for forward-looking projections only.
            return 0.0

        team_id = self.get_player_team_id_as_of(player_id, as_of_gw)
        if not team_id:
            return 0.0

        base_xp = float(self.get_player_xp(player_id, as_of_gw) or 0.0)

        # If the team has no fixture, the player scores 0 expected points.
        target_fixtures = self._get_team_fixture_count(team_id, target_gw)
        if target_fixtures <= 0:
            return 0.0

        base_fdr = self.get_fixture_difficulty(team_id, as_of_gw)
        target_fdr = self.get_fixture_difficulty(team_id, target_gw)

        base_mult = self._fdr_multiplier(base_fdr)
        target_mult = self._fdr_multiplier(target_fdr)

        # Convert baseline into a "neutral" per-fixture xP and then re-apply target context.
        neutral_xp = base_xp / max(base_mult, 0.01)
        forecast = neutral_xp * target_mult * float(target_fixtures)
        return float(max(0.0, forecast))

    def get_player_cumulative_stats(self, through_gw: int) -> pd.DataFrame:
        """Get cumulative points/minutes through a GW (inclusive).

        Used to construct a deadline-safe player snapshot for GW decisions:
        for GW N decisions, use `through_gw = N-1` (no current/future leakage).
        """
        through_gw = int(through_gw)
        if through_gw <= 0:
            return pd.DataFrame(columns=['element', 'total_points', 'minutes'])

        cached = self._cumulative_stats_cache.get(through_gw)
        if cached is not None:
            return cached

        totals: Dict[int, Dict[str, int]] = defaultdict(lambda: {'total_points': 0, 'minutes': 0})
        for gw in range(1, through_gw + 1):
            df = self.gw_data.get(gw)
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                pid = int(row.get('element', 0) or 0)
                if pid <= 0:
                    continue
                totals[pid]['total_points'] += int(row.get('total_points', 0) or 0)
                totals[pid]['minutes'] += int(row.get('minutes', 0) or 0)

        out = pd.DataFrame(
            [{'element': pid, **vals} for pid, vals in totals.items()]
        )
        self._cumulative_stats_cache[through_gw] = out
        return out

    def build_players_df_for_solver(self, as_of_gw: int) -> pd.DataFrame:
        """Build a deadline-safe players DataFrame for the MIP solver.

        The MIP solver expects a "bootstrap-like" players_df with season-to-date
        minutes/points. For GW N decisions, we use:
        - Metadata + prices from the GW N snapshot (deadline view)
        - Cumulative minutes/points from GWs 1..N-1 (no GW N results leakage)
        """
        as_of_gw = int(as_of_gw)
        if as_of_gw < 1:
            return pd.DataFrame()

        # Prefer the as_of_gw file; if missing, fall back to the latest prior GW.
        snapshot_gw = as_of_gw
        while snapshot_gw > 0 and (snapshot_gw not in self.gw_data or self.gw_data[snapshot_gw].empty):
            snapshot_gw -= 1
        if snapshot_gw <= 0:
            return pd.DataFrame()

        gw_df = self.gw_data[snapshot_gw].copy()

        # Map position to element_type for solver
        pos_to_type = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}

        # Create team name to ID mapping (reverse of team_lookup)
        team_name_to_id = {v: k for k, v in self.team_lookup.items()}

        # Join cumulative stats through the prior GW (prevents leakage)
        cumulative = self.get_player_cumulative_stats(as_of_gw - 1)
        if not cumulative.empty:
            gw_df = gw_df.merge(cumulative, on='element', how='left', suffixes=('', '_cum'))
            # If source file already has total_points/minutes (GW outcomes), ignore them.
            gw_df['total_points'] = gw_df['total_points_cum'].fillna(0).astype(int)
            gw_df['minutes'] = gw_df['minutes_cum'].fillna(0).astype(int)
        else:
            gw_df['total_points'] = 0
            gw_df['minutes'] = 0

        df = gw_df.copy()
        df['element_type'] = df['position'].map(lambda x: pos_to_type.get(x, 3))
        df['now_cost'] = df['value']  # Already in tenths
        df['id'] = df['element']
        df['price'] = df['value'] / 10.0  # Price in millions
        df['web_name'] = df['name']
        df['status'] = 'a'
        df['chance_of_playing_next_round'] = 100
        df['team_name'] = df['team']
        df['team'] = df['team'].map(lambda x: team_name_to_id.get(x, 0))

        return df
