"""Data Fetcher Module

Handles all data retrieval from FPL API and local CSV files.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import getters
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import Dict, List, Optional, Any
from scraping.fpl_api import (
    get_data,
    get_individual_player_data,
    get_entry_data,
    get_entry_personal_data,
    get_entry_gws_data,
    get_entry_transfers_data,
    get_fixtures_data,
    get_classic_league_standings,
    get_entry_picks_for_gw
)
from etl.fetchers import ClubEloFetcher, FixtureDifficultyCalculator, FPLFetcher as ETLFPLFetcher
from .cache_manager import CacheManager, cached
from .session_cache import SessionCacheManager
from utils.config import SEASON as DEFAULT_SEASON


class FPLDataFetcher:
    """Fetches and consolidates FPL data from API and CSV files."""

    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    def __init__(self, team_id: int, season: str = None, use_cache: bool = True, session_cache: Optional['SessionCacheManager'] = None):
        if season is None:
            season = DEFAULT_SEASON
        self.team_id = team_id
        self.season = season
        self.base_path = Path(__file__).parent.parent.parent / "data" / season

        # Cache for API data
        self._bootstrap_data = None
        self._players_df = None
        self._fixtures_df = None
        self._team_data = None
        self._personal_data = None
        self._player_history_cache = {}
        
        # Initialize fetchers for Elo calculation
        # NOTE: ClubElo API is disabled - using FPL Core Insights Elo ratings instead
        self._elo_fetcher = ClubEloFetcher()
        # Initialize ETL FPL Fetcher for the calculator
        self._etl_fpl_fetcher = ETLFPLFetcher()
        try:
            self._difficulty_calculator = FixtureDifficultyCalculator(self._etl_fpl_fetcher, self._elo_fetcher)
        except Exception as e:
            logger.warning(f"FixtureDifficultyCalculator initialization failed: {e}")
            self._difficulty_calculator = None
        
        # Initialize cache manager (use session cache if provided, otherwise fall back to legacy CacheManager)
        if session_cache is not None:
            self.cache = session_cache
        else:
            self.cache = CacheManager(enabled=use_cache)

    @property
    def bootstrap_data(self) -> Dict:
        """Lazy load bootstrap data from API with caching."""
        if self._bootstrap_data is None:
            # Try cache first
            cached = self.cache.get('bootstrap')
            if cached is not None:
                self._bootstrap_data = cached
            else:
                self._bootstrap_data = get_data()
                self.cache.set('bootstrap', self._bootstrap_data)
        return self._bootstrap_data

    @property
    def players_df(self) -> pd.DataFrame:
        """Lazy load players data from API (live data).

        Note: We use API data instead of CSV because CSV files in the repo
        may be outdated. The API always has current season stats.
        """
        if self._players_df is None:
            # Always use API data for current/live stats
            self._players_df = pd.DataFrame(self.bootstrap_data['elements'])
        return self._players_df

    @property
    def fixtures_df(self) -> pd.DataFrame:
        """Lazy load fixtures data."""
        if self._fixtures_df is None:
            csv_path = self.base_path / "fixtures.csv"
            if csv_path.exists():
                self._fixtures_df = pd.read_csv(csv_path)
            else:
                self._fixtures_df = pd.DataFrame(get_fixtures_data())
        return self._fixtures_df

    @property
    def team_data(self) -> Dict:
        """Lazy load team history data with caching."""
        if self._team_data is None:
            # Try cache first
            cached = self.cache.get('team_data', self.team_id)
            if cached is not None:
                self._team_data = cached
            else:
                self._team_data = get_entry_data(self.team_id)
                self.cache.set('team_data', self._team_data, self.team_id)
        return self._team_data

    @property
    def personal_data(self) -> Dict:
        """Lazy load team personal data (name, manager, etc) with caching."""
        if self._personal_data is None:
            # Try cache first
            cached = self.cache.get('team_data', self.team_id, 'personal')
            if cached is not None:
                self._personal_data = cached
            else:
                self._personal_data = get_entry_personal_data(self.team_id)
                self.cache.set('team_data', self._personal_data, self.team_id, 'personal')
        return self._personal_data

    def get_team_info(self, gameweek: Optional[int] = None) -> Dict:
        """Get basic team information."""
        # Default to current summary data
        points = self.personal_data.get('summary_overall_points', 0)
        rank = self.personal_data.get('summary_overall_rank', 0)
        gw_points = self.personal_data.get('summary_event_points', 0)
        gw_rank = self.personal_data.get('summary_event_rank', 0)
        
        # If specific gameweek requested, try to find it in history
        if gameweek:
            history = self.get_gw_history()
            # Sort to ensure we find the right one
            gw_data = next((g for g in history if g['event'] == gameweek), None)
            
            if gw_data:
                points = gw_data.get('total_points', 0)
                rank = gw_data.get('overall_rank', 0)
                gw_points = gw_data.get('points', 0)
                gw_rank = gw_data.get('rank', 0)
            elif gameweek < self.get_current_gameweek():
                 # If data not found but it's in the past, maybe return last known?
                 # For now, let's just stick to defaults if missing, or maybe log warning
                 pass

        return {
            'team_id': self.team_id,
            'team_name': self.personal_data.get('name', 'Unknown'),
            'manager_name': f"{self.personal_data.get('player_first_name', '')} {self.personal_data.get('player_last_name', '')}".strip(),
            'overall_points': points,
            'overall_rank': rank,
            'gameweek_points': gw_points,
            'gameweek_rank': gw_rank,
            'season': self.season
        }

    def get_current_gameweek(self) -> int:
        """Get the current gameweek number."""
        events = self.bootstrap_data.get('events', [])
        for event in events:
            if event.get('is_current', False):
                return event['id']
        # If no current, return the latest finished
        for event in reversed(events):
            if event.get('finished', False):
                return event['id']
        return 1

    def calculate_free_transfers(self, target_gw: int) -> int:
        """Calculate accumulated free transfers available for the NEXT gameweek.
        
        Args:
            target_gw: The gameweek that just finished (or the one we are analyzing).
                      The calculation returns FTs available for target_gw + 1.
        
        Returns:
            Number of free transfers (1-5).
        """
        gw_history = self.get_gw_history()
        chips_used = self.get_chips_used()
        chip_usage = {c['event']: c['name'] for c in chips_used}
        
        # Filter history up to target_gw
        sorted_history = sorted([gw for gw in gw_history if gw['event'] <= target_gw], 
                              key=lambda x: x['event'])
        
        # Start with 1 FT for GW2 (after GW1)
        current_ft = 1
        
        if not sorted_history:
            return 1
            
        last_gw = sorted_history[-1]['event']
        
        # Iterate from GW2 to last_gw to simulate FT accumulation
        for gw in range(2, last_gw + 1):
            gw_data = next((g for g in sorted_history if g['event'] == gw), None)
            
            # Check if chip used in THIS GW
            chip = chip_usage.get(gw)
            
            if chip in ['wildcard', 'freehit']:
                # Chip used: Transfers don't cost FTs.
                # Balance for NEXT week (gw+1) resets to 1.
                current_ft = 1
            else:
                # Regular GW: Deduct transfers used
                transfers_made = gw_data.get('event_transfers', 0) if gw_data else 0
                
                current_ft -= transfers_made
                current_ft = max(0, current_ft)
                
                # Add 1 for next week
                current_ft += 1
                current_ft = min(5, current_ft)
                
        return current_ft

    def get_current_squad(self, gameweek: Optional[int] = None) -> List[Dict]:
        """Get current squad with full player details and caching.

        Args:
            gameweek: Specific gameweek to get picks from. If None, uses current.

        Returns:
            List of player dictionaries with stats and pick info.
        """
        gw = gameweek or self.get_current_gameweek()
        
        # Try cache first
        cached = self.cache.get('gw_picks', self.team_id, gw)
        if cached is not None:
            return cached
        
        gw_data = get_entry_gws_data(self.team_id, gw, start_gw=gw)[0]

        picks = gw_data.get('picks', [])
        squad = []

        for pick in picks:
            player_id = pick['element']
            player_stats = self.get_player_stats(player_id)

            # Extract purchase and selling prices from pick data (in tenths of millions)
            # The FPL API returns these as integers, e.g. 100 = 10.0m
            purchase_price_raw = pick.get('purchase_price')
            selling_price_raw = pick.get('selling_price')
            
            # Convert to millions (float) for easier calculations
            purchase_price_m = round(float(purchase_price_raw) / 10, 1) if purchase_price_raw else None
            selling_price_m = round(float(selling_price_raw) / 10, 1) if selling_price_raw else None
            
            # Fallback to now_cost if selling_price not available
            now_cost = player_stats.get('now_cost', 0)
            if selling_price_m is None and now_cost:
                selling_price_m = round(float(now_cost) / 10, 1)

            squad.append({
                'id': player_id,
                'name': player_stats.get('web_name', 'Unknown'),
                'full_name': f"{player_stats.get('first_name', '')} {player_stats.get('second_name', '')}".strip(),
                'position': self.POSITION_MAP.get(player_stats.get('element_type', 0), 'UNK'),
                'team': self._get_team_name(player_stats.get('team', 0)),
                'team_id': player_stats.get('team', 0),
                'is_captain': pick.get('is_captain', False),
                'is_vice_captain': pick.get('is_vice_captain', False),
                'multiplier': pick.get('multiplier', 1),
                'position_in_squad': pick.get('position', 0),
                'purchase_price': purchase_price_raw,
                'selling_price': selling_price_raw,
                'purchase_price_m': purchase_price_m,
                'selling_price_m': selling_price_m,
                'stats': player_stats
            })

        # Cache the result
        self.cache.set('gw_picks', squad, self.team_id, gw)
        
        return squad

    def get_player_stats(self, player_id: int) -> Dict:
        """Get comprehensive stats for a player from CSV data.

        Args:
            player_id: FPL element ID of the player.

        Returns:
            Dictionary with all player statistics.
        """
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            # Fallback to API
            elements = self.bootstrap_data.get('elements', [])
            for el in elements:
                if el['id'] == player_id:
                    return el
            return {}
        return player_row.iloc[0].to_dict()

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Get gameweek-by-gameweek history for a player with caching.

        Args:
            player_id: FPL element ID of the player.

        Returns:
            DataFrame with GW history (from API element-summary).
        """
        if player_id in self._player_history_cache:
            return self._player_history_cache[player_id]

        # Try cache first
        cached = self.cache.get('player_history', player_id)
        if cached is not None:
            self._player_history_cache[player_id] = cached
            return cached

        try:
            player_data = get_individual_player_data(player_id)
            history = player_data.get('history', [])
            if history:
                df = pd.DataFrame(history)
                self._player_history_cache[player_id] = df
                self.cache.set('player_history', df, player_id)
                return df
        except Exception:
            pass
        
        return pd.DataFrame()

    def get_player_gw_data_from_csv(self, player_name: str) -> pd.DataFrame:
        """Get player's GW data from merged CSV file.

        Args:
            player_name: Web name of the player.

        Returns:
            DataFrame with all GW performances.
        """
        merged_path = self.base_path / "gws" / "merged_gw.csv"
        if merged_path.exists():
            df = pd.read_csv(merged_path)
            return df[df['name'] == player_name]
        return pd.DataFrame()

    def get_upcoming_fixtures(self, team_id: int, num_fixtures: int = 5) -> List[Dict]:
        """Get upcoming fixtures for a team with dynamic Elo-based difficulty.

        Args:
            team_id: FPL team ID (1-20).
            num_fixtures: Number of upcoming fixtures to return.

        Returns:
            List of fixture dictionaries with difficulty info and probabilities.
        """
        # Calculate difficulties for all teams
        all_difficulties = self._difficulty_calculator.get_fixture_difficulties()
        
        # Get fixtures for the requested team
        team_fixtures = all_difficulties.get(team_id, [])
        
        # Sort by gameweek and limit
        upcoming = sorted(team_fixtures, key=lambda x: x['gameweek'])[:num_fixtures]
        
        # Map fields to match what the report expects (backward compatibility)
        # while adding the new Elo fields
        mapped_fixtures = []
        for fix in upcoming:
            mapped_fixtures.append({
                'gameweek': fix['gameweek'],
                'opponent': fix['opponent'],
                'is_home': fix['is_home'],
                'difficulty': fix['fdr_elo'],  # Use Elo-based FDR as default difficulty
                'difficulty_ordinal': fix['fdr_original'], # Keep original just in case
                'win_prob': fix['win_prob'],
                'draw_prob': fix['draw_prob'],
                'loss_prob': fix['loss_prob'],
                'opponent_elo': fix.get('opponent_elo', 0),
                'own_elo': fix.get('own_elo', 0)
            })
            
        return mapped_fixtures

    def get_fixtures_by_gw(self, team_id: int, start_gw: int, end_gw: int) -> Dict[int, List[Dict]]:
        """Get fixtures for a team keyed by gameweek number.

        Unlike get_upcoming_fixtures (flat list limited by count), this returns
        a dict keyed by GW so callers can detect BGW (empty list) and DGW
        (two-element list) per gameweek.

        Args:
            team_id: FPL team ID (1-20).
            start_gw: First gameweek (inclusive).
            end_gw: Last gameweek (inclusive).

        Returns:
            Dict mapping each GW in [start_gw, end_gw] to a list of fixture
            dicts.  Empty list = BGW, two-element list = DGW.
        """
        by_gw: Dict[int, List[Dict]] = {gw: [] for gw in range(start_gw, end_gw + 1)}

        if self._difficulty_calculator is None:
            return by_gw

        # Pass start_gw-1 so the calculator includes fixtures from start_gw
        # (it excludes gw <= current_gw_override)
        all_difficulties = self._difficulty_calculator.get_fixture_difficulties(
            current_gw_override=start_gw - 1
        )
        team_fixtures = all_difficulties.get(team_id, [])

        for fix in team_fixtures:
            gw = fix['gameweek']
            if gw not in by_gw:
                continue
            by_gw[gw].append({
                'gameweek': gw,
                'opponent': fix['opponent'],
                'is_home': fix['is_home'],
                'difficulty': fix['fdr_elo'],
                'difficulty_ordinal': fix['fdr_original'],
                'win_prob': fix['win_prob'],
                'draw_prob': fix['draw_prob'],
                'loss_prob': fix['loss_prob'],
                'opponent_elo': fix.get('opponent_elo', 0),
                'own_elo': fix.get('own_elo', 0),
            })

        return by_gw

    def get_position_peers(self, position: str, min_minutes: int = 90) -> pd.DataFrame:
        """Get all players of a certain position for comparison.

        Args:
            position: Position code (GKP, DEF, MID, FWD).
            min_minutes: Minimum minutes played to be included.

        Returns:
            DataFrame of players in that position.
        """
        pos_code = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}.get(position, 0)
        df = self.players_df[
            (self.players_df['element_type'] == pos_code) &
            (self.players_df['minutes'] >= min_minutes)
        ].copy()
        return df

    def get_gw_history(self) -> List[Dict]:
        """Get gameweek-by-gameweek history for the team."""
        return self.team_data.get('current', [])

    def get_season_history(self) -> List[Dict]:
        """Get full season history including squad picks for every gameweek.
        
        Returns:
            List of dictionaries, one per GW, containing:
            - gameweek: int
            - points: int
            - rank: int
            - squad: List[Dict] (players with stats)
        """
        history = []
        current_gw = self.get_current_gameweek()
        
        # Get basic history first
        gw_summaries = {gw['event']: gw for gw in self.get_gw_history()}
        
        for gw in range(1, current_gw + 1):
            try:
                # Fetch specific GW data
                gw_picks = self.get_current_squad(gameweek=gw)
                
                # Update picks with ACTUAL GW points
                for player in gw_picks:
                    pid = player['id']
                    ph = self.get_player_history(pid)
                    if not ph.empty:
                        # Find row for this GW
                        gw_stat = ph[ph['round'] == gw]
                        if not gw_stat.empty:
                            # Update stats with GW specific data
                            # We keep the 'stats' dict but overwrite 'event_points'
                            actual_points = int(gw_stat.iloc[0]['total_points'])
                            player['stats']['event_points'] = actual_points
                        else:
                            player['stats']['event_points'] = 0
                    else:
                        player['stats']['event_points'] = 0

                summary = gw_summaries.get(gw, {})
                
                history.append({
                    'gameweek': gw,
                    'points': summary.get('points', 0),
                    'total_points': summary.get('total_points', 0),
                    'rank': summary.get('overall_rank', 0),
                    'rank_sort': summary.get('rank_sort', 0),
                    'event_transfers': summary.get('event_transfers', 0),
                    'event_transfers_cost': summary.get('event_transfers_cost', 0),
                    'value': summary.get('value', 0),
                    'bank': summary.get('bank', 0),
                    'squad': gw_picks
                })
            except Exception as e:
                print(f"[WARN] Failed to fetch history for GW{gw}: {e}")
                
        return history

    def get_chips_used(self) -> List[Dict]:
        """Get list of chips used this season."""
        return self.team_data.get('chips', [])

    def get_squad_issues(self, squad: List[Dict] = None, gameweek: int = None) -> Dict:
        """Detect squad issues that inform chip recommendations.

        Analyzes the current squad for:
        - Injuries (chance_of_playing < 75%)
        - Suspension risk (4 or 9 yellow cards)
        - Price drop risk (negative cost_change_event or trending down)
        - Ownership decline (transfers_out > transfers_in significantly)

        Args:
            squad: List of player dicts with stats. If None, fetches current squad.
            gameweek: Target gameweek. If None, uses current.

        Returns:
            Dict with categorized issues:
            {
                'injuries': [{'name': str, 'chance': int, 'return_estimate': str}],
                'suspension_risk': [{'name': str, 'yellows': int, 'threshold': int}],
                'price_drops': [{'name': str, 'change': float, 'trend': str}],
                'ownership_decline': [{'name': str, 'net_transfers': int, 'ownership': float}],
                'total_issues': int,
                'summary': str
            }
        """
        if squad is None:
            squad = self.get_current_squad(gameweek)

        issues = {
            'injuries': [],
            'suspension_risk': [],
            'price_drops': [],
            'ownership_decline': [],
            'total_issues': 0,
            'summary': ''
        }

        for player in squad:
            stats = player.get('stats', {})
            name = player.get('name', stats.get('web_name', 'Unknown'))

            # Check injuries (chance_of_playing < 75%)
            chance = stats.get('chance_of_playing_next_round')
            if chance is not None and chance < 75:
                # Estimate return based on news if available
                news = stats.get('news', '')
                return_est = 'Unknown'
                if 'unknown' in news.lower():
                    return_est = 'Unknown'
                elif any(x in news.lower() for x in ['week', 'wk']):
                    return_est = news[:50] if news else 'Short-term'
                elif any(x in news.lower() for x in ['month', 'mth']):
                    return_est = 'Long-term'
                elif chance == 0:
                    return_est = 'Out'
                else:
                    return_est = 'Doubt'

                issues['injuries'].append({
                    'name': name,
                    'chance': chance,
                    'return_estimate': return_est,
                    'news': news[:100] if news else ''
                })

            # Check suspension risk (4 yellows = 1 away from 5-game ban, 9 = 1 away from 10-game ban)
            yellows = int(stats.get('yellow_cards', 0) or 0)
            if yellows == 4:
                issues['suspension_risk'].append({
                    'name': name,
                    'yellows': yellows,
                    'threshold': 5,
                    'risk_level': 'high'
                })
            elif yellows == 9:
                issues['suspension_risk'].append({
                    'name': name,
                    'yellows': yellows,
                    'threshold': 10,
                    'risk_level': 'high'
                })
            elif yellows == 3 or yellows == 8:
                issues['suspension_risk'].append({
                    'name': name,
                    'yellows': yellows,
                    'threshold': 5 if yellows == 3 else 10,
                    'risk_level': 'medium'
                })

            # Check price drop risk
            cost_change = stats.get('cost_change_event', 0) or 0
            cost_change_start = stats.get('cost_change_start', 0) or 0
            if cost_change < 0 or cost_change_start < -2:  # Lost 0.2m+ this season
                trend = 'falling' if cost_change < 0 else 'down this season'
                issues['price_drops'].append({
                    'name': name,
                    'change': cost_change / 10,  # Convert to millions
                    'change_season': cost_change_start / 10,
                    'trend': trend
                })

            # Check ownership decline (significant net negative transfers)
            transfers_in = int(stats.get('transfers_in_event', 0) or 0)
            transfers_out = int(stats.get('transfers_out_event', 0) or 0)
            ownership = float(stats.get('selected_by_percent', 0) or 0)
            net_transfers = transfers_in - transfers_out

            # Significant decline: losing > 100k managers this GW, or > 2% ownership drop potential
            if net_transfers < -100000 or (ownership > 5 and net_transfers < -50000):
                issues['ownership_decline'].append({
                    'name': name,
                    'net_transfers': net_transfers,
                    'transfers_in': transfers_in,
                    'transfers_out': transfers_out,
                    'ownership': ownership
                })

        # Calculate totals and summary
        total = (len(issues['injuries']) + len(issues['suspension_risk']) +
                 len(issues['price_drops']) + len(issues['ownership_decline']))
        issues['total_issues'] = total

        # Build summary string
        summary_parts = []
        if issues['injuries']:
            summary_parts.append(f"{len(issues['injuries'])} injured")
        if issues['suspension_risk']:
            summary_parts.append(f"{len(issues['suspension_risk'])} suspension risk")
        if issues['price_drops']:
            summary_parts.append(f"{len(issues['price_drops'])} price drop risk")
        if issues['ownership_decline']:
            summary_parts.append(f"{len(issues['ownership_decline'])} losing ownership")

        issues['summary'] = ', '.join(summary_parts) if summary_parts else 'No major issues detected'

        return issues

    def get_transfers(self) -> List[Dict]:
        """Get transfer history for the team."""
        transfers = get_entry_transfers_data(self.team_id) or []

        enriched: List[Dict] = []
        for t in transfers:
            element_in = t.get('element_in')
            element_out = t.get('element_out')

            in_stats = self.get_player_stats(int(element_in)) if element_in else {}
            out_stats = self.get_player_stats(int(element_out)) if element_out else {}

            in_team_id = int(in_stats.get('team', 0) or 0) if in_stats else 0
            out_team_id = int(out_stats.get('team', 0) or 0) if out_stats else 0

            in_pos = self.POSITION_MAP.get(int(in_stats.get('element_type', 0) or 0), 'UNK') if in_stats else 'UNK'
            out_pos = self.POSITION_MAP.get(int(out_stats.get('element_type', 0) or 0), 'UNK') if out_stats else 'UNK'

            element_in_cost = t.get('element_in_cost')
            element_out_cost = t.get('element_out_cost')

            # Costs are returned as ints in tenths of millions (e.g., 71 -> 7.1)
            in_cost_m = None
            out_cost_m = None
            if element_in_cost is not None:
                try:
                    in_cost_m = round(float(element_in_cost) / 10, 1)
                except (TypeError, ValueError):
                    in_cost_m = None
            if element_out_cost is not None:
                try:
                    out_cost_m = round(float(element_out_cost) / 10, 1)
                except (TypeError, ValueError):
                    out_cost_m = None

            enriched_transfer = dict(t)
            enriched_transfer.update(
                {
                    'element_in_name': in_stats.get('web_name', 'Unknown') if in_stats else 'Unknown',
                    'element_out_name': out_stats.get('web_name', 'Unknown') if out_stats else 'Unknown',
                    'element_in_team': self._get_team_name(in_team_id) if in_team_id else 'UNK',
                    'element_out_team': self._get_team_name(out_team_id) if out_team_id else 'UNK',
                    'element_in_position': in_pos,
                    'element_out_position': out_pos,
                    'element_in_cost_m': in_cost_m,
                    'element_out_cost_m': out_cost_m,
                }
            )
            enriched.append(enriched_transfer)

        # Stable ordering for report output
        def sort_key(x: Dict) -> tuple:
            event = x.get('event')
            try:
                event_i = int(event)
            except (TypeError, ValueError):
                event_i = 10_000
            time_s = x.get('time') or ''
            return (event_i, time_s)

        enriched.sort(key=sort_key)
        return enriched

    def get_team_value(self, gameweek: Optional[int] = None) -> float:
        """Get current team value in millions.
        
        Args:
            gameweek: If provided, gets value at end of specific gameweek.
        """
        history = self.get_gw_history()
        
        # Filter if gameweek provided
        if gameweek:
            history = [h for h in history if h['event'] <= gameweek]
            
        if history:
            return history[-1].get('value', 1000) / 10
        return 100.0

    def get_bank(self, gameweek: Optional[int] = None) -> float:
        """Get money in the bank in millions.
        
        Args:
            gameweek: If provided, gets bank at end of specific gameweek.
        """
        history = self.get_gw_history()
        
        # Filter if gameweek provided
        if gameweek:
            history = [h for h in history if h['event'] <= gameweek]
            
        if history:
            return history[-1].get('bank', 0) / 10
        return 0.0

    def _get_team_name(self, team_id: int) -> str:
        """Get team short name from ID."""
        teams = self.bootstrap_data.get('teams', [])
        for team in teams:
            if team['id'] == team_id:
                return team['short_name']
        return 'UNK'

    def get_all_players_by_position(self, position: str) -> pd.DataFrame:
        """Get all players of a position, sorted by form."""
        peers = self.get_position_peers(position, min_minutes=0)
        return peers.sort_values('form', ascending=False)


def build_competitive_dataset(
    entry_ids: List[int],
    season: str = None,
    gameweek: Optional[int] = None,
    use_cache: bool = True,
    session_cache: Optional['SessionCacheManager'] = None
) -> List[Dict]:
    """Build competitive dataset for multiple FPL entries.

    Fetches team info, GW history, current squad, chips used, and derived
    metrics for each entry to enable comparative analysis.

    Args:
        entry_ids: List of FPL entry/team IDs to compare.
        season: Season folder name (default from config).
        gameweek: Specific gameweek for squad snapshot. If None, uses current.
    """
    if season is None:
        season = DEFAULT_SEASON
    """

    Returns:
        List of dictionaries, one per entry, containing:
        - entry_id: int
        - team_info: Dict (team_name, manager_name, overall_points, overall_rank)
        - gw_history: List[Dict] (event, points, total_points, overall_rank, etc.)
        - squad: List[Dict] (player picks for the specified gameweek)
        - season_history: List[Dict] (full season history with squad for each GW)
        - chips_used: List[Dict] (chips used this season)
        - total_hits: int (sum of transfer costs across all GWs)
        - team_value: float (current team value in millions)
        - bank: float (money in bank in millions)
        - gw_transfers: Dict (transfer activity for current GW vs prior GW)
        - transfer_history: Dict (transfer history over past 5 GWs with visual data)
    """
    # Create a cache manager for competitive data (use session cache if provided)
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    # Try to get entire dataset from cache
    cache_key_args = tuple(sorted(entry_ids))
    cached_dataset = cache.get('competitive', season, gameweek, *cache_key_args)
    if cached_dataset is not None:
        return cached_dataset
    
    def _build_single_entry(entry_id: int) -> Dict[str, Any]:
        fetcher = FPLDataFetcher(entry_id, season, use_cache=use_cache, session_cache=session_cache)

        # Determine gameweek
        gw = gameweek or fetcher.get_current_gameweek()

        # Get team info
        team_info = fetcher.get_team_info(gameweek=gw)

        # Get GW history
        gw_history = fetcher.get_gw_history()

        # Filter history if gameweek specified to prevent leakage
        if gw:
            gw_history = [h for h in gw_history if h['event'] <= gw]

        # Compute total hits from GW history
        total_hits = sum(
            gw_entry.get('event_transfers_cost', 0)
            for gw_entry in gw_history
        )

        # Get team value and bank from latest GW (respecting gameweek)
        team_value = fetcher.get_team_value(gameweek=gw)
        bank = fetcher.get_bank(gameweek=gw)

        # Get chips used
        chips_used = fetcher.get_chips_used()
        if gw:
            chips_used = [c for c in chips_used if c['event'] <= gw]

        # Get current squad for the specified gameweek
        try:
            squad = fetcher.get_current_squad(gameweek=gw)
        except Exception:
            squad = []

        # Get full season history (needed for accurate treemap contribution calculations)
        # IMPORTANT: Filter to target gameweek to prevent data leakage
        try:
            full_season_history = fetcher.get_season_history()
            # Filter to only include gameweeks up to and including target GW
            season_history = [h for h in full_season_history if h.get('gameweek', 0) <= gw]
        except Exception:
            season_history = []

        # Get transfer activity for current GW vs prior GW
        try:
            gw_transfers = compute_gw_transfers(entry_id, gw, season, use_cache, session_cache)
        except Exception:
            gw_transfers = {
                'transfers_in': [],
                'transfers_out': [],
                'net_points': 0,
                'chip_used': None,
                'prior_chip_used': None,
                'transfer_cost': 0,
                'is_wildcard': False,
                'is_free_hit': False,
                'num_changes': 0
            }

        # Get transfer history for past 5 GWs (for visual progression)
        try:
            transfer_history = compute_transfer_history(
                entry_id,
                gw,
                num_gws=5,
                season=season,
                use_cache=use_cache,
                session_cache=session_cache
            )
        except Exception:
            transfer_history = {
                'current_xi': [],
                'current_bench': [],
                'transfer_timeline': [],
                'player_history': {},
                'chips_timeline': {},
                'gw_range': []
            }

        return {
            'entry_id': entry_id,
            'team_info': team_info,
            'gw_history': gw_history,
            'squad': squad,
            'season_history': season_history,
            'chips_used': chips_used,
            'total_hits': total_hits,
            'team_value': team_value,
            'bank': bank,
            'gw_transfers': gw_transfers,
            'transfer_history': transfer_history
        }

    max_workers = max(1, min(5, len(entry_ids)))
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(entry_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_build_single_entry, entry_id): idx
            for idx, entry_id in enumerate(entry_ids)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            entry_id = entry_ids[idx]
            try:
                ordered_results[idx] = future.result()
            except Exception as exc:
                print(f"[WARN] Competitive dataset failed for entry {entry_id}: {exc}")

    results = [item for item in ordered_results if item is not None]

    # Cache the complete dataset
    cache.set('competitive', results, season, gameweek, *cache_key_args)
    
    return results


def get_league_entry_ids(
    league_id: int,
    sample_n: int,
    focus_entry_id: Optional[int] = None,
    use_cache: bool = True,
    session_cache: Optional['SessionCacheManager'] = None
) -> List[int]:
    """Get entry IDs from a classic league, sampling around focus_entry or from the top.

    Args:
        league_id: Classic league ID.
        sample_n: Number of entries to return.
        focus_entry_id: If provided, sample entries around this entry's rank.
                       Otherwise sample from top of league.
        use_cache: Whether to use caching.

    Returns:
        List of entry IDs from the league.
    """
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    # Try cache first
    cache_key = ('league_entries', league_id, sample_n, focus_entry_id)
    cached = cache.get('league_standings', *cache_key)
    if cached is not None:
        return cached

    all_entries = []
    page = 1
    max_pages = 20  # Limit to avoid excessive API calls

    # Fetch league standings with pagination
    while page <= max_pages:
        try:
            data = get_classic_league_standings(league_id, page)
            standings = data.get('standings', {})
            results = standings.get('results', [])
            
            if not results:
                break
            
            for entry in results:
                all_entries.append({
                    'entry': entry.get('entry'),
                    'rank': entry.get('rank'),
                    'player_name': entry.get('player_name'),
                    'entry_name': entry.get('entry_name'),
                    'total': entry.get('total')
                })
            
            # Check if there are more pages
            if not standings.get('has_next', False):
                break
            
            page += 1
            
        except Exception as e:
            print(f"[WARN] Failed to fetch league page {page}: {e}")
            break

    if not all_entries:
        return []

    # Sample entries
    entry_ids = []
    
    if focus_entry_id is not None:
        # Find focus entry's rank
        focus_rank = None
        for e in all_entries:
            if e['entry'] == focus_entry_id:
                focus_rank = e['rank']
                break
        
        if focus_rank is not None:
            # Sort by rank and get entries around focus
            all_entries.sort(key=lambda x: x['rank'])
            
            # Find index of focus entry
            focus_idx = None
            for i, e in enumerate(all_entries):
                if e['entry'] == focus_entry_id:
                    focus_idx = i
                    break
            
            if focus_idx is not None:
                # Get N/2 above and N/2 below (plus focus entry)
                half = sample_n // 2
                start_idx = max(0, focus_idx - half)
                end_idx = min(len(all_entries), start_idx + sample_n)
                
                # Adjust start if we hit the end
                if end_idx == len(all_entries):
                    start_idx = max(0, end_idx - sample_n)
                
                entry_ids = [e['entry'] for e in all_entries[start_idx:end_idx]]
        
        # Fallback if focus entry not found
        if not entry_ids:
            entry_ids = [e['entry'] for e in all_entries[:sample_n]]
    else:
        # Sample from top of league
        entry_ids = [e['entry'] for e in all_entries[:sample_n]]

    # Cache the result
    cache.set('league_standings', entry_ids, *cache_key)
    
    return entry_ids


def get_top_global_teams(
    n: int = 5,
    use_cache: bool = True,
    session_cache: Optional['SessionCacheManager'] = None
) -> List[Dict]:
    """Get the top N teams from the overall FPL league.
    
    Args:
        n: Number of top teams to return.
        use_cache: Whether to use caching.
    
    Returns:
        List of dicts with entry_id, manager_name, team_name, total_points, rank
    """
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    # Try cache first
    cached = cache.get('top_global_teams', n)
    if cached is not None:
        return cached
    
    # Overall FPL league ID is 314
    OVERALL_LEAGUE_ID = 314
    
    try:
        data = get_classic_league_standings(OVERALL_LEAGUE_ID, page=1)
        standings = data.get('standings', {})
        results = standings.get('results', [])
        
        top_teams = []
        for entry in results[:n]:
            top_teams.append({
                'entry_id': entry.get('entry'),
                'manager_name': entry.get('player_name'),
                'team_name': entry.get('entry_name'),
                'total_points': entry.get('total'),
                'rank': entry.get('rank')
            })
        
        cache.set('top_global_teams', top_teams, n)
        return top_teams
        
    except Exception as e:
        print(f"[WARN] Failed to fetch top global teams: {e}")
        return []


def get_bgw_dgw_gameweeks(use_cache: bool = True, session_cache: Optional['SessionCacheManager'] = None) -> Dict:
    """Detect Blank Gameweeks (BGW) and Double Gameweeks (DGW) from fixtures.
    
    A BGW occurs when some teams have no fixtures in a gameweek.
    A DGW occurs when some teams have 2+ fixtures in a gameweek.
    
    Args:
        use_cache: Whether to use caching.
    
    Returns:
        Dict with:
        - bgw: List of {gw, teams_missing, count}
        - dgw: List of {gw, teams_doubled, count}
        - normal: List of normal GW numbers
    """
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    cached = cache.get('bgw_dgw_info')
    if cached is not None:
        return cached
    
    try:
        fixtures = get_fixtures_data()
        
        # Count fixtures per team per gameweek
        gw_team_counts = {}  # {gw: {team_id: fixture_count}}
        
        for fix in fixtures:
            gw = fix.get('event')
            if gw is None:
                continue
                
            if gw not in gw_team_counts:
                gw_team_counts[gw] = {}
            
            team_h = fix.get('team_h')
            team_a = fix.get('team_a')
            
            gw_team_counts[gw][team_h] = gw_team_counts[gw].get(team_h, 0) + 1
            gw_team_counts[gw][team_a] = gw_team_counts[gw].get(team_a, 0) + 1
        
        bgws = []
        dgws = []
        normal = []
        
        for gw in sorted(gw_team_counts.keys()):
            team_counts = gw_team_counts[gw]
            
            # Check for teams with 0 fixtures (BGW)
            all_teams = set(range(1, 21))  # Teams 1-20
            teams_with_fixtures = set(team_counts.keys())
            teams_missing = all_teams - teams_with_fixtures
            
            # Check for teams with 2+ fixtures (DGW)
            teams_doubled = [t for t, count in team_counts.items() if count >= 2]
            
            if teams_missing:
                bgws.append({
                    'gw': gw,
                    'teams_missing': len(teams_missing),
                    'team_ids': list(teams_missing)
                })
            if teams_doubled:
                dgws.append({
                    'gw': gw,
                    'teams_doubled': len(teams_doubled),
                    'team_ids': teams_doubled
                })
            else:
                normal.append(gw)
        
        result = {'bgw': bgws, 'dgw': dgws, 'normal': normal}
        cache.set('bgw_dgw_info', result)
        return result
        
    except Exception as e:
        print(f"[WARN] Failed to detect BGW/DGW: {e}")
        return {'bgw': [], 'dgw': [], 'normal': []}

def compute_league_ownership(
    entry_ids: List[int],
    gw: int,
    use_cache: bool = True,
    session_cache: Optional['SessionCacheManager'] = None
) -> Dict:
    """Compute player ownership percentages within a set of league entries.

    Args:
        entry_ids: List of entry IDs to analyze.
        gw: Gameweek to check picks for.
        use_cache: Whether to use caching.

    Returns:
        Dict containing:
            - ownership: {player_id: ownership_fraction} (0.0 to 1.0)
            - captain_counts: {player_id: count}
            - sample_size: number of entries analyzed
            - top_owned: List of (player_id, ownership) sorted by ownership desc
    """
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    # Try cache first
    cache_key_args = tuple(sorted(entry_ids))
    cached = cache.get('league_ownership', gw, *cache_key_args)
    if cached is not None:
        return cached

    player_counts = {}  # player_id -> count of entries owning
    captain_counts = {}  # player_id -> count of captaincies
    valid_entries = 0

    def _fetch_entry_picks(entry_id: int) -> List[Dict]:
        try:
            picks_data = get_entry_picks_for_gw(entry_id, gw)
            picks = picks_data.get('picks', [])
            return picks if picks else []
        except Exception:
            # Entry might not have played this GW yet
            return []

    max_workers = max(1, min(10, len(entry_ids)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(_fetch_entry_picks, entry_id): entry_id
            for entry_id in entry_ids
        }
        for future in as_completed(future_to_entry):
            try:
                picks = future.result()
            except Exception:
                continue
            if not picks:
                continue

            valid_entries += 1
            for pick in picks:
                player_id = pick.get('element')
                if player_id is None:
                    continue

                # Count ownership (only count starting XI + bench = 15 players)
                player_counts[player_id] = player_counts.get(player_id, 0) + 1

                # Count captaincies
                if pick.get('is_captain', False):
                    captain_counts[player_id] = captain_counts.get(player_id, 0) + 1

    if valid_entries == 0:
        result = {
            'ownership': {},
            'captain_counts': {},
            'sample_size': 0,
            'top_owned': []
        }
        return result

    # Convert counts to fractions
    ownership = {
        pid: count / valid_entries 
        for pid, count in player_counts.items()
    }
    
    # Sort by ownership descending
    top_owned = sorted(ownership.items(), key=lambda x: x[1], reverse=True)

    result = {
        'ownership': ownership,
        'captain_counts': captain_counts,
        'sample_size': valid_entries,
        'top_owned': top_owned
    }
    
    # Cache the result
    cache.set('league_ownership', result, gw, *cache_key_args)
    
    return result


def compute_gw_transfers(
    entry_id: int,
    current_gw: int,
    season: str = None,
    use_cache: bool = True,
    session_cache: Optional['SessionCacheManager'] = None
) -> Dict:
    """Compute transfers made between prior GW and current GW for an entry.

    Compares squad picks between consecutive gameweeks to identify transfers,
    handles chip usage (Free Hit, Wildcard), and enriches with player stats.

    Args:
        entry_id: FPL entry/team ID.
        current_gw: Current gameweek to analyze.
        season: Season folder name (default from config).
        use_cache: Whether to use caching.
    """
    if season is None:
        season = DEFAULT_SEASON
    """

    Returns:
        Dict containing:
            - transfers_in: List[Dict] (player dicts with name, position, gw_points)
            - transfers_out: List[Dict] (player dicts with name, position, gw_points)
            - net_points: int (points gained from ins minus points lost from outs)
            - chip_used: Optional[str] (wildcard, freehit, etc. if used in current GW)
            - prior_chip_used: Optional[str] (chip used in prior GW, affects comparison)
            - transfer_cost: int (hit cost for this GW)
            - is_wildcard: bool (whether WC was used)
            - is_free_hit: bool (whether FH was used)
            - num_changes: int (total number of players changed)
    """
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    # Try cache first
    cached = cache.get('gw_transfers', entry_id, current_gw)
    if cached is not None:
        return cached

    fetcher = FPLDataFetcher(entry_id, season, use_cache=use_cache, session_cache=session_cache)
    
    result = {
        'transfers_in': [],
        'transfers_out': [],
        'net_points': 0,
        'chip_used': None,
        'prior_chip_used': None,
        'transfer_cost': 0,
        'is_wildcard': False,
        'is_free_hit': False,
        'num_changes': 0
    }
    
    # Handle GW1 edge case
    if current_gw <= 1:
        result['chip_used'] = 'initial_squad'
        cache.set('gw_transfers', result, entry_id, current_gw)
        return result
    
    prior_gw = current_gw - 1
    
    # Get chips used to check for WC/FH
    chips_used = fetcher.get_chips_used()
    chip_map = {c['event']: c['name'] for c in chips_used}
    
    current_chip = chip_map.get(current_gw)
    prior_chip = chip_map.get(prior_gw)
    
    result['chip_used'] = current_chip
    result['prior_chip_used'] = prior_chip
    
    # Get GW history for transfer cost
    gw_history = fetcher.get_gw_history()
    gw_data = next((g for g in gw_history if g.get('event') == current_gw), None)
    if gw_data:
        result['transfer_cost'] = gw_data.get('event_transfers_cost', 0)
    
    # Handle Wildcard in current GW
    if current_chip == 'wildcard':
        result['is_wildcard'] = True
        # Still compute changes to show count
        try:
            current_picks = get_entry_picks_for_gw(entry_id, current_gw)
            prior_picks = get_entry_picks_for_gw(entry_id, prior_gw)
            
            current_ids = set(p['element'] for p in current_picks.get('picks', []))
            prior_ids = set(p['element'] for p in prior_picks.get('picks', []))
            
            result['num_changes'] = len(current_ids - prior_ids)
        except Exception:
            result['num_changes'] = 0
        
        cache.set('gw_transfers', result, entry_id, current_gw)
        return result
    
    # Handle Free Hit in current GW
    if current_chip == 'freehit':
        result['is_free_hit'] = True
        # FH squad is temporary, show the FH picks but mark it
        try:
            current_picks = get_entry_picks_for_gw(entry_id, current_gw)
            prior_picks = get_entry_picks_for_gw(entry_id, prior_gw)
            
            current_ids = set(p['element'] for p in current_picks.get('picks', []))
            prior_ids = set(p['element'] for p in prior_picks.get('picks', []))
            
            result['num_changes'] = len(current_ids - prior_ids)
        except Exception:
            result['num_changes'] = 0
        
        cache.set('gw_transfers', result, entry_id, current_gw)
        return result
    
    # Handle Free Hit in PRIOR GW - compare pre-FH squad to current
    comparison_gw = prior_gw
    if prior_chip == 'freehit' and prior_gw > 1:
        # FH reverts, so compare to GW before FH
        comparison_gw = prior_gw - 1
    
    # Fetch picks for both GWs
    try:
        current_picks_data = get_entry_picks_for_gw(entry_id, current_gw)
        comparison_picks_data = get_entry_picks_for_gw(entry_id, comparison_gw)
    except Exception:
        # Entry might not have data for these GWs
        cache.set('gw_transfers', result, entry_id, current_gw)
        return result
    
    current_picks = current_picks_data.get('picks', [])
    comparison_picks = comparison_picks_data.get('picks', [])
    
    if not current_picks or not comparison_picks:
        cache.set('gw_transfers', result, entry_id, current_gw)
        return result
    
    current_ids = set(p['element'] for p in current_picks)
    comparison_ids = set(p['element'] for p in comparison_picks)
    
    # Identify transfers
    transfers_in_ids = current_ids - comparison_ids
    transfers_out_ids = comparison_ids - current_ids
    
    result['num_changes'] = len(transfers_in_ids)
    
    # Enrich transfers with player data
    position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    def get_player_gw_points(player_id: int, gw: int) -> int:
        """Get points a player scored in a specific GW."""
        try:
            history = fetcher.get_player_history(player_id)
            if history.empty:
                return 0
            gw_row = history[history['round'] == gw]
            if gw_row.empty:
                return 0
            return int(gw_row.iloc[0]['total_points'])
        except Exception:
            return 0
    
    # Process transfers in
    in_points_total = 0
    for pid in transfers_in_ids:
        stats = fetcher.get_player_stats(pid)
        gw_points = get_player_gw_points(pid, current_gw)
        in_points_total += gw_points
        
        result['transfers_in'].append({
            'player_id': pid,
            'name': stats.get('web_name', 'Unknown'),
            'position': position_map.get(stats.get('element_type', 0), 'UNK'),
            'team': fetcher._get_team_name(stats.get('team', 0)),
            'gw_points': gw_points,
            'price': round(float(stats.get('now_cost', 0)) / 10, 1)
        })
    
    # Process transfers out
    out_points_total = 0
    for pid in transfers_out_ids:
        stats = fetcher.get_player_stats(pid)
        gw_points = get_player_gw_points(pid, current_gw)
        out_points_total += gw_points
        
        result['transfers_out'].append({
            'player_id': pid,
            'name': stats.get('web_name', 'Unknown'),
            'position': position_map.get(stats.get('element_type', 0), 'UNK'),
            'team': fetcher._get_team_name(stats.get('team', 0)),
            'gw_points': gw_points,
            'price': round(float(stats.get('now_cost', 0)) / 10, 1)
        })
    
    # Sort by points descending
    result['transfers_in'].sort(key=lambda x: x['gw_points'], reverse=True)
    result['transfers_out'].sort(key=lambda x: x['gw_points'], reverse=True)
    
    # Calculate net points (in points minus out points)
    result['net_points'] = in_points_total - out_points_total
    
    # Cache the result
    cache.set('gw_transfers', result, entry_id, current_gw)
    
    return result


def compute_transfer_history(
    entry_id: int,
    current_gw: int,
    num_gws: int = 5,
    season: str = None,
    use_cache: bool = True,
    session_cache: Optional['SessionCacheManager'] = None
) -> Dict:
    """Compute transfer history over multiple gameweeks for an entry.

    Tracks squad evolution and when each player was brought in/out,
    providing data for visual progression display.

    Args:
        entry_id: FPL entry/team ID.
        current_gw: Current gameweek.
        num_gws: Number of gameweeks to look back.
        season: Season folder name (default from config).
        use_cache: Whether to use caching.
    """
    if season is None:
        season = DEFAULT_SEASON
    """

    Returns:
        Dict containing:
            - current_xi: List[Dict] (current starting XI with transfer info)
            - current_bench: List[Dict] (current bench with transfer info)
            - transfer_timeline: List[Dict] (transfers per GW over the window)
            - player_history: Dict[player_id] -> {joined_gw, left_gw, is_current}
            - chips_timeline: Dict[gw] -> chip_name
    """
    cache = session_cache if session_cache is not None else CacheManager(enabled=use_cache)
    
    # Try cache first
    cached = cache.get('transfer_history', entry_id, current_gw, num_gws)
    if cached is not None:
        return cached

    fetcher = FPLDataFetcher(entry_id, season, use_cache=use_cache, session_cache=session_cache)
    position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    result = {
        'current_xi': [],
        'current_bench': [],
        'transfer_timeline': [],
        'player_history': {},
        'chips_timeline': {},
        'gw_range': []
    }
    
    # Determine GW range
    start_gw = max(1, current_gw - num_gws + 1)
    end_gw = current_gw
    result['gw_range'] = list(range(start_gw, end_gw + 1))
    
    # Get chips used
    chips_used = fetcher.get_chips_used()
    result['chips_timeline'] = {c['event']: c['name'] for c in chips_used}
    
    # Track squads for each GW in the window
    gw_squads = {}
    for gw in result['gw_range']:
        try:
            picks_data = get_entry_picks_for_gw(entry_id, gw)
            picks = picks_data.get('picks', [])
            gw_squads[gw] = {
                'player_ids': set(p['element'] for p in picks),
                'picks': picks,
                'chip': result['chips_timeline'].get(gw)
            }
        except Exception:
            gw_squads[gw] = {'player_ids': set(), 'picks': [], 'chip': None}
    
    # Build player history - when each player joined/left
    all_players = set()
    for gw_data in gw_squads.values():
        all_players.update(gw_data['player_ids'])
    
    for pid in all_players:
        gws_in_squad = [gw for gw in result['gw_range'] if pid in gw_squads[gw]['player_ids']]
        if gws_in_squad:
            joined_gw = min(gws_in_squad)
            # Check if they were already in squad before window
            if start_gw > 1:
                try:
                    pre_window = get_entry_picks_for_gw(entry_id, start_gw - 1)
                    pre_ids = set(p['element'] for p in pre_window.get('picks', []))
                    if pid in pre_ids:
                        joined_gw = None  # Was already in squad before window
                except Exception:
                    pass
            
            left_gw = None
            is_current = pid in gw_squads[current_gw]['player_ids']
            if not is_current:
                # Find when they left
                for gw in result['gw_range']:
                    if pid in gw_squads[gw]['player_ids']:
                        # Check if next GW they're gone
                        next_gw = gw + 1
                        if next_gw <= current_gw and pid not in gw_squads.get(next_gw, {}).get('player_ids', set()):
                            left_gw = next_gw
                            break
            
            result['player_history'][pid] = {
                'joined_gw': joined_gw,
                'left_gw': left_gw,
                'is_current': is_current,
                'gws_owned': gws_in_squad
            }
    
    # Build transfer timeline (transfers per GW)
    for gw in result['gw_range']:
        if gw == start_gw:
            continue  # No comparison for first GW in window
        
        prior_gw = gw - 1
        # Handle Free Hit in prior GW
        if gw_squads.get(prior_gw, {}).get('chip') == 'freehit' and prior_gw > 1:
            prior_gw = prior_gw - 1
        
        current_ids = gw_squads[gw]['player_ids']
        prior_ids = gw_squads.get(prior_gw, {}).get('player_ids', set())
        
        transfers_in = current_ids - prior_ids
        transfers_out = prior_ids - current_ids
        
        gw_transfers = {
            'gw': gw,
            'chip': gw_squads[gw].get('chip'),
            'transfers_in': [],
            'transfers_out': []
        }
        
        for pid in transfers_in:
            stats = fetcher.get_player_stats(pid)
            gw_transfers['transfers_in'].append({
                'player_id': pid,
                'name': stats.get('web_name', 'Unknown'),
                'position': position_map.get(stats.get('element_type', 0), 'UNK'),
                'team': fetcher._get_team_name(stats.get('team', 0))
            })
        
        for pid in transfers_out:
            stats = fetcher.get_player_stats(pid)
            gw_transfers['transfers_out'].append({
                'player_id': pid,
                'name': stats.get('web_name', 'Unknown'),
                'position': position_map.get(stats.get('element_type', 0), 'UNK'),
                'team': fetcher._get_team_name(stats.get('team', 0))
            })
        
        result['transfer_timeline'].append(gw_transfers)
    
    # Build current XI and bench with transfer info
    current_picks = gw_squads.get(current_gw, {}).get('picks', [])
    for pick in current_picks:
        pid = pick['element']
        stats = fetcher.get_player_stats(pid)
        history = result['player_history'].get(pid, {})
        
        player_info = {
            'player_id': pid,
            'name': stats.get('web_name', 'Unknown'),
            'position': position_map.get(stats.get('element_type', 0), 'UNK'),
            'team': fetcher._get_team_name(stats.get('team', 0)),
            'is_captain': pick.get('is_captain', False),
            'is_vice_captain': pick.get('is_vice_captain', False),
            'position_in_squad': pick.get('position', 0),
            'joined_gw': history.get('joined_gw'),
            'is_new': history.get('joined_gw') is not None and history.get('joined_gw') >= start_gw,
            'gws_owned': len(history.get('gws_owned', []))
        }
        
        if pick.get('position', 0) <= 11:
            result['current_xi'].append(player_info)
        else:
            result['current_bench'].append(player_info)
    
    # Sort XI by position in squad
    result['current_xi'].sort(key=lambda x: x['position_in_squad'])
    result['current_bench'].sort(key=lambda x: x['position_in_squad'])
    
    # Build GW-by-GW squad data for horizontal display
    result['gw_squads_data'] = {}
    for gw in result['gw_range']:
        gw_picks = gw_squads.get(gw, {}).get('picks', [])
        gw_xi = []
        gw_bench = []
        
        # Get prior GW for comparison (to identify ins/outs)
        prior_gw = gw - 1
        prior_ids = gw_squads.get(prior_gw, {}).get('player_ids', set()) if prior_gw >= start_gw else set()
        current_ids = gw_squads.get(gw, {}).get('player_ids', set())
        
        # Handle Free Hit in prior GW
        if prior_gw >= start_gw and gw_squads.get(prior_gw, {}).get('chip') == 'freehit':
            # After FH, squad reverts - compare to pre-FH
            pre_fh_gw = prior_gw - 1
            if pre_fh_gw >= 1:
                try:
                    pre_fh_picks = get_entry_picks_for_gw(entry_id, pre_fh_gw)
                    prior_ids = set(p['element'] for p in pre_fh_picks.get('picks', []))
                except Exception:
                    pass
        
        transfers_in_this_gw = current_ids - prior_ids if prior_gw >= start_gw else set()
        transfers_out_this_gw = prior_ids - current_ids if prior_gw >= start_gw else set()
        
        for pick in gw_picks:
            pid = pick['element']
            stats = fetcher.get_player_stats(pid)
            
            player_info = {
                'player_id': pid,
                'name': stats.get('web_name', 'Unknown'),
                'position': position_map.get(stats.get('element_type', 0), 'UNK'),
                'team': fetcher._get_team_name(stats.get('team', 0)),
                'is_captain': pick.get('is_captain', False),
                'is_vice_captain': pick.get('is_vice_captain', False),
                'position_in_squad': pick.get('position', 0),
                'is_new_this_gw': pid in transfers_in_this_gw
            }
            
            if pick.get('position', 0) <= 11:
                gw_xi.append(player_info)
            else:
                gw_bench.append(player_info)
        
        gw_xi.sort(key=lambda x: x['position_in_squad'])
        gw_bench.sort(key=lambda x: x['position_in_squad'])
        
        # Also track who left this GW
        players_out = []
        for pid in transfers_out_this_gw:
            stats = fetcher.get_player_stats(pid)
            players_out.append({
                'player_id': pid,
                'name': stats.get('web_name', 'Unknown'),
                'position': position_map.get(stats.get('element_type', 0), 'UNK')
            })
        
        result['gw_squads_data'][gw] = {
            'xi': gw_xi,
            'bench': gw_bench,
            'chip': gw_squads.get(gw, {}).get('chip'),
            'transfers_out': players_out
        }
    
    # Cache the result
    cache.set('transfer_history', result, entry_id, current_gw, num_gws)
    
    return result
