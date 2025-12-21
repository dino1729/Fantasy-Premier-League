"""ETL Fetchers Module

Extracts data from multiple sources for the FPL Data Warehouse.

Sources:
- FPL API: Bootstrap static data, player histories, fixture events
- ClubElo: Dynamic team strength ratings
- Future: Understat, FBref for xG/xA

All fetchers save raw JSON/CSV to data/raw/ with timestamps for archiving.
"""

import json
import os
import time
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_MAPPINGS = PROJECT_ROOT / 'data' / 'mappings'

# Ensure directories exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_MAPPINGS.mkdir(parents=True, exist_ok=True)


class FPLFetcher:
    """Fetches data from the official FPL API.
    
    Endpoints:
    - /bootstrap-static/: Players, teams, events (gameweeks), game settings
    - /element-summary/{id}/: Player-specific history and fixtures
    - /fixtures/: All season fixtures with stats
    - /event/{gw}/live/: Live gameweek data with BPS breakdowns
    """
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self, cache_duration: int = 300):
        """Initialize FPL fetcher.
        
        Args:
            cache_duration: Seconds to cache in-memory responses.
        """
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_duration = cache_duration
        
    def _request(self, endpoint: str, retries: int = 3) -> Dict:
        """Make API request with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def _get_cached(self, key: str, fetcher_fn) -> Any:
        """Get data from cache or fetch fresh."""
        now = time.time()
        if key in self._cache:
            if now - self._cache_timestamps.get(key, 0) < self.cache_duration:
                return self._cache[key]
        
        data = fetcher_fn()
        self._cache[key] = data
        self._cache_timestamps[key] = now
        return data
    
    def get_bootstrap_static(self, save_raw: bool = True) -> Dict:
        """Fetch bootstrap-static data (players, teams, events).
        
        Returns dict with keys:
        - elements: All players with stats
        - teams: All 20 PL teams
        - events: All gameweeks
        - element_types: Position definitions (GKP, DEF, MID, FWD)
        - game_settings: Transfer/chip rules
        """
        data = self._get_cached('bootstrap', 
                                lambda: self._request('/bootstrap-static/'))
        
        if save_raw:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = DATA_RAW / f'bootstrap_static_{timestamp}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved bootstrap-static to {filepath}")
        
        return data
    
    def get_player_history(self, player_id: int, save_raw: bool = False) -> Dict:
        """Fetch detailed history for a specific player.
        
        Returns:
        - history: Past gameweek performance this season
        - history_past: Season-by-season totals (previous years)
        - fixtures: Upcoming fixtures with difficulty
        """
        data = self._request(f'/element-summary/{player_id}/')
        
        if save_raw:
            filepath = DATA_RAW / f'player_{player_id}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data
    
    def get_fixtures(self, save_raw: bool = True) -> List[Dict]:
        """Fetch all fixtures with detailed stats.
        
        Each fixture includes:
        - teams, event (gameweek), scores
        - team_h_difficulty / team_a_difficulty (FDR 1-5)
        - stats: goals_scored, assists, bonus, bps breakdown
        - CBIT: stats include clearances_blocks_interceptions for defenders
        """
        data = self._get_cached('fixtures', 
                                lambda: self._request('/fixtures/'))
        
        if save_raw:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = DATA_RAW / f'fixtures_{timestamp}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved fixtures to {filepath}")
        
        return data
    
    def get_live_gameweek(self, gameweek: int, save_raw: bool = False) -> Dict:
        """Fetch live data for a specific gameweek.
        
        Contains detailed BPS breakdowns and per-player stats for the GW.
        Includes CBIT metrics: blocks, interceptions, clearances, tackles.
        """
        data = self._request(f'/event/{gameweek}/live/')
        
        if save_raw:
            filepath = DATA_RAW / f'live_gw{gameweek}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data
    
    def get_current_gameweek(self) -> int:
        """Get the current active gameweek number."""
        bootstrap = self.get_bootstrap_static(save_raw=False)
        events = bootstrap.get('events', [])
        
        for event in events:
            if event.get('is_current', False):
                return event['id']
        
        # Fallback: return latest finished GW
        finished = [e for e in events if e.get('finished', False)]
        if finished:
            return finished[-1]['id']
        
        return 1
    
    def extract_player_cbit_stats(self, gameweek: int) -> Dict[int, Dict]:
        """Extract CBIT (Clearances, Blocks, Interceptions, Tackles) per player.
        
        These stats are critical for the 2025/26 BPS changes that reward
        defensive actions. Returns dict of player_id -> CBIT metrics.
        """
        live_data = self.get_live_gameweek(gameweek)
        elements = live_data.get('elements', [])
        
        cbit_stats = {}
        for element in elements:
            pid = element.get('id')
            stats = element.get('stats', {})
            
            cbit_stats[pid] = {
                'clearances_blocks_interceptions': stats.get('clearances_blocks_interceptions', 0),
                'tackles': stats.get('tackles', 0),
                'recoveries': stats.get('recoveries', 0),
                'saves': stats.get('saves', 0),
                'penalties_saved': stats.get('penalties_saved', 0),
                'bonus': stats.get('bonus', 0),
                'bps': stats.get('bps', 0)
            }
        
        return cbit_stats


class ClubEloFetcher:
    """Fetches team Elo ratings from ClubElo.com.
    
    ClubElo provides dynamic team strength ratings that are more nuanced
    than FPL's static 1-5 FDR scale. This allows calculating exact
    win/draw/loss probabilities for fixture difficulty.
    
    API: http://api.clubelo.com/{YYYY-MM-DD}
    Returns CSV with: Rank,Club,Country,Level,Elo,From,To
    """
    
    API_URL = "http://api.clubelo.com"
    
    # Mapping from ClubElo names to FPL team IDs (2025-26)
    # FPL team IDs from bootstrap-static teams array
    CLUBELO_TO_FPL = {
        'Arsenal': 1,
        'Aston Villa': 2,
        'Bournemouth': 3,
        'Brentford': 4,
        'Brighton': 5,
        'Chelsea': 6,
        'Crystal Palace': 7,
        'Everton': 8,
        'Fulham': 9,
        'Ipswich': 10,
        'Leicester': 11,
        'Liverpool': 12,
        'Man City': 13,
        'Man United': 14,
        'Newcastle': 15,
        'Nott\'m Forest': 16,
        'Southampton': 17,
        'Spurs': 18,
        'West Ham': 19,
        'Wolves': 20,
        # ClubElo alternative names
        'ManCity': 13,
        'ManUnited': 14,
        'NottmForest': 16,
        'Nottingham Forest': 16,
        'Tottenham': 18,
    }
    
    def __init__(self):
        self._cache = {}
    
    def get_ratings(self, for_date: Optional[date] = None, 
                    save_raw: bool = True) -> Dict[int, float]:
        """Fetch Elo ratings for all Premier League teams.
        
        Args:
            for_date: Date to fetch ratings for (default: today).
            save_raw: Whether to save raw CSV to data/raw/.
            
        Returns:
            Dict mapping FPL team_id -> Elo rating.
        """
        if for_date is None:
            for_date = date.today()
        
        date_str = for_date.strftime('%Y-%m-%d')
        
        if date_str in self._cache:
            return self._cache[date_str]
        
        url = f"{self.API_URL}/{date_str}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw_csv = response.text
            
            if save_raw:
                filepath = DATA_RAW / f'clubelo_{date_str}.csv'
                with open(filepath, 'w') as f:
                    f.write(raw_csv)
                logger.info(f"Saved ClubElo data to {filepath}")
            
            # Parse CSV
            ratings = self._parse_csv(raw_csv)
            self._cache[date_str] = ratings
            return ratings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch ClubElo data: {e}")
            return self._get_fallback_ratings()
    
    def _parse_csv(self, csv_text: str) -> Dict[int, float]:
        """Parse ClubElo CSV into FPL team_id -> Elo mapping."""
        ratings = {}
        lines = csv_text.strip().split('\n')
        
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 5:
                club_name = parts[1].strip()
                country = parts[2].strip()
                elo = float(parts[4])
                
                # Only include English teams
                if country == 'ENG':
                    fpl_id = self.CLUBELO_TO_FPL.get(club_name)
                    if fpl_id:
                        ratings[fpl_id] = elo
        
        return ratings
    
    def _get_fallback_ratings(self) -> Dict[int, float]:
        """Return approximate Elo ratings if API fails.
        
        Based on 2024-25 season end positions as rough estimates.
        """
        return {
            1: 1900,   # Arsenal
            2: 1800,   # Aston Villa
            3: 1650,   # Bournemouth
            4: 1700,   # Brentford
            5: 1750,   # Brighton
            6: 1850,   # Chelsea
            7: 1650,   # Crystal Palace
            8: 1600,   # Everton
            9: 1700,   # Fulham
            10: 1500,  # Ipswich (promoted)
            11: 1550,  # Leicester (promoted)
            12: 2000,  # Liverpool
            13: 2050,  # Man City
            14: 1750,  # Man United
            15: 1800,  # Newcastle
            16: 1700,  # Nott'm Forest
            17: 1450,  # Southampton (promoted)
            18: 1800,  # Spurs
            19: 1700,  # West Ham
            20: 1600,  # Wolves
        }
    
    @staticmethod
    def calculate_win_probability(elo_home: float, elo_away: float,
                                   home_advantage: float = 65.0) -> Tuple[float, float, float]:
        """Calculate match probabilities using Elo formula.
        
        Args:
            elo_home: Home team's Elo rating.
            elo_away: Away team's Elo rating.
            home_advantage: Elo points added for home team (default 65).
            
        Returns:
            Tuple of (home_win, draw, away_win) probabilities.
        """
        # Adjust for home advantage
        elo_home_adj = elo_home + home_advantage
        
        # Elo expected score formula
        dr = elo_away - elo_home_adj
        expected_home = 1 / (1 + 10 ** (dr / 400))
        expected_away = 1 - expected_home
        
        # Approximate draw probability based on Elo difference
        # Smaller Elo gap = higher draw chance
        elo_diff = abs(elo_home_adj - elo_away)
        draw_base = 0.25 * (1 - elo_diff / 600)  # ~25% baseline, decreases with gap
        draw_prob = max(0.1, min(0.35, draw_base))
        
        # Adjust win probs for draw
        home_win = expected_home * (1 - draw_prob)
        away_win = expected_away * (1 - draw_prob)
        
        # Normalize
        total = home_win + draw_prob + away_win
        return home_win / total, draw_prob / total, away_win / total


class FixtureDifficultyCalculator:
    """Calculates dynamic fixture difficulty using Elo ratings.
    
    Replaces FPL's static 1-5 FDR with probability-based difficulty
    that accounts for home/away and current team form.
    """
    
    def __init__(self, fpl_fetcher: FPLFetcher, elo_fetcher: ClubEloFetcher):
        self.fpl = fpl_fetcher
        self.elo = elo_fetcher
    
    def get_fixture_difficulties(self, for_date: Optional[date] = None
                                  ) -> Dict[int, List[Dict]]:
        """Calculate Elo-based difficulties for all upcoming fixtures.
        
        Returns:
            Dict mapping team_id -> list of fixture dicts with:
            - gameweek, opponent, is_home
            - fdr_original (FPL's 1-5)
            - fdr_elo (our 1-5 based on Elo)
            - win_prob, draw_prob, loss_prob
        """
        fixtures = self.fpl.get_fixtures(save_raw=False)
        elo_ratings = self.elo.get_ratings(for_date)
        current_gw = self.fpl.get_current_gameweek()
        
        # Build team name map
        bootstrap = self.fpl.get_bootstrap_static(save_raw=False)
        teams = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}
        
        # Group future fixtures by team
        team_fixtures = {tid: [] for tid in range(1, 21)}
        
        for fix in fixtures:
            gw = fix.get('event')
            if gw is None or gw <= current_gw:
                continue  # Skip past/current GW
            
            home_id = fix['team_h']
            away_id = fix['team_a']
            
            home_elo = elo_ratings.get(home_id, 1600)
            away_elo = elo_ratings.get(away_id, 1600)
            
            # Calculate probabilities
            h_win, draw, a_win = ClubEloFetcher.calculate_win_probability(home_elo, away_elo)
            
            # Convert to 1-5 difficulty (from perspective of each team)
            # Higher opponent win prob = higher difficulty
            home_diff_elo = self._prob_to_fdr(a_win)
            away_diff_elo = self._prob_to_fdr(h_win)
            
            # Home team's fixture
            team_fixtures[home_id].append({
                'gameweek': gw,
                'opponent': teams.get(away_id, '?'),
                'opponent_id': away_id,
                'is_home': True,
                'fdr_original': fix.get('team_h_difficulty', 3),
                'fdr_elo': home_diff_elo,
                'win_prob': round(h_win, 3),
                'draw_prob': round(draw, 3),
                'loss_prob': round(a_win, 3),
                'opponent_elo': away_elo,
                'own_elo': home_elo
            })
            
            # Away team's fixture
            team_fixtures[away_id].append({
                'gameweek': gw,
                'opponent': teams.get(home_id, '?'),
                'opponent_id': home_id,
                'is_home': False,
                'fdr_original': fix.get('team_a_difficulty', 3),
                'fdr_elo': away_diff_elo,
                'win_prob': round(a_win, 3),
                'draw_prob': round(draw, 3),
                'loss_prob': round(h_win, 3),
                'opponent_elo': home_elo,
                'own_elo': away_elo
            })
        
        # Sort each team's fixtures by gameweek
        for tid in team_fixtures:
            team_fixtures[tid].sort(key=lambda x: x['gameweek'])
        
        return team_fixtures
    
    def _prob_to_fdr(self, loss_prob: float) -> int:
        """Convert loss probability to FDR 1-5 scale.
        
        FDR 1 = Easy (< 20% loss chance)
        FDR 2 = Fairly Easy (20-35%)
        FDR 3 = Medium (35-50%)
        FDR 4 = Hard (50-65%)
        FDR 5 = Very Hard (> 65%)
        """
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


def fetch_all_data(save_raw: bool = True) -> Dict[str, Any]:
    """Convenience function to fetch all data sources.
    
    Returns dict with:
    - bootstrap: FPL bootstrap-static data
    - fixtures: All fixtures with stats
    - elo_ratings: Current ClubElo ratings
    - fixture_difficulties: Elo-based FDR for all teams
    - current_gw: Current gameweek number
    """
    logger.info("Starting full data fetch...")
    
    fpl = FPLFetcher()
    elo = ClubEloFetcher()
    calc = FixtureDifficultyCalculator(fpl, elo)
    
    bootstrap = fpl.get_bootstrap_static(save_raw=save_raw)
    fixtures = fpl.get_fixtures(save_raw=save_raw)
    elo_ratings = elo.get_ratings(save_raw=save_raw)
    difficulties = calc.get_fixture_difficulties()
    current_gw = fpl.get_current_gameweek()
    
    logger.info(f"Fetched {len(bootstrap.get('elements', []))} players, "
                f"{len(fixtures)} fixtures, "
                f"{len(elo_ratings)} team Elo ratings. "
                f"Current GW: {current_gw}")
    
    return {
        'bootstrap': bootstrap,
        'fixtures': fixtures,
        'elo_ratings': elo_ratings,
        'fixture_difficulties': difficulties,
        'current_gw': current_gw
    }


if __name__ == '__main__':
    # Test fetching
    data = fetch_all_data()
    
    # Print sample Elo-based difficulties
    print("\nSample fixture difficulties (Team ID 1 = Arsenal):")
    arsenal_fixtures = data['fixture_difficulties'].get(1, [])[:5]
    for fix in arsenal_fixtures:
        print(f"  GW{fix['gameweek']}: vs {fix['opponent']} "
              f"({'H' if fix['is_home'] else 'A'}) - "
              f"FDR: {fix['fdr_original']}->{fix['fdr_elo']} "
              f"(Win: {fix['win_prob']:.1%})")

