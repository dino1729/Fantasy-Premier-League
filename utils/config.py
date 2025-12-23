"""Central Configuration Module

Provides a single source of truth for all FPL Report configuration values.
Loads settings from config.yml and exposes typed constants for use throughout
the codebase.

Usage:
    from utils.config import TEAM_ID, SEASON, COMPETITORS
    from utils.config import FREE_HIT, WILDCARD, MIP_SOLVER
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yml.
    
    Checks project root first, then reports/ for backward compatibility.
    
    Returns:
        Dict with config values or empty dict if not found.
    """
    project_root = _get_project_root()
    
    # Try config.yml in project root first
    config_path = project_root / 'config.yml'
    if not config_path.exists():
        # Fall back to reports/ for backward compatibility
        config_path = project_root / 'reports' / 'config.yml'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


# Load config at module level (singleton pattern)
_CONFIG = load_config()


# =============================================================================
# TEAM SETTINGS
# =============================================================================

# Your FPL Team ID (required)
TEAM_ID: int = _CONFIG.get('team_id', 847569)

# Competitor team IDs for mini-league comparison
COMPETITORS: List[int] = _CONFIG.get('competitors', [21023, 1827604, 489166])


# =============================================================================
# GAMEWEEK & SEASON
# =============================================================================

# Target gameweek to analyze (None = auto-detect)
GAMEWEEK: Optional[int] = _CONFIG.get('gameweek')

# Season identifier (used for data folder paths)
SEASON: str = _CONFIG.get('season', '2025-26')


# =============================================================================
# FREE HIT CHIP STRATEGY
# =============================================================================

_FREE_HIT_CONFIG = _CONFIG.get('free_hit', {})

FREE_HIT: Dict[str, Any] = {
    'target_gw': _FREE_HIT_CONFIG.get('target_gw'),
    'strategy': _FREE_HIT_CONFIG.get('strategy', 'balanced'),
}

# Convenience accessors
FREE_HIT_TARGET_GW: Optional[int] = FREE_HIT['target_gw']
FREE_HIT_STRATEGY: str = FREE_HIT['strategy']


# =============================================================================
# WILDCARD CHIP STRATEGY
# =============================================================================

_WILDCARD_CONFIG = _CONFIG.get('wildcard', {})

WILDCARD: Dict[str, Any] = {
    'strategy': _WILDCARD_CONFIG.get('strategy', 'balanced'),
}

# Convenience accessor
WILDCARD_STRATEGY: str = WILDCARD['strategy']


# =============================================================================
# TRANSFER PLANNER
# =============================================================================

_TRANSFER_CONFIG = _CONFIG.get('transfer_planner', {})

TRANSFER_PLANNER: Dict[str, Any] = {
    'horizon': _TRANSFER_CONFIG.get('horizon', 5),
    'free_transfers': _TRANSFER_CONFIG.get('free_transfers'),
}

# Convenience accessors
TRANSFER_HORIZON: int = TRANSFER_PLANNER['horizon']
FREE_TRANSFERS_OVERRIDE: Optional[int] = TRANSFER_PLANNER['free_transfers']


# =============================================================================
# MIP SOLVER (Mathematical Optimization)
# =============================================================================

_MIP_CONFIG = _CONFIG.get('mip_solver', {})

MIP_SOLVER: Dict[str, Any] = {
    'enabled': _MIP_CONFIG.get('enabled', True),
    'time_limit': _MIP_CONFIG.get('time_limit', 60),
    'candidate_pool': _MIP_CONFIG.get('candidate_pool', 30),
}

# Convenience accessors
MIP_ENABLED: bool = MIP_SOLVER['enabled']
MIP_TIME_LIMIT: int = MIP_SOLVER['time_limit']
MIP_CANDIDATE_POOL: int = MIP_SOLVER['candidate_pool']


# =============================================================================
# LEAGUE OWNERSHIP ANALYSIS
# =============================================================================

_LEAGUE_CONFIG = _CONFIG.get('league', {})

LEAGUE: Dict[str, Any] = {
    'league_id': _LEAGUE_CONFIG.get('league_id'),
    'sample_size': _LEAGUE_CONFIG.get('sample_size', 20),
}

# Convenience accessors
LEAGUE_ID: Optional[int] = LEAGUE['league_id']
LEAGUE_SAMPLE_SIZE: int = LEAGUE['sample_size']


# =============================================================================
# CACHE SETTINGS
# =============================================================================

_CACHE_CONFIG = _CONFIG.get('cache', {})

CACHE: Dict[str, Any] = {
    'session_ttl': _CACHE_CONFIG.get('session_ttl', 3600),
    'max_sessions': _CACHE_CONFIG.get('max_sessions', 10),
    'auto_cleanup': _CACHE_CONFIG.get('auto_cleanup', True),
}

# Convenience accessors
CACHE_SESSION_TTL: int = CACHE['session_ttl']
CACHE_MAX_SESSIONS: int = CACHE['max_sessions']
CACHE_AUTO_CLEANUP: bool = CACHE['auto_cleanup']


# =============================================================================
# DATA SETTINGS
# =============================================================================

_DATA_CONFIG = _CONFIG.get('data', {})

DATA: Dict[str, Any] = {
    'staleness_hours': _DATA_CONFIG.get('staleness_hours', 6),
    'min_minutes': _DATA_CONFIG.get('min_minutes', 90),
    'top_players_pool': _DATA_CONFIG.get('top_players_pool', 200),
    'use_cross_season_data': _DATA_CONFIG.get('use_cross_season_data', True),
}

# Convenience accessors
DATA_STALENESS_HOURS: int = DATA['staleness_hours']
DATA_MIN_MINUTES: int = DATA['min_minutes']
DATA_TOP_PLAYERS_POOL: int = DATA['top_players_pool']
DATA_USE_CROSS_SEASON: bool = DATA['use_cross_season_data']


# =============================================================================
# COMPETITIVE ANALYSIS
# =============================================================================

_COMPETITIVE_CONFIG = _CONFIG.get('competitive', {})

COMPETITIVE: Dict[str, Any] = {
    'top_global_count': _COMPETITIVE_CONFIG.get('top_global_count', 5),
    'transfer_history_gws': _COMPETITIVE_CONFIG.get('transfer_history_gws', 5),
}

# Convenience accessors
TOP_GLOBAL_COUNT: int = COMPETITIVE['top_global_count']
TRANSFER_HISTORY_GWS: int = COMPETITIVE['transfer_history_gws']


# =============================================================================
# OUTPUT OPTIONS
# =============================================================================

_OUTPUT_CONFIG = _CONFIG.get('output', {})

OUTPUT: Dict[str, Any] = {
    'no_competitive': _OUTPUT_CONFIG.get('no_competitive', False),
    'verbose': _OUTPUT_CONFIG.get('verbose', False),
}

# Convenience accessors
NO_COMPETITIVE: bool = OUTPUT['no_competitive']
VERBOSE: bool = OUTPUT['verbose']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_path(subpath: str = '') -> Path:
    """Get path to data directory for the configured season.
    
    Args:
        subpath: Optional subdirectory or file within the season folder.
        
    Returns:
        Path to data/{SEASON}/{subpath}
    """
    base = _get_project_root() / 'data' / SEASON
    if subpath:
        return base / subpath
    return base


def reload_config() -> None:
    """Reload configuration from disk.
    
    Updates all module-level constants. Useful for testing or when
    config.yml changes during runtime.
    """
    global _CONFIG, TEAM_ID, COMPETITORS, GAMEWEEK, SEASON
    global FREE_HIT, FREE_HIT_TARGET_GW, FREE_HIT_STRATEGY
    global WILDCARD, WILDCARD_STRATEGY
    global TRANSFER_PLANNER, TRANSFER_HORIZON, FREE_TRANSFERS_OVERRIDE
    global MIP_SOLVER, MIP_ENABLED, MIP_TIME_LIMIT, MIP_CANDIDATE_POOL
    global LEAGUE, LEAGUE_ID, LEAGUE_SAMPLE_SIZE
    global CACHE, CACHE_SESSION_TTL, CACHE_MAX_SESSIONS, CACHE_AUTO_CLEANUP
    global DATA, DATA_STALENESS_HOURS, DATA_MIN_MINUTES, DATA_TOP_PLAYERS_POOL
    global COMPETITIVE, TOP_GLOBAL_COUNT, TRANSFER_HISTORY_GWS
    global OUTPUT, NO_COMPETITIVE, VERBOSE
    
    _CONFIG = load_config()
    
    # Reload all values
    TEAM_ID = _CONFIG.get('team_id', 847569)
    COMPETITORS = _CONFIG.get('competitors', [21023, 1827604, 489166])
    GAMEWEEK = _CONFIG.get('gameweek')
    SEASON = _CONFIG.get('season', '2025-26')
    
    _fh = _CONFIG.get('free_hit', {})
    FREE_HIT = {'target_gw': _fh.get('target_gw'), 'strategy': _fh.get('strategy', 'balanced')}
    FREE_HIT_TARGET_GW = FREE_HIT['target_gw']
    FREE_HIT_STRATEGY = FREE_HIT['strategy']
    
    _wc = _CONFIG.get('wildcard', {})
    WILDCARD = {'strategy': _wc.get('strategy', 'balanced')}
    WILDCARD_STRATEGY = WILDCARD['strategy']
    
    _tp = _CONFIG.get('transfer_planner', {})
    TRANSFER_PLANNER = {'horizon': _tp.get('horizon', 5), 'free_transfers': _tp.get('free_transfers')}
    TRANSFER_HORIZON = TRANSFER_PLANNER['horizon']
    FREE_TRANSFERS_OVERRIDE = TRANSFER_PLANNER['free_transfers']
    
    _mip = _CONFIG.get('mip_solver', {})
    MIP_SOLVER = {'enabled': _mip.get('enabled', True), 'time_limit': _mip.get('time_limit', 60), 'candidate_pool': _mip.get('candidate_pool', 30)}
    MIP_ENABLED = MIP_SOLVER['enabled']
    MIP_TIME_LIMIT = MIP_SOLVER['time_limit']
    MIP_CANDIDATE_POOL = MIP_SOLVER['candidate_pool']
    
    _lg = _CONFIG.get('league', {})
    LEAGUE = {'league_id': _lg.get('league_id'), 'sample_size': _lg.get('sample_size', 20)}
    LEAGUE_ID = LEAGUE['league_id']
    LEAGUE_SAMPLE_SIZE = LEAGUE['sample_size']
    
    _cache = _CONFIG.get('cache', {})
    CACHE = {'session_ttl': _cache.get('session_ttl', 3600), 'max_sessions': _cache.get('max_sessions', 10), 'auto_cleanup': _cache.get('auto_cleanup', True)}
    CACHE_SESSION_TTL = CACHE['session_ttl']
    CACHE_MAX_SESSIONS = CACHE['max_sessions']
    CACHE_AUTO_CLEANUP = CACHE['auto_cleanup']
    
    _data = _CONFIG.get('data', {})
    DATA = {
        'staleness_hours': _data.get('staleness_hours', 6), 
        'min_minutes': _data.get('min_minutes', 90), 
        'top_players_pool': _data.get('top_players_pool', 200),
        'use_cross_season_data': _data.get('use_cross_season_data', True)
    }
    DATA_STALENESS_HOURS = DATA['staleness_hours']
    DATA_MIN_MINUTES = DATA['min_minutes']
    DATA_TOP_PLAYERS_POOL = DATA['top_players_pool']
    DATA_USE_CROSS_SEASON = DATA['use_cross_season_data']
    
    _comp = _CONFIG.get('competitive', {})
    COMPETITIVE = {'top_global_count': _comp.get('top_global_count', 5), 'transfer_history_gws': _comp.get('transfer_history_gws', 5)}
    TOP_GLOBAL_COUNT = COMPETITIVE['top_global_count']
    TRANSFER_HISTORY_GWS = COMPETITIVE['transfer_history_gws']
    
    _out = _CONFIG.get('output', {})
    OUTPUT = {'no_competitive': _out.get('no_competitive', False), 'verbose': _out.get('verbose', False)}
    NO_COMPETITIVE = OUTPUT['no_competitive']
    VERBOSE = OUTPUT['verbose']

