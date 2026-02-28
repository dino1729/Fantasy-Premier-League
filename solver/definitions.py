"""Solver type definitions and data classes."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Constants
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POSITION_QUOTAS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
MAX_PER_TEAM = 3

# Scenario definitions for multi-period solver
SCENARIO_CONFIGS = {
    'conservative': {
        'max_hits': 0,
        'max_transfers_per_week': 1,
        'description': 'No hits, max 1 transfer per week'
    },
    'balanced': {
        'max_hits': 8,  # Up to 2 hits over horizon
        'max_transfers_per_week': 2,
        'description': 'Hits allowed if profitable, max 2 per week'
    },
    'aggressive': {
        'max_hits': 16,  # Up to 4 hits
        'max_transfers_per_week': 3,
        'description': 'Chase upside with multiple hits if needed'
    }
}


@dataclass
class WeeklyPlan:
    """Plan for a single gameweek within the planning horizon."""
    gameweek: int
    transfers_in: List[Dict] = field(default_factory=list)
    transfers_out: List[Dict] = field(default_factory=list)
    ft_available: int = 1
    ft_used: int = 0
    ft_remaining: int = 0
    hit_cost: int = 0
    lineup: List[Dict] = field(default_factory=list)
    bench: List[Dict] = field(default_factory=list)
    captain: Optional[Dict] = None
    differential_captain: Optional[Dict] = None  # Alternative lower-EO option
    formation: str = ''
    expected_xp: float = 0.0
    confidence: str = 'high'  # 'high', 'moderate', 'low'
    reasoning: str = ''
    is_hold: bool = False  # True if no transfers this week


@dataclass
class TransferAction:
    """A single transfer action with timing and context."""
    gameweek: int
    player_out: Dict
    player_in: Dict
    expected_gain: float  # xP gain over remaining horizon
    cost: str  # 'free' or '-4 hit'
    reasoning: str
    backup: Optional[Dict] = None  # Backup player if primary is doubtful
    backup_xp: float = 0.0
    price_alert: Optional[str] = None  # 'ACT BY Fri 6pm - predicted rise'
    is_sell_rebuy: bool = False  # Flagged if selling player you'll rebuy


@dataclass
class MIPSolverResult:
    """Result from the MIP transfer optimizer.

    Supports both single-period (legacy) and multi-period planning modes.
    """
    status: str  # 'optimal', 'infeasible', 'timeout', 'error', 'unavailable'

    # Transfer summary (aggregated across all weeks)
    transfers_out: List[Dict] = field(default_factory=list)
    transfers_in: List[Dict] = field(default_factory=list)
    new_squad: List[Dict] = field(default_factory=list)

    # First week lineup (for backward compatibility)
    starting_xi: List[Dict] = field(default_factory=list)
    bench: List[Dict] = field(default_factory=list)
    formation: str = ''
    captain: Optional[Dict] = None
    vice_captain: Optional[Dict] = None

    # Cost summary
    hit_cost: int = 0  # Total hits across all weeks
    num_transfers: int = 0  # Total transfers across all weeks
    free_transfers_used: int = 0
    budget_remaining: float = 0.0

    # Expected value
    expected_points: float = 0.0  # Total xP over horizon minus hits
    per_gw_xp: List[float] = field(default_factory=list)
    baseline_xp: float = 0.0  # xP if no transfers made (for comparison)

    # Multi-period planning (NEW)
    weekly_plans: List[WeeklyPlan] = field(default_factory=list)
    transfer_sequence: List[TransferAction] = field(default_factory=list)

    # Backup plans for injury doubts (NEW)
    backup_transfers: List[Dict] = field(default_factory=list)

    # Sell-rebuy warnings (NEW)
    sell_rebuy_warnings: List[str] = field(default_factory=list)

    # Confidence per gameweek (NEW)
    confidence_per_gw: List[str] = field(default_factory=list)

    # Price alerts (NEW)
    price_alerts: List[Dict] = field(default_factory=list)
    # Format: [{'player': 'Palmer', 'action': 'BUY', 'deadline': 'Fri 6pm', 'reason': 'predicted +0.1m'}]

    # Chip consideration (NEW)
    chip_recommendation: Optional[str] = None  # 'free_hit', 'bench_boost', None
    chip_reasoning: Optional[str] = None

    # Watchlist for zero-transfer scenarios (NEW)
    watchlist: List[Dict] = field(default_factory=list)

    # Solver metadata
    solver_time: float = 0.0
    optimality_gap: float = 0.0  # Gap from proven optimal (NEW)
    scenario: str = 'balanced'  # Which scenario this result is for (NEW)
    message: str = ''


@dataclass
class MultiPeriodResult:
    """Container for all three scenario results."""
    conservative: Optional[MIPSolverResult] = None
    balanced: Optional[MIPSolverResult] = None
    aggressive: Optional[MIPSolverResult] = None
    recommended: str = 'balanced'  # Which scenario is recommended
    chip_recommendation: Optional[str] = None
    chip_reasoning: Optional[str] = None
    watchlist: List[Dict] = field(default_factory=list)
    baseline_xp: float = 0.0
