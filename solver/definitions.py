"""Solver type definitions and data classes."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Constants
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POSITION_QUOTAS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
MAX_PER_TEAM = 3

@dataclass
class MIPSolverResult:
    """Result from the MIP transfer optimizer."""
    status: str  # 'optimal', 'infeasible', 'timeout', 'error', 'unavailable'
    transfers_out: List[Dict] = field(default_factory=list)
    transfers_in: List[Dict] = field(default_factory=list)
    new_squad: List[Dict] = field(default_factory=list)
    starting_xi: List[Dict] = field(default_factory=list)
    bench: List[Dict] = field(default_factory=list)
    formation: str = ''
    captain: Optional[Dict] = None
    vice_captain: Optional[Dict] = None
    hit_cost: int = 0
    num_transfers: int = 0
    free_transfers_used: int = 0
    budget_remaining: float = 0.0
    expected_points: float = 0.0
    per_gw_xp: List[float] = field(default_factory=list)
    solver_time: float = 0.0
    message: str = ''
