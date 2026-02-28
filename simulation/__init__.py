"""FPL Season Simulation Package.

This package provides tools for backtesting FPL strategies by simulating
full seasons with historical data.

Main components:
- SimulationEngine: Orchestrates the GW-by-GW simulation loop
- HistoricalDataAdapter: Loads and provides access to historical CSV data
- ChipOptimizer: Determines optimal chip timing
- AutoSubSimulator: Applies FPL auto-substitution rules
- InitialSquadBuilder: Builds starting squad with hybrid template
- RiskAdjustedCaptainSelector: Selects captain with consistency weighting
- NoTransferBaseline: Baseline strategy for comparison
- BacktestReportGenerator: Generates JSON and PDF reports
"""

from simulation.state import (
    GameweekState,
    PlayerState,
    TransferRecord,
    AutoSubRecord,
    GameweekDecisions,
    GameweekResults,
    SimulationResult,
    ChipType,
    INITIAL_CHIPS,
    STARTING_BUDGET,
    MAX_FREE_TRANSFERS,
    HIT_COST,
)
from simulation.data_adapter import HistoricalDataAdapter
from simulation.auto_sub import AutoSubSimulator
from simulation.chip_optimizer import ChipOptimizer
from simulation.squad_builder import InitialSquadBuilder
from simulation.captain_selector import RiskAdjustedCaptainSelector
from simulation.baseline import NoTransferBaseline
from simulation.report_generator import BacktestReportGenerator

__all__ = [
    # State classes
    'GameweekState',
    'PlayerState',
    'TransferRecord',
    'AutoSubRecord',
    'GameweekDecisions',
    'GameweekResults',
    'SimulationResult',
    'ChipType',
    # Constants
    'INITIAL_CHIPS',
    'STARTING_BUDGET',
    'MAX_FREE_TRANSFERS',
    'HIT_COST',
    # Components
    'HistoricalDataAdapter',
    'AutoSubSimulator',
    'ChipOptimizer',
    'InitialSquadBuilder',
    'RiskAdjustedCaptainSelector',
    'NoTransferBaseline',
    'BacktestReportGenerator',
]
