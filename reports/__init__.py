"""Reports module - FPL Report Generation.

This module contains the report generation pipeline:
- fpl_report/ - Core analysis and report modules
- analytics.py - ROI analysis
- strategy_reporter.py - Strategy reporting
"""

from .analytics import ROICalculator, analyze_transfer_roi
from .strategy_reporter import StrategyReporter, print_strategy_report

