"""Strategy Reporter Module

Generates human-readable reports for transfer strategies.

Outputs:
- ASCII tables for terminal display
- Transfer schedules with ROI analysis  
- Squad depth maps with problem area highlighting
- Chip usage recommendations

Uses tabulate for clean table formatting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Try importing tabulate, provide fallback
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

from solver.interpreter import StrategyPlan, WeeklyPlan
from reports.analytics import StrategyROI, TransferROI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class StrategyReporter:
    """Generates formatted reports for transfer strategies."""
    
    def __init__(self, use_colors: bool = True):
        """Initialize reporter.
        
        Args:
            use_colors: Whether to use ANSI colors in output.
        """
        self.use_colors = use_colors
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.END}"
        return text
    
    def print_header(self, title: str):
        """Print formatted section header."""
        width = 60
        print()
        print(self._color("=" * width, Colors.BLUE))
        print(self._color(f"  {title}", Colors.BOLD))
        print(self._color("=" * width, Colors.BLUE))
        print()
    
    def print_subheader(self, title: str):
        """Print formatted subsection header."""
        print()
        print(self._color(f"─── {title} ───", Colors.CYAN))
        print()
    
    def print_transfer_schedule(self, plan: StrategyPlan):
        """Print transfer schedule as ASCII table.
        
        Example:
        | GW | Transfer In    | Transfer Out   | Cost | Cap | Net xP |
        |----|----------------|----------------|------|-----|--------|
        | 17 | Cole Palmer    | Mo Salah       | -4   | HAA | +3.2   |
        | 18 | (Roll)         | -              | 0    | PAL | +0.0   |
        """
        self.print_header("TRANSFER SCHEDULE")
        
        rows = []
        for gw, weekly in sorted(plan.weekly_plans.items()):
            # Transfer info
            if weekly.transfers_in:
                t_in = ', '.join([t.get('name', '?')[:12] for t in weekly.transfers_in])
            else:
                t_in = '(Roll)'
            
            if weekly.transfers_out:
                t_out = ', '.join([t.get('name', '?')[:12] for t in weekly.transfers_out])
            else:
                t_out = '-'
            
            # Cost
            cost = f"-{weekly.hits_taken}" if weekly.hits_taken > 0 else "FREE"
            
            # Captain
            cap = weekly.captain.get('name', '?')[:3].upper() if weekly.captain else '?'
            
            # Expected points
            xp = f"+{weekly.expected_points:.1f}" if weekly.expected_points > 0 else f"{weekly.expected_points:.1f}"
            
            # Chip
            chip = weekly.chip_played.upper()[:3] if weekly.chip_played else '-'
            
            rows.append([
                f"GW{gw}",
                t_in,
                t_out,
                cost,
                cap,
                chip,
                xp
            ])
        
        headers = ['GW', 'Transfer In', 'Transfer Out', 'Cost', 'Cap', 'Chip', 'Net xP']
        
        if TABULATE_AVAILABLE:
            print(tabulate(rows, headers=headers, tablefmt='simple'))
        else:
            self._print_simple_table(headers, rows)
        
        # Summary
        print()
        print(f"Total Expected Points: {self._color(f'{plan.total_expected_points:.1f}', Colors.GREEN)}")
        print(f"Total Hits: {self._color(f'-{plan.total_hits}', Colors.RED if plan.total_hits > 0 else Colors.GREEN)}")
        print(f"Net Gain: {self._color(f'+{plan.net_expected_gain:.1f}', Colors.GREEN if plan.net_expected_gain > 0 else Colors.RED)}")
    
    def print_squad_map(self, plan: StrategyPlan, 
                        fixtures_df: Optional[pd.DataFrame] = None):
        """Print squad depth map with problem area highlighting.
        
        Format:
        GKP: Raya (ARS) ★ | Flekken (BRE)
        DEF: Gabriel (ARS) | Saliba (ARS) | Timber (ARS) ⚠️ | ...
        MID: Saka (ARS) ★★ | Palmer (CHE) | ...
        FWD: Haaland (MCI) ★★★ | Isak (NEW) | ...
        
        ★ = Form indicator, ⚠️ = Injury/rotation risk, 🔴 = Tough fixtures
        """
        self.print_subheader("SQUAD DEPTH MAP")
        
        # Get first week's lineup
        first_gw = min(plan.weekly_plans.keys()) if plan.weekly_plans else None
        if not first_gw:
            print("No squad data available")
            return
        
        weekly = plan.weekly_plans[first_gw]
        all_players = weekly.starting_xi + weekly.bench
        
        # Group by position
        by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for p in all_players:
            pos = p.get('position', 'UNK')
            if pos in by_pos:
                # Add indicators
                name = p.get('name', '?')
                team = p.get('team', '?')[:3]
                
                # Check if in starting XI
                in_xi = p in weekly.starting_xi
                
                # Captain indicator
                is_cap = weekly.captain and p.get('id') == weekly.captain.get('id')
                is_vc = weekly.vice_captain and p.get('id') == weekly.vice_captain.get('id')
                
                indicators = ''
                if is_cap:
                    indicators += ' (C)'
                elif is_vc:
                    indicators += ' (V)'
                
                if not in_xi:
                    indicators += ' [B]'  # Bench
                
                by_pos[pos].append(f"{name} ({team}){indicators}")
        
        # Print each position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            players = by_pos[pos]
            pos_color = {
                'GKP': Colors.YELLOW,
                'DEF': Colors.GREEN,
                'MID': Colors.CYAN,
                'FWD': Colors.RED,
            }.get(pos, Colors.END)
            
            print(f"{self._color(f'{pos}:', pos_color)} {' | '.join(players)}")
        
        print()
        print(f"Formation: {self._color(weekly.formation, Colors.BOLD)}")
        print(f"Budget Remaining: £{weekly.budget_remaining:.1f}m")
    
    def print_roi_analysis(self, roi: StrategyROI):
        """Print ROI analysis for the strategy."""
        self.print_subheader("RETURN ON INVESTMENT ANALYSIS")
        
        # Main comparison
        print(f"Expected Points (No Transfers): {roi.baseline_xp:.1f}")
        print(f"Expected Points (With Transfers): {roi.optimal_xp:.1f}")
        print()
        
        gain_color = Colors.GREEN if roi.gross_gain > 0 else Colors.RED
        print(f"Gross Gain: {self._color(f'{roi.gross_gain:+.1f} xP', gain_color)}")
        
        if roi.hit_cost > 0:
            print(f"Hit Cost: {self._color(f'-{roi.hit_cost} pts', Colors.RED)}")
        
        net_color = Colors.GREEN if roi.net_gain > 0 else Colors.RED
        print(f"Net Gain: {self._color(f'{roi.net_gain:+.1f} xP', net_color)}")
        
        if roi.hit_cost > 0:
            print(f"ROI: {roi.roi_percentage:.0f}%")
        
        # Recommendation
        print()
        rec_color = Colors.GREEN if 'STRONG' in roi.recommendation or 'RECOMMENDED' in roi.recommendation else Colors.YELLOW
        print(self._color(f"⚡ {roi.recommendation}", rec_color))
        
        # Individual transfers
        if roi.transfers:
            self.print_subheader("TRANSFER BREAKDOWN")
            
            rows = []
            for t in roi.transfers:
                cost_str = f"-{t.cost}" if t.cost > 0 else "FREE"
                gain_str = f"{t.net_gain:+.1f}"
                rows.append([
                    f"{t.player_out} → {t.player_in}",
                    t.position,
                    f"{t.xp_out:.1f} → {t.xp_in:.1f}",
                    cost_str,
                    gain_str,
                    t.confidence.upper()
                ])
            
            headers = ['Transfer', 'Pos', 'xP', 'Cost', 'Net', 'Conf']
            
            if TABULATE_AVAILABLE:
                print(tabulate(rows, headers=headers, tablefmt='simple'))
            else:
                self._print_simple_table(headers, rows)
    
    def print_chip_alerts(self, plan: StrategyPlan):
        """Print chip usage alerts."""
        if not plan.chips_to_play:
            return
        
        self.print_subheader("⚠️ CHIP STRATEGY ALERTS")
        
        for gw, chip in plan.chips_to_play:
            chip_name = chip.upper().replace('_', ' ')
            alert = f"ACTIVATE {chip_name} IN GW{gw}"
            print(self._color(f"  🎯 {alert}", Colors.YELLOW + Colors.BOLD))
        
        print()
    
    def print_warnings(self, plan: StrategyPlan):
        """Print any warnings from the plan."""
        if not plan.warnings:
            return
        
        self.print_subheader("⚠️ WARNINGS")
        
        for warning in plan.warnings:
            if 'VIOLATION' in warning:
                color = Colors.RED
            elif 'ALERT' in warning:
                color = Colors.YELLOW
            else:
                color = Colors.YELLOW
            
            print(self._color(f"  {warning}", color))
        
        print()
    
    def print_full_report(self, plan: StrategyPlan, 
                          roi: Optional[StrategyROI] = None):
        """Print complete strategy report."""
        self.print_header(f"FPL STRATEGY REPORT - GW{plan.current_gw}")
        
        print(f"Generated: {plan.generated_at[:19]}")
        print(f"Planning Horizon: {plan.horizon} gameweeks")
        
        # Transfer schedule
        self.print_transfer_schedule(plan)
        
        # Squad map
        self.print_squad_map(plan)
        
        # ROI analysis
        if roi:
            self.print_roi_analysis(roi)
        
        # Chip alerts
        self.print_chip_alerts(plan)
        
        # Warnings
        self.print_warnings(plan)
        
        # Footer
        print()
        print(self._color("=" * 60, Colors.BLUE))
        print(self._color("  END OF REPORT", Colors.BOLD))
        print(self._color("=" * 60, Colors.BLUE))
        print()
    
    def _print_simple_table(self, headers: List[str], rows: List[List]):
        """Print simple table without tabulate."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Print header
        header_line = ' | '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print('-' * len(header_line))
        
        # Print rows
        for row in rows:
            row_line = ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            print(row_line)


def print_strategy_report(plan: StrategyPlan,
                          roi: Optional[StrategyROI] = None,
                          use_colors: bool = True):
    """Convenience function to print full strategy report.
    
    Args:
        plan: StrategyPlan from solver interpreter.
        roi: Optional ROI analysis.
        use_colors: Whether to use terminal colors.
    """
    reporter = StrategyReporter(use_colors=use_colors)
    reporter.print_full_report(plan, roi)


def generate_cli_report(mip_result: Dict,
                        current_squad: List[Dict],
                        current_gw: int,
                        horizon: int = 5,
                        use_colors: bool = True) -> str:
    """Generate complete CLI report from MIP result.
    
    Args:
        mip_result: Dict from MIP solver.
        current_squad: Current squad for ROI baseline.
        current_gw: Current gameweek.
        horizon: Planning horizon.
        use_colors: Whether to use terminal colors.
        
    Returns:
        Formatted report string.
    """
    from solver.interpreter import SolverInterpreter
    from reports.analytics import ROICalculator
    
    # Interpret solver output
    interpreter = SolverInterpreter()
    plan = interpreter.interpret_mip_result(mip_result, current_gw, horizon)
    
    # Calculate ROI
    calculator = ROICalculator()
    roi = calculator.analyze_strategy(
        current_squad,
        mip_result.get('transfers_out', []),
        mip_result.get('transfers_in', []),
        current_gw + 1,
        horizon
    )
    
    # Print report
    print_strategy_report(plan, roi, use_colors)
    
    return plan


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    from solver.interpreter import StrategyPlan, WeeklyPlan
    
    # Create sample plan
    plan = StrategyPlan(current_gw=16, horizon=5)
    
    plan.weekly_plans[17] = WeeklyPlan(
        gameweek=17,
        transfers_in=[{'name': 'Palmer', 'position': 'MID', 'team': 'CHE'}],
        transfers_out=[{'name': 'Salah', 'position': 'MID', 'team': 'LIV'}],
        starting_xi=[
            {'id': 1, 'name': 'Raya', 'position': 'GKP', 'team': 'ARS'},
            {'id': 2, 'name': 'Gabriel', 'position': 'DEF', 'team': 'ARS'},
            {'id': 3, 'name': 'Saliba', 'position': 'DEF', 'team': 'ARS'},
            {'id': 4, 'name': 'Timber', 'position': 'DEF', 'team': 'ARS'},
            {'id': 5, 'name': 'Walker', 'position': 'DEF', 'team': 'MCI'},
            {'id': 6, 'name': 'Saka', 'position': 'MID', 'team': 'ARS'},
            {'id': 7, 'name': 'Palmer', 'position': 'MID', 'team': 'CHE'},
            {'id': 8, 'name': 'Gordon', 'position': 'MID', 'team': 'NEW'},
            {'id': 9, 'name': 'Foden', 'position': 'MID', 'team': 'MCI'},
            {'id': 10, 'name': 'Haaland', 'position': 'FWD', 'team': 'MCI'},
            {'id': 11, 'name': 'Isak', 'position': 'FWD', 'team': 'NEW'},
        ],
        bench=[
            {'id': 12, 'name': 'Flekken', 'position': 'GKP', 'team': 'BRE'},
            {'id': 13, 'name': 'Harwood-Bellis', 'position': 'DEF', 'team': 'SOU'},
            {'id': 14, 'name': 'Winks', 'position': 'MID', 'team': 'LEI'},
            {'id': 15, 'name': 'Archer', 'position': 'FWD', 'team': 'SOU'},
        ],
        captain={'id': 10, 'name': 'Haaland'},
        vice_captain={'id': 6, 'name': 'Saka'},
        formation='4-4-2',
        hits_taken=0,
        expected_points=65.2,
        budget_remaining=0.5
    )
    
    plan.total_expected_points = 325.0
    plan.total_hits = 4
    plan.net_expected_gain = 12.5
    
    # Print report
    reporter = StrategyReporter()
    reporter.print_full_report(plan)

