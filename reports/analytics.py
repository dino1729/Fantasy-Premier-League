"""Analytics Module - The "Why" Engine

Provides explainable AI for transfer decisions by calculating ROI.

Key Question: "Why is the bot selling my best player?"

Answer: We compare the Optimal Strategy vs a Baseline (no transfers) and
calculate the net gain after accounting for transfer costs.

Example Output:
"This transfer plan costs 4 points but generates 12.5 extra points over 5 weeks.
Net Gain: +8.5 points."
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TransferROI:
    """Return on Investment analysis for a transfer."""
    player_out: str
    player_in: str
    position: str
    cost: int  # Hit cost (0 or 4)
    xp_out: float  # Expected points if kept
    xp_in: float  # Expected points from replacement
    xp_diff: float  # Raw difference
    net_gain: float  # After hit cost
    payback_weeks: Optional[float] = None  # Weeks to break even
    reasoning: str = ''
    confidence: str = 'medium'


@dataclass
class StrategyROI:
    """Complete ROI analysis for a strategy."""
    baseline_xp: float  # Points with no transfers
    optimal_xp: float  # Points with optimal transfers
    gross_gain: float  # optimal - baseline
    hit_cost: int  # Total hit points
    net_gain: float  # gross - hits
    roi_percentage: float  # net_gain / hit_cost (if hits > 0)
    transfers: List[TransferROI] = field(default_factory=list)
    captain_analysis: Dict = field(default_factory=dict)
    risk_assessment: str = 'medium'
    recommendation: str = ''


class ROICalculator:
    """Calculates Return on Investment for transfer decisions.
    
    Compares:
    1. Baseline: Keep current squad, no transfers
    2. Optimal: Make recommended transfers
    
    Then calculates net expected gain after hit costs.
    """
    
    HIT_COST = 4  # Points per extra transfer
    
    def __init__(self, projections_df: Optional[pd.DataFrame] = None):
        """Initialize ROI calculator.
        
        Args:
            projections_df: DataFrame with xP projections per player/GW.
        """
        if projections_df is None:
            from etl.transformers import load_parquet
            self.projections_df = load_parquet('projections_horizon.parquet')
        else:
            self.projections_df = projections_df
        
        # Build xP lookup: player_id -> {gw: xp}
        self._build_xp_lookup()
    
    def _build_xp_lookup(self):
        """Build lookup table for expected points."""
        self.xp_lookup = {}
        
        for _, row in self.projections_df.iterrows():
            pid = int(row['player_id'])
            gw = int(row['gameweek'])
            xp = float(row['xp'])
            
            if pid not in self.xp_lookup:
                self.xp_lookup[pid] = {}
            self.xp_lookup[pid][gw] = xp
    
    def get_player_xp(self, player_id: int, 
                      start_gw: int, 
                      horizon: int = 5) -> float:
        """Get total expected points for a player over horizon."""
        total = 0.0
        player_xp = self.xp_lookup.get(player_id, {})
        
        for w in range(horizon):
            gw = start_gw + w
            total += player_xp.get(gw, 0.0)
        
        return total
    
    def calculate_baseline(self, current_squad: List[Dict],
                           start_gw: int,
                           horizon: int = 5) -> float:
        """Calculate expected points with no transfers.
        
        Args:
            current_squad: List of player dicts with 'id' key.
            start_gw: First gameweek to project.
            horizon: Number of weeks to project.
            
        Returns:
            Total expected points for current squad over horizon.
        """
        total_xp = 0.0
        
        # Get top 11 by xP each week (simplified - assumes best lineup)
        for w in range(horizon):
            gw = start_gw + w
            
            # Get xP for each squad player this GW
            weekly_xp = []
            for player in current_squad:
                pid = player.get('id') or player.get('player_id')
                if pid:
                    xp = self.xp_lookup.get(pid, {}).get(gw, 0.0)
                    weekly_xp.append((pid, xp, player.get('position', 'UNK')))
            
            # Select best 11 (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)
            best_11_xp = self._select_best_11(weekly_xp)
            
            # Add captain bonus (double captain points)
            if best_11_xp:
                captain_xp = max(best_11_xp)
                total_xp += sum(best_11_xp) + captain_xp  # Captain gets 2x
        
        return total_xp
    
    def _select_best_11(self, player_xp: List[Tuple[int, float, str]]) -> List[float]:
        """Select best 11 players from squad respecting formation rules."""
        # Group by position
        by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for pid, xp, pos in player_xp:
            if pos in by_pos:
                by_pos[pos].append(xp)
        
        # Sort each position by xP
        for pos in by_pos:
            by_pos[pos].sort(reverse=True)
        
        # Build best 11 (1 GK + 10 outfield)
        selected = []
        
        # 1 GK
        if by_pos['GKP']:
            selected.append(by_pos['GKP'][0])
        
        # Minimum: 3 DEF, 2 MID, 1 FWD
        selected.extend(by_pos['DEF'][:3])
        selected.extend(by_pos['MID'][:2])
        selected.extend(by_pos['FWD'][:1])
        
        # Fill remaining 4 spots with highest remaining
        remaining = (
            by_pos['DEF'][3:] + 
            by_pos['MID'][2:] + 
            by_pos['FWD'][1:]
        )
        remaining.sort(reverse=True)
        selected.extend(remaining[:4])
        
        return selected[:11]
    
    def calculate_transfer_roi(self, player_out: Dict,
                               player_in: Dict,
                               start_gw: int,
                               horizon: int = 5,
                               is_hit: bool = False) -> TransferROI:
        """Calculate ROI for a single transfer.
        
        Args:
            player_out: Dict with player being sold.
            player_in: Dict with player being bought.
            start_gw: First gameweek for analysis.
            horizon: Number of weeks to analyze.
            is_hit: Whether this transfer incurs a -4 hit.
            
        Returns:
            TransferROI with gain analysis.
        """
        out_id = player_out.get('id') or player_out.get('player_id')
        in_id = player_in.get('id') or player_in.get('player_id')
        
        # Calculate xP over horizon
        xp_out = self.get_player_xp(out_id, start_gw, horizon) if out_id else 0
        xp_in = self.get_player_xp(in_id, start_gw, horizon) if in_id else 0
        
        xp_diff = xp_in - xp_out
        hit_cost = self.HIT_COST if is_hit else 0
        net_gain = xp_diff - hit_cost
        
        # Calculate payback period
        if is_hit and xp_diff > 0:
            weekly_gain = xp_diff / horizon
            payback_weeks = hit_cost / weekly_gain if weekly_gain > 0 else None
        else:
            payback_weeks = None
        
        # Generate reasoning
        if net_gain > 5:
            reasoning = f"Strong upgrade: +{net_gain:.1f} net xP over {horizon} GWs"
            confidence = 'high'
        elif net_gain > 0:
            reasoning = f"Positive move: +{net_gain:.1f} net xP after{'hit cost' if is_hit else ' free transfer'}"
            confidence = 'medium'
        elif net_gain > -2:
            reasoning = f"Marginal benefit: {net_gain:.1f} xP difference - consider waiting"
            confidence = 'low'
        else:
            reasoning = f"Questionable: {net_gain:.1f} xP - transfer may not be worth it"
            confidence = 'low'
        
        return TransferROI(
            player_out=player_out.get('name', 'Unknown'),
            player_in=player_in.get('name', 'Unknown'),
            position=player_in.get('position', 'UNK'),
            cost=hit_cost,
            xp_out=round(xp_out, 1),
            xp_in=round(xp_in, 1),
            xp_diff=round(xp_diff, 1),
            net_gain=round(net_gain, 1),
            payback_weeks=round(payback_weeks, 1) if payback_weeks else None,
            reasoning=reasoning,
            confidence=confidence,
        )
    
    def analyze_strategy(self, current_squad: List[Dict],
                         transfers_out: List[Dict],
                         transfers_in: List[Dict],
                         start_gw: int,
                         horizon: int = 5,
                         free_transfers: int = 1) -> StrategyROI:
        """Analyze complete transfer strategy ROI.
        
        Args:
            current_squad: Current 15-player squad.
            transfers_out: Players being sold.
            transfers_in: Players being bought.
            start_gw: First gameweek for analysis.
            horizon: Planning horizon.
            free_transfers: Available free transfers.
            
        Returns:
            Complete StrategyROI analysis.
        """
        # Calculate baseline (no transfers)
        baseline_xp = self.calculate_baseline(current_squad, start_gw, horizon)
        
        # Build new squad
        new_squad = [p for p in current_squad 
                    if p.get('id') not in {t.get('id') for t in transfers_out}]
        new_squad.extend(transfers_in)
        
        # Calculate optimal (with transfers)
        optimal_xp = self.calculate_baseline(new_squad, start_gw, horizon)
        
        # Gross gain
        gross_gain = optimal_xp - baseline_xp
        
        # Hit cost
        num_transfers = len(transfers_in)
        hits = max(0, num_transfers - free_transfers)
        hit_cost = hits * self.HIT_COST
        
        # Net gain
        net_gain = gross_gain - hit_cost
        
        # ROI percentage
        roi_pct = (net_gain / hit_cost * 100) if hit_cost > 0 else float('inf')
        
        # Analyze individual transfers
        transfer_rois = []
        for i, (out, inp) in enumerate(zip(transfers_out, transfers_in)):
            is_hit = i >= free_transfers
            roi = self.calculate_transfer_roi(out, inp, start_gw, horizon, is_hit)
            transfer_rois.append(roi)
        
        # Risk assessment
        if net_gain > 10:
            risk = 'low'
            recommendation = f"STRONG BUY: {net_gain:.1f} net expected gain justifies the moves"
        elif net_gain > 5:
            risk = 'medium'
            recommendation = f"RECOMMENDED: {net_gain:.1f} xP gain is worthwhile"
        elif net_gain > 0:
            risk = 'medium'
            recommendation = f"MARGINAL: {net_gain:.1f} xP gain - proceed with caution"
        else:
            risk = 'high'
            recommendation = f"NOT RECOMMENDED: {net_gain:.1f} xP - consider holding transfers"
        
        return StrategyROI(
            baseline_xp=round(baseline_xp, 1),
            optimal_xp=round(optimal_xp, 1),
            gross_gain=round(gross_gain, 1),
            hit_cost=hit_cost,
            net_gain=round(net_gain, 1),
            roi_percentage=round(roi_pct, 1) if hit_cost > 0 else 0,
            transfers=transfer_rois,
            risk_assessment=risk,
            recommendation=recommendation,
        )
    
    def explain_decision(self, transfer_roi: TransferROI) -> str:
        """Generate human-readable explanation for a transfer decision.
        
        Args:
            transfer_roi: ROI analysis for the transfer.
            
        Returns:
            Explanation string.
        """
        lines = [
            f"Transfer: {transfer_roi.player_out} → {transfer_roi.player_in}",
            f"Position: {transfer_roi.position}",
            "",
            f"Expected points comparison (5 GWs):",
            f"  • {transfer_roi.player_out}: {transfer_roi.xp_out} xP",
            f"  • {transfer_roi.player_in}: {transfer_roi.xp_in} xP",
            f"  • Raw gain: {transfer_roi.xp_diff:+.1f} xP",
        ]
        
        if transfer_roi.cost > 0:
            lines.extend([
                "",
                f"Transfer cost: -{transfer_roi.cost} points (hit)",
                f"Net gain: {transfer_roi.net_gain:+.1f} xP",
            ])
            
            if transfer_roi.payback_weeks:
                lines.append(f"Payback period: {transfer_roi.payback_weeks:.1f} weeks")
        else:
            lines.extend([
                "",
                "Transfer cost: FREE",
                f"Net gain: {transfer_roi.net_gain:+.1f} xP",
            ])
        
        lines.extend([
            "",
            f"Assessment: {transfer_roi.reasoning}",
            f"Confidence: {transfer_roi.confidence.upper()}",
        ])
        
        return '\n'.join(lines)
    
    def generate_summary(self, strategy_roi: StrategyROI) -> str:
        """Generate summary report for strategy ROI.
        
        Args:
            strategy_roi: Complete strategy analysis.
            
        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 60,
            "TRANSFER STRATEGY ROI ANALYSIS",
            "=" * 60,
            "",
            "EXPECTED POINTS COMPARISON:",
            f"  Baseline (no transfers): {strategy_roi.baseline_xp} xP",
            f"  With transfers: {strategy_roi.optimal_xp} xP",
            f"  Gross gain: {strategy_roi.gross_gain:+.1f} xP",
            "",
        ]
        
        if strategy_roi.hit_cost > 0:
            lines.extend([
                f"TRANSFER COST:",
                f"  Hit points: -{strategy_roi.hit_cost} pts",
                f"  Net gain: {strategy_roi.net_gain:+.1f} xP",
                f"  ROI: {strategy_roi.roi_percentage:.0f}%",
                "",
            ])
        else:
            lines.extend([
                f"Transfer cost: FREE (using free transfer)",
                f"Net gain: {strategy_roi.net_gain:+.1f} xP",
                "",
            ])
        
        lines.extend([
            f"RECOMMENDATION: {strategy_roi.recommendation}",
            f"Risk Level: {strategy_roi.risk_assessment.upper()}",
            "",
            "-" * 60,
            "TRANSFER BREAKDOWN:",
            "-" * 60,
        ])
        
        for i, t in enumerate(strategy_roi.transfers, 1):
            lines.extend([
                f"\n{i}. {t.player_out} → {t.player_in} ({t.position})",
                f"   xP: {t.xp_out} → {t.xp_in} ({t.xp_diff:+.1f})",
                f"   {'HIT: -4 pts | ' if t.cost > 0 else ''}Net: {t.net_gain:+.1f} xP",
                f"   {t.reasoning}",
            ])
        
        return '\n'.join(lines)


def analyze_transfer_roi(current_squad: List[Dict],
                         transfers_out: List[Dict],
                         transfers_in: List[Dict],
                         start_gw: int,
                         horizon: int = 5,
                         free_transfers: int = 1) -> StrategyROI:
    """Convenience function for ROI analysis.
    
    Returns:
        StrategyROI with complete analysis.
    """
    calculator = ROICalculator()
    return calculator.analyze_strategy(
        current_squad, transfers_out, transfers_in,
        start_gw, horizon, free_transfers
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    calculator = ROICalculator()
    
    # Sample transfer
    roi = calculator.calculate_transfer_roi(
        {'id': 1, 'name': 'Salah', 'position': 'MID'},
        {'id': 2, 'name': 'Palmer', 'position': 'MID'},
        start_gw=17,
        horizon=5,
        is_hit=False
    )
    
    print(calculator.explain_decision(roi))

