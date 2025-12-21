"""Solver Interpreter Module

Translates raw MIP solver output into human-readable action plans.

The solver generates thousands of variable states (x[p,w] = 0 or 1).
This module extracts the "True" decisions and maps them to player names,
creating actionable weekly plans with transfers, lineups, and captaincy.

Output format:
{
    "GW12": {
        "transfers_in": ["Palmer"],
        "transfers_out": ["Salah"],
        "starting_xi": [...],
        "captain": "Haaland",
        "vice_captain": "Saka",
        "bench": [...],
        "chip_played": None,
        "hits_taken": 4,
        "expected_points": 65.2
    },
    ...
}
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PARQUET = PROJECT_ROOT / 'data' / 'parquet'


@dataclass
class WeeklyPlan:
    """Represents the action plan for a single gameweek."""
    gameweek: int
    transfers_in: List[Dict] = field(default_factory=list)
    transfers_out: List[Dict] = field(default_factory=list)
    starting_xi: List[Dict] = field(default_factory=list)
    bench: List[Dict] = field(default_factory=list)
    captain: Optional[Dict] = None
    vice_captain: Optional[Dict] = None
    formation: str = ''
    chip_played: Optional[str] = None  # 'wildcard', 'free_hit', 'bench_boost', 'triple_captain'
    hits_taken: int = 0
    free_transfers_used: int = 0
    expected_points: float = 0.0
    budget_remaining: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'gameweek': self.gameweek,
            'transfers_in': self.transfers_in,
            'transfers_out': self.transfers_out,
            'starting_xi': [p['name'] if isinstance(p, dict) else p for p in self.starting_xi],
            'bench': [p['name'] if isinstance(p, dict) else p for p in self.bench],
            'captain': self.captain.get('name') if self.captain else None,
            'vice_captain': self.vice_captain.get('name') if self.vice_captain else None,
            'formation': self.formation,
            'chip_played': self.chip_played,
            'hits_taken': self.hits_taken,
            'free_transfers_used': self.free_transfers_used,
            'expected_points': round(self.expected_points, 1),
            'budget_remaining': round(self.budget_remaining, 1),
        }


@dataclass
class StrategyPlan:
    """Complete multi-week strategy plan."""
    current_gw: int
    horizon: int
    weekly_plans: Dict[int, WeeklyPlan] = field(default_factory=dict)
    total_expected_points: float = 0.0
    total_hits: int = 0
    total_transfers: int = 0
    net_expected_gain: float = 0.0
    baseline_expected: float = 0.0  # Points without any transfers
    chips_to_play: List[Tuple[int, str]] = field(default_factory=list)  # (gw, chip_name)
    warnings: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'current_gw': self.current_gw,
            'horizon': self.horizon,
            'weekly_plans': {
                str(gw): plan.to_dict() 
                for gw, plan in self.weekly_plans.items()
            },
            'summary': {
                'total_expected_points': round(self.total_expected_points, 1),
                'total_hits': self.total_hits,
                'total_transfers': self.total_transfers,
                'net_expected_gain': round(self.net_expected_gain, 1),
                'baseline_expected': round(self.baseline_expected, 1),
            },
            'chips_to_play': [
                {'gameweek': gw, 'chip': chip} 
                for gw, chip in self.chips_to_play
            ],
            'warnings': self.warnings,
            'generated_at': self.generated_at,
        }
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save strategy plan to JSON."""
        if filepath is None:
            filepath = PROJECT_ROOT / 'reports' / f'strategy_plan_gw{self.current_gw}.json'
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved strategy plan to {filepath}")
        return filepath


class SolverInterpreter:
    """Interprets MIP solver output into actionable strategy plans.
    
    Maps solver variable states (binary 0/1 values) to human-readable
    decisions like "Buy Palmer, Sell Salah, Captain Haaland".
    """
    
    POSITION_ORDER = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    
    def __init__(self, players_df: Optional[pd.DataFrame] = None):
        """Initialize interpreter.
        
        Args:
            players_df: Players DataFrame for name lookup.
        """
        if players_df is None:
            from etl.transformers import load_parquet
            self.players_df = load_parquet('players.parquet')
        else:
            self.players_df = players_df
        
        # Build player lookup
        self._build_player_lookup()
    
    def _build_player_lookup(self):
        """Build lookup tables for player info."""
        self.player_lookup = {}
        
        for _, row in self.players_df.iterrows():
            pid = int(row['player_id'])
            self.player_lookup[pid] = {
                'id': pid,
                'name': row['web_name'],
                'full_name': row.get('full_name', row['web_name']),
                'position': row['position'],
                'team': row['team_name'],
                'team_id': int(row['team_id']),
                'price': row.get('cost_m', row.get('cost', 50) / 10.0),
            }
    
    def interpret_mip_result(self, mip_result: Dict, 
                             current_gw: int,
                             horizon: int = 5) -> StrategyPlan:
        """Interpret MIP solver result into a strategy plan.
        
        Args:
            mip_result: Dict from TransferMIPSolver with keys:
                - status, transfers_in, transfers_out, starting_xi, bench
                - captain, vice_captain, formation, hit_cost, etc.
            current_gw: Current gameweek number.
            horizon: Planning horizon in gameweeks.
            
        Returns:
            Complete StrategyPlan with weekly breakdowns.
        """
        plan = StrategyPlan(
            current_gw=current_gw,
            horizon=horizon
        )
        
        if mip_result.get('status') != 'optimal':
            plan.warnings.append(f"Solver status: {mip_result.get('status', 'unknown')}")
            plan.warnings.append(mip_result.get('message', 'No solution found'))
            return plan
        
        # Extract transfers
        transfers_in = self._enrich_players(mip_result.get('transfers_in', []))
        transfers_out = self._enrich_players(mip_result.get('transfers_out', []))
        
        # Extract lineup
        starting_xi = self._enrich_players(mip_result.get('starting_xi', []))
        bench = self._enrich_players(mip_result.get('bench', []))
        
        # Captain info
        captain = mip_result.get('captain')
        if captain and isinstance(captain, dict):
            captain = self._enrich_player(captain)
        
        vice_captain = mip_result.get('vice_captain')
        if vice_captain and isinstance(vice_captain, dict):
            vice_captain = self._enrich_player(vice_captain)
        
        # Per-GW expected points
        per_gw_xp = mip_result.get('per_gw_xp', [])
        
        # Build weekly plans
        for w in range(horizon):
            gw = current_gw + w + 1
            
            weekly = WeeklyPlan(gameweek=gw)
            
            # Transfers only in first week
            if w == 0:
                weekly.transfers_in = transfers_in
                weekly.transfers_out = transfers_out
                weekly.hits_taken = mip_result.get('hit_cost', 0)
                weekly.free_transfers_used = mip_result.get('free_transfers_used', 0)
            
            # Lineup and captain (same for all weeks in basic solver)
            weekly.starting_xi = starting_xi
            weekly.bench = bench
            weekly.captain = captain
            weekly.vice_captain = vice_captain
            weekly.formation = mip_result.get('formation', '')
            weekly.budget_remaining = mip_result.get('budget_remaining', 0)
            
            # Expected points for this week
            if w < len(per_gw_xp):
                weekly.expected_points = per_gw_xp[w]
            
            plan.weekly_plans[gw] = weekly
        
        # Summary stats
        plan.total_expected_points = mip_result.get('expected_points', 0)
        plan.total_hits = mip_result.get('hit_cost', 0)
        plan.total_transfers = mip_result.get('num_transfers', 0)
        
        # Detect chip usage
        self._detect_chips(mip_result, plan)
        
        # Validation warnings
        self._validate_plan(plan)
        
        return plan
    
    def interpret_heuristic_result(self, strategy_result: Dict,
                                   current_gw: int) -> StrategyPlan:
        """Interpret heuristic strategy result into a plan.
        
        Args:
            strategy_result: Dict from TransferStrategyPlanner.generate_strategy()
            current_gw: Current gameweek number.
            
        Returns:
            StrategyPlan converted from heuristic recommendations.
        """
        plan = StrategyPlan(
            current_gw=current_gw,
            horizon=strategy_result.get('planning_horizon', 5)
        )
        
        # Extract immediate recommendations
        recommendations = strategy_result.get('immediate_recommendations', [])
        planned_transfers = strategy_result.get('planned_transfers', [])
        
        # Expected value analysis
        ev = strategy_result.get('expected_value', {})
        plan.baseline_expected = ev.get('current_squad', 0)
        plan.total_expected_points = ev.get('optimized_squad', 0)
        plan.net_expected_gain = ev.get('potential_gain', 0)
        
        # Build weekly plans from planned transfers
        for transfer in planned_transfers:
            gw = transfer.get('gameweek', current_gw + 1)
            
            if gw not in plan.weekly_plans:
                plan.weekly_plans[gw] = WeeklyPlan(gameweek=gw)
            
            weekly = plan.weekly_plans[gw]
            
            # Add transfer
            if transfer.get('action') == 'transfer':
                weekly.transfers_out.append({
                    'name': transfer.get('out', 'Unknown'),
                    'position': transfer.get('out_position', 'UNK'),
                })
                weekly.transfers_in.append({
                    'name': transfer.get('in', 'Unknown'),
                    'position': transfer.get('in_position', 'UNK'),
                })
                
                if transfer.get('take_hit', False):
                    weekly.hits_taken = 4
                    plan.total_hits += 4
                
                plan.total_transfers += 1
            
            weekly.expected_points = transfer.get('expected_gain', 0)
        
        # Check for MIP recommendation
        mip_rec = strategy_result.get('mip_recommendation')
        if mip_rec and mip_rec.get('status') == 'optimal':
            # Overlay MIP results
            mip_plan = self.interpret_mip_result(mip_rec, current_gw)
            plan.weekly_plans = mip_plan.weekly_plans
            plan.total_expected_points = mip_plan.total_expected_points
            plan.total_hits = mip_plan.total_hits
            plan.total_transfers = mip_plan.total_transfers
        
        return plan
    
    def _enrich_players(self, players: List[Dict]) -> List[Dict]:
        """Enrich player list with full info from lookup."""
        enriched = []
        for p in players:
            enriched.append(self._enrich_player(p))
        return enriched
    
    def _enrich_player(self, player: Dict) -> Dict:
        """Enrich single player with lookup info."""
        pid = player.get('id') or player.get('player_id')
        
        if pid and pid in self.player_lookup:
            lookup = self.player_lookup[pid]
            return {
                **player,
                'name': player.get('name') or lookup['name'],
                'full_name': lookup['full_name'],
                'position': player.get('position') or lookup['position'],
                'team': player.get('team') or lookup['team'],
                'price': player.get('buy_price') or player.get('price') or lookup['price'],
            }
        
        return player
    
    def _detect_chips(self, mip_result: Dict, plan: StrategyPlan):
        """Detect if any chips should be played based on solver output."""
        # Check for chip variables in solver result
        # These would be set by an extended solver that models chip usage
        
        chip_vars = {
            'use_wildcard': 'wildcard',
            'use_free_hit': 'free_hit', 
            'use_bench_boost': 'bench_boost',
            'use_triple_captain': 'triple_captain',
        }
        
        for var_name, chip_name in chip_vars.items():
            # Check if chip activated for any week
            for gw, weekly in plan.weekly_plans.items():
                chip_key = f'{var_name}_gw{gw}'
                if mip_result.get(chip_key, 0) > 0.5:
                    weekly.chip_played = chip_name
                    plan.chips_to_play.append((gw, chip_name))
                    plan.warnings.append(
                        f"⚠️ STRATEGY ALERT: ACTIVATE {chip_name.upper().replace('_', ' ')} IN GW{gw}"
                    )
    
    def _validate_plan(self, plan: StrategyPlan):
        """Validate the plan and add warnings for potential issues."""
        for gw, weekly in plan.weekly_plans.items():
            # Check team limits
            team_counts = {}
            all_players = weekly.starting_xi + weekly.bench
            
            for p in all_players:
                team = p.get('team', 'UNK')
                team_counts[team] = team_counts.get(team, 0) + 1
            
            for team, count in team_counts.items():
                if count > 3:
                    plan.warnings.append(
                        f"⚠️ CONSTRAINT VIOLATION: {count} players from {team} in GW{gw} (max 3)"
                    )
            
            # Check formation validity
            if weekly.starting_xi:
                pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
                for p in weekly.starting_xi:
                    pos = p.get('position', 'UNK')
                    if pos in pos_counts:
                        pos_counts[pos] += 1
                
                if pos_counts['GKP'] != 1:
                    plan.warnings.append(f"⚠️ Invalid formation in GW{gw}: {pos_counts['GKP']} goalkeepers")
                if pos_counts['DEF'] < 3:
                    plan.warnings.append(f"⚠️ Invalid formation in GW{gw}: only {pos_counts['DEF']} defenders")
                if pos_counts['FWD'] < 1:
                    plan.warnings.append(f"⚠️ Invalid formation in GW{gw}: no forwards")
            
            # Check for suspicious transfers (selling premium players before easy fixtures)
            for transfer_out in weekly.transfers_out:
                name = transfer_out.get('name', '')
                # Premium player warning (this would need fixture data)
                if any(premium in name.lower() for premium in ['haaland', 'salah', 'palmer']):
                    plan.warnings.append(
                        f"⚠️ EYE TEST: Selling premium player {name} in GW{gw} - verify fixtures"
                    )
    
    def calculate_transfer_roi(self, player_in: Dict, player_out: Dict, 
                               horizon: int = 5, hit_taken: bool = False) -> Dict:
        """Calculate ROI metrics for a transfer.
        
        Args:
            player_in: Incoming player dict with 'price', 'xp' (list of xP per GW).
            player_out: Outgoing player dict with 'price', 'xp' (list of xP per GW).
            horizon: Number of GWs to calculate ROI over.
            hit_taken: Whether a 4-point hit was taken for this transfer.
            
        Returns:
            Dict with ROI metrics:
            - xp_gain: Total xP gain over horizon
            - price_diff: Price difference (in minus out, positive = savings)
            - roi_per_million: xP gain per £1m invested (if price_diff > 0)
            - net_gain: xP gain minus hit cost
            - payback_gws: GWs needed to recover hit cost (if hit taken)
            - classification: 'excellent', 'good', 'marginal', 'poor'
        """
        # Get xP projections
        in_xp = player_in.get('xp', [0] * horizon)
        out_xp = player_out.get('xp', [0] * horizon)
        
        # Calculate total xP for each over horizon
        in_total = sum(in_xp[:horizon]) if isinstance(in_xp, list) else in_xp
        out_total = sum(out_xp[:horizon]) if isinstance(out_xp, list) else out_xp
        
        xp_gain = in_total - out_total
        hit_cost = 4 if hit_taken else 0
        net_gain = xp_gain - hit_cost
        
        # Price analysis
        in_price = player_in.get('price', player_in.get('buy_price', 0))
        out_price = player_out.get('price', player_out.get('sell_price', 0))
        price_diff = out_price - in_price  # Positive = freed up funds
        
        # ROI per million (only meaningful if investing more)
        if price_diff < 0:  # Spending more
            extra_investment = abs(price_diff)
            roi_per_million = round(xp_gain / extra_investment, 2) if extra_investment > 0 else 0
        else:
            roi_per_million = float('inf')  # Gaining both points AND value
        
        # Payback period (GWs to recover hit)
        payback_gws = None
        if hit_taken and xp_gain > 0:
            # Average xP gain per GW
            avg_weekly_gain = xp_gain / horizon
            if avg_weekly_gain > 0:
                payback_gws = round(hit_cost / avg_weekly_gain, 1)
        
        # Classification
        if net_gain >= 5:
            classification = 'excellent'
        elif net_gain >= 2:
            classification = 'good' 
        elif net_gain >= 0:
            classification = 'marginal'
        else:
            classification = 'poor'
        
        return {
            'xp_gain': round(xp_gain, 1),
            'hit_cost': hit_cost,
            'net_gain': round(net_gain, 1),
            'price_diff': round(price_diff, 1),
            'roi_per_million': roi_per_million,
            'payback_gws': payback_gws,
            'classification': classification,
        }
    
    def generate_transfer_explanation(self, player_in: Dict, player_out: Dict,
                                       roi_metrics: Dict, fixtures_in: List[Dict] = None,
                                       fixtures_out: List[Dict] = None) -> str:
        """Generate "Why This Move?" explanation for a transfer.
        
        Args:
            player_in: Incoming player dict.
            player_out: Outgoing player dict.
            roi_metrics: Dict from calculate_transfer_roi().
            fixtures_in: Optional upcoming fixtures for incoming player.
            fixtures_out: Optional upcoming fixtures for outgoing player.
            
        Returns:
            Human-readable explanation string.
        """
        reasons = []
        
        in_name = player_in.get('name', 'Unknown')
        out_name = player_out.get('name', 'Unknown')
        
        # Headline based on classification
        classification = roi_metrics.get('classification', 'marginal')
        if classification == 'excellent':
            headline = f"🎯 **Strong Move**: {in_name} projects +{roi_metrics['xp_gain']:.1f} xP over {out_name}"
        elif classification == 'good':
            headline = f"✅ **Solid Move**: {in_name} expected to outperform {out_name}"
        elif classification == 'marginal':
            headline = f"⚠️ **Marginal**: Small edge to {in_name}, consider waiting"
        else:
            headline = f"❌ **Reconsider**: {out_name} may actually be better"
        
        reasons.append(headline)
        
        # XP reasoning
        net_gain = roi_metrics.get('net_gain', 0)
        if roi_metrics.get('hit_cost', 0) > 0:
            reasons.append(f"• Net gain after -4 hit: +{net_gain:.1f} xP")
            if roi_metrics.get('payback_gws'):
                reasons.append(f"• Hit payback period: ~{roi_metrics['payback_gws']} GWs")
        else:
            reasons.append(f"• Expected gain (free transfer): +{net_gain:.1f} xP")
        
        # Price reasoning
        price_diff = roi_metrics.get('price_diff', 0)
        if price_diff > 0.3:
            reasons.append(f"• Frees up £{price_diff:.1f}m for future upgrades")
        elif price_diff < -0.5:
            reasons.append(f"• Requires £{abs(price_diff):.1f}m extra investment")
            if roi_metrics.get('roi_per_million', 0) > 3:
                reasons.append(f"• ROI: +{roi_metrics['roi_per_million']:.1f} xP per £1m spent (good)")
        
        # Fixture reasoning (if provided)
        if fixtures_in and fixtures_out:
            in_avg_diff = sum(f.get('difficulty', 3) for f in fixtures_in[:3]) / 3 if fixtures_in else 3
            out_avg_diff = sum(f.get('difficulty', 3) for f in fixtures_out[:3]) / 3 if fixtures_out else 3
            
            if in_avg_diff < out_avg_diff - 0.5:
                reasons.append(f"• {in_name} has easier fixtures ahead")
            elif out_avg_diff < in_avg_diff - 0.5:
                reasons.append(f"• ⚠️ {out_name} actually has easier fixtures")
        
        return '\n'.join(reasons)


def interpret_solver_output(mip_result: Dict, 
                            current_gw: int,
                            horizon: int = 5) -> StrategyPlan:
    """Convenience function to interpret solver output.
    
    Args:
        mip_result: Dict from MIP solver.
        current_gw: Current gameweek.
        horizon: Planning horizon.
        
    Returns:
        Interpreted StrategyPlan.
    """
    interpreter = SolverInterpreter()
    return interpreter.interpret_mip_result(mip_result, current_gw, horizon)


if __name__ == '__main__':
    # Test with sample MIP result
    logging.basicConfig(level=logging.INFO)
    
    sample_result = {
        'status': 'optimal',
        'transfers_in': [{'id': 1, 'name': 'Palmer', 'position': 'MID', 'buy_price': 10.5}],
        'transfers_out': [{'id': 2, 'name': 'Salah', 'position': 'MID', 'sell_price': 12.8}],
        'starting_xi': [
            {'id': 10, 'name': 'Raya', 'position': 'GKP'},
            {'id': 11, 'name': 'Gabriel', 'position': 'DEF'},
            # ... more players
        ],
        'bench': [{'id': 20, 'name': 'Flekken', 'position': 'GKP'}],
        'captain': {'id': 5, 'name': 'Haaland'},
        'vice_captain': {'id': 1, 'name': 'Palmer'},
        'formation': '4-4-2',
        'hit_cost': 4,
        'num_transfers': 2,
        'expected_points': 65.2,
        'per_gw_xp': [12.5, 13.2, 14.1, 12.8, 12.6],
        'budget_remaining': 0.5,
    }
    
    interpreter = SolverInterpreter()
    plan = interpreter.interpret_mip_result(sample_result, current_gw=16, horizon=5)
    
    print(json.dumps(plan.to_dict(), indent=2))

