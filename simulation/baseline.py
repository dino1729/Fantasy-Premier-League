"""No-transfer baseline for FPL simulation comparison.

This module provides a baseline strategy that makes no transfers,
using the GW1 squad throughout the season with only captain selection.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from simulation.state import (
    GameweekState,
    PlayerState,
    GameweekDecisions,
    GameweekResults,
    SimulationResult,
    INITIAL_CHIPS,
    STARTING_FREE_TRANSFERS,
)
from simulation.data_adapter import HistoricalDataAdapter
from simulation.auto_sub import AutoSubSimulator
from simulation.captain_selector import RiskAdjustedCaptainSelector


class NoTransferBaseline:
    """Baseline strategy with no transfers.

    This baseline:
    - Uses the initial GW1 squad throughout
    - Selects captain each week using risk-adjusted logic
    - Applies auto-subs when starters don't play
    - Never uses chips (to isolate pure transfer value)
    """

    def __init__(
        self,
        data_adapter: HistoricalDataAdapter,
        initial_squad: List[PlayerState],
        initial_bank: float,
    ):
        """Initialize baseline.

        Args:
            data_adapter: Historical data adapter
            initial_squad: Starting squad (same as main simulation)
            initial_bank: Starting bank balance
        """
        self.data = data_adapter
        self.initial_squad = initial_squad
        self.initial_bank = initial_bank

        self.auto_sub = AutoSubSimulator()
        self.captain_selector = RiskAdjustedCaptainSelector(data_adapter)

    def run_baseline(
        self,
        start_gw: int = 1,
        end_gw: int = 38,
    ) -> SimulationResult:
        """Run baseline simulation.

        Args:
            start_gw: Starting gameweek
            end_gw: Ending gameweek

        Returns:
            SimulationResult with baseline performance
        """
        states: List[GameweekState] = []
        total_points = 0

        # Current squad (never changes in baseline)
        squad = list(self.initial_squad)

        for gw in range(start_gw, end_gw + 1):
            # Update player prices to current GW values
            for player in squad:
                player.current_price = self.data.get_player_price(player.id, gw)

            # Get xP predictions for captain selection
            xp_predictions = {}
            for player in squad:
                xp_predictions[player.id] = self.data.get_player_xp(player.id, gw)

            # Select captain using same logic as main simulation
            captain_id, vice_id = self.captain_selector.select_captain_from_squad(
                squad, xp_predictions, gw
            )

            # Build lineup (top 11 by xP)
            lineup, bench = self._build_lineup(squad, xp_predictions)

            # Get actual minutes and points
            actual_minutes = {p.id: self.data.get_player_minutes(p.id, gw) for p in squad}
            actual_points = {p.id: self.data.get_player_actual_points(p.id, gw) for p in squad}

            # Apply auto-subs
            lineup_players = [p for p in squad if p.id in lineup]
            bench_players = [p for p in squad if p.id in bench]

            final_xi, auto_subs, effective_captain, _ = self.auto_sub.apply_auto_subs(
                lineup_players,
                bench_players,
                actual_minutes,
                captain_id,
                vice_id,
            )

            # Calculate points
            gw_points, points_before_hits, captain_pts = self.auto_sub.calculate_gw_points(
                final_xi,
                actual_points,
                effective_captain,
                chip_used=None,
                hit_cost=0,
            )

            total_points += gw_points

            # Create state
            state = GameweekState(
                gameweek=gw,
                squad=squad,
                bank=self.initial_bank,  # Never spend bank in baseline
                free_transfers=STARTING_FREE_TRANSFERS,  # Always 1 (unused)
                total_points=total_points,
                chips_available=set(INITIAL_CHIPS),  # Never use chips
                decisions=GameweekDecisions(
                    transfers=[],
                    lineup=lineup,
                    bench_order=bench,
                    captain_id=captain_id,
                    vice_captain_id=vice_id,
                    chip_used=None,
                    formation=self._get_formation(lineup_players),
                ),
                results=GameweekResults(
                    gw_points=gw_points,
                    gw_points_before_hits=points_before_hits,
                    hit_cost=0,
                    auto_subs=auto_subs,
                    effective_captain_id=effective_captain,
                    captain_points=captain_pts,
                    bench_points=sum(actual_points.get(p.id, 0) for p in bench_players),
                ),
            )
            states.append(state)

        return SimulationResult(
            season='2024-25',
            states=states,
            total_points=total_points,
            total_hits=0,
            chips_used={},
            transfers_made=0,
        )

    def _build_lineup(
        self,
        squad: List[PlayerState],
        xp_predictions: Dict[int, float],
    ) -> Tuple[List[int], List[int]]:
        """Build lineup and bench from squad.

        Picks best 11 by xP while maintaining valid formation.

        Args:
            squad: Full squad
            xp_predictions: Dict mapping player_id to xP

        Returns:
            Tuple of (lineup IDs, bench IDs in order)
        """
        # Group by position
        by_position = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        for player in squad:
            by_position[player.position].append({
                'id': player.id,
                'xp': xp_predictions.get(player.id, 0),
                'player': player,
            })

        # Sort each position by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x['xp'], reverse=True)

        lineup = []
        bench = []

        # Must have 1 GKP in lineup
        lineup.append(by_position['GKP'][0]['id'])
        if len(by_position['GKP']) > 1:
            bench.append(by_position['GKP'][1]['id'])

        # Minimum formation: 3-2-1
        # Fill minimums first
        for i, pos in enumerate(['DEF', 'MID', 'FWD']):
            min_count = [3, 2, 1][i]
            for j in range(min_count):
                if j < len(by_position[pos]):
                    lineup.append(by_position[pos][j]['id'])

        # Fill remaining spots (need 11 total)
        remaining_needed = 11 - len(lineup)
        remaining_candidates = []

        for pos in ['DEF', 'MID', 'FWD']:
            min_count = {'DEF': 3, 'MID': 2, 'FWD': 1}[pos]
            for i, p in enumerate(by_position[pos]):
                if i >= min_count:  # Already picked minimum
                    remaining_candidates.append(p)

        # Sort remaining by xP and take top N
        remaining_candidates.sort(key=lambda x: x['xp'], reverse=True)
        for p in remaining_candidates[:remaining_needed]:
            lineup.append(p['id'])

        # Rest go to bench
        for p in remaining_candidates[remaining_needed:]:
            bench.append(p['id'])

        # Order bench: outfield first, GKP last (already added)
        return lineup, bench

    def _get_formation(self, lineup: List[PlayerState]) -> str:
        """Get formation string from lineup."""
        counts = {'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in lineup:
            if player.position in counts:
                counts[player.position] += 1
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"

    def save_results(self, result: SimulationResult, output_dir: Path) -> Path:
        """Save baseline results to JSON.

        Args:
            result: Baseline simulation result
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'baseline_results.json'

        data = {
            'type': 'no_transfer_baseline',
            'season': result.season,
            'total_points': result.total_points,
            'total_hits': result.total_hits,
            'transfers_made': result.transfers_made,
            'gameweeks': [
                {
                    'gw': s.gameweek,
                    'points': s.results.gw_points,
                    'captain_id': s.decisions.captain_id,
                    'auto_subs_count': len(s.results.auto_subs),
                }
                for s in result.states
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path
