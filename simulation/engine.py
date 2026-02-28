"""Main simulation engine for FPL season backtesting.

This module orchestrates the full season simulation, coordinating
data loading, transfer optimization, scoring, and checkpointing.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd

from simulation.state import (
    GameweekState,
    PlayerState,
    TransferRecord,
    GameweekDecisions,
    GameweekResults,
    SimulationResult,
    ChipType,
    INITIAL_CHIPS,
    STARTING_BUDGET,
    STARTING_FREE_TRANSFERS,
    MAX_FREE_TRANSFERS,
    HIT_COST,
    SQUAD_SIZE,
)
from simulation.data_adapter import HistoricalDataAdapter
from simulation.auto_sub import AutoSubSimulator
from simulation.chip_optimizer import ChipOptimizer
from simulation.squad_builder import InitialSquadBuilder
from simulation.captain_selector import RiskAdjustedCaptainSelector
from simulation.baseline import NoTransferBaseline

# Try to import MIP solver
try:
    from solver.optimizer import TransferMIPSolver, MIP_AVAILABLE
except ImportError:
    MIP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Main orchestrator for FPL season simulation.

    Coordinates:
    - Initial squad building
    - GW-by-GW transfer optimization
    - Chip timing decisions
    - Scoring with auto-subs
    - Checkpointing and reporting
    """

    # Transfer limit to reduce hits (max transfers without chip)
    MAX_TRANSFERS_WITHOUT_CHIP = 2

    def __init__(
        self,
        data_adapter: HistoricalDataAdapter,
        output_dir: Path = None,
        candidate_pool_size: int = 30,
        solver_time_limit: float = 60.0,
        max_transfers: int = None,
        use_future_gw_xp: bool = False,
    ):
        """Initialize simulation engine.

        Args:
            data_adapter: Historical data adapter
            output_dir: Directory for checkpoints and results
            candidate_pool_size: Players per position for MIP solver
            solver_time_limit: MIP solver time limit in seconds
            max_transfers: Max transfers per GW without chip (default: 2)
        """
        self.data = data_adapter
        self.output_dir = Path(output_dir or "simulation_results")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.candidate_pool_size = candidate_pool_size
        self.solver_time_limit = solver_time_limit
        self.max_transfers = max_transfers or self.MAX_TRANSFERS_WITHOUT_CHIP
        self.use_future_gw_xp = use_future_gw_xp

        # Initialize components
        self.auto_sub = AutoSubSimulator()
        self.chip_optimizer = ChipOptimizer(data_adapter)
        self.squad_builder = InitialSquadBuilder(data_adapter)
        self.captain_selector = RiskAdjustedCaptainSelector(data_adapter)

        # Results storage
        self.states: List[GameweekState] = []
        self.decision_log: List[Dict] = []

    def run_simulation(
        self,
        start_gw: int = 1,
        end_gw: int = 38,
        checkpoint_interval: int = 1,
    ) -> SimulationResult:
        """Run full season simulation.

        Args:
            start_gw: Starting gameweek (1 for fresh, or resume from checkpoint)
            end_gw: Ending gameweek
            checkpoint_interval: Save state every N gameweeks

        Returns:
            SimulationResult with full audit trail
        """
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load state
        if start_gw == 1:
            state = self._initialize_state()
        else:
            state = self._load_checkpoint(start_gw - 1)

        for gw in range(start_gw, end_gw + 1):
            logger.info(f"Processing GW{gw}...")
            print(f"Processing GW{gw}...")

            # Update player prices to current GW
            state = self._update_squad_prices(state, gw)

            # Pre-transfer xP (used only for chip decision heuristics)
            pre_xp_predictions = self._build_xp_predictions(state.squad, gw)
            captain_candidate = self._get_captain_candidate(
                state, pre_xp_predictions, gw
            )

            # Check chip triggers
            chip_decision = self.chip_optimizer.decide_chip(
                state, gw, captain_candidate
            )

            # Calculate adaptive horizon
            horizon = min(5, 38 - gw + 1)

            # Run transfer optimization (produces the squad used for this GW)
            transfers: List[TransferRecord] = []
            new_squad: List[PlayerState] = list(state.squad)
            hit_cost = 0

            if chip_decision == ChipType.FREE_HIT.value:
                transfers, new_squad, hit_cost = self._run_free_hit(state, gw, horizon)
                active_squad = new_squad
            elif chip_decision == ChipType.WILDCARD_1.value:
                transfers, new_squad, hit_cost = self._run_wildcard(state, gw, horizon)
                state.squad = new_squad
                active_squad = state.squad
            else:
                transfers, new_squad, hit_cost = self._run_normal_transfers(
                    state, gw, horizon
                )
                state.squad = new_squad
                active_squad = state.squad

            # Update bank after transfers (free hit does NOT persist squad/bank changes)
            if chip_decision != ChipType.FREE_HIT.value:
                bank_change = sum(t.price_out for t in transfers) - sum(
                    t.price_in for t in transfers
                )
                state.bank += bank_change

            # Build xP predictions for the active squad (must include transferred-in players)
            xp_predictions = self._build_xp_predictions(active_squad, gw)

            # Build lineup + bench from active squad
            lineup, bench = self._build_lineup_from_squad(active_squad, xp_predictions)

            # Select captain and vice-captain
            lineup_ids = [p.id for p in active_squad if p.id in lineup]
            captain_id, vice_id = self.captain_selector.select_captain_from_squad(
                active_squad, xp_predictions, gw, lineup_ids
            )

            # Get actual data for scoring
            actual_minutes = {
                p.id: self.data.get_player_minutes(p.id, gw) for p in active_squad
            }
            actual_points = {
                p.id: self.data.get_player_actual_points(p.id, gw) for p in active_squad
            }

            # Apply auto-subs
            lineup_players = [p for p in active_squad if p.id in lineup]
            bench_players = [p for p in active_squad if p.id in bench]

            final_xi, auto_subs, effective_captain, _ = self.auto_sub.apply_auto_subs(
                lineup_players,
                bench_players,
                actual_minutes,
                captain_id,
                vice_id,
            )

            # Calculate points
            gw_points, points_before_hits, captain_pts = (
                self.auto_sub.calculate_gw_points(
                    final_xi,
                    actual_points,
                    effective_captain,
                    chip_used=chip_decision,
                    hit_cost=hit_cost,
                )
            )

            # Bench points should never count unless Bench Boost is active.
            # Also, exclude any bench players that were auto-subbed into the XI
            # to avoid double-counting.
            remaining_bench_points = self.auto_sub.calculate_remaining_bench_points(
                bench_players, actual_points, auto_subs=auto_subs
            )
            if chip_decision == ChipType.BENCH_BOOST.value:
                gw_points += remaining_bench_points
                points_before_hits += remaining_bench_points

            state.total_points += gw_points

            # Update chips
            if chip_decision:
                state.chips_available.discard(chip_decision)

            # 2025-26 season: Reset chips at GW20 for second set
            from simulation.state import CHIP_RESET_GW, INITIAL_CHIPS

            if gw == CHIP_RESET_GW - 1:  # Reset before GW20 (after GW19 results)
                state.chips_available = INITIAL_CHIPS.copy()

            # Build formation string
            formation = self._get_formation(lineup_players)

            # Create decisions and results
            decisions = GameweekDecisions(
                transfers=transfers,
                lineup=lineup,
                bench_order=bench,
                captain_id=captain_id,
                vice_captain_id=vice_id,
                chip_used=chip_decision,
                formation=formation,
            )

            results = GameweekResults(
                gw_points=gw_points,
                gw_points_before_hits=points_before_hits,
                hit_cost=hit_cost,
                auto_subs=auto_subs,
                effective_captain_id=effective_captain,
                captain_points=captain_pts,
                bench_points=remaining_bench_points,
            )

            # Update state
            state.gameweek = gw
            state.decisions = decisions
            state.results = results

            # Free hit: persistent squad already unchanged (active_squad was temporary)

            # Update free transfers for next GW
            num_transfers = len(transfers)
            if chip_decision in [
                ChipType.WILDCARD_1.value,
                ChipType.FREE_HIT.value,
            ]:
                state.free_transfers = 1  # Reset to 1 after chip
            else:
                if num_transfers == 0:
                    state.free_transfers = min(
                        state.free_transfers + 1, MAX_FREE_TRANSFERS
                    )
                else:
                    # Used transfers, reset to 1 for next week
                    state.free_transfers = 1

            # Store state
            history_state = GameweekState(
                gameweek=state.gameweek,
                squad=list(active_squad),
                bank=state.bank,
                free_transfers=state.free_transfers,
                total_points=state.total_points,
                chips_available=set(state.chips_available),
                decisions=decisions,
                results=results,
            )
            self.states.append(history_state)

            # Log progress
            self._log_gw_summary(gw, gw_points, chip_decision, len(transfers), hit_cost)

            # Checkpoint
            if gw % checkpoint_interval == 0:
                # Checkpoints should reflect the persistent squad used for next GW.
                checkpoint_state = history_state
                if chip_decision == ChipType.FREE_HIT.value:
                    checkpoint_state = GameweekState(
                        gameweek=state.gameweek,
                        squad=list(state.squad),
                        bank=state.bank,
                        free_transfers=state.free_transfers,
                        total_points=state.total_points,
                        chips_available=set(state.chips_available),
                        decisions=decisions,
                        results=results,
                    )
                checkpoint_state.save_checkpoint(self.checkpoint_dir)

        return self._compile_results()

    def _initialize_state(self) -> GameweekState:
        """Initialize fresh state for GW1."""
        squad, bank = self.squad_builder.build_starting_squad(gw=1)

        return GameweekState(
            gameweek=0,  # Will be updated in loop
            squad=squad,
            bank=bank,
            free_transfers=STARTING_FREE_TRANSFERS,
            total_points=0,
            chips_available=set(INITIAL_CHIPS),
            decisions=GameweekDecisions(),
            results=GameweekResults(),
        )

    def _load_checkpoint(self, gw: int) -> GameweekState:
        """Load state from checkpoint file."""
        checkpoint_path = self.checkpoint_dir / f"gw{gw}.json"
        return GameweekState.load_checkpoint(checkpoint_path)

    def _update_squad_prices(self, state: GameweekState, gw: int) -> GameweekState:
        """Update squad player prices to current GW values."""
        for player in state.squad:
            player.current_price = self.data.get_player_price(player.id, gw)
        return state

    def _build_xp_predictions(
        self, squad: List[PlayerState], gw: int
    ) -> Dict[int, float]:
        """Build xP predictions for a given squad."""
        predictions = {}
        for player in squad:
            if self.use_future_gw_xp:
                # Get xP from the GW snapshot file.
                xp = self.data.get_player_xp(player.id, gw)

                # Handle DGW: double the xP
                team_id = player.team_id or self.data.get_player_team_id(player.id, gw)
                if team_id and self.data.is_dgw(team_id, gw):
                    xp *= 2
            else:
                # Backtest-safe mode: use deadline snapshot only (no forward file reads).
                xp = self.data.get_player_xp_forecast(
                    player.id, as_of_gw=gw, target_gw=gw
                )

            predictions[player.id] = xp
        return predictions

    def _get_captain_candidate(
        self,
        state: GameweekState,
        xp_predictions: Dict[int, float],
        gw: int,
    ) -> Dict:
        """Get best captain candidate for chip decisions."""
        candidates = self.captain_selector.get_captain_candidates(
            state.squad, xp_predictions, gw, top_n=1
        )
        if candidates:
            return candidates[0]
        return {}

    def _run_normal_transfers(
        self,
        state: GameweekState,
        gw: int,
        horizon: int,
    ) -> Tuple[List[TransferRecord], List[PlayerState], int]:
        """Run normal transfer optimization with transfer cap."""
        if not MIP_AVAILABLE:
            # No transfers if MIP not available
            return [], list(state.squad), 0

        try:
            # Candidate pool (deadline-safe; based on data strictly before GW decisions)
            candidate_ids = self._get_candidate_pool(gw)

            # Build xP matrix for solver
            xp_matrix = self._build_xp_matrix(
                state, gw, horizon, candidate_ids=candidate_ids
            )

            # Convert squad to solver format
            current_squad = self._state_to_squad_list(state)

            # Build players DataFrame for solver
            players_df = self._build_players_df(gw)

            # Cap effective free transfers to limit hits
            # Solver will think we have limited FTs, penalizing extra transfers
            effective_ft = min(state.free_transfers, self.max_transfers)

            # Run solver
            solver = TransferMIPSolver(
                current_squad=current_squad,
                bank=state.bank,
                players_df=players_df,
                xp_matrix=xp_matrix,
                free_transfers=effective_ft,
                horizon=horizon,
                candidate_pool_size=self.candidate_pool_size,
                time_limit=self.solver_time_limit,
                candidate_ids=candidate_ids,
            )

            result = solver.solve()

            if result.status != "optimal":
                logger.warning(f"MIP solver status: {result.status}")
                return [], list(state.squad), 0

            # Cap transfers to max_transfers to minimize hits
            num_transfers = len(result.transfers_in)
            if num_transfers > self.max_transfers:
                # Only apply first max_transfers transfers
                result.transfers_in = result.transfers_in[: self.max_transfers]
                result.transfers_out = result.transfers_out[: self.max_transfers]

            # Recalculate hit cost after capping
            actual_transfers = len(result.transfers_in)
            actual_hit_cost = max(0, actual_transfers - state.free_transfers) * HIT_COST

            # Convert result to our format
            transfers = self._convert_transfers(result, state)
            new_squad = self._apply_transfers_to_squad(state.squad, result, gw)

            return transfers, new_squad, actual_hit_cost

        except Exception as e:
            logger.error(f"MIP solver error: {e}")
            return [], list(state.squad), 0

    def _run_wildcard(
        self,
        state: GameweekState,
        gw: int,
        horizon: int,
    ) -> Tuple[List[TransferRecord], List[PlayerState], int]:
        """Run wildcard optimization (unlimited free transfers)."""
        # Same as normal but with max free transfers
        saved_ft = state.free_transfers
        state.free_transfers = 15  # Effectively unlimited

        transfers, new_squad, _ = self._run_normal_transfers(state, gw, horizon)

        state.free_transfers = saved_ft
        return transfers, new_squad, 0  # No hit cost on wildcard

    def _run_free_hit(
        self,
        state: GameweekState,
        gw: int,
        horizon: int,
    ) -> Tuple[List[TransferRecord], List[PlayerState], int]:
        """Run free hit optimization (one-week squad)."""
        # For free hit, optimize for just this GW
        transfers, new_squad, _ = self._run_wildcard(state, gw, 1)
        return transfers, new_squad, 0

    def _build_xp_matrix(
        self,
        state: GameweekState,
        current_gw: int,
        horizon: int,
        candidate_ids: Optional[List[int]] = None,
    ) -> Dict[int, List[float]]:
        """Build xP prediction matrix for MIP solver."""
        squad_ids = {p.id for p in state.squad}
        if candidate_ids is None:
            candidate_ids = self._get_candidate_pool(current_gw)
        all_ids = squad_ids | set(candidate_ids)

        xp_matrix = {}
        for pid in all_ids:
            xp_list = []
            for gw_offset in range(horizon):
                target_gw = current_gw + gw_offset
                if target_gw > 38:
                    xp_list.append(0.0)
                else:
                    if self.use_future_gw_xp:
                        # WARNING: This reads xP from the target GW snapshot file, which can
                        # leak future information in backtests if those snapshots were built
                        # later with updated injuries/form.
                        xp = self.data.get_player_xp(pid, target_gw)

                        # Handle DGW
                        team_id = self.data.get_player_team_id(pid, target_gw)
                        if team_id and self.data.is_dgw(team_id, target_gw):
                            xp *= 2
                    else:
                        # Backtest-safe mode: forecast from the current GW deadline only.
                        xp = self.data.get_player_xp_forecast(
                            pid, as_of_gw=current_gw, target_gw=target_gw
                        )

                    xp_list.append(xp)
            xp_matrix[pid] = xp_list

        return xp_matrix

    def _get_candidate_pool(self, gw: int) -> List[int]:
        """Get candidate player IDs for transfer consideration."""
        candidates = []
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            top_players = self.data.get_top_players_by_total_points(
                gw, position=pos, n=self.candidate_pool_size
            )
            candidates.extend([p["id"] for p in top_players])
        return candidates

    def _state_to_squad_list(self, state: GameweekState) -> List[Dict]:
        """Convert state squad to solver format."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "position": p.position,
                "selling_price_m": p.current_price,
                "team_id": p.team_id,
                "team": p.team_name,
                "stats": {
                    "now_cost": int(p.current_price * 10),
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                },
            }
            for p in state.squad
        ]

    def _build_players_df(self, gw: int) -> pd.DataFrame:
        """Build players DataFrame for solver (deadline-safe)."""
        return self.data.build_players_df_for_solver(gw)

    def _convert_transfers(
        self,
        result,
        state: GameweekState,
    ) -> List[TransferRecord]:
        """Convert MIP result to TransferRecord list."""
        transfers = []

        for t_out, t_in in zip(result.transfers_out, result.transfers_in):
            transfers.append(
                TransferRecord(
                    player_out_id=t_out["id"],
                    player_out_name=t_out.get("name", "Unknown"),
                    player_in_id=t_in["id"],
                    player_in_name=t_in.get("name", "Unknown"),
                    price_out=t_out.get("sell_price", 0),
                    price_in=t_in.get("buy_price", 0),
                    is_hit=len(transfers) >= state.free_transfers,
                )
            )

        return transfers

    def _apply_transfers_to_squad(
        self,
        squad: List[PlayerState],
        result,
        gw: int,
    ) -> List[PlayerState]:
        """Apply MIP result transfers to squad."""
        # Build new squad from result
        new_squad = []
        transferred_out_ids = {t["id"] for t in result.transfers_out}
        transferred_in = {t["id"]: t for t in result.transfers_in}

        # Keep players not transferred out
        for player in squad:
            if player.id not in transferred_out_ids:
                new_squad.append(player)

        # Add transferred in players
        for pid, p_data in transferred_in.items():
            team_id = p_data.get("team_id", 0)
            team_name = self.data.get_team_name(team_id) if team_id else ""

            new_squad.append(
                PlayerState(
                    id=pid,
                    name=p_data.get("name", f"Player_{pid}"),
                    position=p_data.get("position", "MID"),
                    team_id=team_id,
                    team_name=team_name,
                    purchase_price=p_data.get("buy_price", 5.0),
                    current_price=p_data.get("buy_price", 5.0),
                )
            )

        return new_squad

    def _build_lineup_from_squad(
        self,
        squad: List[PlayerState],
        xp_predictions: Dict[int, float],
    ) -> Tuple[List[int], List[int]]:
        """Build lineup and bench from squad."""
        # Group by position
        by_position = defaultdict(list)
        for player in squad:
            by_position[player.position].append(
                {
                    "id": player.id,
                    "xp": xp_predictions.get(player.id, 0),
                }
            )

        # Sort by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x["xp"], reverse=True)

        lineup = []
        bench = []

        # 1 GKP in lineup
        if by_position["GKP"]:
            lineup.append(by_position["GKP"][0]["id"])
            bench.extend([p["id"] for p in by_position["GKP"][1:]])

        # Minimum formation: 3 DEF, 2 MID, 1 FWD
        minimums = {"DEF": 3, "MID": 2, "FWD": 1}
        for pos, min_count in minimums.items():
            for i in range(min(min_count, len(by_position[pos]))):
                lineup.append(by_position[pos][i]["id"])

        # Fill remaining spots (need 11 total)
        remaining_needed = 11 - len(lineup)
        candidates = []
        for pos in ["DEF", "MID", "FWD"]:
            min_count = minimums[pos]
            for i, p in enumerate(by_position[pos]):
                if i >= min_count:
                    candidates.append(p)

        candidates.sort(key=lambda x: x["xp"], reverse=True)
        for p in candidates[:remaining_needed]:
            lineup.append(p["id"])
        for p in candidates[remaining_needed:]:
            bench.append(p["id"])

        return lineup, bench

    def _get_formation(self, lineup: List[PlayerState]) -> str:
        """Get formation string."""
        counts = {"DEF": 0, "MID": 0, "FWD": 0}
        for p in lineup:
            if p.position in counts:
                counts[p.position] += 1
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"

    def _log_gw_summary(
        self,
        gw: int,
        points: int,
        chip: Optional[str],
        transfers: int,
        hits: int,
    ):
        """Log gameweek summary."""
        chip_str = f" [{chip}]" if chip else ""
        hit_str = f" (-{hits} hits)" if hits > 0 else ""
        print(f"  GW{gw}: {points} pts{chip_str}, {transfers} transfers{hit_str}")

    def _compile_results(self) -> SimulationResult:
        """Compile final simulation results."""
        total_hits = sum(s.results.hit_cost for s in self.states)
        chips_used = {}
        transfers_made = 0

        for state in self.states:
            if state.decisions.chip_used:
                chips_used[state.decisions.chip_used] = state.gameweek
            transfers_made += len(state.decisions.transfers)

        return SimulationResult(
            season="2024-25",
            states=self.states,
            total_points=self.states[-1].total_points if self.states else 0,
            total_hits=total_hits,
            chips_used=chips_used,
            transfers_made=transfers_made,
        )


def main():
    """CLI entry point for simulation."""
    parser = argparse.ArgumentParser(description="Run FPL season simulation")
    parser.add_argument("--start-gw", type=int, default=1, help="Starting gameweek")
    parser.add_argument("--end-gw", type=int, default=38, help="Ending gameweek")
    parser.add_argument(
        "--output-dir", type=str, default="simulation_results", help="Output directory"
    )
    parser.add_argument(
        "--season-path", type=str, default="data/2024-25", help="Path to season data"
    )
    parser.add_argument(
        "--with-baseline", action="store_true", help="Also run no-transfer baseline"
    )
    parser.add_argument(
        "--max-transfers",
        type=int,
        default=2,
        help="Max transfers per GW without chip (default: 2)",
    )
    parser.add_argument(
        "--use-future-gw-xp",
        action="store_true",
        help="Use xP from each future GW snapshot file (leak-prone for backtests)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize data adapter
    data_adapter = HistoricalDataAdapter(Path(args.season_path))

    # Run simulation
    engine = SimulationEngine(
        data_adapter,
        output_dir=Path(args.output_dir),
        max_transfers=args.max_transfers,
        use_future_gw_xp=args.use_future_gw_xp,
    )
    result = engine.run_simulation(
        start_gw=args.start_gw,
        end_gw=args.end_gw,
    )

    print(f"\n{'=' * 50}")
    print(f"Simulation Complete!")
    print(f"Total Points: {result.total_points}")
    print(f"Total Hits: {result.total_hits}")
    print(f"Transfers Made: {result.transfers_made}")
    print(f"Chips Used: {result.chips_used}")

    baseline_result = None
    if args.with_baseline:
        # Get initial squad from first state
        initial_squad = engine.states[0].squad if engine.states else []
        initial_bank = STARTING_BUDGET - sum(p.purchase_price for p in initial_squad)

        baseline = NoTransferBaseline(data_adapter, initial_squad, initial_bank)
        baseline_result = baseline.run_baseline(args.start_gw, args.end_gw)

        print(f"\nBaseline (No Transfers): {baseline_result.total_points}")
        print(f"Improvement: {result.total_points - baseline_result.total_points}")

        baseline.save_results(baseline_result, Path(args.output_dir))

    # Generate report artifacts (JSON + LaTeX + optional PDF)
    from simulation.report_generator import BacktestReportGenerator

    report_gen = BacktestReportGenerator(Path(args.output_dir))
    json_path, pdf_path = report_gen.generate_report(
        result, baseline_result, data_adapter=data_adapter
    )
    print(f"\nReport JSON: {json_path}")
    tex_path = Path(args.output_dir) / "simulation_report.tex"
    print(f"Report LaTeX: {tex_path}")
    if pdf_path:
        print(f"Report PDF: {pdf_path}")


if __name__ == "__main__":
    main()
