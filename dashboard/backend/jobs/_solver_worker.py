"""Standalone solver worker for subprocess execution.

Runs transfer strategy generation and prints JSON payload for parent process.
All diagnostic output is redirected to stderr so stdout remains valid JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


_project_root = str(Path(__file__).parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from etl.fetchers import FPLCoreInsightsFetcher
from reports.fpl_report.data_fetcher import FPLDataFetcher
from reports.fpl_report.player_analyzer import PlayerAnalyzer
from reports.fpl_report.transfer_recommender import TransferRecommender
from reports.fpl_report.transfer_strategy import TransferStrategyPlanner
from utils.config import (
    FREE_TRANSFERS_OVERRIDE,
    MIP_CANDIDATE_POOL,
    MIP_TIME_LIMIT,
    SEASON,
    TEAM_ID,
    TRANSFER_HORIZON,
)

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _season_for_fpl_core(season: str) -> str:
    parts = season.split("-")
    if len(parts) != 2:
        return season

    start = parts[0]
    end = parts[1]
    if len(end) == 4:
        return f"{start}-{end}"
    if len(end) == 2 and len(start) == 4:
        return f"{start}-{start[:2]}{end}"
    return season


def _build_squad_analysis(fetcher: FPLDataFetcher, analyzer: PlayerAnalyzer, gameweek: int) -> List[Dict]:
    squad = fetcher.get_current_squad(gameweek)
    analyses: List[Dict] = []

    for player in squad:
        player_id = player["id"]
        position = player["position"]
        summary = analyzer.generate_player_summary(player_id, position)
        summary["player_id"] = summary.get("player_id", player_id)
        summary["name"] = summary.get("name", player.get("name", "Unknown"))
        summary["position"] = summary.get("position", position)
        summary["position_in_squad"] = player.get("position_in_squad", 0)
        summary["is_captain"] = player.get("is_captain", False)
        summary["is_vice_captain"] = player.get("is_vice_captain", False)
        analyses.append(summary)

    return analyses


def _build_recommender(
    fetcher: FPLDataFetcher,
    analyzer: PlayerAnalyzer,
    gameweek: int,
    season: str,
) -> TransferRecommender:
    """Try cached FPL Core predictor wiring first, fallback to base predictor."""
    core_season = _season_for_fpl_core(season)

    try:
        logger.info("Initializing FPL Core fetcher for season %s", core_season)
        core_fetcher = FPLCoreInsightsFetcher(season=core_season)
        if not _core_cache_ready(core_fetcher, gameweek):
            raise RuntimeError("FPL Core cache is incomplete")

        season_data = core_fetcher.fetch_all()
        all_gw_data = core_fetcher.fetch_all_gameweeks(up_to_gw=gameweek)
        all_gw_data_pred = copy.deepcopy(all_gw_data)

        next_gw = gameweek + 1
        next_fixtures = core_fetcher.get_fixtures(gameweek=next_gw)
        next_teams = core_fetcher.get_teams(gameweek=next_gw)
        if next_fixtures is not None or next_teams is not None:
            all_gw_data_pred[next_gw] = {
                "fixtures": next_fixtures,
                "teams": next_teams,
            }

        season_data["fpl_api_fixtures"] = fetcher.fixtures_df

        logger.info("Using FPL Core predictor path for solver worker")
        return TransferRecommender(
            fetcher,
            analyzer,
            use_new_models=True,
            use_fpl_core_predictor=True,
            all_gw_data=all_gw_data_pred,
            fpl_core_season_data=season_data,
            current_gw=gameweek,
        )
    except Exception as exc:
        logger.warning("FPL Core predictor setup failed, using fallback predictor: %s", exc)
        return TransferRecommender(
            fetcher,
            analyzer,
            use_new_models=True,
            use_fpl_core_predictor=False,
        )


def _core_cache_ready(core_fetcher: FPLCoreInsightsFetcher, gameweek: int) -> bool:
    """Require full local cache for Core predictor to avoid heavy runtime downloads."""
    season_dir = core_fetcher.season_cache_dir

    for dataset in core_fetcher.SEASON_DATASETS:
        if not (season_dir / f"{dataset}.csv").exists():
            return False

    for gw in range(1, gameweek + 1):
        gw_dir = season_dir / f"gw{gw}"
        for dataset in core_fetcher.GW_DATASETS:
            if not (gw_dir / f"{dataset}.csv").exists():
                return False

    return True


def _format_underperformers(underperformers: List[Dict]) -> List[Dict]:
    formatted = []
    for player in underperformers:
        formatted.append(
            {
                "player_id": player.get("player_id"),
                "name": player.get("name"),
                "position": player.get("position"),
                "team": player.get("team"),
                "price": _safe_float(player.get("price"), 0.0),
                "severity": _safe_int(player.get("severity"), 0),
                "reasons": [str(reason) for reason in player.get("reasons", [])][:6],
                "current_form": _safe_float(player.get("current_form"), 0.0),
                "total_points": _safe_int(player.get("total_points"), 0),
            }
        )
    return sorted(formatted, key=lambda item: item["severity"], reverse=True)


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if hasattr(obj, "item"):
        # numpy scalar support
        return _sanitize_json(obj.item())
    return str(obj)


def _build_solver_payload(
    gameweek: int,
    strategy: Dict,
    underperformers: List[Dict],
) -> Dict:
    mip = strategy.get("mip_recommendation") or {}
    scenarios = mip.get("scenarios") or {}
    recommended = str(mip.get("recommended", "balanced")).lower()
    if recommended not in {"conservative", "balanced", "aggressive"}:
        recommended = "balanced"

    baseline_xp = _safe_float(
        mip.get("baseline_xp", strategy.get("expected_value", {}).get("current_squad", 0.0)),
        0.0,
    )
    mip_status = str(mip.get("status", "pending"))
    mip_message = str(mip.get("message", ""))

    def build_scenario(name: str) -> Dict:
        source = scenarios.get(name)
        if source is None and mip_status == "optimal" and name == recommended:
            # Compatibility fallback when scenario breakdown is absent.
            expected = _safe_float(mip.get("expected_points"), baseline_xp)
            source = {
                "num_transfers": _safe_int(mip.get("num_transfers")),
                "hit_cost": _safe_int(mip.get("hit_cost")),
                "expected_points": expected,
                "baseline_xp": baseline_xp,
                "net_gain": round(expected - baseline_xp, 1),
                "weekly_plans": mip.get("weekly_plans", []),
                "transfer_sequence": mip.get("transfer_sequence", []),
            }

        scenario: Dict[str, Any] = dict(source) if source else {}
        if source:
            scenario_status = "optimal"
        elif name == recommended:
            scenario_status = mip_status
        else:
            scenario_status = "pending" if mip_status == "optimal" else "unavailable"

        expected_points = _safe_float(scenario.get("expected_points"), baseline_xp)
        scenario_baseline = _safe_float(scenario.get("baseline_xp"), baseline_xp)

        scenario.setdefault("num_transfers", 0)
        scenario.setdefault("hit_cost", 0)
        scenario.setdefault("expected_points", expected_points)
        scenario.setdefault("baseline_xp", scenario_baseline)
        scenario.setdefault("net_gain", round(expected_points - scenario_baseline, 1))
        scenario.setdefault("weekly_plans", [])
        scenario.setdefault("transfer_sequence", [])
        scenario["scenario"] = name
        scenario["status"] = scenario_status
        scenario["message"] = str(scenario.get("message", mip_message))
        scenario["underperformers"] = underperformers
        return scenario

    return {
        "gameweek": gameweek,
        "conservative": build_scenario("conservative"),
        "balanced": build_scenario("balanced"),
        "aggressive": build_scenario("aggressive"),
        "recommended": recommended,
        "baseline_xp": baseline_xp,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FPL transfer solver worker")
    parser.add_argument("--team-id", type=int, default=TEAM_ID)
    parser.add_argument("--season", type=str, default=SEASON)
    parser.add_argument("--gameweek", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=TRANSFER_HORIZON)
    parser.add_argument("--time-limit", type=float, default=float(MIP_TIME_LIMIT))
    parser.add_argument("--candidate-pool", type=int, default=int(MIP_CANDIDATE_POOL))
    parser.add_argument("--free-transfers", type=int, default=FREE_TRANSFERS_OVERRIDE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Keep diagnostics out of stdout, parent expects JSON there.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [solver-worker] %(message)s",
        stream=sys.stderr,
    )

    with contextlib.redirect_stdout(sys.stderr):
        logger.info("Solver worker starting for team=%s season=%s", args.team_id, args.season)
        fetcher = FPLDataFetcher(args.team_id, args.season, use_cache=False)
        gameweek = args.gameweek or fetcher.get_current_gameweek()
        analyzer = PlayerAnalyzer(fetcher)
        squad_analysis = _build_squad_analysis(fetcher, analyzer, gameweek)
        recommender = _build_recommender(fetcher, analyzer, gameweek, args.season)

        underperformers_raw = recommender.identify_underperformers(squad_analysis)
        underperformers = _format_underperformers(underperformers_raw)

        free_transfers = args.free_transfers
        if free_transfers is None:
            try:
                free_transfers = fetcher.calculate_free_transfers(gameweek)
            except Exception as exc:
                logger.warning("Failed to calculate free transfers, defaulting to 1: %s", exc)
                free_transfers = 1

        planner = TransferStrategyPlanner(fetcher, analyzer, recommender)
        strategy = planner.generate_strategy(
            squad_analysis=squad_analysis,
            num_weeks=args.horizon,
            use_mip=True,
            mip_time_limit=args.time_limit,
            mip_candidate_pool=args.candidate_pool,
            free_transfers=_safe_int(free_transfers, 1),
            current_gw=gameweek,
        )
        payload = _build_solver_payload(gameweek, strategy, underperformers)

    print(json.dumps(_sanitize_json(payload), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
