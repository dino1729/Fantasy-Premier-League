"""League job - competitive dataset + global managers."""

from __future__ import annotations

import logging
from typing import Iterable, List

from dashboard.backend.database import log_refresh, write_singleton
from reports.fpl_report.data_fetcher import build_competitive_dataset, get_top_global_teams
from utils.config import COMPETITORS, SEASON, TEAM_ID, TOP_GLOBAL_COUNT

logger = logging.getLogger(__name__)


def _dedupe_ids(ids: Iterable[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for raw in ids:
        try:
            entry_id = int(raw)
        except (TypeError, ValueError):
            continue
        if entry_id <= 0 or entry_id in seen:
            continue
        seen.add(entry_id)
        ordered.append(entry_id)
    return ordered


def run_league_job(
    gameweek: int | None = None,
    competitor_ids: List[int] | None = None,
    top_global_count: int | None = None,
    use_cache: bool = False,
) -> None:
    """Fetch competitor + top global datasets and persist them."""
    logger.info("League job starting")

    try:
        resolved_competitor_ids = _dedupe_ids(competitor_ids or [TEAM_ID, *COMPETITORS])
        competitors = build_competitive_dataset(
            resolved_competitor_ids,
            season=SEASON,
            gameweek=gameweek,
            use_cache=use_cache,
        )

        resolved_top_global_count = top_global_count or TOP_GLOBAL_COUNT
        top_global = get_top_global_teams(n=resolved_top_global_count, use_cache=use_cache)
        top_global_ids = _dedupe_ids([TEAM_ID, *[entry.get("entry_id") for entry in top_global]])
        global_managers = build_competitive_dataset(
            top_global_ids,
            season=SEASON,
            gameweek=gameweek,
            use_cache=use_cache,
        )

        write_singleton(
            "competitive_data",
            {
                "competitors": competitors,
                "global_managers": global_managers,
                "top_global_summary": top_global,
                "season": SEASON,
                "current_gameweek": gameweek,
            },
        )

        message = (
            f"competitors={len(competitors)} "
            f"global_managers={len(global_managers)} "
            f"top_global_summary={len(top_global)}"
        )
        log_refresh("league", "ok", message)
        logger.info("League job completed: %s", message)

    except Exception as exc:
        logger.exception("League job failed")
        log_refresh("league", "error", str(exc))
        raise
