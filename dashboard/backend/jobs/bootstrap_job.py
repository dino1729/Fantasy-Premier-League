"""Bootstrap job - fetches core FPL data every 30 minutes.

Populates: bootstrap_cache, squad_data, gw_history, fixtures
"""

import logging
import sys
from pathlib import Path

# Ensure project root is importable
_project_root = str(Path(__file__).parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dashboard.backend.database import (
    log_refresh,
    write_singleton,
)
from reports.fpl_report.data_fetcher import (
    FPLDataFetcher,
    build_competitive_dataset,
    get_bgw_dgw_gameweeks,
    get_top_global_teams,
)
from utils.config import COMPETITORS, SEASON, TEAM_ID

logger = logging.getLogger(__name__)


def run_bootstrap_job(gameweek: int = None):
    """Fetch squad, fixtures, history and write to SQLite.

    Args:
        gameweek: Target GW. None = auto-detect.
    """
    logger.info("Bootstrap job starting")
    try:
        fetcher = FPLDataFetcher(TEAM_ID, SEASON, use_cache=False)
        gw = gameweek or fetcher.get_current_gameweek()

        # Team info
        team_info = fetcher.get_team_info(gameweek=gw)

        # Squad
        squad = fetcher.get_current_squad(gw)
        squad_issues = fetcher.get_squad_issues(squad, gw)

        write_singleton("squad_data", {
            "squad": squad,
            "issues": squad_issues,
            "team_info": team_info,
        }, gameweek=gw)

        # GW history + season history + chips + transfers
        gw_history = fetcher.get_gw_history()
        season_history = fetcher.get_season_history()
        chips_used = fetcher.get_chips_used()
        transfers = fetcher.get_transfers()
        free_transfers = fetcher.calculate_free_transfers(gw)
        bank = fetcher.get_bank(gw)
        team_value = fetcher.get_team_value(gw)

        write_singleton("gw_history", {
            "gw_history": gw_history,
            "season_history": season_history,
            "chips_used": chips_used,
            "transfers": transfers,
            "free_transfers": free_transfers,
            "bank": bank,
            "team_value": team_value,
            "current_gameweek": gw,
        })

        # Fixtures: upcoming for each squad player's team + BGW/DGW + FDR grid
        team_ids_in_squad = list({p["team_id"] for p in squad})
        fixtures_by_team = {}
        for tid in team_ids_in_squad:
            fixtures_by_team[tid] = fetcher.get_upcoming_fixtures(tid, num_fixtures=8)

        # Full FDR grid: all 20 teams
        all_teams = fetcher.bootstrap_data.get("teams", [])
        fdr_grid = {}
        for team in all_teams:
            tid = team["id"]
            fdr_grid[tid] = {
                "name": team["name"],
                "short_name": team["short_name"],
                "fixtures": fetcher.get_upcoming_fixtures(tid, num_fixtures=8),
            }

        bgw_dgw = get_bgw_dgw_gameweeks(use_cache=False)

        write_singleton("fixtures", {
            "squad_fixtures": fixtures_by_team,
            "fdr_grid": fdr_grid,
            "bgw_dgw": bgw_dgw,
            "current_gameweek": gw,
        })

        # Bootstrap cache (teams list, positions, etc.)
        teams = [
            {"id": t["id"], "name": t["name"], "short_name": t["short_name"]}
            for t in all_teams
        ]
        write_singleton("bootstrap_cache", {
            "teams": teams,
            "current_gameweek": gw,
            "season": SEASON,
            "team_info": team_info,
        })

        log_refresh("bootstrap", "ok", f"GW{gw} data refreshed")
        logger.info("Bootstrap job completed (GW%d)", gw)

    except Exception as e:
        logger.exception("Bootstrap job failed")
        log_refresh("bootstrap", "error", str(e))
        raise
