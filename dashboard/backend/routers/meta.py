"""Meta endpoint - current GW, season, team info, refresh times."""

from fastapi import APIRouter
from dashboard.backend.database import get_refresh_status, read_singleton

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/meta")
async def get_meta():
    bootstrap = read_singleton("bootstrap_cache")
    status = get_refresh_status()

    if bootstrap is None:
        return {"ready": False, "refresh_status": status}

    data = bootstrap["data"]
    return {
        "ready": True,
        "current_gameweek": data.get("current_gameweek"),
        "season": data.get("season"),
        "team_info": data.get("team_info"),
        "teams": data.get("teams"),
        "refresh_status": status,
    }
