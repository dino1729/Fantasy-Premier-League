"""Squad endpoints - current squad and issues."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_singleton

router = APIRouter(prefix="/api", tags=["squad"])


@router.get("/squad")
async def get_squad():
    row = read_singleton("squad_data")
    if row is None:
        return JSONResponse(status_code=503, content={"error": "Warming up - squad data not yet loaded"})
    data = row["data"]
    return {
        "gameweek": row.get("gameweek"),
        "squad": data.get("squad"),
        "team_info": data.get("team_info"),
        "refreshed_at": row.get("refreshed_at"),
    }


@router.get("/squad/issues")
async def get_squad_issues():
    row = read_singleton("squad_data")
    if row is None:
        return JSONResponse(status_code=503, content={"error": "Warming up"})
    return row["data"].get("issues", {})
