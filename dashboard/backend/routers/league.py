"""League endpoints - competitor and global manager data."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_singleton

router = APIRouter(prefix="/api", tags=["league"])


@router.get("/league/competitors")
async def get_competitors():
    row = read_singleton("competitive_data")
    if row is None:
        return JSONResponse(status_code=503, content={"error": "Computing league data..."})
    data = row["data"]
    return {
        "competitors": data.get("competitors", []),
        "refreshed_at": row.get("refreshed_at"),
    }


@router.get("/league/global")
async def get_global_managers():
    row = read_singleton("competitive_data")
    if row is None:
        return JSONResponse(status_code=503, content={"error": "Computing league data..."})
    data = row["data"]
    return {
        "global_managers": data.get("global_managers", []),
        "refreshed_at": row.get("refreshed_at"),
    }
