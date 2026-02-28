"""Fixtures endpoint - FDR grid data."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_singleton

router = APIRouter(prefix="/api", tags=["fixtures"])


@router.get("/fixtures/fdr-grid")
async def get_fdr_grid():
    row = read_singleton("fixtures")
    if row is None:
        return JSONResponse(status_code=503, content={"error": "Warming up"})
    data = row["data"]
    return {
        "fdr_grid": data.get("fdr_grid"),
        "bgw_dgw": data.get("bgw_dgw"),
        "current_gameweek": data.get("current_gameweek"),
        "refreshed_at": row.get("refreshed_at"),
    }
