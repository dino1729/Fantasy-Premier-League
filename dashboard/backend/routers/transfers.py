"""Transfer endpoints - solver results and transfer history."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_solver_results, read_singleton

router = APIRouter(prefix="/api", tags=["transfers"])


@router.get("/transfers/solver")
async def get_solver():
    results = read_solver_results()
    if results is None:
        gw_row = read_singleton("squad_data")
        gameweek = gw_row.get("gameweek") if gw_row else 0
        return {
            "gameweek": int(gameweek or 0),
            "conservative": None,
            "balanced": None,
            "aggressive": None,
            "recommended": "balanced",
            "baseline_xp": 0.0,
            "refreshed_at": None,
            "status": "pending",
            "message": "Solver has not run yet - computing in background.",
        }
    return results


@router.get("/transfers/history")
async def get_transfer_history():
    row = read_singleton("gw_history")
    if row is None:
        return JSONResponse(status_code=503, content={"error": "Warming up"})
    data = row["data"]
    return {
        "transfers": data.get("transfers", []),
        "refreshed_at": row.get("refreshed_at"),
    }
