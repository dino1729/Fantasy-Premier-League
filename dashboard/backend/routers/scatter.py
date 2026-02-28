"""Scatter plot data endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_scatter

router = APIRouter(prefix="/api", tags=["scatter"])

VALID_CHART_TYPES = {"xg_goals", "xa_assists", "usage_output", "defensive"}


@router.get("/scatter/{chart_type}")
async def get_scatter(chart_type: str):
    if chart_type not in VALID_CHART_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid chart type. Must be one of: {sorted(VALID_CHART_TYPES)}"},
        )
    data = read_scatter(chart_type)
    if data is None:
        return JSONResponse(status_code=503, content={"error": "Analysis not yet complete"})
    return {"chart_type": chart_type, "data": data}
