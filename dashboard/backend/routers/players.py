"""Player endpoints - paginated/filtered player stats."""

from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_players, read_player

router = APIRouter(prefix="/api", tags=["players"])


@router.get("/players")
async def get_players(
    position: Optional[str] = Query(None, pattern="^(GKP|DEF|MID|FWD)$"),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_minutes: Optional[int] = Query(None, ge=0),
    sort_by: str = Query("total_points"),
    sort_dir: str = Query("DESC", pattern="^(ASC|DESC)$"),
):
    players = read_players(
        position=position,
        min_price=min_price,
        max_price=max_price,
        min_minutes=min_minutes,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    return {"players": players, "total": len(players)}


@router.get("/players/{player_id}")
async def get_player_detail(player_id: int):
    player = read_player(player_id)
    if player is None:
        return JSONResponse(status_code=404, content={"error": "Player not found"})
    return player
