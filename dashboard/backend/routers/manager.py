"""Manager report endpoints - season overview and captain analysis."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dashboard.backend.database import read_singleton

router = APIRouter(prefix="/api", tags=["manager"])


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_captain_analysis(season_history: list) -> dict:
    """Build captain picks with optimal counterfactual from season squad snapshots."""
    rows = []
    total_earned = 0
    total_optimal = 0
    total_correct = 0

    for gw_entry in season_history:
        squad = gw_entry.get("squad", [])
        if not squad:
            continue

        captain = next((p for p in squad if p.get("is_captain")), None)
        if not captain:
            continue

        captain_points = _safe_int(captain.get("stats", {}).get("event_points", 0))
        multiplier = _safe_int(captain.get("multiplier", 2), default=2)
        captain_return = captain_points * multiplier

        optimal_player = max(
            squad,
            key=lambda player: (
                _safe_int(player.get("stats", {}).get("event_points", 0)),
                1 if player.get("is_captain") else 0,
            ),
        )
        optimal_points = _safe_int(optimal_player.get("stats", {}).get("event_points", 0))
        optimal_return = optimal_points * multiplier

        was_optimal = captain_points == optimal_points
        if was_optimal:
            total_correct += 1

        total_earned += captain_return
        total_optimal += optimal_return

        rows.append(
            {
                "gameweek": _safe_int(gw_entry.get("gameweek")),
                "captain_id": _safe_int(captain.get("id")),
                "captain_name": captain.get("name"),
                "captain_points": captain_points,
                "captain_multiplier": multiplier,
                "captain_return": captain_return,
                "optimal_id": _safe_int(optimal_player.get("id")),
                "optimal_name": optimal_player.get("name"),
                "optimal_points": optimal_points,
                "optimal_return": optimal_return,
                "points_lost": max(0, optimal_return - captain_return),
                "was_optimal": was_optimal,
            }
        )

    total_gws = len(rows)
    accuracy = (total_correct / total_gws) if total_gws > 0 else 0.0
    total_lost = max(0, total_optimal - total_earned)

    return {
        "captain_picks": rows,
        "summary": {
            "total_gws": total_gws,
            "correct_picks": total_correct,
            "accuracy": round(accuracy, 4),
            "captain_points_earned": total_earned,
            "captain_points_optimal": total_optimal,
            "captain_points_lost": total_lost,
        },
    }


@router.get("/manager/overview")
async def get_manager_overview():
    gw_row = read_singleton("gw_history")
    squad_row = read_singleton("squad_data")
    if gw_row is None or squad_row is None:
        return JSONResponse(status_code=503, content={"error": "Warming up"})

    gw_data = gw_row["data"]
    squad_data = squad_row["data"]

    return {
        "team_info": squad_data.get("team_info"),
        "gw_history": gw_data.get("gw_history"),
        "season_history": gw_data.get("season_history"),
        "chips_used": gw_data.get("chips_used"),
        "team_value": gw_data.get("team_value"),
        "bank": gw_data.get("bank"),
        "free_transfers": gw_data.get("free_transfers"),
        "refreshed_at": gw_row.get("refreshed_at"),
    }


@router.get("/manager/captains")
async def get_captain_analysis():
    gw_row = read_singleton("gw_history")
    squad_row = read_singleton("squad_data")
    if gw_row is None or squad_row is None:
        return JSONResponse(status_code=503, content={"error": "Warming up"})

    gw_data = gw_row["data"]
    season_history = gw_data.get("season_history", [])
    analysis = _build_captain_analysis(season_history)

    return {
        "captain_picks": analysis["captain_picks"],
        "summary": analysis["summary"],
        "refreshed_at": gw_row.get("refreshed_at"),
    }
