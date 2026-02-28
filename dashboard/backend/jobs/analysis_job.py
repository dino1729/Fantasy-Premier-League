"""Analysis job - populates player_analysis table with stats + predictions.

Runs PlayerAnalyzer for top ~200 players (form trends, ICT, percentiles)
and FPLPointsPredictor for xP predictions across 5 GWs.
All ~600 players get basic stats written; only top players get deep analysis.
"""

import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dashboard.backend.database import log_refresh, write_player_analyses, write_scatter
from reports.fpl_report.data_fetcher import FPLDataFetcher
from reports.fpl_report.player_analyzer import PlayerAnalyzer
from reports.fpl_report.predictor import FPLPointsPredictor
from utils.config import SEASON, TEAM_ID

logger = logging.getLogger(__name__)

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DEEP_ANALYSIS_MIN_MINUTES = 90
DEEP_ANALYSIS_MAX_PLAYERS = 250


def _safe_float(val, default=0.0):
    """Convert value to float, returning default for None/NaN/errors."""
    if val is None:
        return default
    try:
        f = float(val)
        import math
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _build_basic_row(element: dict) -> dict:
    """Build a player_analysis row from raw bootstrap element data."""
    pos = POSITION_MAP.get(element.get("element_type", 0), "MID")
    price = _safe_float(element.get("now_cost", 0)) / 10.0

    return {
        "player_id": element["id"],
        "web_name": element.get("web_name", "Unknown"),
        "position": pos,
        "team": str(element.get("team", 0)),  # team ID as string; bootstrap_job resolves names
        "price": round(price, 1),
        "form": _safe_float(element.get("form")),
        "total_points": _safe_int(element.get("total_points")),
        "minutes": _safe_int(element.get("minutes")),
        "goals": _safe_int(element.get("goals_scored")),
        "assists": _safe_int(element.get("assists")),
        "clean_sheets": _safe_int(element.get("clean_sheets")),
        "bps": _safe_int(element.get("bps")),
        "xg": _safe_float(element.get("expected_goals")),
        "xa": _safe_float(element.get("expected_assists")),
        "xg_diff": round(
            _safe_int(element.get("goals_scored")) - _safe_float(element.get("expected_goals")), 2
        ),
        "xa_diff": round(
            _safe_int(element.get("assists")) - _safe_float(element.get("expected_assists")), 2
        ),
        "influence": _safe_float(element.get("influence")),
        "creativity": _safe_float(element.get("creativity")),
        "threat": _safe_float(element.get("threat")),
        "ict_index": _safe_float(element.get("ict_index")),
        # Predictions - null until predictor runs
        "xp_gw1": None,
        "xp_gw2": None,
        "xp_gw3": None,
        "xp_gw4": None,
        "xp_gw5": None,
        "xp_confidence": None,
        # Percentiles - null until deep analysis
        "pct_form": None,
        "pct_ict": None,
        "pct_xg": None,
        "pct_xp": None,
        # Ownership
        "transfers_in_event": _safe_int(element.get("transfers_in_event")),
        "transfers_out_event": _safe_int(element.get("transfers_out_event")),
        "selected_by_percent": _safe_float(element.get("selected_by_percent")),
        # Deep analysis blobs - null until analyzed
        "form_trend": None,
        "ict_breakdown": None,
        "raw_stats": None,
    }


def _enrich_with_deep_analysis(row: dict, analyzer: PlayerAnalyzer, fetcher: FPLDataFetcher) -> None:
    """Add form_trend, ict_breakdown, percentiles, raw_stats from PlayerAnalyzer."""
    pid = row["player_id"]
    pos = row["position"]

    try:
        stats = fetcher.get_player_stats(pid)
        if not stats:
            return

        # Form trend
        form_trend = analyzer.analyze_form_trend(pid)
        row["form_trend"] = form_trend

        # ICT breakdown
        ict_breakdown = analyzer.analyze_ict_breakdown(stats)
        row["ict_breakdown"] = ict_breakdown

        # Peer comparison -> percentiles
        peers = analyzer.compare_to_peers(stats, pos)
        pcts = peers.get("percentiles", {})
        row["pct_form"] = pcts.get("form")
        row["pct_ict"] = pcts.get("ict_index")
        row["pct_xg"] = pcts.get("expected_goals")
        row["pct_xp"] = pcts.get("total_points")

        # Raw stats blob
        row["raw_stats"] = {
            "team_id": _safe_int(stats.get("team")),
            "minutes": _safe_int(stats.get("minutes")),
            "total_points": _safe_int(stats.get("total_points")),
            "goals": _safe_int(stats.get("goals_scored")),
            "assists": _safe_int(stats.get("assists")),
            "clean_sheets": _safe_int(stats.get("clean_sheets")),
            "saves": _safe_int(stats.get("saves")),
            "goals_conceded": _safe_int(stats.get("goals_conceded")),
            "bonus": _safe_int(stats.get("bonus")),
            "bps": _safe_int(stats.get("bps")),
            "photo_code": _safe_int(stats.get("code")),
            "team_code": _safe_int(stats.get("team_code")),
        }
    except Exception as e:
        logger.warning("Deep analysis failed for player %d: %s", pid, e)


def _enrich_with_predictions(rows: list, predictor: FPLPointsPredictor, player_ids: list) -> None:
    """Add xP predictions to player rows."""
    try:
        predictions = predictor.predict_multiple_gws(player_ids, num_gws=5)
    except Exception as e:
        logger.warning("Predictions failed: %s", e)
        return

    pid_to_row = {r["player_id"]: r for r in rows}
    for pid, pred in predictions.items():
        row = pid_to_row.get(pid)
        if not row:
            continue
        gw_preds = pred.get("predictions", [])
        for i in range(min(5, len(gw_preds))):
            row[f"xp_gw{i+1}"] = round(gw_preds[i], 2) if gw_preds[i] is not None else None
        row["xp_confidence"] = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }.get(pred.get("confidence", "low"), 0.3)


def _resolve_team_names(rows: list, teams: list) -> None:
    """Replace team IDs with short names (e.g. '1' -> 'ARS')."""
    team_map = {str(t["id"]): t["short_name"] for t in teams}
    for row in rows:
        row["team"] = team_map.get(str(row["team"]), row["team"])


def _extract_scatter_data(rows: list) -> None:
    """Pre-compute scatter chart data from player rows and write to scatter_data table."""
    # Only include players with meaningful minutes
    eligible = [r for r in rows if _safe_int(r.get("minutes")) >= 90]

    charts = {
        "xg_goals": {"x_field": "xg", "y_field": "goals", "x_label": "xG", "y_label": "Goals"},
        "xa_assists": {"x_field": "xa", "y_field": "assists", "x_label": "xA", "y_label": "Assists"},
        "usage_output": {"x_field": "minutes", "y_field": "total_points", "x_label": "Minutes", "y_label": "Total Points"},
        "defensive": {"x_field": "clean_sheets", "y_field": "bps", "x_label": "Clean Sheets", "y_label": "BPS"},
    }

    for chart_type, cfg in charts.items():
        points = []
        for r in eligible:
            x = _safe_float(r.get(cfg["x_field"]))
            y = _safe_float(r.get(cfg["y_field"]))
            # Skip zero-zero for scatter relevance (except usage_output)
            if chart_type != "usage_output" and x == 0 and y == 0:
                continue
            points.append({
                "player_id": r["player_id"],
                "name": r["web_name"],
                "team": r["team"],
                "position": r["position"],
                "x": x,
                "y": y,
                "minutes": _safe_int(r.get("minutes")),
            })
        write_scatter(chart_type, points)


def run_analysis_job():
    """Analyze all players and write stats + predictions to SQLite."""
    logger.info("Analysis job starting")
    try:
        fetcher = FPLDataFetcher(TEAM_ID, SEASON, use_cache=False)
        elements = fetcher.bootstrap_data.get("elements", [])
        teams = fetcher.bootstrap_data.get("teams", [])
        logger.info("Loaded %d players from bootstrap", len(elements))

        # Build basic rows for ALL players
        rows = [_build_basic_row(el) for el in elements]
        _resolve_team_names(rows, teams)

        # Identify top players for deep analysis (sorted by total_points, min minutes)
        deep_candidates = sorted(
            [r for r in rows if _safe_int(r.get("minutes")) >= DEEP_ANALYSIS_MIN_MINUTES],
            key=lambda r: r.get("total_points", 0),
            reverse=True,
        )[:DEEP_ANALYSIS_MAX_PLAYERS]
        deep_ids = {r["player_id"] for r in deep_candidates}
        logger.info("Running deep analysis on %d players", len(deep_candidates))

        # Deep analysis: form trends, ICT, percentiles
        analyzer = PlayerAnalyzer(fetcher)
        for row in rows:
            if row["player_id"] in deep_ids:
                _enrich_with_deep_analysis(row, analyzer, fetcher)

        # Predictions: train model then predict
        logger.info("Training FPLPointsPredictor...")
        predictor = FPLPointsPredictor(fetcher)
        try:
            predictor.train(player_ids=list(deep_ids))
        except TypeError:
            # Backward compatibility for predictor implementations that do not
            # require explicit player_ids in train().
            predictor.train()

        if predictor.is_trained:
            logger.info("Generating predictions for %d players", len(deep_candidates))
            _enrich_with_predictions(rows, predictor, list(deep_ids))
        else:
            logger.warning("Predictor failed to train - skipping predictions")

        # Write all rows to DB
        write_player_analyses(rows)
        logger.info("Wrote %d player rows to database", len(rows))

        # Extract scatter data (Sprint 7 will use this, but we build it now)
        _extract_scatter_data(rows)
        logger.info("Scatter data written for 4 chart types")

        log_refresh("analysis", "ok", f"Analyzed {len(rows)} players ({len(deep_candidates)} deep)")

    except Exception as e:
        logger.exception("Analysis job failed")
        log_refresh("analysis", "error", str(e))
        raise
