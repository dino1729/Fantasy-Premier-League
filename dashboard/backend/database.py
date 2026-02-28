"""SQLite database layer for FPL Dashboard.

Provides schema creation and read/write helpers for pre-computed analytics data.
All data tables use a singleton-blob pattern (id=1) except player_analysis which
has flat columns for SQL-level filtering.
"""

import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path(__file__).parent.parent / "fpl_dashboard.db"


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to None."""

    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

    def encode(self, o):
        return super().encode(_sanitize_floats(o))


def _sanitize_floats(obj):
    """Recursively replace NaN/Inf floats with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _dumps(obj) -> str:
    """JSON serialize with NaN/Inf safety."""
    return json.dumps(obj, default=str, cls=_SafeEncoder)

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


@contextmanager
def get_db():
    """Context manager yielding a thread-local SQLite connection."""
    conn = _get_conn()
    try:
        yield conn
    finally:
        conn.commit()


def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS refresh_log (
                job_name TEXT PRIMARY KEY,
                last_run_at TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS bootstrap_cache (
                id INTEGER PRIMARY KEY CHECK(id=1),
                data TEXT,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS squad_data (
                id INTEGER PRIMARY KEY CHECK(id=1),
                gameweek INTEGER,
                data TEXT,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS gw_history (
                id INTEGER PRIMARY KEY CHECK(id=1),
                data TEXT,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS fixtures (
                id INTEGER PRIMARY KEY CHECK(id=1),
                data TEXT,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS solver_results (
                id INTEGER PRIMARY KEY CHECK(id=1),
                gameweek INTEGER,
                conservative TEXT,
                balanced TEXT,
                aggressive TEXT,
                recommended TEXT,
                baseline_xp REAL,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS competitive_data (
                id INTEGER PRIMARY KEY CHECK(id=1),
                data TEXT,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS player_analysis (
                player_id INTEGER PRIMARY KEY,
                web_name TEXT,
                position TEXT,
                team TEXT,
                price REAL,
                form REAL,
                total_points INTEGER,
                minutes INTEGER,
                goals INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                bps INTEGER,
                xg REAL,
                xa REAL,
                xg_diff REAL,
                xa_diff REAL,
                influence REAL,
                creativity REAL,
                threat REAL,
                ict_index REAL,
                xp_gw1 REAL,
                xp_gw2 REAL,
                xp_gw3 REAL,
                xp_gw4 REAL,
                xp_gw5 REAL,
                xp_confidence REAL,
                pct_form REAL,
                pct_ict REAL,
                pct_xg REAL,
                pct_xp REAL,
                transfers_in_event INTEGER,
                transfers_out_event INTEGER,
                selected_by_percent REAL,
                form_trend TEXT,
                ict_breakdown TEXT,
                raw_stats TEXT,
                refreshed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS scatter_data (
                chart_type TEXT PRIMARY KEY,
                data TEXT,
                refreshed_at TEXT
            );
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Singleton blob helpers ---

def write_singleton(table: str, data: Any, **extra_cols):
    """Upsert a singleton row (id=1) with JSON-serialized data."""
    cols = ["id", "data", "refreshed_at"]
    vals = [1, _dumps(data), _now()]
    for k, v in extra_cols.items():
        cols.append(k)
        vals.append(v)
    placeholders = ", ".join("?" * len(cols))
    col_str = ", ".join(cols)
    with get_db() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
            vals,
        )


def read_singleton(table: str) -> Optional[Dict]:
    """Read a singleton row, parsing the JSON data column."""
    with get_db() as conn:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = 1"
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("data"):
        result["data"] = json.loads(result["data"])
    return result


# --- Refresh log ---

def log_refresh(job_name: str, status: str, message: str = ""):
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO refresh_log (job_name, last_run_at, status, message) "
            "VALUES (?, ?, ?, ?)",
            (job_name, _now(), status, message),
        )


def get_refresh_status() -> Dict[str, Dict]:
    """Get last refresh info for all jobs."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM refresh_log").fetchall()
    return {row["job_name"]: dict(row) for row in rows}


# --- Player analysis (flat columns) ---

def write_player_analyses(players: List[Dict]):
    """Bulk write player analysis rows. Replaces all existing rows."""
    if not players:
        return
    now = _now()
    with get_db() as conn:
        conn.execute("DELETE FROM player_analysis")
        for p in players:
            conn.execute(
                """INSERT INTO player_analysis (
                    player_id, web_name, position, team, price, form,
                    total_points, minutes, goals, assists, clean_sheets, bps,
                    xg, xa, xg_diff, xa_diff,
                    influence, creativity, threat, ict_index,
                    xp_gw1, xp_gw2, xp_gw3, xp_gw4, xp_gw5, xp_confidence,
                    pct_form, pct_ict, pct_xg, pct_xp,
                    transfers_in_event, transfers_out_event, selected_by_percent,
                    form_trend, ict_breakdown, raw_stats, refreshed_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )""",
                (
                    p["player_id"], p.get("web_name"), p.get("position"),
                    p.get("team"), p.get("price"), p.get("form"),
                    p.get("total_points"), p.get("minutes"),
                    p.get("goals"), p.get("assists"),
                    p.get("clean_sheets"), p.get("bps"),
                    p.get("xg"), p.get("xa"),
                    p.get("xg_diff"), p.get("xa_diff"),
                    p.get("influence"), p.get("creativity"),
                    p.get("threat"), p.get("ict_index"),
                    p.get("xp_gw1"), p.get("xp_gw2"),
                    p.get("xp_gw3"), p.get("xp_gw4"),
                    p.get("xp_gw5"), p.get("xp_confidence"),
                    p.get("pct_form"), p.get("pct_ict"),
                    p.get("pct_xg"), p.get("pct_xp"),
                    p.get("transfers_in_event"), p.get("transfers_out_event"),
                    p.get("selected_by_percent"),
                    _dumps(p.get("form_trend")) if p.get("form_trend") else None,
                    _dumps(p.get("ict_breakdown")) if p.get("ict_breakdown") else None,
                    _dumps(p.get("raw_stats")) if p.get("raw_stats") else None,
                    now,
                ),
            )


def read_players(
    position: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_minutes: Optional[int] = None,
    sort_by: str = "total_points",
    sort_dir: str = "DESC",
) -> List[Dict]:
    """Read player analysis rows with optional filters."""
    conditions = []
    params = []
    if position:
        conditions.append("position = ?")
        params.append(position)
    if min_price is not None:
        conditions.append("price >= ?")
        params.append(min_price)
    if max_price is not None:
        conditions.append("price <= ?")
        params.append(max_price)
    if min_minutes is not None:
        conditions.append("minutes >= ?")
        params.append(min_minutes)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    allowed_sorts = {
        "total_points", "price", "form", "minutes", "goals", "assists",
        "xg", "xa", "ict_index", "xp_gw1", "selected_by_percent",
        "web_name", "bps", "clean_sheets",
    }
    if sort_by not in allowed_sorts:
        sort_by = "total_points"
    direction = "ASC" if sort_dir.upper() == "ASC" else "DESC"

    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM player_analysis {where} ORDER BY {sort_by} {direction}",
            params,
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        for json_col in ("form_trend", "ict_breakdown", "raw_stats"):
            if d.get(json_col):
                d[json_col] = json.loads(d[json_col])
        result.append(d)
    return result


def read_player(player_id: int) -> Optional[Dict]:
    """Read a single player analysis row."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM player_analysis WHERE player_id = ?", (player_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for json_col in ("form_trend", "ict_breakdown", "raw_stats"):
        if d.get(json_col):
            d[json_col] = json.loads(d[json_col])
    return d


# --- Scatter data ---

def write_scatter(chart_type: str, data: Any):
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scatter_data (chart_type, data, refreshed_at) "
            "VALUES (?, ?, ?)",
            (chart_type, _dumps(data), _now()),
        )


def read_scatter(chart_type: str) -> Optional[List]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT data FROM scatter_data WHERE chart_type = ?", (chart_type,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["data"])


# --- Solver results ---

def write_solver_results(
    gameweek: int,
    conservative: Any,
    balanced: Any,
    aggressive: Any,
    recommended: str,
    baseline_xp: float,
):
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO solver_results
               (id, gameweek, conservative, balanced, aggressive, recommended, baseline_xp, refreshed_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?)""",
            (
                gameweek,
                _dumps(conservative),
                _dumps(balanced),
                _dumps(aggressive),
                recommended,
                baseline_xp,
                _now(),
            ),
        )


def read_solver_results() -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM solver_results WHERE id = 1").fetchone()
    if row is None:
        return None
    d = dict(row)
    for col in ("conservative", "balanced", "aggressive"):
        if d.get(col):
            d[col] = json.loads(d[col])
    return d
