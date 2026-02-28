"""FastAPI application for FPL Dashboard.

Single process serving both API endpoints and the React SPA static files.
APScheduler runs in-process for periodic data refresh.
"""

import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dashboard.backend.database import init_db, get_refresh_status, _dumps
from dashboard.backend.scheduler import start_scheduler, shutdown_scheduler

logger = logging.getLogger(__name__)

# Track whether initial blocking jobs have completed
_warmup_complete = asyncio.Event()


def _run_startup_job(job_name: str, fn) -> None:
    """Run a startup job and keep failures isolated to refresh logs."""
    try:
        fn()
    except Exception:
        logger.exception("Startup %s job failed", job_name)


def _run_blocking_jobs():
    """Run bootstrap on startup (blocking), then kick off non-blocking jobs."""
    from dashboard.backend.jobs.bootstrap_job import run_bootstrap_job
    try:
        run_bootstrap_job()
    except Exception:
        logger.exception("Startup bootstrap job failed")
    _warmup_complete.set()

    # Kick off heavy jobs immediately after bootstrap so routes like
    # /api/transfers/solver are populated early, not only on interval triggers.
    from dashboard.backend.jobs.analysis_job import run_analysis_job
    from dashboard.backend.jobs.solver_job import run_solver_job
    from dashboard.backend.jobs.league_job import run_league_job

    for job_name, fn in (
        ("analysis", run_analysis_job),
        ("solver", run_solver_job),
        ("league", run_league_job),
    ):
        threading.Thread(
            target=_run_startup_job,
            args=(job_name, fn),
            daemon=True,
            name=f"startup-{job_name}",
        ).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init DB schema
    init_db()
    logger.info("Database initialized")

    # Run blocking startup jobs in a thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_blocking_jobs)

    # Start scheduler for periodic refreshes
    start_scheduler()
    logger.info("Scheduler started")

    yield

    # Shutdown
    shutdown_scheduler()
    logger.info("Scheduler stopped")


class SafeJSONResponse(JSONResponse):
    """JSONResponse that handles NaN/Inf floats."""

    def render(self, content: Any) -> bytes:
        return _dumps(content).encode("utf-8")


app = FastAPI(
    title="FPL Analytics Dashboard",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)

# CORS for Vite dev server (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --- Routers ---
from dashboard.backend.routers.meta import router as meta_router
from dashboard.backend.routers.squad import router as squad_router
from dashboard.backend.routers.players import router as players_router
from dashboard.backend.routers.fixtures import router as fixtures_router
from dashboard.backend.routers.transfers import router as transfers_router
from dashboard.backend.routers.league import router as league_router
from dashboard.backend.routers.scatter import router as scatter_router
from dashboard.backend.routers.manager import router as manager_router

app.include_router(meta_router)
app.include_router(squad_router)
app.include_router(players_router)
app.include_router(fixtures_router)
app.include_router(transfers_router)
app.include_router(league_router)
app.include_router(scatter_router)
app.include_router(manager_router)


@app.get("/api/health")
async def health():
    """Startup readiness check."""
    is_ready = _warmup_complete.is_set()
    status = get_refresh_status()
    return {
        "ready": is_ready,
        "jobs": status,
    }


# --- Static SPA serving (production) ---
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    # Serve index.html for all non-API routes (SPA fallback)
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        # Try serving exact file first, then fall back to index.html
        file_path = _frontend_dist / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dist / "index.html"))
