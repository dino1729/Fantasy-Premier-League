"""APScheduler setup for periodic data refresh jobs."""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler = None


def _bootstrap():
    from dashboard.backend.jobs.bootstrap_job import run_bootstrap_job
    try:
        run_bootstrap_job()
    except Exception:
        logger.exception("Scheduled bootstrap job failed")


def _analysis():
    from dashboard.backend.jobs.analysis_job import run_analysis_job
    try:
        run_analysis_job()
    except Exception:
        logger.exception("Scheduled analysis job failed")


def _solver():
    from dashboard.backend.jobs.solver_job import run_solver_job
    try:
        run_solver_job()
    except Exception:
        logger.exception("Scheduled solver job failed")


def _league():
    from dashboard.backend.jobs.league_job import run_league_job
    try:
        run_league_job()
    except Exception:
        logger.exception("Scheduled league job failed")


def start_scheduler():
    global _scheduler
    _scheduler = BackgroundScheduler(daemon=True)

    _scheduler.add_job(_bootstrap, IntervalTrigger(minutes=30), id="bootstrap",
                       name="Bootstrap (FPL API + fixtures)", replace_existing=True)
    _scheduler.add_job(_analysis, IntervalTrigger(hours=2), id="analysis",
                       name="Player analysis + predictions", replace_existing=True)
    _scheduler.add_job(_solver, IntervalTrigger(hours=6), id="solver",
                       name="MIP solver (3 scenarios)", replace_existing=True)
    _scheduler.add_job(_league, IntervalTrigger(hours=1), id="league",
                       name="League + global managers", replace_existing=True)

    _scheduler.start()
    logger.info("Scheduler started with 4 periodic jobs")


def shutdown_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
