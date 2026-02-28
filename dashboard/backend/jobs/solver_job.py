"""Solver job - runs MIP strategy in subprocess and writes results."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from dashboard.backend.database import log_refresh, write_solver_results
from utils.config import (
    FREE_TRANSFERS_OVERRIDE,
    MIP_CANDIDATE_POOL,
    MIP_TIME_LIMIT,
    SEASON,
    TEAM_ID,
    TRANSFER_HORIZON,
)

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parents[3]
_worker_path = Path(__file__).resolve().with_name("_solver_worker.py")


def _read_worker_payload(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("solver worker produced no stdout payload")
    # Worker is expected to emit exactly one JSON line at the end.
    return json.loads(lines[-1])


def run_solver_job(gameweek: int | None = None) -> None:
    """Run transfer solver via subprocess and persist scenario results."""
    logger.info("Solver job starting")

    cmd = [
        sys.executable,
        str(_worker_path),
        "--team-id",
        str(TEAM_ID),
        "--season",
        SEASON,
        "--horizon",
        str(TRANSFER_HORIZON),
        "--time-limit",
        str(MIP_TIME_LIMIT),
        "--candidate-pool",
        str(MIP_CANDIDATE_POOL),
    ]
    if gameweek is not None:
        cmd.extend(["--gameweek", str(gameweek)])
    if FREE_TRANSFERS_OVERRIDE is not None:
        cmd.extend(["--free-transfers", str(FREE_TRANSFERS_OVERRIDE)])

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(_project_root),
            capture_output=True,
            text=True,
            timeout=60 * 60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        message = f"solver worker timed out after {exc.timeout}s"
        logger.exception(message)
        log_refresh("solver", "error", message)
        raise RuntimeError(message) from exc
    except Exception as exc:
        message = f"solver worker execution failed: {exc}"
        logger.exception(message)
        log_refresh("solver", "error", message)
        raise

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        summary = stderr.splitlines()[-1] if stderr else "worker exited with non-zero status"
        logger.error("Solver worker failed: %s", summary)
        log_refresh("solver", "error", summary[:500])
        raise RuntimeError(summary)

    try:
        payload = _read_worker_payload(completed.stdout or "")
    except Exception as exc:
        stderr_tail = "\n".join((completed.stderr or "").splitlines()[-5:])
        message = f"invalid solver worker payload: {exc}"
        logger.error("%s\nstderr tail:\n%s", message, stderr_tail)
        log_refresh("solver", "error", message[:500])
        raise RuntimeError(message) from exc

    solver_gameweek = int(payload.get("gameweek") or gameweek or 0)
    conservative = payload.get("conservative")
    balanced = payload.get("balanced")
    aggressive = payload.get("aggressive")
    recommended = str(payload.get("recommended", "balanced"))
    baseline_xp = float(payload.get("baseline_xp") or 0.0)

    write_solver_results(
        gameweek=solver_gameweek,
        conservative=conservative,
        balanced=balanced,
        aggressive=aggressive,
        recommended=recommended,
        baseline_xp=baseline_xp,
    )

    status = "ok"
    message = f"GW{solver_gameweek} solver refreshed ({recommended})"
    log_refresh("solver", status, message)
    logger.info("Solver job completed: %s", message)
