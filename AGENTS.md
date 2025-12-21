# Repository Guidelines

## Project Structure & Module Organization

The codebase is organized into functional directories:

- **`scraping/`** - FPL API and external data source fetchers
  - `fpl_api.py` - Core FPL API functions
  - `understat.py` - xG/xA data from Understat
  - `fbref.py` - Detailed player statistics from FBref
  - `teams_scraper.py` - Team-specific data scraping
  - `top_managers.py` - Top managers data
  - `global_scraper.py` - Main scraping orchestrator

- **`processing/`** - Data cleaning, parsing, and merging
  - `cleaners.py` - Data cleaning utilities
  - `parsers.py` - Data parsing utilities
  - `mergers.py` - Data merging utilities
  - `collector.py` - GW data collection
  - `aggregated_points_goals.py` - Team-level aggregation
  - `global_merger.py` - Multi-season merger

- **`utils/`** - General utilities
  - `utility.py` - General helper functions
  - `schedule.py` - Schedule generation
  - `gameweek.py` - Gameweek helpers
  - `new_position_checker.py` - Position change calculator
  - `top_players.py` - Top players extraction

- **`etl/`** - Modern ETL pipeline
  - `fetchers.py` - Modern data fetchers
  - `transformers.py` - Data transformers
  - `pipeline.py` - Pipeline orchestration

- **`models/`** - ML model infrastructure
  - `feature_engineering.py` - Feature engineering
  - `train.py` - Model training
  - `inference.py` - Model inference

- **`solver/`** - MIP optimization
  - `optimizer.py` - Transfer MIP solver
  - `definitions.py` - Solver definitions
  - `interpreter.py` - Result interpretation

- **`reports/`** - Report generation
  - `fpl_report/` - Core report modules
  - `analytics.py` - ROI analysis
  - `strategy_reporter.py` - Strategy reporting

- **Root-level entry points:**
  - `generate_fpl_report.py` - Report generation CLI (primary)
  - `run_report.sh` - Shell wrapper for report generation
  - `main.py` - Full pipeline orchestrator
  - `global_scraper.py` - Data scraping entry point
  - `global_merger.py` - Data merging entry point
  - `config.yml` - Configuration file

## Architecture Overview (Report Pipeline)

- Entrypoint: `./run_report.sh` → runs `python3 generate_fpl_report.py --no-pdf`, then compiles PDF via `pdflatex`.
- Data layer: `reports/fpl_report/data_fetcher.py` wraps `scraping/fpl_api.py` (FPL API) and enriches picks/history/fixtures.
- Caching: `reports/fpl_report/cache_manager.py` stores pickled responses under `reports/cache/` with TTLs.
- Analysis: `reports/fpl_report/player_analyzer.py` (form regression, ICT, xGI vs returns, peer percentiles).
- Prediction: `reports/fpl_report/predictor.py` (`FPLPointsPredictor`) trains a time-split ensemble.
- Transfers: `reports/fpl_report/transfer_recommender.py` and `reports/fpl_report/transfer_strategy.py`.
- Rendering: `reports/fpl_report/plot_generator.py` outputs plots to `reports/plots/`, and `reports/fpl_report/latex_generator.py` assembles sections into `report_<TEAM_ID>.tex`.

## Build, Test, and Development Commands

> **Important:** GW17 is ongoing. Please use **GW16** for all report generation and testing commands until further notice.

- Always activate the repo's virtualenv first: `source venv/bin/activate` (macOS/Linux)
- If you don't want to rely on activation, use the repo wrappers: `bash scripts/py ...` and `bash scripts/pip ...`
- Install deps: `python -m pip install -r requirements.txt`
- Refresh current-season data: `python global_scraper.py` (writes into `data/<season>/`)
- Merge multi-season data: `python global_merger.py`
- Preferred report entrypoint: `./run_report.sh <TEAM_ID> 16` (requires `zsh`, internet, and `pdflatex` for PDF)
- (Alternative) Generate LaTeX only: `python generate_fpl_report.py --team <TEAM_ID> --season 2025-26 --gw 16 --no-pdf`
- Run unit tests: `python -m unittest discover -s tests -v`

## Coding Style & Naming Conventions

- Python style: 4-space indentation, PEP8-ish formatting, minimal side effects outside script entrypoints.
- Naming: `snake_case` for files/functions, `CapWords` for classes, constants in `UPPER_SNAKE_CASE`.
- Data contracts matter: avoid changing CSV column names/types without updating the generating script(s) and any downstream consumers.

## Import Patterns

Use the new import paths:
```python
from scraping.fpl_api import get_data
from processing.parsers import parse_players
from utils.utility import uprint
from reports.fpl_report.data_fetcher import FPLDataFetcher
```

Legacy imports work but show deprecation warnings:
```python
from getters import get_data  # Deprecated
```

## Testing Guidelines

- Tests use the standard library `unittest`; prefer small, deterministic unit tests with mocks (avoid live HTTP calls).
- Name new tests `tests/test_<area>.py` and keep fixtures in-repo (avoid large binary assets).

## Commit & Pull Request Guidelines

- Match existing history: short, imperative subjects (for example `Add GW9 data`, `Update README`), optionally with `(#PR)` when applicable.
- PRs should describe the season/GW affected, data source/assumptions, and any schema changes.

## Data & Generated Artifacts

- Keep season folder naming consistent (`YYYY-YY`) and avoid mixing generated outputs with source data unless the output is part of the published dataset.
- Do not commit local artifacts like `venv/`, `.pytest_cache/`, or `__pycache__/` (see `.gitignore`).
