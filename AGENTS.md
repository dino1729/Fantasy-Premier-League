# Repository Guidelines

## Project Structure & Module Organization

- Primary focus: generating FPL data analysis reports (LaTeX/PDF) under `reports/` (preferred entrypoint: `reports/run_report.sh`).
- Top-level `*.py` scripts handle scraping/parsing/merging (for example `global_scraper.py`, `global_merger.py`).
- `data/<season>/` contains the published dataset (for example `data/2025-26/`), including `gws/`, `players/`, and season-wide CSVs like `cleaned_players.csv`.
- `reports/` contains the report generator CLI (`reports/generate_fpl_report.py`) and its modules in `reports/fpl_report/`.
- `tests/` contains unit tests (currently focused on report generation).

## Architecture Overview (Report Pipeline)

- Entrypoint: `./reports/run_report.sh` → runs `python3 reports/generate_fpl_report.py --no-pdf`, then compiles PDF via `pdflatex`.
- Data layer: `reports/fpl_report/data_fetcher.py` wraps `getters.py` (FPL API) and enriches picks/history/fixtures; optional CSV fallbacks live under `data/<season>/`.
- Caching: `reports/fpl_report/cache_manager.py` stores pickled responses under `reports/cache/` with TTLs; debug via `--cache-stats` and reset with `--clear-cache`.
- Analysis: `reports/fpl_report/player_analyzer.py` (form regression, ICT, xGI vs returns, peer percentiles).
- Prediction: `reports/fpl_report/predictor.py` (`FPLPointsPredictor`) trains a time-split ensemble (Gradient Boosting + Random Forest) on rolling 4-GW features + fixture/team-strength context; exposes MAE/RMSE/R².
- Transfers: `reports/fpl_report/transfer_recommender.py` scores replacements via weighted components (form/fixtures/xGI+PPG/model xP/ownership/value) and `reports/fpl_report/transfer_strategy.py` builds 5-GW EV and transfer plans.
- Rendering: `reports/fpl_report/plot_generator.py` outputs plots to `reports/plots/`, and `reports/fpl_report/latex_generator.py` assembles sections into `report_<TEAM_ID>.tex`.

## Build, Test, and Development Commands

> **Important:** GW17 is ongoing. Please use **GW16** for all report generation and testing commands until further notice.
> TODO: Update to allow current GW once GW17 concludes.

- Always activate the repo’s virtualenv first to avoid reinstalling packages or creating duplicate envs: `source venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\activate` (Windows)
- If you don’t want to rely on activation, use the repo wrappers: `bash scripts/py ...` and `bash scripts/pip ...` (they prefer `./venv/bin/python` when present).
- Install deps: `python -m pip install -r requirements.txt`
- Refresh current-season data: `python global_scraper.py` (writes into `data/<season>/`)
- Merge multi-season data: `python global_merger.py`
- Preferred report entrypoint: `./reports/run_report.sh <TEAM_ID> 16` (uses `reports/generate_fpl_report.py`; requires `zsh`, internet, and `pdflatex` for PDF)
- (Alternative) Generate LaTeX only: `python reports/generate_fpl_report.py --team <TEAM_ID> --season 2025-26 --gw 16 --no-pdf` (see `--help`)
- Run unit tests: `python -m unittest discover -s tests -v`

## Coding Style & Naming Conventions

- Python style: 4-space indentation, PEP8-ish formatting, minimal side effects outside script entrypoints.
- Naming: `snake_case` for files/functions, `CapWords` for classes, constants in `UPPER_SNAKE_CASE`.
- Data contracts matter: avoid changing CSV column names/types without updating the generating script(s) and any downstream consumers (notably `reports/fpl_report/`).

## Testing Guidelines

- Tests use the standard library `unittest`; prefer small, deterministic unit tests with mocks (avoid live HTTP calls).
- Name new tests `tests/test_<area>.py` and keep fixtures in-repo (avoid large binary assets).

## Commit & Pull Request Guidelines

- Match existing history: short, imperative subjects (for example `Add GW9 data`, `Update README`, `Handle assistant managers`), optionally with `(#PR)` when applicable.
- PRs should describe the season/GW affected, data source/assumptions, and any schema changes; include a quick “how to reproduce” command when adding or updating scripts.

## Data & Generated Artifacts

- Keep season folder naming consistent (`YYYY-YY`) and avoid mixing generated outputs with source data unless the output is part of the published dataset.
- Do not commit local artifacts like `venv/`, `.pytest_cache/`, or `__pycache__/` (see `.gitignore`).
