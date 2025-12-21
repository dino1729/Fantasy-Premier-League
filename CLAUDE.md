# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fantasy Premier League (FPL) analytics system that generates professional LaTeX/PDF reports with squad analysis, transfer recommendations, and multi-week strategy optimization. The architecture follows a three-layer approach: Data Engineering → Predictive Modeling → Prescriptive Analytics.

## Build, Test, and Development Commands

```bash
# Activate virtual environment (always do this first)
source venv/bin/activate

# If you want a command that always uses the repo venv (no activation required):
bash scripts/py -m unittest discover -s tests -v
bash scripts/pip install -r requirements.txt

# Install dependencies
python -m pip install -r requirements.txt

# Generate report (preferred entrypoint)
./reports/run_report.sh <TEAM_ID> [GAMEWEEK]
# Example: ./reports/run_report.sh 847569 16

# Generate LaTeX only (no PDF)
python reports/generate_fpl_report.py --team <TEAM_ID> --gw <N> --no-pdf

# Run unit tests
python -m unittest discover -s tests -v

# Run single test file
python -m unittest tests.test_report_enhancements -v

# Refresh current-season data from FPL API
python global_scraper.py

# Merge multi-season datasets
python global_merger.py

# Full pipeline with model training
python main.py --update-data --train-models --team <TEAM_ID>
```

## Architecture

### Layer 1: Data Engineering
- **FPL API integration**: `getters.py` (legacy) or `etl/fetchers.py` (modern)
- **Data consolidation**: `reports/fpl_report/data_fetcher.py` wraps API calls with enrichment
- **Caching**: `reports/fpl_report/cache_manager.py` - persistent caching with TTLs (bootstrap: 3600s, team_data: 300s)
- **Storage**: Parquet files in `data/parquet/`, CSVs in `data/<season>/`

### Layer 2: Predictive Modeling
- **Feature engineering**: `models/feature_engineering.py` - rolling form, ICT, xG/xA, team strength
- **Model training**: `models/train.py` - position-specific XGBoost + Ridge ensemble
- **Inference**: `reports/fpl_report/predictor.py` - GradientBoosting + RandomForest ensemble for xP predictions

### Layer 3: Prescriptive Analytics
- **MIP Solver**: `solver/optimizer.py` - TransferMIPSolver using sasoptpy + HiGHS for mathematically optimal transfers
- **Transfer recommendations**: `reports/fpl_report/transfer_recommender.py` - weighted scoring (form/fixtures/xP/ownership/value)
- **Multi-week strategy**: `reports/fpl_report/transfer_strategy.py` - 5-GW horizon planning with fixture swing detection
- **Report generation**: `reports/fpl_report/latex_generator.py` → pdflatex

### Data Flow
```
FPL API → FPLDataFetcher → PlayerAnalyzer + FPLPointsPredictor
  → TransferRecommender + TransferStrategyPlanner
  → PlotGenerator → LaTeXReportGenerator → pdflatex
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `reports/` | Report generation pipeline - primary focus of development |
| `reports/fpl_report/` | Core analysis modules (data_fetcher, predictor, transfer_recommender, etc.) |
| `etl/` | Modern ETL pipeline (fetchers, transformers, pipeline orchestration) |
| `models/` | ML model training infrastructure and artifacts |
| `solver/` | MIP optimizer using sasoptpy + HiGHS |
| `data/<season>/` | Historical + current season data (gws/, players/, CSVs) |
| `tests/` | Unit tests using unittest framework |

## Configuration

User settings are in `reports/config.yml`:
- `team_id`: Your FPL team ID
- `competitors`: List of team IDs for mini-league comparison
- `gameweek`: Target GW (null = auto-detect)
- `free_hit`/`wildcard`: Chip strategy settings (safe/balanced/aggressive)
- `mip_solver`: Solver settings (enabled, time_limit, candidate_pool)

CLI args override config values.

## Code Conventions

- Python style: 4-space indentation, PEP8-ish, minimal side effects outside entrypoints
- Naming: `snake_case` for files/functions, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants
- Position codes: `GKP`, `DEF`, `MID`, `FWD` (API uses numeric 1, 2, 3, 4)
- Season folder naming: `YYYY-YY` (e.g., `2025-26`)

## Data Contracts

- FPL API schema is stable; local CSVs are optional fallbacks
- All `cleaned_players.csv` exports must include expected schema columns
- Changing CSV column names/types requires updating generating scripts AND downstream consumers in `reports/fpl_report/`

## Cache Management

Cache stored in `reports/cache/`. Control via CLI:
- `--no-cache`: Bypass cache
- `--clear-cache`: Clear all cached data
- `--cache-stats`: Show cache statistics

## Testing Guidelines

- Use unittest framework with mocks - avoid live HTTP calls in tests
- Name new tests `tests/test_<area>.py`
- Keep fixtures in-repo (no large binary assets)

## Commit Style

Match existing history:
- Short, imperative subjects: "Add GW9 data", "Fix transfer calculation"
- Include `(#PR)` when referencing PRs
- Describe season/GW affected and data source for data updates
