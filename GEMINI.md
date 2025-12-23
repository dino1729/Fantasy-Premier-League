# Fantasy Premier League (FPL) Bot & Analysis Tool

## Project Overview
This project is a comprehensive toolkit for Fantasy Premier League (FPL) managers. It combines data engineering, machine learning, and mathematical optimization to provide data-driven transfer strategies and detailed team analysis reports.

**Key Capabilities:**
*   **Data Collection (ETL):** Scrapes live FPL API data and historical stats.
*   **Machine Learning (ML):** Predicts player points (xP) using position-specific ensembles (XGBoost + Ridge).
*   **Optimization (Solver):** Uses Mixed-Integer Programming (MIP) to find the mathematically optimal transfer strategy over a multi-gameweek horizon.
*   **Reporting:** Generates professional PDF reports with squad analysis, transfer recommendations, and visual deep dives.

## Architecture

The system is organized into four main modules managed by `main.py`:

1.  **ETL Pipeline (`etl/`, `scraping/`, `processing/`)**
    *   Fetches data from FPL API and external sources (Understat, ClubElo).
    *   Transforms raw data into clean CSVs and Parquet files in `data/`.
    *   Key modules: `scraping.global_scraper` (season data), `etl/pipeline.py`.

2.  **Machine Learning (`models/`)**
    *   **Training (`models/train.py`):** Trains custom models for GKP, DEF, MID, FWD.
    *   **Inference (`models/inference.py`):** Generates expected points (xP) for future gameweeks.
    *   **Model:** Weighted ensemble of XGBoost (0.7) and Ridge Regression (0.3), trained using TimeSeriesSplit to prevent leakage.

3.  **Optimization Solver (`solver/`)**
    *   **Engine:** Uses `sasoptpy` for modeling and `highspy` as the backend solver.
    *   **Logic (`solver/optimizer.py`):** Maximizes projected points over a horizon (default 5 weeks) subject to budget, squad quotas, and transfer limits.

4.  **Reporting (`reports/`)**
    *   **Generator (`generate_fpl_report.py`):** Detailed analysis logic (at project root).
    *   **Output:** LaTeX-based PDFs containing squad breakdown, transfer advice, and plots.
    *   **Wrapper:** `./run_report.sh` handles the full generation flow including PDF compilation.

## Key Files & Directories

*   `main.py`: **Primary Entry Point**. Orchestrates the full pipeline (Data -> Model -> Solver -> Report).
*   `generate_fpl_report.py`: Report generation CLI (at project root).
*   `run_report.sh`: Shell wrapper to generate PDF reports (requires `pdflatex`).
*   `scraping/global_scraper.py`: Updates the current season's dataset.
*   `processing/global_merger.py`: Merges multi-season data.
*   `models/train.py`: Script to retrain the point prediction models.
*   `solver/optimizer.py`: The core MIP solver implementation.
*   `data/`: Data warehouse (not tracked in git). Contains season folders and parquet files.
*   `requirements.txt`: Python dependencies.

## Usage Guide

### 1. Setup
Activate the virtual environment and install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Full Pipeline (Recommended)
Run the complete flow: update data, predict points, solve for strategy, and report.
```bash
python main.py --update-data --team <YOUR_TEAM_ID> --gameweek <N>
```

### 3. Quick Run (Cached Data)
If data is already fresh, skip the update to save time:
```bash
python main.py --team <YOUR_TEAM_ID> --gameweek <N>
```

### 4. Report Generation Only
To generate a visual PDF report without running the full solver:
```bash
./run_report.sh <YOUR_TEAM_ID> [GAMEWEEK]
```

### 5. Data Refresh
To update the current season's dataset:
```bash
python -m scraping.global_scraper
```

### 6. Training Models
To retrain the ML models with the latest data:
```bash
python main.py --train-models
```

## Development Conventions

*   **Code Style:** PEP 8.
*   **Data Management:**
    *   Raw data goes to `data/<season>/` (not tracked in git).
    *   Processed ML data goes to `data/parquet/`.
    *   **Crucial:** Do not change CSV column names without updating `processing/` and `reports/` parsers.
*   **Testing:** Run unit tests via `python -m unittest discover -s tests -v`.
*   **Solver:** If `sasoptpy`/`highspy` are missing, the system falls back to heuristic strategies.

## Import Patterns

```python
from scraping.fpl_api import get_data
from processing.parsers import parse_players
from utils.utility import uprint
from reports.fpl_report.data_fetcher import FPLDataFetcher
```
