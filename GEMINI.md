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

1.  **ETL Pipeline (`etl/`, root scrapers)**
    *   Fetches data from FPL API and external sources (Understat, ClubElo).
    *   Transforms raw data into clean CSVs and Parquet files in `data/`.
    *   Key scripts: `global_scraper.py` (season data), `etl/pipeline.py`.

2.  **Machine Learning (`models/`)**
    *   **Training (`models/train.py`):** Trains custom models for GKP, DEF, MID, FWD.
    *   **Inference (`models/inference.py`):** Generates expected points (xP) for future gameweeks.
    *   **Model:** Weighted ensemble of XGBoost (0.7) and Ridge Regression (0.3), trained using TimeSeriesSplit to prevent leakage.

3.  **Optimization Solver (`solver/`)**
    *   **Engine:** Uses `sasoptpy` for modeling and `highspy` as the backend solver.
    *   **Logic (`solver/optimizer.py`):** Maximizes projected points over a horizon (default 5 weeks) subject to budget, squad quotas, and transfer limits.

4.  **Reporting (`reports/`)**
    *   **Generator (`reports/generate_fpl_report.py`):** detailed analysis logic.
    *   **Output:** LaTeX-based PDFs containing squad breakdown, transfer advice, and plots.
    *   **Wrapper:** `reports/run_report.sh` handles the full generation flow including PDF compilation.

## Key Files & Directories

*   `main.py`: **Primary Entry Point**. Orchestrates the full pipeline (Data -> Model -> Solver -> Report).
*   `global_scraper.py`: Updates the current season's dataset in `data/`.
*   `reports/run_report.sh`: Helper script to generate PDF reports (requires `pdflatex`).
*   `models/train.py`: Script to retrain the point prediction models.
*   `solver/optimizer.py`: The core MIP solver implementation.
*   `data/`: Data warehouse. Contains season folders (e.g., `2024-25/`) and `parquet/` files for the ML pipeline.
*   `requirements.txt`: Python dependencies.

## Usage Guide

**Note:** As GW17 is currently ongoing, please only generate reports up to **GW16** during this testing phase.

### 1. Setup
Activate the virtual environment and install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Full Pipeline (Recommended)
Run the complete flow: update data, predict points, solve for strategy, and report.
```bash
# Example for GW16
python main.py --update-data --team <YOUR_TEAM_ID> --gameweek 16
```

### 3. Quick Run (Cached Data)
If data is already fresh, skip the update to save time:
```bash
# Example for GW16
python main.py --team <YOUR_TEAM_ID> --gameweek 16
```

### 4. Report Generation Only
To generate a visual PDF report without running the full solver:
```bash
# Example for GW16
./reports/run_report.sh <YOUR_TEAM_ID> 16
```

### 5. Training Models
To retrain the ML models with the latest data:
```bash
python main.py --train-models
```

## Development Conventions

*   **Code Style:** PEP 8.
*   **Data Management:**
    *   Raw data goes to `data/<season>/`.
    *   Processed ML data goes to `data/parquet/`.
    *   **Crucial:** Do not change CSV column names without updating `etl/` and `reports/` parsers.
*   **Testing:** Run unit tests via `python -m unittest discover -s tests -v`.
*   **Solver:** If `sasoptpy`/`highspy` are missing, the system falls back to heuristic strategies.
