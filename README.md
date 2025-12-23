# Fantasy Premier League (FPL) Report Generator

Generate a detailed LaTeX/PDF report for a Fantasy Premier League team: squad breakdown, performance trends, transfer recommendations, and optional competitive/league-aware analysis.

## Quick Start

1. Activate the virtual environment:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `.\venv\Scripts\activate`

2. Install dependencies: `python -m pip install -r requirements.txt`

3. Generate a report:
   ```bash
   ./run_report.sh <TEAM_ID> [GAMEWEEK]
   # Example: ./run_report.sh 847569 17
   ```
   
   Or use Python directly:
   ```bash
   python generate_fpl_report.py --team <TEAM_ID> --gw 17
   ```

Outputs: `reports/report_<TEAM_ID>.tex` and `reports/report_<TEAM_ID>.pdf`

## Requirements

- Python 3
- `zsh` (for `./run_report.sh`)
- Internet access (fetches live data from the official FPL API)
- `pdflatex` (TeX Live / MacTeX) for PDF compilation

## Project Structure

```
Fantasy-Premier-League/
├── generate_fpl_report.py    # Report generation CLI (main entry point)
├── run_report.sh             # Shell wrapper for report generation
├── main.py                   # Full pipeline orchestrator
├── config.yml                # Configuration file
│
├── scraping/                 # FPL API and external data fetchers
│   ├── fpl_api.py           # Core FPL API functions
│   ├── understat.py         # xG/xA data from Understat
│   ├── fbref.py             # Detailed stats from FBref
│   ├── global_scraper.py    # Main scraping orchestrator
│   └── ...
│
├── processing/               # Data cleaning, parsing, merging
│   ├── cleaners.py          # Data cleaning utilities
│   ├── parsers.py           # Data parsing utilities
│   ├── mergers.py           # Data merging utilities
│   ├── global_merger.py     # Multi-season merger
│   └── ...
│
├── utils/                    # General utilities
│   ├── utility.py           # Helper functions
│   ├── gameweek.py          # Gameweek utilities
│   └── ...
│
├── etl/                      # Modern ETL pipeline
│   ├── fetchers.py          # Data fetchers
│   ├── transformers.py      # Data transformers
│   └── pipeline.py          # Pipeline orchestration
│
├── models/                   # ML model infrastructure
│   ├── train.py             # Model training
│   ├── inference.py         # Model inference
│   └── feature_engineering.py
│
├── solver/                   # MIP optimization
│   ├── optimizer.py         # Transfer MIP solver
│   └── ...
│
├── reports/                  # Report generation
│   ├── fpl_report/          # Core report modules
│   ├── analytics.py         # ROI analysis
│   └── strategy_reporter.py # Strategy reporting
│
└── tests/                    # Unit tests
```

## Architecture

### End-to-End Pipeline

```
FPL API → FPLDataFetcher → PlayerAnalyzer + FPLPointsPredictor
  → TransferRecommender + TransferStrategyPlanner
  → PlotGenerator → LaTeXReportGenerator → pdflatex
```

1. `./run_report.sh` runs `python generate_fpl_report.py --no-pdf` to generate LaTeX and plots
2. Then runs `pdflatex` twice to compile the PDF

### Data Layer (`reports/fpl_report/data_fetcher.py`, `scraping/fpl_api.py`)

Live FPL API data with optional local CSV fallbacks:

- `GET /api/bootstrap-static/`: player master data, teams, events
- `GET /api/element-summary/<player_id>/`: per-player GW history
- `GET /api/entry/<team_id>/history/`: entry season history and chips
- `GET /api/entry/<team_id>/event/<gw>/picks/`: squad picks for a GW
- `GET /api/entry/<team_id>/transfers/`: transfer log
- `GET /api/fixtures/`: fixture list with FDR

### Caching (`reports/fpl_report/cache_manager.py`)

API responses cached as pickled objects with TTLs:
- Control via CLI: `--no-cache`, `--clear-cache`, `--cache-stats`

### Player Analysis (`reports/fpl_report/player_analyzer.py`)

- Form trend analysis with linear regression
- ICT decomposition (influence/creativity/threat)
- xGI vs actual returns comparison
- Peer percentile rankings

### Points Prediction (`reports/fpl_report/predictor.py`)

Ensemble model (GradientBoosting + RandomForest) predicting player points:
- Rolling form features (4 GW window)
- Fixture difficulty context
- Team strength indicators

### Transfer Recommendations (`reports/fpl_report/transfer_recommender.py`)

Weighted scoring for replacements:
```
score = 0.20*form + 0.20*fixtures + 0.10*expected_points
      + 0.30*predicted_points + 0.10*ownership_diff + 0.10*value
```

### Multi-Week Strategy (`reports/fpl_report/transfer_strategy.py`)

- 5-GW expected points projections
- Fixture swing detection
- Transfer sequencing with hit accounting
- Wildcard and Free Hit draft optimization

## CLI Reference

```bash
python generate_fpl_report.py --help
```

**Common options:**
- `--team <ID>`: Your FPL team ID (required)
- `--gw <N>`: Gameweek to analyze
- `--no-pdf`: Generate LaTeX only
- `--verbose`: Detailed output

**Caching:**
- `--no-cache`: Bypass cache
- `--clear-cache`: Clear cached data
- `--cache-stats`: Show cache statistics

**Competitive analysis:**
- `--compare <id...>`: Compare against other teams
- `--no-competitive`: Skip competitive section

**Free Hit:**
- `--league-id <ID>`: League for ownership analysis
- `--free-hit-strategy safe|balanced|aggressive`

## Development

**Run tests:**
```bash
python -m unittest discover -s tests -v
```

**Refresh season data:**
```bash
python -m scraping.global_scraper
```

**Merge multi-season data:**
```bash
python -m processing.global_merger
```

**Full pipeline with model training:**
```bash
python main.py --update-data --train-models --team <TEAM_ID>
```

## Import Patterns

```python
# Scraping / FPL API
from scraping.fpl_api import get_data, get_fixtures_data

# Data processing
from processing.cleaners import clean_players
from processing.parsers import parse_players

# Utilities
from utils.utility import uprint
from utils.gameweek import get_recent_gameweek_id

# Report modules
from reports.fpl_report.data_fetcher import FPLDataFetcher
from reports.fpl_report.player_analyzer import PlayerAnalyzer
```

## Data Files

Data files are not tracked in git. To populate the `data/` directory:
1. Run `python -m scraping.global_scraper` to fetch current season data
2. Or download historical data from the original dataset

### Dataset Structure

```
data/<season>/
├── cleaned_players.csv      # Season overview
├── fixtures.csv             # Season fixtures
├── teams.csv                # Team information
├── gws/
│   ├── gw<N>.csv           # Per-gameweek data
│   └── merged_gw.csv       # All GWs merged
└── players/
    └── <player_name>/
        ├── gw.csv          # Player GW history
        └── history.csv     # Prior seasons
```

## License

See [LICENSE](LICENSE) for details.
