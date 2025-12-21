# Fantasy Premier League (FPL) Report Generator

Generate a detailed LaTeX/PDF report for a Fantasy Premier League team: squad breakdown, performance trends, transfer recommendations, and optional competitive/league-aware analysis.

This repo also includes an embedded historical dataset under `data/` used by some scripts and downstream analyses.

## Quick Start (Recommended)

> **Important Note:** GW17 is currently ongoing. During this phase, please limit report generation to **GW16** to ensure data stability.
> TODO: Remove this restriction once GW17 is complete.

1. Activate the existing virtual environment (avoid reinstalling packages / duplicate envs):
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `.\venv\Scripts\activate`

2. Install Python dependencies: `python -m pip install -r requirements.txt`

3. Generate a report (preferred entrypoint): `./reports/run_report.sh <TEAM_ID> [GAMEWEEK]`
   - Example (use GW16): `./reports/run_report.sh 847569 16`
   - Help: `./reports/run_report.sh --help`
   - LaTeX only (no PDF): `python reports/generate_fpl_report.py --team <TEAM_ID> --gw 16 --no-pdf`

Outputs are written to `reports/report_<TEAM_ID>.tex` and `reports/report_<TEAM_ID>.pdf`.

## Requirements

- Python 3
- `zsh` (for `./reports/run_report.sh`)
- Internet access (fetches live data from the official FPL API)
- `pdflatex` (TeX Live / MacTeX) for PDF compilation

## Project Layout

- `reports/run_report.sh`: one-command report generation workflow
- `reports/generate_fpl_report.py`: report generator CLI (used by `run_report.sh`)
- `reports/fpl_report/`: analysis, plotting, caching, LaTeX generation modules
- `data/<season>/`: season datasets (CSV structure described below)
- `tests/`: unit tests (`unittest`)

## Architecture (Technical)

### End-to-End Pipeline

1. `./reports/run_report.sh` runs `python3 reports/generate_fpl_report.py --team <TEAM_ID> --no-pdf` to generate `reports/report_<TEAM_ID>.tex` and plots in `reports/plots/`.
2. The script then runs `pdflatex` twice to compile `reports/report_<TEAM_ID>.pdf` (cross-references/TOC).

At a high level:

`FPL API + local fixtures` → `FPLDataFetcher` → `PlayerAnalyzer` + `FPLPointsPredictor` → `TransferRecommender` + `TransferStrategyPlanner` + draft optimizers → `PlotGenerator` → `LaTeXReportGenerator.compile_report()` → `pdflatex`

### Data Layer (`reports/fpl_report/data_fetcher.py`, `getters.py`)

The report is driven primarily by live FPL API data (via `getters.py`), with optional local CSV fallbacks:

- `GET /api/bootstrap-static/`: player master data (`elements`), teams, events (GW metadata).
- `GET /api/element-summary/<player_id>/`: per-player GW history (targets for modelling + form trends).
- `GET /api/entry/<team_id>/history/`: entry season history (`current`) and chips.
- `GET /api/entry/<team_id>/event/<gw>/picks/`: squad picks for a GW (captaincy, multipliers, bench order).
- `GET /api/entry/<team_id>/transfers/`: transfer log (enriched for reporting).
- `GET /api/fixtures/`: fixture list (FDR + home/away).

Key derived datasets in `FPLDataFetcher`:

- `get_current_squad(gw)`: pick list + player stats snapshot.
- `get_season_history()`: “full season” structure with GW points/rank and the actual XI contributions (injects each player’s `event_points` from their history row for that GW).
- `get_upcoming_fixtures(team_id)`: FDR-aware next fixtures.

### Caching (`reports/fpl_report/cache_manager.py`, `reports/cache/`)

API responses and computed artifacts are cached as pickled objects with TTLs (seconds) to speed iteration:

- Examples: `bootstrap` (3600), `team_data` (300), `player_history` (3600), `gw_picks` (300), `competitive` (300).
- Control via CLI: `--no-cache`, `--clear-cache`, `--cache-stats`.

Because cached objects are pickles, upgrading Python/pandas can require clearing `reports/cache/`.

### Player Analysis (`reports/fpl_report/player_analyzer.py`)

Per-player deep dives compute:

- Form trend: linear regression slope over last `window` GWs (`scipy.stats.linregress`) + volatility (CV%).
- ICT decomposition: influence/creativity/threat shares + ranks.
- xGI vs returns: compares `expected_*` (from FPL API) to goals/assists and classifies over/underperformance.
- Peer percentiles: percentile ranks vs same-position players with `min_minutes`.

These outputs feed LaTeX “deep dive” sections and underperformer detection.

### Points Prediction Model (`reports/fpl_report/predictor.py`)

`FPLPointsPredictor` predicts player points for upcoming GWs and powers transfer EV/strategy:

- Models: ensemble mean of `GradientBoostingRegressor` and `RandomForestRegressor` (features standardized via `StandardScaler`).
- Training rows: for each `(player_id, gw)` with `gw >= 5`, target is `total_points` in GW `gw`, features computed from the prior 4 GWs.
- Split: time-ordered train/validation split to reduce leakage.

Feature families (see `FPLPointsPredictor.feature_cols`):

- Rolling form (last 4): avg points/minutes/ICT/bonus/BPS, xG/xA/xGI, clean sheets, saves.
- Context: fixture difficulty (`team_*_difficulty`), home/away.
- Team strength: derived from finished fixtures as goals scored/conceded per game for team and opponent.
- Momentum: short-window deltas for points and minutes.

Outputs:

- `predict(player_ids)`: next-GW expected points.
- `predict_multiple_gws(player_ids, n)`: per-GW vector + `cumulative` and a simple consistency-based `confidence`.
- Metrics: MAE/RMSE/R² available via `get_model_metrics()` and summarized in the report.

Note: The module includes scaffolding for Understat ingestion, but Understat data is not currently wired into feature generation.

### Transfer Suggestions (`reports/fpl_report/transfer_recommender.py`)

Pipeline:

1. `identify_underperformers()` flags players using heuristics (low form, falling trend, low minutes proxy, bottom peer quartile, “no minutes”).
2. `get_recommendations()` builds same-position candidate pools constrained by `bank + sell_price`, filters unavailable players (`status != 'a'` or `chance_of_playing_next_round < 75`), trains the predictor on a candidate set, then scores replacements.

Replacement scoring is a weighted sum over normalized components:

```text
score = 0.20*form + 0.20*fixtures + 0.10*expected_points
      + 0.30*predicted_points + 0.10*ownership_diff + 0.10*value
```

Where:

- `fixtures`: inverse mean FDR of next 3 fixtures.
- `expected_points`: function of xGI and PPG.
- `predicted_points`: model’s next-GW prediction (capped/normalized).
- `ownership_diff`: favors lower `selected_by_percent` for differential picks.
- `value`: points-per-million proxy within budget.

### Multi-Week Strategy (`reports/fpl_report/transfer_strategy.py`)

`TransferStrategyPlanner.generate_strategy()` composes:

- Multi-GW xP for the current squad via `predict_multiple_gws(..., 5)`.
- Fixture swing detection: compares early vs late FDR averages and identifies “improving/worsening” runs.
- Enhanced “immediate recommendations”: ranks swaps by 5-GW expected gain and assigns priority from expected gain + underperformance severity + swing alignment.
- Transfer sequencing: a simple heuristic schedule (free transfers first; optional “consider” items).
- Alternative strategies: conservative (1 move) vs aggressive (multiple moves with hit accounting).

The report surfaces both EV deltas and the model performance metrics used to generate them.

### Draft Optimizers (Wildcard + Free Hit)

`WildcardOptimizer` (in `reports/fpl_report/transfer_strategy.py`) builds a 15-player wildcard draft:

- Constraints: positional quotas (2/5/5/3), max 3 per club, budget cap, availability filter.
- Scoring: weighted blend of season-long indicators (PPG, total points, minutes reliability, xGI, form) plus a small bonus from 5-GW model xP if provided.
- Algorithm: greedy XI selection (default target 4-4-2, prioritized by position), then fills bench with cheapest valid players, then selects the best formation among standard FPL formations.

`FreeHitOptimizer` builds a 15-player free hit draft optimized for a single GW:

- Primary score: `ep_next` (FPL API expected points next round), with a form/PPG fallback if missing.
- League-aware differentials: optional differential bonus `diff_bonus * (1 - league_ownership)^2` based on sampled squads (from `compute_league_ownership()` in `reports/fpl_report/data_fetcher.py`).
- Strategy knob: `--free-hit-strategy safe|balanced|aggressive` controls the differential bonus magnitude (currently used) and declares a max differential count (currently not enforced).

### Rendering Pipeline (Plots → LaTeX → PDF)

Plots are generated by `PlotGenerator` (`reports/fpl_report/plot_generator.py`) into `reports/plots/` and then referenced from LaTeX:

- Standard: points per GW, contribution heatmap, contribution treemap, transfer matrix, hindsight fixture analysis.
- Competitive: points/rank progression and per-team treemaps.
- Free Hit: per-GW comparison plot for the next 5 GWs (built from model predictions for apples-to-apples comparison).

LaTeX sections are assembled by `LaTeXReportGenerator` (`reports/fpl_report/latex_generator.py`):

- Core sections: title page, season summary, GW performance charts, formation diagram, player deep dives.
- Strategy sections: transfer recommendations, multi-week strategy, chip strategy, insights.
- Optional sections: competitive analysis, wildcard draft, free hit draft.

## CLI Reference (Advanced)

The underlying CLI is `python reports/generate_fpl_report.py`:

- Required: `--team <TEAM_ID>`
- Common: `--gw <N>`, `--season <YYYY-YY>`, `--no-pdf`, `--output report.tex`, `--verbose`
- Caching: `--no-cache`, `--clear-cache`, `--cache-stats`
- Competitive: `--compare <id...>`, `--no-competitive`
- Free Hit: `--league-id <classic_league_id>`, `--league-sample <N>`, `--free-hit-gw <N>`, `--free-hit-strategy safe|balanced|aggressive`

## Development & Testing

- Run tests: `python -m unittest discover -s tests -v`
- Refresh current-season dataset (writes to `data/<season>/`): `python global_scraper.py`
- Merge multi-season data exports: `python global_merger.py`
- Download your team’s raw data snapshot: `python teams_scraper.py <team_id>`

If report generation fails due to missing scientific/plotting dependencies, install the report stack into the active `venv` (at minimum: `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `squarify`).

## Dataset Notes (Optional)

`data/<season>/` is structured as:

- `cleaned_players.csv`: season overview
- `gws/gw<number>.csv`: gameweek-level rows
- `gws/merged_gw.csv`: all GWs merged
- `players/<player_name>/gws.csv`: per-player GW history
- `players/<player_name>/history.csv`: prior-season history

### Citing the Dataset

```bibtex
@misc{anand2016fantasypremierleague,
  title = {{FPL Historical Dataset}},
  author = {Anand, Vaastav},
  year = {2022},
  howpublished = {Retrieved August 2022 from \\url{https://github.com/vaastav/Fantasy-Premier-League/}}
}
```
