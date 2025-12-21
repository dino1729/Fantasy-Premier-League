# FPL Core Insights Integration

## Overview

This integration adds support for the FPL Core Insights dataset ([github.com/olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights)), which provides enhanced FPL data that combines:

- Official FPL API data
- Detailed match statistics (Opta-like metrics)
- ClubElo team strength ratings
- Historical data across multiple seasons

**Data updates:** Twice daily at 5:00 AM and 5:00 PM UTC

## What Was Implemented

### 1. FPL Core Insights Fetcher (`etl/fetchers.py`)

New `FPLCoreInsightsFetcher` class that downloads and caches CSV data from the FPL Core Insights repository.

**Key Features:**
- Downloads season-level and gameweek-specific datasets
- 6-hour cache duration (aligned with twice-daily updates)
- Automatic fallback to cached data if download fails
- Organized storage in `data/fpl_core/{season}/`

**Available Datasets:**

Season-level:
- `players.csv` - Player information (season aggregate)
- `playerstats.csv` - FPL API equivalent data (season aggregate)
- `teams.csv` - Team info with Elo ratings
- `gameweek_summaries.csv` - Summary stats per gameweek

Gameweek-level (stored in `gw{N}/` subdirectories):
- `fixtures.csv` - Fixtures for this gameweek
- `matches.csv` - Completed matches
- `player_gameweek_stats.csv` - Player stats for this gameweek
- `playermatchstats.csv` - Detailed match-level stats (64+ metrics including shots, xG, passes, tackles, saves)
- `players.csv` - Player info snapshot
- `playerstats.csv` - FPL API data snapshot
- `teams.csv` - Team info snapshot

### 2. Integration with Report Generator

**Modified:** `reports/generate_fpl_report.py`

The report generator now:
1. Fetches all season-level data
2. Fetches **ALL gameweek data from GW1 to current GW** (for historical analysis)
3. Stores data in `data/fpl_core/2025-2026/` with organized subdirectories
4. Provides convenient access to current gameweek data

### 3. ClubElo API Replacement

**Modified:** `etl/fetchers.py` and `reports/fpl_report/data_fetcher.py`

- ClubElo API calls are now disabled (servers are down)
- Elo ratings are sourced from FPL Core Insights `teams.csv` instead
- Fallback ratings provided if both sources fail

### 4. Caching Configuration

**Modified:** `.gitignore`

Added `data/fpl_core/` to ignore list to prevent committing cached CSV files.

## Usage

### Basic Usage (Automatic)

When you run the report generator, FPL Core data is automatically fetched:

```bash
./reports/run_report.sh 847569 16
```

This will:
- Download all gameweek data from GW1 to GW16
- Cache data in `data/fpl_core/2025-2026/`
- Use cached data if less than 6 hours old

### Programmatic Usage

```python
from etl.fetchers import FPLCoreInsightsFetcher

# Initialize fetcher
fetcher = FPLCoreInsightsFetcher(season="2025-2026")

# Fetch season-level data
season_data = fetcher.fetch_all()
teams_df = season_data['teams']  # Includes Elo ratings

# Fetch specific gameweek
gw16_data = fetcher.fetch_all(gameweek=16)
player_stats = gw16_data['playerstats']
match_stats = gw16_data['playermatchstats']  # Detailed metrics

# Fetch all gameweeks (GW1-GW16)
all_gw_data = fetcher.fetch_all_gameweeks(up_to_gw=16)
gw5_match_stats = all_gw_data[5]['playermatchstats']

# Get specific datasets
teams = fetcher.get_teams()  # Season-level
gw16_players = fetcher.get_players(gameweek=16)  # GW-specific

# Check cache status
cache_info = fetcher.get_cache_info()  # Season-level
gw_cache = fetcher.get_cache_info(gameweek=16)  # Gameweek-specific
```

## Data Storage Structure

```
data/fpl_core/2025-2026/
├── players.csv                    # Season aggregate
├── playerstats.csv                # Season aggregate
├── teams.csv                      # Includes Elo ratings
├── gameweek_summaries.csv         # Summary per GW
├── gw1/
│   ├── fixtures.csv
│   ├── matches.csv
│   ├── player_gameweek_stats.csv
│   ├── playermatchstats.csv       # 64+ detailed metrics
│   ├── players.csv
│   ├── playerstats.csv
│   └── teams.csv
├── gw2/
│   └── ... (same structure)
└── gw16/
    └── ... (same structure)
```

## Key Metrics Available

The `playermatchstats.csv` files provide rich analytics including:

**Attacking:**
- `total_shots`, `shots_on_target`, `xg` (expected goals)
- `big_chances_created`, `big_chances_scored`
- `key_passes`, `final_third_passes`

**Passing:**
- `accurate_passes`, `accurate_passes_percent`
- `accurate_long_balls`, `accurate_crosses`

**Defending:**
- `tackles_won`, `interceptions`, `clearances`
- `duels_won`, `recoveries`, `blocks`

**Goalkeeping:**
- `saves`, `saves_inside_box`, `high_claim`
- `goals_prevented`, `sweeper_actions`
- `gk_accurate_passes`, `gk_accurate_long_balls`

## Benefits

1. **Richer Analytics:** 64+ detailed metrics per player/match vs FPL API's basic stats
2. **Historical Data:** Complete gameweek history for trend analysis
3. **Elo Ratings:** Dynamic team strength ratings for better fixture analysis
4. **Match-Level Granularity:** Analyze performance in specific matches
5. **Reliable Updates:** Twice-daily updates from stable GitHub repository

## Next Steps

Now that you have this data available, you can enhance your reports with:

1. **Deeper Player Analysis:**
   - Shot quality metrics (xG per shot)
   - Passing accuracy and creativity metrics
   - Defensive contribution analysis

2. **Historical Trend Analysis:**
   - Form over last N gameweeks using match stats
   - Home vs away performance splits
   - Performance against different quality opponents (using Elo)

3. **Advanced Team Analysis:**
   - Team attacking/defensive patterns
   - Player roles within team tactics
   - Team chemistry and partnerships

4. **Predictive Features:**
   - Use historical match stats for better xP predictions
   - Fixture difficulty based on opponent Elo + team's historical performance
   - Form metrics based on underlying stats (xG, shots, passes)

## Cache Management

Cache is automatically managed:
- **Duration:** 6 hours (aligned with twice-daily updates)
- **Location:** `data/fpl_core/2025-2026/`
- **Fallback:** Uses stale cache if download fails
- **Cleanup:** Run with `force_refresh=True` to invalidate cache

```python
# Force fresh download
fetcher.fetch_all(force_refresh=True)
fetcher.fetch_all_gameweeks(up_to_gw=16, force_refresh=True)
```

## Performance

- **Season-level fetch:** ~1 second (4 CSVs)
- **Single gameweek fetch:** ~1-2 seconds (7 CSVs)
- **All gameweeks (GW1-GW16):** ~20-30 seconds on first run, instant when cached

## Support

FPL Core Insights repository: https://github.com/olbauday/FPL-Core-Insights

For issues or questions about the integration, check:
1. Cache status with `fetcher.get_cache_info()`
2. Network connectivity to GitHub
3. Repository structure hasn't changed

