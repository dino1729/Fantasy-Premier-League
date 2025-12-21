# FPL Report Caching System

## Overview

The FPL report generator now includes an intelligent caching system that dramatically speeds up report generation during development and iteration. The cache stores API responses and computed data with configurable time-to-live (TTL) values.

## Performance Improvements

- **First run (no cache):** ~85 seconds
- **Subsequent runs (with cache):** ~34 seconds
- **Speed improvement:** 60% faster

## How It Works

The cache system automatically stores:
- **Bootstrap data** (1 hour TTL) - Player stats, teams, fixtures
- **Team data** (5 minutes TTL) - Your team's history and info
- **Player history** (1 hour TTL) - Individual player gameweek stats
- **GW picks** (5 minutes TTL) - Squad selections per gameweek
- **Competitive data** (5 minutes TTL) - Multi-team comparison data

Cache entries automatically expire based on their TTL, ensuring you get fresh data when needed while still benefiting from caching during rapid iteration.

## CLI Commands

### View Cache Statistics
```bash
python3 generate_fpl_report.py --cache-stats
```

Shows:
- Cache status (enabled/disabled)
- Total number of cached entries
- Breakdown by cache type

### Clear Cache
```bash
python3 generate_fpl_report.py --clear-cache --team 847569
```

Clears all cached data before generating a new report.

### Disable Cache
```bash
python3 generate_fpl_report.py --no-cache --team 847569
```

Runs report generation without using or updating the cache (fetches fresh data).

### Using with run_report.sh

The shell script automatically uses caching. To disable:
```bash
# Edit run_report.sh and add --no-cache flag to the Python command
# Or run Python script directly with --no-cache
```

## Cache Location

Cache files are stored in:
```
reports/cache/
```

This directory contains:
- `*.pkl` files - Pickled Python objects (cached data)
- `cache_metadata.json` - Metadata about cache entries (timestamps, types)

## When to Clear Cache

Clear the cache when:
- You want to ensure you're working with the latest data
- A gameweek has just finished
- You've made changes to data structures
- Testing cache functionality

## TTL Configuration

Default TTL values (in seconds):
- `bootstrap`: 3600 (1 hour)
- `fixtures`: 3600 (1 hour)
- `team_data`: 300 (5 minutes)
- `player_history`: 3600 (1 hour)
- `gw_picks`: 300 (5 minutes)
- `competitive`: 300 (5 minutes)
- `predictions`: 1800 (30 minutes)
- `default`: 600 (10 minutes)

To modify TTL values, edit `reports/fpl_report/cache_manager.py`:
```python
DEFAULT_TTL = {
    'bootstrap': 3600,  # Change this value
    # ...
}
```

## Development Workflow

### Recommended workflow for rapid iteration:

1. **First run** - Generate report with cache enabled:
   ```bash
   ./run_report.sh 847569
   ```

2. **Subsequent runs** - Cache automatically speeds up generation:
   ```bash
   ./run_report.sh 847569  # Much faster!
   ```

3. **When you need fresh data**:
   ```bash
   python3 generate_fpl_report.py --clear-cache --team 847569
   ```

4. **Check cache status**:
   ```bash
   python3 generate_fpl_report.py --cache-stats
   ```

## Technical Details

### Cache Key Generation

Cache keys are generated using MD5 hashes of:
- Cache type (e.g., 'team_data', 'player_history')
- Function arguments (e.g., team_id, player_id, gameweek)
- Keyword arguments

This ensures each unique combination of parameters gets its own cache entry.

### Automatic Expiration

When you request cached data:
1. System checks if cache entry exists
2. Checks if entry has expired based on TTL
3. If expired, removes old cache and fetches fresh data
4. If valid, returns cached data immediately

### Thread Safety

The current implementation is designed for single-threaded use. If running multiple report generations in parallel, consider disabling cache or implementing file locking.

## Troubleshooting

### Cache not working?

Check if cache is enabled:
```bash
python3 generate_fpl_report.py --cache-stats
```

### Stale data?

Clear the cache:
```bash
python3 generate_fpl_report.py --clear-cache --team 847569
```

### Cache directory issues?

The cache directory is created automatically. If you encounter permissions issues:
```bash
rm -rf reports/cache
mkdir reports/cache
```

## Future Enhancements

Potential improvements:
- Configurable cache directory via environment variable
- Per-cache-type TTL override via CLI
- Cache warming (pre-populate cache for multiple teams)
- Cache compression for large datasets
- Redis/memcached backend for distributed caching

