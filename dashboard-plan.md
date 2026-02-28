# FPL Analytics Dashboard - Implementation Plan

## Context

Currently, all analysis output goes through `generate_fpl_report.py` → LaTeX → PDF. This works but is static, slow to regenerate, and impossible to interact with. We're building a web dashboard that surfaces the same analytics (plus more) in an interactive, always-fresh UI - inspired by FPLCore.com but differentiated by our MIP solver, 40-feature ML predictions, Elo-based FDR, and xG/xA scatter analysis.

## Decisions

| Decision | Choice |
|----------|--------|
| Backend | FastAPI (Python) wrapping existing modules |
| Frontend | Vite + React 19 + TanStack Router |
| UI framework | shadcn/ui + Tailwind (dark only) |
| Charts | Visx (D3-based) |
| Tables | TanStack Table + TanStack Virtual |
| Storage | SQLite (pre-computed, single-user) |
| Scheduler | APScheduler in-process |
| Deployment | Single FastAPI process serving API + built SPA on :8000 |
| Auth | None (local network) |
| Photos | PL CDN (`resources.premierleague.com`) |

## Architecture

```
APScheduler (in-process)
  ├─ bootstrap_job (30min): FPL API + fixtures → SQLite
  ├─ analysis_job (2hr): player analysis + predictions + scatter data → SQLite
  ├─ solver_job (6hr): MIP solver 3 scenarios → SQLite (subprocess isolation)
  └─ league_job (1hr): competitive dataset + global managers → SQLite

FastAPI (:8000)
  ├─ /api/* → reads from SQLite, returns JSON
  └─ /* → serves React SPA static files (production)

React SPA
  ├─ TanStack Query → fetches /api/*
  ├─ TanStack Router → 7 routes
  └─ Visx + shadcn/ui → renders charts/tables
```

## Directory Structure

```
dashboard/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan, static mounting
│   ├── database.py           # SQLite schema, read/write helpers
│   ├── scheduler.py          # APScheduler setup + job registration
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── squad.py          # /api/squad, /api/squad/issues
│   │   ├── players.py        # /api/players (paginated, filtered)
│   │   ├── fixtures.py       # /api/fixtures/fdr-grid
│   │   ├── transfers.py      # /api/transfers/solver, /api/transfers/history
│   │   ├── league.py         # /api/league/competitors, /api/league/global
│   │   ├── scatter.py        # /api/scatter/{chart_type}
│   │   ├── manager.py        # /api/manager/overview, /api/manager/captains
│   │   └── meta.py           # /api/meta (GW, refresh times, team info)
│   └── jobs/
│       ├── __init__.py
│       ├── bootstrap_job.py  # FPL API + squad + fixtures + GW history
│       ├── analysis_job.py   # PlayerAnalyzer + predictions + scatter data
│       ├── solver_job.py     # MIP solver via subprocess
│       ├── league_job.py     # build_competitive_dataset + top global
│       └── _solver_worker.py # standalone script: runs solver, prints JSON
├── frontend/
│   ├── package.json
│   ├── vite.config.ts        # proxy /api → :8000 in dev
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── router.tsx
│       ├── lib/
│       │   ├── api.ts        # fetch helpers, types
│       │   ├── query.ts      # TanStack Query client
│       │   └── utils.ts      # cn(), formatters, FDR colors
│       ├── hooks/             # one per API domain: useSquad, usePlayers, etc.
│       ├── components/
│       │   ├── layout/        # AppShell, Sidebar, Header
│       │   ├── ui/            # shadcn/ui generated (Button, Card, Tabs, etc.)
│       │   ├── charts/        # Visx wrappers: LineChart, BarChart, ScatterPlot, FDRGrid, Treemap
│       │   └── shared/        # PlayerPhoto, PositionBadge, DifficultyPip, StatCard
│       └── routes/
│           ├── __root.tsx     # AppShell layout
│           ├── index.tsx      # Dashboard (squad grid + GW stats + fixtures)
│           ├── players.tsx    # Player Statistics table
│           ├── fixtures.tsx   # FDR grid
│           ├── transfers.tsx  # MIP solver results
│           ├── league.tsx     # Competitor comparison
│           ├── scatter.tsx    # xG/xA scatter analysis
│           └── manager.tsx    # Season overview + captain analysis
├── requirements-dashboard.txt
└── run_dashboard.sh           # activate venv, build frontend, start uvicorn
```

## SQLite Schema

File: `dashboard/fpl_dashboard.db`

```sql
-- Job tracking
CREATE TABLE refresh_log (
    job_name TEXT PRIMARY KEY,
    last_run_at TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'ok' | 'error'
    message TEXT
);

-- Singleton blobs (id=1 constraint)
CREATE TABLE bootstrap_cache (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT, refreshed_at TEXT);
CREATE TABLE squad_data (id INTEGER PRIMARY KEY CHECK(id=1), gameweek INTEGER, data TEXT, refreshed_at TEXT);
CREATE TABLE gw_history (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT, refreshed_at TEXT);
CREATE TABLE fixtures (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT, refreshed_at TEXT);
CREATE TABLE solver_results (id INTEGER PRIMARY KEY CHECK(id=1), gameweek INTEGER,
    conservative TEXT, balanced TEXT, aggressive TEXT,
    recommended TEXT, baseline_xp REAL, refreshed_at TEXT);
CREATE TABLE competitive_data (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT, refreshed_at TEXT);

-- Per-player flat columns (enables SQL filtering without loading all into Python)
CREATE TABLE player_analysis (
    player_id INTEGER PRIMARY KEY,
    web_name TEXT, position TEXT, team TEXT,
    price REAL, form REAL, total_points INTEGER, minutes INTEGER,
    goals INTEGER, assists INTEGER, clean_sheets INTEGER, bps INTEGER,
    xg REAL, xa REAL, xg_diff REAL, xa_diff REAL,
    influence REAL, creativity REAL, threat REAL, ict_index REAL,
    xp_gw1 REAL, xp_gw2 REAL, xp_gw3 REAL, xp_gw4 REAL, xp_gw5 REAL,
    xp_confidence REAL,
    pct_form REAL, pct_ict REAL, pct_xg REAL, pct_xp REAL,
    transfers_in_event INTEGER, transfers_out_event INTEGER, selected_by_percent REAL,
    form_trend TEXT, ict_breakdown TEXT, raw_stats TEXT,  -- JSON blobs
    refreshed_at TEXT
);

-- Pre-computed scatter coordinates
CREATE TABLE scatter_data (
    chart_type TEXT PRIMARY KEY,  -- xg_goals | xa_assists | usage_output | defensive
    data TEXT, refreshed_at TEXT
);
```

## API Endpoints

| Method | Path | Source Job | Description |
|--------|------|-----------|-------------|
| GET | /api/meta | all | Current GW, season, team info, last refresh per job |
| GET | /api/squad | bootstrap | Squad 15 players, captain/VC, prices, stats |
| GET | /api/squad/issues | bootstrap | Injuries, suspension risk, price drops |
| GET | /api/players | analysis | Paginated/filtered player stats (30+ columns) |
| GET | /api/players/:id | analysis | Single player full detail |
| GET | /api/fixtures/fdr-grid | bootstrap | 20-team x N-GW FDR grid + BGW/DGW + Elo probs |
| GET | /api/transfers/solver | solver | 3 scenarios: weekly plans, transfers, captains, xP |
| GET | /api/transfers/history | bootstrap | Historical transfer list |
| GET | /api/league/competitors | league | Competitor entries with full GW data |
| GET | /api/league/global | league | Top N global managers |
| GET | /api/scatter/:type | analysis | Scatter point data (xg_goals, xa_assists, usage_output, defensive) |
| GET | /api/manager/overview | bootstrap | Season summary, rank trajectory, value history |
| GET | /api/manager/captains | bootstrap | Per-GW captain picks + optimal counterfactual |

## Pages

### 1. Dashboard (`/`)
- **GW Stats Bar**: 5 StatCards (avg pts, highest, your GW pts, GW rank, overall rank)
- **Formation Grid**: 4 rows (GK/DEF/MID/FWD) + bench row. PlayerCard = photo + name + price + pts + captain badge + injury indicator
- **Squad Issues**: injuries/suspensions/price drops from `get_squad_issues()`
- **Mini Fixture Strip**: next 5 GW FDR pips per starting XI player

### 2. Player Statistics (`/players`)
- TanStack Table with 30+ columns in groups: Identity | FPL Stats | xG/xA | ICT + Percentiles | ML Predictions | Fixtures
- Column pinning (name col sticky left), virtual rows (600+ players)
- Filter bar: position toggle, price range, min minutes slider
- Column visibility menu persisted to localStorage
- Click row → expandable detail (form trend sparkline, ICT radar concept)

### 3. Fixture Planner (`/fixtures`)
- CSS Grid: 20 teams x 8 GWs, cells colored by Elo FDR (1-5)
- BGW = gray overlay, DGW = gold border
- Hover tooltip: opponent, H/A, win/draw/loss probability bars
- Sortable by avg difficulty

### 4. Transfer Hub (`/transfers`)
- 3-tab layout (Conservative | Balanced | Aggressive) via shadcn Tabs
- Per tab: summary cards (xP gain, hits taken, FTs used) + weekly plan timeline
- WeeklyPlanCard: GW number, transfers in/out with player photos, captain pick, formation, xP
- Bottom section: underperformer list with severity flags + baseline xP comparison bar chart (Visx)

### 5. League Comparison (`/league`)
- Toggle: Competitors | Top Global
- Rank progression line chart (Visx) - all entries on one chart, your line highlighted
- Points progression (cumulative) line chart
- Transfer diff table: side-by-side GW transfers
- Chip timeline: horizontal markers showing when each manager used chips
- Player contribution treemap (Visx)

### 6. Scatter Analysis (`/scatter`)
- 4-button selector: xG vs Goals | xA vs Assists | Usage/Output | Defensive Value
- Visx ScatterPlot: quadrant lines at medians, quadrant labels ("Clinical", "Wasteful", etc.)
- Points colored by position, sized by minutes
- Hover tooltip: player photo, name, team, both axis values
- Position legend

### 7. Manager Report (`/manager`)
- Season overview: 4 stat cards + rank journey area chart (Visx, inverted Y)
- Points per GW: bar chart with green (above avg) / red (below avg) coloring
- Team value area chart over season
- Captain analysis table: per-GW captain, points earned, was_optimal flag, optimal_player if different
- Captain accuracy big number + total missed points

## Scheduler Jobs

### bootstrap_job (30min)
Calls: `FPLDataFetcher(team_id, season, use_cache=False)` → `get_current_squad()`, `get_gw_history()`, `get_team_info()`, `get_upcoming_fixtures()`, `get_squad_issues()`, `get_transfers()`, `get_bgw_dgw_gameweeks()`, `get_chips_used()`
Writes: `bootstrap_cache`, `squad_data`, `gw_history`, `fixtures`

### analysis_job (2hr)
Calls: `FPLDataFetcher` + `FPLCoreInsightsFetcher` + `PlayerAnalyzer` + `FPLCorePredictor`
For each of ~200 top players: `generate_player_summary()`, `predict_multiple_gws()`
Extracts scatter data (xG vs goals, xA vs assists, etc.) from same DataFrames PlotGenerator uses
Writes: `player_analysis` (DELETE + INSERT all rows), `scatter_data`

### solver_job (6hr)
Runs in subprocess (`_solver_worker.py`) to isolate memory/CPU.
Worker mirrors `generate_fpl_report.py` wiring:
  `FPLDataFetcher` → `FPLCoreInsightsFetcher` → `TransferRecommender` → `TransferStrategyPlanner`
Runs `generate_strategy()` which internally calls `MultiPeriodMIPSolver` for 3 scenarios.
Serializes `MultiPeriodResult` dataclass to JSON, prints to stdout.
Parent reads stdout, writes to `solver_results`.

### league_job (1hr)
Calls: `build_competitive_dataset(competitor_ids)` + `get_top_global_teams(n)`
Writes: `competitive_data`

**Startup behavior**: All 4 jobs run immediately on startup. bootstrap + analysis block (must complete before first request). solver + league run in background (pages show "Computing..." until ready).

## Existing Code Reuse

| Dashboard Need | Existing Function | File |
|---------------|-------------------|------|
| Squad data | `FPLDataFetcher.get_current_squad()` | `reports/fpl_report/data_fetcher.py` |
| GW history | `FPLDataFetcher.get_gw_history()` | same |
| Team info | `FPLDataFetcher.get_team_info()` | same |
| Upcoming fixtures | `FPLDataFetcher.get_upcoming_fixtures()` | same |
| Squad issues | `FPLDataFetcher.get_squad_issues()` | same |
| Transfers | `FPLDataFetcher.get_transfers()` | same |
| BGW/DGW | `FPLDataFetcher.get_bgw_dgw_gameweeks()` | same |
| Chips used | `FPLDataFetcher.get_chips_used()` | same |
| Free transfers | `FPLDataFetcher.calculate_free_transfers()` | same |
| Competitive data | `build_competitive_dataset()` | same |
| Top global | `get_top_global_teams()` | same |
| Form analysis | `PlayerAnalyzer.analyze_form_trend()` | `reports/fpl_report/player_analyzer.py` |
| ICT breakdown | `PlayerAnalyzer.analyze_ict_breakdown()` | same |
| xG vs actual | `PlayerAnalyzer.calculate_expected_vs_actual()` | same |
| Peer percentiles | `PlayerAnalyzer.compare_to_peers()` | same |
| Player summary | `PlayerAnalyzer.generate_player_summary()` | same |
| ML predictions | `FPLCorePredictor.predict_multiple_gws()` | `reports/fpl_report/predictor.py` |
| Underperformers | `TransferRecommender.identify_underperformers()` | `reports/fpl_report/transfer_recommender.py` |
| Transfer strategy | `TransferStrategyPlanner.generate_strategy()` | `reports/fpl_report/transfer_strategy.py` |
| MIP solver | `MultiPeriodMIPSolver` | `solver/optimizer.py` |
| Solver types | `MIPSolverResult`, `WeeklyPlan`, `TransferAction` | `solver/definitions.py` |
| Config | `utils.config` (all UPPERCASE constants) | `utils/config.py` |
| Module wiring pattern | `main()` function | `generate_fpl_report.py` (lines 230-400) |

## Implementation Sprints

### Sprint 1: Backend Foundation (3 files)
- `dashboard/backend/database.py` - SQLite schema + CRUD
- `dashboard/backend/jobs/bootstrap_job.py` - first job: squad + fixtures + history
- `dashboard/backend/main.py` - FastAPI with lifespan, scheduler, `/api/squad` + `/api/meta`
- Verify: `curl localhost:8000/api/squad` returns JSON

### Sprint 2: Frontend Scaffold (React shell)
- `npm create vite` + install TanStack Router/Query, shadcn/ui, Tailwind, Visx
- AppShell + Sidebar + Header + all 7 route stubs
- Dashboard page: `useSquad()` hook → formation grid with real data
- Verify: browser shows squad with player photos

### Sprint 3: Player Statistics Table
- `analysis_job.py` - player analysis + predictions → `player_analysis` table
- `/api/players` endpoint with SQL filtering
- TanStack Table + Virtual + column pinning + localStorage column visibility
- Verify: 600+ player table loads, filters by position, sorts, persists prefs

### Sprint 4: Fixture Planner
- `/api/fixtures/fdr-grid` endpoint
- CSS Grid FDR component with Elo-based colors + BGW/DGW markers
- Hover tooltips with win/draw/loss probabilities
- Verify: 20 teams x 8 GWs grid renders correctly

### Sprint 5: Transfer Hub (MIP Solver Display)
- `solver_job.py` + `_solver_worker.py` subprocess
- `/api/transfers/solver` + `/api/transfers/history`
- 3-tab scenario view + weekly plan timeline + underperformer list
- Verify: all 3 scenarios display, weekly plans show transfers

### Sprint 6: League Comparison
- `league_job.py`
- `/api/league/competitors` + `/api/league/global`
- Visx line charts (rank + points progression) + chip timeline
- Verify: competitor data renders with correct charts

### Sprint 7: Scatter Analysis + Manager Report
- Scatter data extraction in `analysis_job.py`
- `/api/scatter/:type` + `/api/manager/*`
- Visx ScatterPlot with quadrants + tooltips
- Manager report charts + captain analysis table
- Verify: all 4 scatter types render, captain accuracy shown

### Sprint 8: Polish + Production Build
- Loading skeletons for slow-loading data (solver)
- Data freshness indicator in header
- Error boundaries per route
- Responsive sidebar (Sheet on mobile)
- `run_dashboard.sh` script
- `npm run build` + static file serving via FastAPI

## Verification Plan

1. **Cold start test**: Delete `fpl_dashboard.db`, start server, confirm all 4 jobs run + DB populates
2. **API smoke test**: `curl` each of the 12 endpoints, verify JSON response shape
3. **Frontend E2E**: Navigate all 7 pages, confirm data renders (no loading spinners stuck)
4. **Table interaction**: Sort, filter, pin column, hide column, refresh page → prefs persist
5. **Solver display**: Confirm 3 scenarios render with correct transfer sequences
6. **Chart rendering**: Each Visx chart (6 total) renders with real data + tooltips work
7. **Responsive**: Sidebar collapses, formation grid reflows on narrow viewport
8. **Production build**: `npm run build` → FastAPI serves SPA at `/` → all pages work without Vite dev server

## FPLCore Reverse Engineering Reference

Full reverse engineering of https://www.fplcore.com/ is available in the brainstorming notes. Key takeaways:
- React 19 + Vite + TanStack Router + Supabase + Recharts + shadcn/ui
- 12 pages with deep interactivity (custom FDR ratings, in-browser ILP optimizer, multi-GW transfer planner)
- Our differentiators: MIP solver (vs manual planning), 40-feature ML predictions (vs simple xP), Elo-based FDR (vs FPL's crude 1-5), xG/xA scatter analysis
