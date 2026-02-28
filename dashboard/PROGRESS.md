# FPL Analytics Dashboard - Progress

## Status: Sprints 1-8 Complete (of 8)

Built on branch `feature/fpl-dashboard` in worktree `.worktrees/dashboard`.

---

## What's Done

### Sprint 1: Backend Foundation
All backend infrastructure is working with real FPL API data.

**Files created:**
- `dashboard/backend/database.py` - SQLite schema (10 tables), CRUD helpers, NaN-safe JSON serialization
- `dashboard/backend/main.py` - FastAPI app with lifespan, CORS, custom SafeJSONResponse, SPA static serving
- `dashboard/backend/scheduler.py` - APScheduler with 4 periodic jobs (30min/2hr/6hr/1hr)
- `dashboard/backend/jobs/bootstrap_job.py` - Fetches squad, fixtures, history, issues, FDR grid from FPL API
- `dashboard/backend/jobs/analysis_job.py` - Full implementation (Sprint 3)
- `dashboard/backend/jobs/solver_job.py` - Implemented (Sprint 5)
- `dashboard/backend/jobs/league_job.py` - Implemented (Sprint 6)
- `dashboard/backend/routers/` - 8 router files serving 13 API endpoints + /api/health

**API endpoints working:**
| Endpoint | Status | Data |
|----------|--------|------|
| GET /api/health | Working | Startup readiness + job status |
| GET /api/meta | Working | GW28, season, team info, teams list |
| GET /api/squad | Working | 15 players with stats, captain, prices |
| GET /api/squad/issues | Working | Injuries, yellows, price drops |
| GET /api/players | Working | 600+ players with 30+ stat columns |
| GET /api/players/:id | Working | Single player full detail |
| GET /api/fixtures/fdr-grid | Working | 20 teams x 8 GWs with Elo FDR |
| GET /api/transfers/solver | Working (503 while computing) | 3 scenario plans from solver job |
| GET /api/transfers/history | Working | 34 transfers with enriched names |
| GET /api/league/competitors | Working (503 while computing) | Competitor dataset with GW history/transfers/chips |
| GET /api/league/global | Working (503 while computing) | Top global manager comparison dataset |
| GET /api/scatter/:type | Working | 4 scatter chart datasets (xg_goals, xa_assists, usage_output, defensive) |
| GET /api/manager/overview | Working | GW history, value, bank, FTs, chips |
| GET /api/manager/captains | Working | Captain picks + optimal counterfactual + accuracy summary |

### Sprint 2: Frontend Scaffold
React SPA with routing, layout, and live dashboard page.

**Stack:** Vite + React 19 + TypeScript + Tailwind CSS v4 + shadcn/ui (zinc dark) + TanStack Router + TanStack Query

**Files created:**
- `dashboard/frontend/src/lib/api.ts` - API types (170+ lines) + fetch helpers
- `dashboard/frontend/src/lib/query.ts` - TanStack Query client config
- `dashboard/frontend/src/lib/fpl.ts` - FDR colors, position colors, photo URLs, formatters
- `dashboard/frontend/src/lib/utils.ts` - cn() utility
- `dashboard/frontend/src/hooks/use-api.ts` - 13 hooks matching all API endpoints
- `dashboard/frontend/src/router.tsx` - Code-based routing with 7 routes
- `dashboard/frontend/src/routes/root-layout.tsx` - Dark theme shell with sidebar + header
- `dashboard/frontend/src/routes/dashboard.tsx` - Formation grid, stat cards, issues panel
- `dashboard/frontend/src/routes/{players,fixtures,transfers,league,scatter,manager}.tsx` - Placeholder stubs
- `dashboard/frontend/src/components/layout/app-sidebar.tsx` - Icon nav with active states
- `dashboard/frontend/src/components/layout/app-header.tsx` - Team name, GW badge, freshness
- `dashboard/frontend/src/components/shared/player-card.tsx` - Photo, position badge, captain badge, price
- `dashboard/frontend/src/components/shared/stat-card.tsx` - Reusable stat display
- 11 shadcn/ui components in `src/components/ui/`

**Production build:** 356KB JS + 53KB CSS (gzipped: 112KB + 9KB)

### Sprint 3: Player Statistics Table + analysis_job
Full player analytics pipeline and interactive data table.

**Backend:**
- `dashboard/backend/jobs/analysis_job.py` - Processes all ~600 players from bootstrap API data, runs deep analysis (form trends, ICT breakdown, peer percentiles) on top ~250, trains FPLPointsPredictor for xP predictions (5 GW horizon), extracts scatter chart data (4 types)
- `dashboard/backend/main.py` - Analysis job runs in background thread after bootstrap completes (non-blocking; players page polls until data arrives)

**Frontend:**
- `dashboard/frontend/src/routes/players.tsx` - Full player stats table with 30+ columns across 7 groups (Identity, FPL Stats, xG/xA, ICT, Percentiles, Predictions, Ownership)
- TanStack Table + TanStack Virtual for 600+ row performance (renders only visible ~30 rows)
- Column pinning: player name sticky left with z-index layering
- Filter bar: position toggle (GKP/DEF/MID/FWD/All), price range slider, min minutes slider
- Column visibility dropdown menu persisted to localStorage
- Color-coded cells: xG/xA diffs (green/red), percentiles (4-tier), confidence badges
- Polling: auto-refetches every 10s while analysis_job hasn't populated data yet

**New shadcn components:** table, dropdown-menu, slider, toggle-group, toggle

**New npm deps:** @tanstack/react-table, @tanstack/react-virtual

**Production build:** 527KB JS + 63KB CSS (gzipped: 164KB + 11KB)

### Sprint 4: Fixture Planner - FDR Grid
Interactive fixtures grid is now fully wired to backend data.

**Frontend:**
- `dashboard/frontend/src/routes/fixtures.tsx` - Full 20-team x 8-GW fixture planner using `/api/fixtures/fdr-grid`
- Grid columns are fixed to the next 8 sequential gameweeks from `current_gameweek`
- Cells colorized by Elo FDR (1-5), with BGW gray cells and DGW gold borders
- Hover tooltips show win/draw/loss probability bars per fixture
- Avg FDR column + sort toggle (easiest/hardest) for fixture difficulty ranking
- Horizontal scrolling support for smaller screens via `ScrollArea`
- Handles warming-up state (`503`) and loading skeletons consistently with other pages

**Production build:** 555KB JS + 65KB CSS (gzipped: 171KB + 11KB)

### Sprint 5: Transfer Hub - MIP Solver Display
Solver backend and Transfer Hub UI are now implemented.

**Backend:**
- `dashboard/backend/jobs/_solver_worker.py` - standalone subprocess worker that builds squad analysis, runs transfer strategy generation, and emits JSON payload for 3 scenarios
- `dashboard/backend/jobs/solver_job.py` - executes worker, parses payload, writes `solver_results`, updates refresh log
- Worker now uses FPL Core predictor path only when full local Core cache is present; otherwise falls back to the existing predictor path to avoid heavy first-run downloads

**Frontend:**
- `dashboard/frontend/src/lib/api.ts` - strong types for solver scenarios, weekly plans, underperformers, and transfer history entries
- `dashboard/frontend/src/routes/transfers.tsx` - full Transfer Hub implementation:
  - 3 scenario tabs (Conservative / Balanced / Aggressive)
  - summary cards (net xP gain, expected xP, hits, FT usage)
  - weekly plan timeline cards (transfers in/out, captain, formation, xP)
  - underperformer severity panel
  - baseline comparison bar chart built with Visx
  - recent transfer history table
  - warming-up/error/loading states with polling support

**Production build:** 612KB JS + 68KB CSS (gzipped: 191KB + 11KB)

### Sprint 6: League Comparison
League data pipeline and League page visualizations are now implemented.

**Backend:**
- `dashboard/backend/jobs/league_job.py` - fetches and writes both datasets:
  - competitors dataset (`TEAM_ID` + configured `COMPETITORS`)
  - top-global dataset (`TEAM_ID` + `TOP_GLOBAL_COUNT` global entries)
- Persists data to `competitive_data` singleton table and updates `refresh_log`
- Supports optional runtime overrides (`gameweek`, IDs, count, cache) for smoke/debug runs

**Frontend:**
- `dashboard/frontend/src/lib/api.ts` - typed league interfaces (`LeagueEntry`, transfer timeline, GW transfer summary, season history)
- `dashboard/frontend/src/hooks/use-api.ts` - league hooks now poll every 30s until data is available
- `dashboard/frontend/src/routes/league.tsx` - full League page implementation:
  - toggle tabs: Competitors | Top Global
  - rank progression line chart (Visx, your entry highlighted when present)
  - points progression line chart (Visx)
  - side-by-side transfer diff table by GW
  - chip timeline per manager
  - player contribution treemap-style chart (Visx)
  - manager snapshot cards (rank/points/value/hits/recent transfer activity)

**Production build:** 660KB JS + 69KB CSS (gzipped: 207KB + 11KB)

### Sprint 7: Scatter Analysis + Manager Report
Scatter and Manager pages are now fully implemented with backend-supported analysis.

**Backend:**
- `dashboard/backend/routers/manager.py` - captain analysis now includes per-GW optimal counterfactual fields (`optimal_*`, `points_lost`, `was_optimal`) plus aggregate summary (`accuracy`, points earned/optimal/lost)
- Tie handling for optimal captain logic now treats equal-point captain picks as correct

**Frontend:**
- `dashboard/frontend/src/lib/api.ts` - added strong types for scatter response, captain picks, and captain summary
- `dashboard/frontend/src/routes/scatter.tsx` - full Scatter page implementation:
  - 4 chart tabs (`xg_goals`, `xa_assists`, `usage_output`, `defensive`)
  - responsive Visx scatter chart with median quadrant lines
  - minute-scaled dots, position-color legend, and hover tooltip with player detail
  - quadrant distribution counters + warmup/error states
- `dashboard/frontend/src/routes/manager.tsx` - full Manager report implementation:
  - rank journey area chart (Visx)
  - points per GW bar chart (Visx)
  - team value trend area chart (Visx)
  - captain analysis table with actual vs optimal comparison
  - captain accuracy badges (accuracy %, lost points, optimal pick count)

**Production build:** 682KB JS + 71KB CSS (gzipped: 212KB + 12KB)

### Sprint 8: Polish + Production Build
Sprint 8 polish and production runtime flow are now implemented.

**Frontend polish:**
- `dashboard/frontend/src/components/layout/app-sidebar.tsx` - responsive sidebar:
  - desktop fixed sidebar remains on `md+`
  - mobile now uses `Sheet` drawer with route links + close-on-navigation
- `dashboard/frontend/src/components/layout/app-header.tsx` - data freshness refinement:
  - freshness badge (`Fresh`, `Stale`, `Degraded`, `Warming`)
  - last refresh computed from all job timestamps
  - per-job status dots (`bootstrap`, `analysis`, `solver`, `league`)
  - mobile sidebar trigger button added
- `dashboard/frontend/src/router.tsx` - error boundaries per route:
  - shared route error fallback wired to root + all 7 page routes
  - retry action without full-page reload
- `dashboard/frontend/src/routes/transfers.tsx` - solver slow-load UX:
  - expanded skeleton layout for initial/warmup states
  - explicit solver-computing status message over skeleton state

**Backend/runtime polish:**
- `dashboard/backend/jobs/analysis_job.py` - predictor training compatibility fix:
  - `predictor.train(player_ids=...)` when required
  - backward-compatible fallback to old zero-arg signature
- `run_dashboard.sh` (worktree root) added:
  - activates venv
  - builds frontend
  - starts `uvicorn dashboard.backend.main:app`

**Production build:** 693KB JS + 72KB CSS (gzipped: 215KB + 12KB)

**Verification run (Sprint 8):**
- Cold start: moved existing DB aside, started FastAPI, confirmed fresh DB recreation and bootstrap completion
- Analysis repopulation: `player_analysis` (818 rows) and `scatter_data` (4 chart types) after compatibility fix
- League smoke job: completed with compact config (`competitors=1`, `global_managers=2`)
- API/static smoke:
  - `/` and built `/assets/*` served by FastAPI
  - Core APIs (`meta`, `squad`, `players`, `fixtures`, `scatter`, `manager`) returned 200
  - League endpoints returned 200 after league smoke job
  - Solver endpoint remained 503 in smoke due long-running compute window
- Full unit suite (`python -m unittest discover -s tests -v`): 92 tests, 3 pre-existing errors (`ModuleNotFoundError: simulation` in report compile tests)

---

## What's Left

No planned sprint items remain. Optional follow-ups:
- Reduce solver cold-start runtime (currently long enough that `/api/transfers/solver` can remain 503 during smoke windows)
- Resolve pre-existing report test import errors (`simulation` module path during LaTeX compile tests)

---

## How to Run (Development)

```bash
# Terminal 1: Backend
cd .worktrees/dashboard
source ../../venv/bin/activate  # or use full path to venv/bin/python
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd .worktrees/dashboard/dashboard/frontend
npm run dev
# Opens at http://localhost:5173 (proxies /api to :8000)
```

## How to Run (Production Build)

```bash
cd .worktrees/dashboard/dashboard/frontend
npm run build
cd ../..
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000
# SPA served at http://localhost:8000
```

## Key Design Decisions
- SQLite singleton-blob pattern for most tables (simple, single-user)
- `player_analysis` has flat columns for SQL filtering (600+ rows need server-side filtering)
- NaN/Inf-safe JSON serializer (FPL API returns non-standard floats)
- APScheduler in-process; bootstrap blocks on startup (503 until ready)
- Code-based TanStack Router (simpler than file-based for 7 static routes)
- Dark-only theme (zinc base color)
- Vite proxy in dev, FastAPI serves SPA in prod (same :8000 origin)
- `.npmrc` with `legacy-peer-deps=true` (visx peer conflicts with React 19)

## Dependencies Added
- **Python:** fastapi, uvicorn, apscheduler
- **Node:** @tanstack/react-router, @tanstack/react-query, @tanstack/react-table, @tanstack/react-virtual, tailwindcss, @tailwindcss/vite, shadcn, tw-animate-css, clsx, tailwind-merge, lucide-react, @visx/* (14 packages)
