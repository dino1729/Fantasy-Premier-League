# Architecture Update Verification Report (ARCH_UPDATE.md → Repo Reality)

**Generated:** 2025-12-21  
**Reference Plan:** `ARCH_UPDATE.md`  
**Primary Question:** Are the planned architectural upgrades (Phases 1–4) (a) implemented in this repo, and (b) actually *used* by the LaTeX report generator (`reports/run_report.sh` → `reports/generate_fpl_report.py`)?

---

## 0) Methodology / Definitions (Strict)

### What counts as **Implemented**
An upgrade is marked **Implemented** only if the repo contains working code/data structures that realize the upgrade beyond stubs (i.e., not just comments/imports). “Partially” means code exists but a key sub-requirement is missing (data not present, not wired, dead code, incompatible schema, etc).

### What counts as **Used in LaTeX Report Generation**
An upgrade is marked **Used** only if it is executed via the *actual report pipeline*:

`reports/run_report.sh` → `reports/generate_fpl_report.py` → `reports/fpl_report/*` → `reports/fpl_report/latex_generator.py`

If a feature exists only in `main.py` or in standalone modules (`etl/`, `models/`) and is not invoked from the report pipeline, it is **NOT USED** (for the purpose of report generation), even if it “works” standalone.

### Evidence Style
Evidence references are given as:
- `path/to/file.py:SymbolName` (preferred), or
- `path/to/file.py` + brief description of the relevant block.

---

## 1) Executive Summary (Delta vs ARCH_UPDATE.md)

### Overall Status (ARCH_UPDATE.md “100x” plan)
- **Phase 1 (Data Infrastructure):** **Partially implemented**, **partially used** (only the Elo-based difficulty slice is used by the report).
- **Phase 2 (Predictive Engine):** **Implemented (new stack exists)**, **NOT used** by report pipeline (report still uses legacy predictor).
- **Phase 3 (MIP Solver):** **Implemented**, **Used**, but **NOT truly multi-period** (static “all transfers now” formulation).
- **Phase 4 (Strategy & Reporting):** **Partially implemented**, **Used** (MIP section + basic explainability), but missing EO + uncertainty captaincy + solver-driven multi-week transfer schedule.

### Highest-Impact Gaps (Blocking “ARCH_UPDATE.md promise”)
1. **Multi-period MIP** is not implemented (no transfer continuity / FT banking / bank rollover across weeks).
2. **Position-specific ML models exist but are not wired into the report pipeline** (MIP uses legacy predictions).
3. **CBIT / defensive-action metrics** are not ingested into the warehouse nor used in modeling/scoring/reporting.
4. **Effective Ownership (EO) + uncertainty-based captaincy matrix** are not implemented (only simple differential/league ownership heuristics exist).

---

## 2) Two Parallel Pipelines Exist (Important Context)

### A) LaTeX Report Pipeline (the one that matters for “Used”)
- Entrypoint: `reports/run_report.sh`
- CLI: `reports/generate_fpl_report.py`
- Core modules:
  - Data: `reports/fpl_report/data_fetcher.py` (wraps `getters.py` + caching)
  - Analysis: `reports/fpl_report/player_analyzer.py`
  - Prediction (legacy): `reports/fpl_report/predictor.py:FPLPointsPredictor`
  - Transfers: `reports/fpl_report/transfer_recommender.py`, `reports/fpl_report/transfer_strategy.py`
  - Rendering: `reports/fpl_report/latex_generator.py`

### B) “2.0 / Warehouse” Pipeline (largely *not* used by LaTeX report)
- Orchestrator: `main.py`
- Data infra: `etl/pipeline.py:ETLPipeline` → `data/parquet/*.parquet` (+ raw archive in `data/raw/`)
- ML: `models/train.py` + `models/inference.py:FPLInferencePipeline`
- Solver: `solver/optimizer.py:TransferMIPSolver`
- Output: mostly terminal / non-LaTeX via `reports/strategy_reporter.py`

**Consequence:** The repo contains many “implemented” upgrades that are not yet consumed by the LaTeX report generator.

---

## 3) Phase-by-Phase Verification (ARCH_UPDATE.md)

### Phase 1 — Data Infrastructure (“Warehouse”)

| Upgrade (ARCH_UPDATE.md intent) | Implemented? | Used in LaTeX report? | Evidence | Notes / Still TODO |
|---|---:|---:|---|---|
| ETL pipeline replacing ad-hoc calls | ✅ (exists) | ❌ | `etl/pipeline.py:ETLPipeline` | Report still uses `getters.py` + `CacheManager` pickles. |
| Raw response archiving under `data/raw/` | ✅ | ⚠️ (indirect) | `etl/fetchers.py:FPLFetcher` writes raw; `data/raw/*` exists | Report’s Elo difficulty calls `etl.fetchers.FPLFetcher` with `save_raw=False`, so report generation does **not** populate raw archives. |
| Parquet “warehouse” (`data/parquet/`) | ✅ | ❌ | `etl/transformers.py:*Transformer.save_parquet`, `data/parquet/*.parquet` | No report module reads parquet for core report generation. |
| Dynamic difficulty (ClubElo → probabilities → FDR) | ✅ | ✅ | `etl/fetchers.py:ClubEloFetcher`, `etl/fetchers.py:FixtureDifficultyCalculator`, `reports/fpl_report/data_fetcher.py:get_upcoming_fixtures` | **Used** for fixture display + transfer scoring + heatmaps, but **not** used in the legacy ML predictor. |
| CBIT “2025/26 defensive actions” ingestion | ⚠️ partial | ❌ | `etl/fetchers.py:FPLFetcher.extract_player_cbit_stats` exists | Extraction helper exists, but no ETL integration into parquet schema + no downstream usage. |
| Entity resolution mapping CSV (`data/mappings/player_id_map.csv`) | ⚠️ partial | ❌ | `etl/transformers.py:PlayerIDMapper`; `data/mappings/` is empty | Mapper exists but no mapping artifact produced/consumed. |
| Understat integration wired into ETL | ❌ | ❌ | (no ETL fetcher) | `understat.py` exists as a scraper, but ETL does not ingest it; report predictor contains unused Understat loader stub. |
| FBref integration wired into ETL | ❌ | ❌ | (no ETL fetcher) | `fbref.py` exists; not part of ETL or report prediction pipeline. |
| Replace pickle caching with stable formats | ❌ | ❌ | `reports/fpl_report/cache_manager.py` uses pickle | Report caching remains pickle-based (plus metadata). |

**Phase 1 “Used in report” reality:** only the **Elo-based fixture difficulty** slice is exercised end-to-end in LaTeX report generation.

---

### Phase 2 — Predictive Engine (“xP Layer”)

| Upgrade (ARCH_UPDATE.md intent) | Implemented? | Used in LaTeX report? | Evidence | Notes / Still TODO |
|---|---:|---:|---|---|
| Position-specific models (per role) | ✅ | ❌ | `models/train.py:PositionModel` | Implemented as XGBoost (fallback GBR) + Ridge ensemble; not used by report generator. |
| Batch inference pipeline → `projections_horizon.parquet` | ✅ | ❌ | `models/inference.py:FPLInferencePipeline` | Writes `data/parquet/projections_horizon.parquet`, but report uses `reports/fpl_report/predictor.py`. |
| Feature engineering uses Elo context | ✅ (new stack) | ❌ | `models/feature_engineering.py:FeatureEngineer` | New feature set includes `fixture_difficulty_elo`, win probability, opponent/own Elo; unused by LaTeX report pipeline. |
| xMins handling via `chance_of_playing_*` + substitution risk | ⚠️ partial | ❌ | `models/inference.py:FPLInferencePipeline._adjust_for_availability` | Uses `chance_playing` and optional availability model; no explicit substitution-risk factor as in plan. |
| Understat xG/xA features | ❌ (effectively) | ❌ | `reports/fpl_report/predictor.py:_load_understat_data` exists but never called | Current report predictor doesn’t consume Understat; new stack uses FPL expected stats and would need CBIT/Understat wiring for full plan. |

#### Report Pipeline Reality (still “legacy predictor”)
- Used predictor: `reports/fpl_report/predictor.py:FPLPointsPredictor`
  - Model: GradientBoostingRegressor + RandomForestRegressor ensemble
  - Features: rolling 4-GW stats + static FDR from fixtures table (`team_*_difficulty`), not Elo.
- Wiring:
  - `reports/fpl_report/transfer_recommender.py` instantiates `FPLPointsPredictor`
  - `reports/fpl_report/transfer_strategy.py:TransferStrategyPlanner` uses `transfer_recommender.predictor`

**Critical integration gap:** MIP solver optimization currently uses predictions from the **legacy predictor**, not the position-specific models.

---

### Phase 3 — Prescriptive Analytics (MIP Solver)

| Upgrade (ARCH_UPDATE.md intent) | Implemented? | Used in LaTeX report? | Evidence | Notes / Still TODO |
|---|---:|---:|---|---|
| MIP solver using sasoptpy + HiGHS | ✅ | ✅ | `solver/optimizer.py:TransferMIPSolver`; invoked via `reports/fpl_report/transfer_strategy.py:TransferStrategyPlanner._run_mip_solver` | Report supports `--no-mip` and prints solver status. |
| Multi-period planning (week-indexed squad + transfers) | ❌ | ❌ | `solver/optimizer.py` uses one `x[p]` squad var (not `x[p,w]`) | Transfers are computed only relative to the *current* squad; no continuity constraints. |
| Transfer continuity constraint (`squad[w]=squad[w-1]+in-out`) | ❌ | ❌ | (absent) | Core “Horizon Effect” not solved. |
| Bank rollover across weeks | ❌ | ❌ | (absent) | Solver budget is single-shot (bank + sell revenue ≥ buys). |
| Saved free transfer value (FT_Value term) | ❌ | ❌ | (absent) | No objective term for banking free transfers. |
| Formation constraints in solver | ✅ | ✅ | `solver/optimizer.py:TransferMIPSolver.XI_MIN/XI_MAX` | Enforced per week for starting XI. |
| Bench optimization | ⚠️ partial | ⚠️ partial | `solver/optimizer.py` models XI only (bench has no value) | Bench is derived post-hoc as “not in lineup week 0”; not optimized for bench points/ordering/BB chip. |

#### “Why multi-period is NOT implemented” (technical proof)
- Decision variable is a **single** squad selection: `x[p]` in `solver/optimizer.py:TransferMIPSolver._solve_mip`.
- Transfers are derived as `transfer_in[p]=x[p]` for non-current players, and `transfer_out[p]=1-x[p]` for current players.
- There is no week index on transfers and no `x[p,w]` / `transfer_in[p,w]` / `bank[w]`.

**Net effect:** The solver chooses a single optimized squad for the entire horizon and performs all changes immediately.

---

### Phase 4 — Strategy & Reporting (“Explainable AI”)

| Upgrade (ARCH_UPDATE.md intent) | Implemented? | Used in LaTeX report? | Evidence | Notes / Still TODO |
|---|---:|---:|---|---|
| MIP results rendered in LaTeX | ✅ | ✅ | `reports/fpl_report/latex_generator.py:_generate_mip_recommendation` | Produces “Optimal Transfer Plan (MIP Solver)” section + per-GW xP timeline. |
| “Why This Move?” ROI explanation | ⚠️ partial | ✅ (basic) | `reports/fpl_report/latex_generator.py:_generate_mip_recommendation` | Shows xP gain + price delta + coarse verdict; does not compute payback weeks/ROI per £. |
| ROI framework in codebase | ✅ (exists) | ❌ (not used in LaTeX) | `solver/interpreter.py:calculate_transfer_roi` + `reports/analytics.py:ROICalculator` | ROI exists but LaTeX generator does not call it (duplication / drift). |
| Effective Ownership (EO) blocker vs differential | ❌ | ❌ | (no EO computation) | There is **league ownership** and **captain counts**, but no EO metric (captain multipliers not applied). |
| Captaincy uncertainty matrix (Gamma haul probability) | ❌ | ❌ | (absent) | Current captain picks are heuristic scoring (`reports/fpl_report/transfer_recommender.py:get_best_captain_picks`). |
| Solver “Gantt chart” / multi-week transfer timeline | ⚠️ partial | ⚠️ partial | Heuristic timeline exists (`latex_generator.py:_generate_transfer_sequence`) | For MIP, only xP timeline is shown. A solver-to-timeline adapter exists but is unused (`reports/fpl_report/transfer_strategy.py:format_timeline_for_latex`). |
| Chip optimization using MIP | ❌ | ❌ | `WildcardOptimizer` / `FreeHitOptimizer` are greedy | No chip decision variables in solver; interpreter supports detecting chip vars but solver never sets them. |
| Sensitivity analysis knobs (e.g., FT_Value) | ❌ | ❌ | (absent) | No CLI/config parameter affecting objective tradeoffs. |

---

## 4) “Hidden Wins” Already Implemented (Beyond ARCH_UPDATE.md Checklist)

These are not explicitly promised in `ARCH_UPDATE.md` but materially move the system toward Phase 4 “Explainable AI / competitive context”:

- **Transfer momentum signals** (bandwagon vs mass-sell):
  - `reports/fpl_report/transfer_recommender.py:calculate_transfer_momentum`
  - Displayed in LaTeX replacement tables via `reports/fpl_report/latex_generator.py` (trend arrows).
- **Dream Team consistency factor** in captain scoring:
  - `reports/fpl_report/transfer_recommender.py:get_best_captain_picks` uses `dreamteam_count`.
- **Competitive / league sampling + league ownership**:
  - `reports/fpl_report/data_fetcher.py:compute_league_ownership`, `get_top_global_teams`
  - Used by Free Hit optimizer and league analysis sections in LaTeX.

These partially overlap with Phase 4 ideas (EO / explainability), but they are not full EO nor uncertainty modeling.

---

## 5) Integration / Drift Findings (Where “Implemented” ≠ “Used”)

### 5.1 Unused-but-present modules (high-value wiring targets)
- New ML inference pipeline is instantiated but not actually used:
  - `reports/fpl_report/transfer_recommender.py` constructs `self.inference_pipeline` but never calls it.
  - `reports/fpl_report/transfer_strategy.py` detects `self.inference_pipeline` and prints a message, but still uses `self.predictor`.
- Solver-to-timeline adapter exists but is unused in LaTeX:
  - `reports/fpl_report/transfer_strategy.py:build_transfer_timeline` / `format_timeline_for_latex`
  - LaTeX generator independently creates its own xP timeline and does not reuse this code.
- ROI is implemented twice but LaTeX uses neither:
  - `solver/interpreter.py:calculate_transfer_roi`
  - `reports/analytics.py:ROICalculator`

### 5.2 Inconsistencies that matter
- **Fixture difficulty inconsistency**:
  - Transfers/fixtures displayed via `get_upcoming_fixtures` use Elo-based difficulty.
  - Legacy predictor used for MIP uses **static FPL FDR** from fixtures table.
  - This means: the report may show “easy Elo fixtures” but the MIP objective is driven by a model that does not consume Elo features.
- **Docstring vs implementation mismatch**:
  - `solver/interpreter.py` docstring references `x[p,w]` variables, but solver uses `x[p]`.
  - This can mislead future work on multi-period planning.

---

## 6) Technical Requirements Checklist (ARCH_UPDATE.md §5)

| Library / capability | Status in repo | Notes |
|---|---:|---|
| `pandas`, `numpy`, `requests`, `beautifulsoup4` | ✅ | Present in `requirements.txt`. |
| `scikit-learn` | ✅ | Used by both legacy predictor + new models. |
| `xgboost` | ✅ (optional fallback) | `models/train.py` falls back to GBR if missing. |
| `sasoptpy`, `highspy` | ✅ (declared), ⚠️ runtime optional | `solver/optimizer.py` gracefully disables if not installed. |
| `pyarrow` (parquet) | ✅ | Used by `etl/*` and `models/inference.py`. |
| `duckdb` (SQL-on-files) | ❌ | Mentioned in ARCH_UPDATE.md; not present in `requirements.txt` and not used. |

---

## 6.1) Refactored File Structure Mapping (ARCH_UPDATE.md §5)

`ARCH_UPDATE.md` proposes a “best-of-breed” refactor layout. This table maps planned paths to repo reality *and* whether the LaTeX report pipeline consumes them.

| Planned path (ARCH_UPDATE.md) | Exists in repo? | Used by LaTeX report? | Actual path / notes |
|---|---:|---:|---|
| `data/raw/` | ✅ | ❌ | Exists and is populated by ETL fetchers; report pipeline doesn’t write to it. |
| `data/parquet/` | ✅ | ❌ | Exists and is populated by ETL / inference; report pipeline doesn’t read it. |
| `data/mappings/` | ✅ (empty) | ❌ | Directory exists; mapping CSVs are not generated/consumed. |
| `etl/fetchers.py` | ✅ | ✅ (partial) | Elo difficulty is consumed indirectly via `reports/fpl_report/data_fetcher.py:get_upcoming_fixtures`. |
| `etl/pipeline.py` | ✅ | ❌ | Used by `main.py`, not by report pipeline. |
| `models/training/*` | ⚠️ | ❌ | Implemented as `models/train.py` (flat) + artifacts. |
| `models/artifacts/*` | ✅ | ❌ | Models exist but are not wired into LaTeX report predictions/MIP. |
| `models/predictor.py` | ⚠️ | ❌ | Implemented as `models/inference.py` (projection generation). |
| `solver/definitions.py` | ✅ | ✅ | Used by `solver/optimizer.py` which is invoked from report. |
| `solver/constraints.py` | ❌ | ❌ | Constraint logic is merged into `solver/optimizer.py`. |
| `solver/optimizer.py` | ✅ | ✅ | Invoked by `reports/fpl_report/transfer_strategy.py`. |
| `reports/latex_templates/` | ❌ | ❌ | LaTeX is generated inline in `reports/fpl_report/latex_generator.py`. |
| `reports/visualizer.py` | ⚠️ | ✅ | Implemented as `reports/fpl_report/plot_generator.py`. |
| `main.py` (CLI orchestrator) | ✅ | ❌ | Orchestrates ETL/models/solver for terminal reporting; LaTeX uses `reports/generate_fpl_report.py`. |

---

## 7) Still-TODO Priority Queue (Actionable, Repo-Specific)

### P0 — Breaks core ARCH_UPDATE.md promise
1. **True multi-period MIP**
   - Add week-indexed squad variables and continuity constraints:
     - `x[p,w]` (in-squad), `t_in[p,w]`, `t_out[p,w]`, `bank[w]`, `ft[w]`
   - Enforce:
     - continuity: `x[p,w] = x[p,w-1] + t_in[p,w] - t_out[p,w]`
     - transfer limits + hit accounting per week
     - bank rollover and sell/buy pricing per week (at least constant prices as first step)
   - Primary file: `solver/optimizer.py`
2. **Wire new models into LaTeX report pipeline**
   - Replace `reports/fpl_report/predictor.py` usage in:
     - `reports/fpl_report/transfer_recommender.py`
     - `reports/fpl_report/transfer_strategy.py`
   - Consume `models/inference.py` outputs directly (either by calling `FPLInferencePipeline` or reading `data/parquet/projections_horizon.parquet`).

### P1 — High value upgrades aligned with ARCH_UPDATE.md
3. **CBIT end-to-end**
   - Extend ETL to build CBIT season aggregates and store in parquet:
     - integrate `etl/fetchers.py:FPLFetcher.get_live_gameweek` (or fixtures stats) → per-player CBIT totals/rolling
   - Extend feature engineering to reliably populate `cbit_rolling_3` (currently often zero if history lacks columns).
4. **Make Elo features the single source of truth**
   - Ensure prediction features (used by MIP) consume Elo-based difficulty, not static FDR.
   - Avoid double-fetching fixtures via both `getters.py` and `etl.fetchers.FPLFetcher` in the same report run.

### P2 — Phase 4 completeness
5. **EO + blocker/differential classification**
   - Extend `compute_league_ownership` to compute **effective ownership**:
     - `EO = (ownership_fraction) + (captain_fraction)` (or captain double-count depending on definition)
   - Render as “Blockers vs Differentials” in LaTeX.
6. **Captaincy uncertainty matrix**
   - Add distributional modeling layer (Gamma/Poisson) and output “safe vs ceiling” probabilities.
7. **Unify ROI / explainability**
   - Pick one ROI engine (`reports/analytics.py` or `solver/interpreter.py`) and have LaTeX call it.
   - Include payback period and hit justification in the MIP section.

---

## 8) Test/Validation Notes (Repo Hygiene)

### Unit tests
- Test entrypoint: `python3 -m unittest discover -s tests -v`
- Current situation in this environment:
  - Some tests pass individually (example: `tests/test_report_enhancements.py:TestReportEnhancements.test_get_transfers_enriches_names_and_costs`).
  - `TestCompetitiveDatasetBuilder.test_build_competitive_dataset_computes_total_hits` hangs under restricted networking because `build_competitive_dataset()` calls `compute_gw_transfers()` / `compute_transfer_history()` which may invoke live API endpoints unless fully mocked.

**Recommendation:** If expanding architecture work, add deterministic unit tests around:
- multi-period solver constraints (no network)
- model inference integration (no network)
- CBIT feature extraction (local fixtures/live snapshots mocked)
