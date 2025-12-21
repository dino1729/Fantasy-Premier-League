# Architecture Update: Implementation Status & TODO

**Last Updated:** 2025-12-20
**Reference:** [ARCH_UPDATE.md](./ARCH_UPDATE.md)
**Status:** ~70% Complete

---

## Quick Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Data Infrastructure | ✅ Mostly Complete | 85% |
| Phase 2: Predictive Engine | ✅ Mostly Complete | 80% |
| Phase 3: MIP Solver | ⚠️ Partially Complete | 65% |
| Phase 4: Strategy & Reporting | ⚠️ Partially Complete | 50% |
| **NEW: FPL API Data** | ❌ Untapped | 20% |

### Key Findings

**High-Value Quick Wins (from FPL API):**
- `transfers_in_event` / `transfers_out_event` - Bandwagon/panic detection (NOT USED)
- `/api/dream-team/{gw}/` - Team of the Week for haul detection (NOT USED)
- `/api/leagues-classic/314/standings/` - Top manager squad analysis (NOT USED)
- `dreamteam_count`, `news`, `status` fields - Available but ignored

---

## Phase 1: Data Infrastructure (The Foundation)

### ✅ Completed

| Feature | Implementation | Files |
|---------|---------------|-------|
| ETL Pipeline | Replaces direct API calls with structured fetch→transform→save flow | `etl/pipeline.py` |
| FPL API Fetcher | Structured data ingestion with caching | `etl/fetchers.py:FPLFetcher` |
| ClubElo Integration | Dynamic team strength ratings | `etl/fetchers.py:ClubEloFetcher` |
| Elo-Based FDR | Win probability calculation: `P(Win) = 1/(1+10^{(Elo_opp-Elo)/400})` | `etl/fetchers.py:FixtureDifficultyCalculator` |
| CBIT Extraction | Defensive metrics (clearances, blocks, interceptions, tackles) | `etl/fetchers.py:40-52` |
| Parquet Warehouse | High-performance columnar storage | `data/parquet/*.parquet` |
| Schema Definitions | Typed dataclasses for players, fixtures, projections | `etl/transformers.py:PlayerSchema`, `FixtureSchema` |
| Player ID Mapper | Fuzzy matching for entity resolution | `etl/transformers.py:PlayerIDMapper` |
| Heuristic Fallback | Graceful degradation when models unavailable | `etl/pipeline.py:HeuristicAdapter` |

### ❌ Not Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| **Understat Integration** | HIGH | xG/xA source not wired into ETL pipeline |
| **FBref Integration** | MEDIUM | Advanced metrics (progressive passes, pressures) |
| **Entity Resolution CSV** | MEDIUM | `mappings/player_id_map.csv` not created |
| **Raw JSON Archiving** | LOW | `data/raw/` directory structure exists but not populated |

### Files Created

```
etl/
├── __init__.py
├── fetchers.py      # FPLFetcher, ClubEloFetcher, FixtureDifficultyCalculator
├── transformers.py  # PlayerSchema, FixtureSchema, PlayerIDMapper
└── pipeline.py      # ETLPipeline, HeuristicAdapter

data/parquet/
├── players.parquet
├── fixtures.parquet
├── training_data.parquet
└── projections_horizon.parquet
```

---

## Phase 2: Predictive Engine (The "xP" Layer)

### ✅ Completed

| Feature | Implementation | Files |
|---------|---------------|-------|
| Position-Specific Models | Separate models for GKP, DEF, MID, FWD | `models/train.py:PositionModel` |
| Ensemble Architecture | XGBoost (0.7) + Ridge (0.3) weighted blend | `models/train.py:78-85` |
| Position-Specific Features | GKP: saves, CS / DEF: cbit, CS / MID-FWD: xg, xa, xgi | `models/feature_engineering.py` |
| TimeSeriesSplit Validation | Prevents data leakage in temporal data | `models/train.py:TimeSeriesSplit` |
| Availability Predictor | LogisticRegression on chance_of_playing flags | `models/train.py:AvailabilityPredictor` |
| Model Artifacts | Persisted trained models per position | `models/artifacts/model_*.pkl` |
| Inference Pipeline | Batch predictions to parquet | `models/inference.py:FPLInferencePipeline` |

### ⚠️ Partial / Needs Work

| Feature | Current State | Gap |
|---------|---------------|-----|
| **Legacy Predictor** | Still used in report pipeline | `reports/fpl_report/predictor.py` uses GradientBoosting+RandomForest, not position-specific |
| **xMins Adjustment** | Basic availability flag used | No substitution risk factor applied |
| **Model Integration** | Models exist but not fully wired | Report uses legacy predictor, not `models/inference.py` |

### ❌ Not Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| **Understat xG Features** | HIGH | Model features reference xG but data not ingested |
| **xGChain (Build-up Play)** | MEDIUM | Not in feature engineering |
| **Opponent xGA Context** | MEDIUM | Opponent defensive strength not fully integrated |

### Files Created

```
models/
├── __init__.py
├── train.py              # PositionModel, AvailabilityPredictor, train_all_models()
├── feature_engineering.py # PositionFeatureEngine, position-specific feature sets
├── inference.py          # FPLInferencePipeline
└── artifacts/
    ├── model_GKP.pkl
    ├── model_DEF.pkl
    ├── model_MID.pkl
    ├── model_FWD.pkl
    └── availability_model.pkl
```

---

## Phase 3: Prescriptive Analytics (The Solver)

### ✅ Completed

| Feature | Implementation | Files |
|---------|---------------|-------|
| MIP Solver Core | sasoptpy + HiGHS integration | `solver/optimizer.py:TransferMIPSolver` |
| Budget Constraint | Squad cost ≤ budget | `solver/optimizer.py:160-165` |
| Squad Size Constraint | Exactly 15 players | `solver/optimizer.py:155-158` |
| Team Limits | Max 3 per club | `solver/optimizer.py:168-175` |
| Position Quotas | GKP:2, DEF:5, MID:5, FWD:3 | `solver/optimizer.py:178-185` |
| Transfer Hit Accounting | -4 points per extra transfer | `solver/optimizer.py:195-200` |
| Discount Factor | Future weeks weighted by γ^(w-1) | `solver/optimizer.py:128` |
| Graceful Fallback | MIP_AVAILABLE flag for missing dependencies | `solver/optimizer.py:18-25` |
| Result Dataclass | Typed solver output | `solver/definitions.py:MIPSolverResult` |
| Human-Readable Plans | Interprets solver output | `solver/interpreter.py:SolverInterpreter` |

### ⚠️ Critical Gap

| Issue | Current State | Impact |
|-------|---------------|--------|
| **STATIC Optimization** | All transfers scheduled for GW1 | No true multi-period planning |
| **No Transfer Continuity** | Missing constraint: squad[w] = squad[w-1] + in - out | Cannot hold transfers for future weeks |
| **Single-Period Horizon** | Objective sums all weeks but transfers not distributed | "Horizon Effect" not solved |

**This is the most significant architectural gap.** The solver finds optimal transfers but executes them all immediately rather than distributing across the planning horizon.

### ❌ Not Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| **True Multi-Period Variables** | CRITICAL | Need `x[player, week]` binary variables |
| **Bank Rollover Constraint** | HIGH | Budget should carry forward |
| **Saved FT Value** | HIGH | FT_Value term in objective missing |
| **Bench Optimization** | MEDIUM | Solver ignores bench selection |
| **Formation Constraint** | MEDIUM | Only in greedy optimizers, not MIP |
| **DGW/BGW Handling** | LOW | Double/blank gameweek awareness |

### Files Created

```
solver/
├── __init__.py
├── definitions.py    # MIPSolverResult, POSITION_MAP, constants
├── optimizer.py      # TransferMIPSolver, solve(), _build_constraints()
└── interpreter.py    # SolverInterpreter, format_plan()
```

---

## Phase 4: Strategy & Reporting

### ✅ Completed

| Feature | Implementation | Files |
|---------|---------------|-------|
| MIP Integration | Strategy planner calls solver, results flow to report | `reports/fpl_report/transfer_strategy.py:145-187` |
| Heuristic Suppression | When MIP optimal, removes conflicting recommendations | `transfer_strategy.py:171-173` |
| TikZ Timeline | Per-GW expected points visualization | `latex_generator.py:1426-1458` |
| Transfer Table | OUT/IN display with prices | `latex_generator.py:1391-1423` |
| EV Analysis Box | Current vs optimized squad xP | `latex_generator.py:1365-1384` |
| Solver Metrics | Time, status, hit cost display | `latex_generator.py:1359-1384` |
| Wildcard Optimizer | Greedy draft builder | `transfer_strategy.py:WildcardOptimizer` |
| Free Hit Optimizer | League-aware differential drafts | `transfer_strategy.py:FreeHitOptimizer` |

### ❌ Not Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| **"Why This Move?" ROI** | HIGH | Explicit payback period calculation |
| **Effective Ownership (EO)** | MEDIUM | Blocker vs Differential classification |
| **Captaincy Matrix** | MEDIUM | Gamma distribution ceiling simulation |
| **Gantt Chart** | LOW | Multi-week transfer timeline visual |
| **Chip MIP Optimization** | HIGH | Wildcard/FreeHit are greedy, not MIP |
| **Sensitivity Analysis** | LOW | FT_Value slider for aggressive/conservative |

---

## Implementation Priority Queue

### P0: Critical (Breaks Core Promise)

1. **True Multi-Period MIP Optimization**
   - Add week-indexed decision variables: `x[player, week]`
   - Add transfer continuity constraints
   - Add bank rollover tracking
   - **Files:** `solver/optimizer.py`
   - **Effort:** ~2-3 days

2. **Wire New Models to Report Pipeline**
   - Replace `reports/fpl_report/predictor.py` with `models/inference.py`
   - Load position-specific model artifacts
   - **Files:** `reports/fpl_report/transfer_strategy.py`, `predictor.py`
   - **Effort:** ~1 day

### P1: High Priority (Significant Value)

3. **Understat/FBref Data Integration**
   - Add scraper classes to `etl/fetchers.py`
   - Create entity resolution mapping
   - Wire xG/xA features to models
   - **Files:** `etl/fetchers.py`, `models/feature_engineering.py`
   - **Effort:** ~2-3 days

4. **"Why This Move?" Explainability**
   - Calculate per-transfer ROI with payback period
   - Add to LaTeX output
   - **Files:** `solver/interpreter.py`, `latex_generator.py`
   - **Effort:** ~1 day

5. **Chip MIP Optimization**
   - Extend solver for Wildcard/FreeHit chip decisions
   - Compare global EV of playing now vs holding
   - **Files:** `solver/optimizer.py`, `transfer_strategy.py`
   - **Effort:** ~2 days

### P2: Medium Priority (Nice to Have)

6. **Captaincy Matrix with Uncertainty**
   - Gamma distribution for haul probability
   - "Safe vs Ceiling" comparison
   - **Files:** `reports/fpl_report/transfer_recommender.py`
   - **Effort:** ~1 day

7. **Effective Ownership Analysis**
   - Classify players as Blockers vs Differentials
   - Add to competitive analysis section
   - **Files:** `reports/fpl_report/data_fetcher.py`, `latex_generator.py`
   - **Effort:** ~1 day

8. **Formation Constraint in MIP**
   - Add min/max position limits to solver
   - Currently only in greedy optimizers
   - **Files:** `solver/optimizer.py`
   - **Effort:** ~0.5 days

### P3: Low Priority (Polish)

9. **Gantt Chart Visualization**
   - Multi-week transfer timeline
   - TikZ or matplotlib implementation
   - **Files:** `latex_generator.py`
   - **Effort:** ~0.5 days

10. **Sensitivity Analysis Mode**
    - CLI flag for FT_Value adjustment
    - Aggressive vs Conservative modes
    - **Files:** `reports/config.yml`, `solver/optimizer.py`
    - **Effort:** ~0.5 days

---

## File Mapping: ARCH_UPDATE.md → Implementation

| Planned Path | Actual Implementation | Status |
|--------------|----------------------|--------|
| `data/raw/` | Not used | ❌ |
| `data/parquet/` | `data/parquet/*.parquet` | ✅ |
| `data/mappings/` | Not created | ❌ |
| `etl/fetchers.py` | `etl/fetchers.py` | ✅ |
| `etl/pipeline.py` | `etl/pipeline.py` | ✅ |
| `models/training/` | `models/train.py` | ✅ (flat) |
| `models/artifacts/` | `models/artifacts/*.pkl` | ✅ |
| `models/predictor.py` | `models/inference.py` | ✅ |
| `solver/definitions.py` | `solver/definitions.py` | ✅ |
| `solver/constraints.py` | Merged into `optimizer.py` | ⚠️ |
| `solver/optimizer.py` | `solver/optimizer.py` | ✅ |
| `reports/latex_templates/` | Inline in `latex_generator.py` | ⚠️ |
| `reports/visualizer.py` | `reports/fpl_report/plot_generator.py` | ✅ |
| `main.py` | `reports/generate_fpl_report.py` | ✅ |

---

## Testing Status

| Component | Test Coverage | Notes |
|-----------|--------------|-------|
| ETL Pipeline | ❌ None | Needs unit tests |
| Models | ❌ None | Needs train/inference tests |
| Solver | ❌ None | Needs constraint validation tests |
| Reports | ⚠️ Partial | `tests/test_*.py` exist |

**Recommendation:** Add tests before extending multi-period MIP.

---

## Next Actions (Recommended Order)

1. [ ] **Test current solver** - Run full report with MIP enabled, verify output
2. [ ] **Implement P0.1** - True multi-period variables in solver
3. [ ] **Wire new models** - Replace legacy predictor with position-specific inference
4. [ ] **Add Understat** - Complete data layer before next season
5. [ ] **Add explainability** - "Why This Move?" for user trust

---

## Untapped FPL API Data Sources

The official FPL API exposes rich data that is NOT currently utilized by our pipeline. These represent high-value, low-effort improvements.

### API Endpoints Available

| Endpoint | Data Available | Current Usage |
|----------|---------------|---------------|
| `/api/bootstrap-static/` | Master player/team/event data | ✅ Partial |
| `/api/dream-team/{gw}/` | Team of the Gameweek (11 players) | ❌ Not used |
| `/api/event/{gw}/live/` | Live GW stats with `in_dreamteam` flag | ❌ Not used |
| `/api/leagues-classic/314/standings/` | Top 50 overall managers | ❌ Not used |
| `/api/entry/{id}/event/{gw}/picks/` | Any manager's squad for a GW | ⚠️ Only for user's team |
| `/api/entry/{id}/history/` | Manager's full season history | ⚠️ Only for user's team |

### High-Value Unused Fields in `bootstrap-static/elements`

These fields exist in the API but are NOT currently factored into transfer recommendations:

| Field | Description | Signal Value |
|-------|-------------|--------------|
| `transfers_in_event` | Transfers IN this gameweek | 🔥 **Bandwagon detection** - what the crowd is buying |
| `transfers_out_event` | Transfers OUT this gameweek | 🔥 **Panic sell detection** - who's being dumped |
| `cost_change_event` | Price change this GW | 💰 Value timing signal |
| `cost_change_event_fall` | Price fall this GW | 💰 Buy-the-dip opportunities |
| `dreamteam_count` | Times in Dream Team this season | ⭐ Haul consistency metric |
| `in_dreamteam` | Currently in Dream Team | ⭐ Recent form indicator |
| `news` | Injury/availability text | 🏥 Context for availability flags |
| `news_added` | When news was updated | 🏥 Freshness of availability info |

### Proposed New Features (P1 Priority)

#### 1. Transfer Momentum Signal
**API Source:** `transfers_in_event`, `transfers_out_event`

```python
# Net transfer momentum as % of ownership
transfer_momentum = (transfers_in_event - transfers_out_event) / (selected_by_percent * total_players)
```

**Use Cases:**
- Identify "bandwagon" picks before price rises
- Detect "falling knives" being mass-sold
- Contrarian signal: high transfers out + good fixtures = differential opportunity

**Files to modify:** `reports/fpl_report/transfer_recommender.py`

#### 2. Dream Team Consistency Score
**API Source:** `/api/dream-team/{gw}/`, `dreamteam_count`, `in_dreamteam`

```python
# Dream team appearances as % of games played
dream_team_rate = dreamteam_count / games_played
```

**Use Cases:**
- Identify "haul merchants" - players who score big when they score
- Weight captain picks by Dream Team rate
- Compare against ownership for differential hunting

**Files to modify:** `reports/fpl_report/data_fetcher.py`, `transfer_recommender.py`

#### 3. Top Manager Squad Analysis
**API Source:** `/api/leagues-classic/314/standings/`, `/api/entry/{id}/event/{gw}/picks/`

**Implementation:**
1. Fetch top 50-100 manager entry IDs from overall standings
2. Fetch their current squads via picks endpoint
3. Compute "elite ownership" for each player
4. Compare against general ownership for template detection

**Use Cases:**
- Identify what top 1k managers are doing differently
- "Smart money" signal for transfers
- Detect template players vs punts

**Files to modify:** `reports/fpl_report/data_fetcher.py`, `latex_generator.py`

#### 4. Enhanced Availability Intelligence
**API Source:** `status`, `chance_of_playing_this_round`, `chance_of_playing_next_round`, `news`

**Current Gap:** We use `chance_of_playing_next_round` but ignore:
- The actual `news` text (injury type, expected return)
- `status` flag ('a'=available, 'd'=doubtful, 'i'=injured, 's'=suspended)
- Trend: is availability improving or worsening?

**Files to modify:** `reports/fpl_report/transfer_recommender.py`, `latex_generator.py`

### Implementation Priority

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Transfer Momentum Signal | 0.5 day | HIGH | P1 |
| Enhanced Availability | 0.5 day | HIGH | P1 |
| Dream Team Score | 1 day | MEDIUM | P2 |
| Top Manager Analysis | 1-2 days | HIGH | P1 |

---

## Future Improvements (Roadmap)

### 🔮 Vision Features

These are longer-term ideas that would significantly enhance the bot's capabilities.

#### 1. Screenshot-Based Transfer Analysis
**User Story:** User provides a screenshot of their planned transfers, and the bot analyzes those specific moves.

**Implementation Approach:**
1. Accept image input (screenshot of FPL transfer page)
2. Use OCR or vision model to extract:
   - Players being transferred out
   - Players being transferred in
   - Transfer cost/hits
3. Run xP model on the extracted transfers
4. Generate mini-report comparing current squad vs proposed squad

**Technical Requirements:**
- Vision/OCR capability (Claude vision, Tesseract, or similar)
- Player name fuzzy matching to FPL IDs
- Existing xP prediction pipeline

**Output Format:**
```
📊 Transfer Analysis: Your Planned Moves

OUT: Salah (£13.2m) → IN: Foden (£9.1m)
- 5-GW xP Change: -4.2 pts
- Budget freed: £4.1m
- Ownership delta: -45% → +32%

OUT: Haaland (£15.0m) → IN: Isak (£8.9m)
- 5-GW xP Change: -8.1 pts
- Budget freed: £6.1m
- Fixtures: Isak has better run (avg FDR 2.4 vs 3.1)

VERDICT: Net -12.3 xP, but frees £10.2m for future moves.
Recommendation: Consider keeping Haaland, downgrade elsewhere.
```

**Files to create:** `reports/fpl_report/screenshot_analyzer.py`
**Priority:** P3 (Future)
**Effort:** 2-3 days

#### 2. Natural Language Transfer Queries
**User Story:** User asks "Should I sell Salah for Palmer?" and gets a detailed analysis.

**Implementation:**
- Parse natural language for player names
- Fuzzy match to FPL database
- Run head-to-head xP comparison
- Consider fixtures, form, ownership, price

#### 3. Real-Time Price Change Alerts
**User Story:** Bot monitors price change predictions and alerts before deadline.

**Implementation:**
- Track `transfers_in_event` / `transfers_out_event` velocity
- Predict price rises/falls using FPL Towers algorithm approximation
- Push notification before deadline

#### 4. Historical Decision Audit
**User Story:** "How did my GW10 transfers perform vs what you recommended?"

**Implementation:**
- Store recommendations per GW
- Compare against actual outcomes
- Calculate "recommendation alpha" over time

#### 5. League-Specific Optimization
**User Story:** Optimize transfers specifically to beat rivals in my mini-league.

**Implementation:**
- Analyze rival squads (already have this data)
- Weight differentials higher/lower based on league position
- "Catch up" mode vs "protect lead" mode

#### 6. Chip Timing Advisor
**User Story:** "When should I play my Wildcard/Free Hit?"

**Implementation:**
- Simulate chip usage across remaining GWs
- Account for DGWs, BGWs, fixture swings
- Compare EV of chip now vs holding

---

## Appendix: Key Code References

### MIP Integration Flow
```
generate_fpl_report.py:314-325
    └── TransferStrategyPlanner.generate_strategy(use_mip=True)
            └── transfer_strategy.py:145-154 → _run_mip_solver()
                    └── solver/optimizer.py:TransferMIPSolver.solve()
                            └── Returns mip_result dict
            └── transfer_strategy.py:156-173 → Updates result, clears heuristics
    └── multi_week_strategy passed to compile_report()
            └── latex_generator.py:995-996 → Checks is_mip_optimal
                    └── latex_generator.py:1044 → _generate_mip_recommendation()
```

### Model Training Flow
```
models/train.py:train_all_models()
    └── For each position in [GKP, DEF, MID, FWD]:
        └── PositionModel(position).train(data)
            └── XGBoostRegressor + RidgeCV ensemble
            └── Saves to models/artifacts/model_{position}.pkl
```

### Prediction Flow (Legacy - Still Active)
```
reports/fpl_report/transfer_strategy.py:TransferStrategyPlanner
    └── self.predictor = FPLPointsPredictor(fetcher)  # Legacy!
    └── predictor.predict_multiple_gws(player_ids, 5)
        └── Uses GradientBoosting + RandomForest (not position-specific)
```

### Prediction Flow (New - Not Wired)
```
models/inference.py:FPLInferencePipeline
    └── Loads position-specific models from artifacts/
    └── Generates projections_horizon.parquet
    └── NOT consumed by report pipeline yet
```
