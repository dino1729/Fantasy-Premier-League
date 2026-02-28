# Bug Fixes (Solver + Report Wiring)

This document captures technical issues found in the solver layer and how they impact
prediction accuracy and report generation. It is a review artifact; no code changes
are included here.

## Scope

Files reviewed:
- solver/definitions.py
- solver/optimizer.py
- solver/interpreter.py
- reports/fpl_report/transfer_strategy.py
- generate_fpl_report.py
- reports/fpl_report/latex_generator.py
- main.py

## Findings

### 1) Multi-period budget constraint reuses the initial bank (infeasible sequences)

Location:
- solver/optimizer.py:974

What happens:
- The multi-period solver applies this weekly constraint:
  - buy_cost <= bank + sell_revenue
- It does not track rolling bank across weeks or the cumulative effect of previous
  transfers. This means the model can spend the same initial bank repeatedly in
  later weeks.

Why it is a bug:
- The solver can propose transfer sequences that are infeasible in real FPL finances
  (e.g., buying an expensive player in multiple weeks without having the funds).
- This inflates expected_points because the optimizer assumes more budget than is
  actually available.

Impact:
- Over-optimistic transfer plans.
- Potentially impossible sequences shown in the report.

Detailed fix:
1) Add a per-week bank state variable (integer/continuous, lb=0):
   - bank[w] represents funds available before transfers for week w.
2) Initialize bank at week 0:
   - bank[0] == self.bank
3) Add flow constraints to carry bank forward:
   - bank[w+1] == bank[w]
     + sum(sell_price[p] * tout[p,w])
     - sum(buy_price[p] * tin[p,w])
4) Enforce affordability per week using bank[w]:
   - sum(buy_price[p] * tin[p,w]) <= bank[w] + sum(sell_price[p] * tout[p,w])
5) Remove the current per-week constraint that reuses the initial bank.
6) Optional (if price_trajectory is modeled):
   - Use sell_price[w] and buy_price[w] per week instead of static prices.

---

### 2) Candidate pool mismatch causes missing predictions to be treated as 0 xP

Locations:
- reports/fpl_report/transfer_strategy.py:301
- solver/optimizer.py:706

What happens:
- Predictions are generated for `all_ids = squad_ids + top_players` where top_players
  is built as the global top (candidate_pool * 4) by total_points.
- The solver builds its candidate pool by selecting the top N per position.
- Some candidates in the solver pool are not in all_ids, so they lack predictions.
  The xP matrix fills these with zeros.

Why it is a bug:
- A subset of solver candidates are systematically penalized with 0 xP, even though
  they are legitimate candidates. This distorts the optimization, and can block
  otherwise optimal transfers.

Impact:
- Biased transfer recommendations.
- Lower-quality lineups when missing predictions overlap strong positional options.

Detailed fix:
1) Make candidate selection deterministic and shared:
   - Extract a helper that computes candidate_ids using the same filters as
     the solver (status == 'a', minutes >= 90, top N per position).
2) Build predictions for exactly those candidate_ids plus current squad:
   - all_ids = current_squad_ids U candidate_ids
   - predictions = _predict_multiple_gws(all_ids, num_weeks)
3) Ensure MultiPeriodMIPSolver uses the same candidate_ids:
   - Option A: add a candidate_ids param to MultiPeriodMIPSolver and filter
     the players_df inside _build_candidate_pool.
   - Option B: keep solver logic but guarantee prediction coverage by using
     the same selection logic in TransferStrategyPlanner.
4) Add a guardrail:
   - If any candidate has no predictions, log a warning and exclude it from
     the candidate pool (or compute a fallback prediction).

---

### 3) Baseline xP can violate formation constraints (inflated baseline_xp)

Location:
- solver/optimizer.py:825

What happens:
- `_calculate_week_xp_for_squad` first picks the top 11 by xP with only max-pos
  constraints, then tries to add missing positions to satisfy minimums.
- If it adds players after already reaching 11, it still slices `xi[:11]` when
  computing xP. The appended players may be dropped, leaving a lineup that still
  violates minimum requirements.

Why it is a bug:
- The baseline_xp can be overestimated because it may include a too-strong,
  invalid formation. This affects the net gain comparison used to recommend the
  best scenario.

Impact:
- Scenario comparison table may show incorrect xP gains.
- Conservative path could appear worse/better incorrectly.

Detailed fix:
1) Build the XI in two phases:
   - Phase A (minima): select the highest-xP players for each position to
     satisfy XI_MIN (1 GKP, 3 DEF, 2 MID, 1 FWD).
   - Phase B (fill): fill remaining slots up to 11 with the highest-xP
     remaining players while respecting XI_MAX.
2) Ensure the final XI always contains exactly 11 and respects all position
   constraints before computing xP.
3) Add a fallback warning if minima cannot be satisfied (e.g., missing data).
4) Use the same XI builder in _calculate_baseline_xp to keep comparisons valid.

---

### 4) Single-period solver reports per-GW xP and total xP using week-0 XI only

Locations:
- solver/optimizer.py:341
- solver/optimizer.py:437

What happens:
- The model optimizes lineup and captain decisions for each week, but the result
  extraction computes per_gw_xp using only the week-0 starting_xi.
- It also sums per_gw_xp without discounting, while the objective uses discounting.

Why it is a bug:
- The reported per_gw_xp does not match the optimized weekly lineups.
- expected_points can diverge from the solver objective because it ignores the
  discount factor used during optimization.

Impact:
- Mismatch between what is optimized and what is reported (confusing MIP output).
- Inconsistent numbers in the report and console output.

Detailed fix:
1) During solution extraction, reconstruct week-specific lineups:
   - For each w, use lineup[p,w] and captain[p,w] to build a week_lineup list
     and week_captain.
2) Compute per_gw_xp from the week_lineup (not week 0 only):
   - week_xp = sum(xp[p,w]) + captain_xp (captain counted twice)
3) Recompute expected_points consistently with the objective:
   - expected_points = sum((discount_factor ** w) * per_gw_xp[w]) - hit_cost
4) Keep starting_xi, bench, captain, vice_captain for week 0 only (as a summary),
   but store per_gw_xp from all weeks for report accuracy.

---

### 5) Free transfer banking not modeled (hit cost mis-estimation)

Location:
- solver/optimizer.py:1023

What happens:
- ft_avail is set to free_transfers for week 0, then forced to 1 for all later
  weeks. There is no state tracking for banked FTs.

Why it is a bug:
- In FPL, if no transfers are made, free transfers can be banked up to 5.
- The solver understates available FTs and overstates hit costs for delayed
  transfers.

Impact:
- Suboptimal timing recommendations.
- Weekly plan hit costs and summary numbers can be wrong.

Detailed fix:
1) Add an integer ft_avail[w] variable (range 1..5) and set:
   - ft_avail[0] == min(5, self.free_transfers)
2) Track transfers_used[w] = sum(tin[p,w]).
3) Model FT banking with a capped recurrence:
   - ft_raw[w] = ft_avail[w-1] - transfers_used[w-1] + 1
   - ft_avail[w] = min(5, ft_raw[w]) (linearized)
4) Example linearization (Big-M, M=5):
   - ft_avail[w] <= ft_raw[w]
   - ft_avail[w] <= 5
   - ft_avail[w] >= ft_raw[w] - M * cap[w]
   - ft_avail[w] >= 5 - M * (1 - cap[w])
   - cap[w] in {0,1}
5) Update hit constraint using ft_avail[w]:
   - hits[w] >= HIT_COST * (transfers_used[w] - ft_avail[w])
6) Replace the current fixed ft_avail logic (week 0 only).

---

### 6) Bench Boost / Triple Captain projections read the wrong prediction key (and TC can crash)

Locations:
- reports/fpl_report/transfer_strategy.py:1544
- reports/fpl_report/transfer_strategy.py:1652
- reports/fpl_report/transfer_strategy.py:1667

What happens:
- Both BB and TC projections read `player_pred.get('gw_predictions', [])`, but the
  predictor outputs use the key `predictions`.
- In TC projections, `base_pts` is referenced even when the prediction list is
  missing, which can raise `UnboundLocalError`.

Why it is a bug:
- BB/TC projections become zero or defaulted, producing misleading chip advice.
- TC projections can crash if a player has no predictions (base_pts undefined).

Impact:
- Chip projections in the report are wrong or fail entirely.

Detailed fix:
1) Replace `gw_predictions` with `predictions` in both BB and TC sections.
2) Ensure `base_pts` is initialized in all branches:
   - If no prediction available, set `base_pts = default_value` before use.
3) Add a small fallback when predictions are missing:
   - Use `ep_next` (if available) or a conservative default (e.g., 4.0).
4) Add a unit test to confirm both BB and TC paths accept missing predictions.

---

### 7) Free Hit EV baseline undercounts when predictions are only built for top 200 players

Locations:
- generate_fpl_report.py:555
- generate_fpl_report.py:917
- generate_fpl_report.py:960

What happens:
- `all_player_predictions` is built only for a top-200 subset.
- Current XI and Free Hit XI xP sums ignore players not in that subset (treated as 0).

Why it is a bug:
- EV comparisons can be significantly understated, especially for lower-owned or
  budget players not in the top-200 list.

Impact:
- Free Hit EV sections in the report are biased downward.

Detailed fix:
1) Build predictions for all IDs needed for EV:
   - current_xi_ids U fh_candidate_ids (or at least current_xi_ids + chosen XI).
2) If prediction budget is a concern, lazily extend `all_player_predictions`:
   - If a pid is missing, predict just that pid and add to the dict.
3) Add a warning if any XI player lacks predictions after the fix.

---

### 8) Current squad EV sums all 15 players instead of best XI (inflated EV)

Location:
- reports/fpl_report/transfer_strategy.py:974

What happens:
- `_calculate_squad_ev` sums cumulative predictions for every player in the squad,
  which counts bench points that should not be included in XI-only scoring.

Why it is a bug:
- EV (current_squad) and derived `potential_gain` are inflated in the report.

Impact:
- Misleading EV analysis; transfers can appear less beneficial than they are.

Detailed fix:
1) For each GW in the horizon, build the best XI under formation constraints:
   - 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD.
2) Sum XI xP + captain bonus for each GW, then aggregate over horizon.
3) Replace the current sum-of-all-players approach with the XI-based total.

---

### 9) ROI baseline ignores XI max constraints (inflated ROI)

Location:
- reports/analytics.py:149

What happens:
- `_select_best_11` enforces minimums but not max caps (e.g., can select 6 DEF).

Why it is a bug:
- Baseline_xp and optimal_xp can exceed legal formation limits, inflating ROI.

Impact:
- ROI analysis in CLI report can misstate gains.

Detailed fix:
1) Enforce XI_MAX caps during selection (GKP=1, DEF<=5, MID<=5, FWD<=3).
2) After meeting minima, fill remaining slots by xP but block any position that
   would exceed its max.
3) Add tests for lineups that would otherwise violate max caps.

---

### 10) MIP CLI report shows zero net gain (missing baseline/net gain in interpreter)

Locations:
- solver/interpreter.py:198
- reports/strategy_reporter.py:133

What happens:
- `SolverInterpreter.interpret_mip_result` does not set `baseline_expected` or
  `net_expected_gain`. StrategyReporter prints 0.0 for net gain.

Why it is a bug:
- The CLI report misrepresents the expected impact of MIP recommendations.

Impact:
- Confusing and incorrect summary in `print_strategy_report`.

Detailed fix:
1) If `mip_result` contains `baseline_xp`, set:
   - plan.baseline_expected = baseline_xp
   - plan.net_expected_gain = plan.total_expected_points - baseline_xp
2) Otherwise, compute baseline via ROICalculator or a helper in the interpreter.
3) Update StrategyReporter to display baseline when available.

---

### 11) Historical reports leak current-GW context in FPLPointsPredictor

Locations:
- reports/fpl_report/predictor.py:525
- reports/fpl_report/transfer_strategy.py:113

What happens:
- `FPLPointsPredictor.predict_multiple_gws` always uses
  `fetcher.get_current_gameweek()` rather than the report’s target GW.

Why it is a bug:
- When generating past reports, predictions are built using current fixtures
  and recent history, leaking future data.

Impact:
- Historical reports are not backtest-safe.

Detailed fix:
1) Add an optional `current_gw` parameter to `predict_multiple_gws`.
2) Thread `current_gw` through TransferStrategyPlanner and callers.
3) Default to fetcher.get_current_gameweek() only when `current_gw` is None.

---

### 12) Free Hit fixture context can ignore blanks and over-score non-playing teams

Locations:
- reports/fpl_report/transfer_strategy.py:2408
- reports/fpl_report/transfer_strategy.py:2501

What happens:
- `_build_fixture_map` selects the first fixture with `gw >= target_gw`. If a team
  is blanking in the target GW, it will use the next available fixture instead.
- `_calculate_score` then treats that fixture as the target GW fixture.

Why it is a bug:
- Players with no fixture in the target GW receive positive fixture bonuses and
  can be selected for Free Hit squads, inflating expected points.

Impact:
- Free Hit recommendations can include blanking players.
- EV comparisons are biased upward.

Detailed fix:
1) Build fixture context strictly for the target GW:
   - If no fixture for target_gw, store a sentinel (e.g., `None` or `is_blank=True`).
2) In `_calculate_score`, if target fixture is missing:
   - Set base_score to 0 (or a minimal value) and skip fixture bonuses.
3) For multi-fixture context (DGW), allow multiple fixtures only for the target GW,
   not beyond it.

---

### 13) Free Hit scoring treats zero predictions as “missing” and falls back to ep_next

Location:
- reports/fpl_report/transfer_strategy.py:2475

What happens:
- If the prediction value is 0 (valid for blanks), `base_score <= 0` triggers
  fallback to `ep_next`, which is for the next GW, not the target GW.

Why it is a bug:
- A blank in the target GW can be scored as if the player has a normal fixture.

Impact:
- Overestimates target GW expected points.

Detailed fix:
1) Distinguish “missing prediction” from “prediction value 0”:
   - Use a sentinel (`None`) or a separate boolean flag when predictions exist.
2) Only fallback to `ep_next` when predictions are missing entirely.
3) If target_gw is beyond prediction horizon, prefer a conservative 0/low
   estimate rather than `ep_next`.

---

### 14) FPL Core predictor uses team_code instead of team_id (fixture cache miss)

Locations:
- reports/fpl_report/fpl_core_predictor.py:2159
- reports/fpl_report/fpl_core_predictor.py:525

What happens:
- `predict_multiple_gws` uses `team_code` from the Core players table as the
  fixture lookup key.
- The fixture cache is keyed by `home_team/away_team` IDs (FPL team IDs).

Why it is a bug:
- If `team_code` != `team_id`, fixture context defaults to neutral values, and
  Elo/FDR features are effectively ignored.

Impact:
- Degraded prediction accuracy, especially for fixture-heavy reasoning.

Detailed fix:
1) Build a mapping between team_code and team_id using the Core teams table:
   - `teams_df` should expose both `id` and `code` (or equivalent).
2) Convert player team_code to the corresponding team_id before calling
   `_get_fixture_context`.
3) If no mapping is found, log a warning and fall back to neutral context.

---

### 15) Transfer recommender drops predictions arbitrarily (set truncation)

Location:
- reports/fpl_report/transfer_recommender.py:366

What happens:
- Candidate IDs are deduped via `set`, then truncated to 200 using list slicing.
- Set ordering is non-deterministic, so the truncated list changes run-to-run.

Why it is a bug:
- Some candidates appear with `predicted_points=0` simply because they were
  excluded from the prediction batch.

Impact:
- Inconsistent transfer recommendations; scoring can be biased by missing preds.

Detailed fix:
1) Use a deterministic ordering before truncation:
   - Sort by `total_points`, `form`, or `minutes` (descending).
2) Or, keep the full candidate list but only score candidates that have
   predictions available.
3) Add a debug warning when candidates are dropped due to the cap.

---

### 16) Wildcard EV undercounts when predictions are incomplete

Locations:
- generate_fpl_report.py:813
- reports/fpl_report/transfer_strategy.py:2213

What happens:
- Wildcard EV sums `xp_5gw` for the starting XI.
- If a player has no prediction entry, `xp_5gw` is set to 0.

Why it is a bug:
- The wildcard EV output is biased downward when predictions are limited to a
  top-200 subset or otherwise incomplete.

Impact:
- Misleading “5-GW xP” comparisons in the report.

Detailed fix:
1) Ensure predictions are built for all players selected in the wildcard squad:
   - Extend prediction IDs to include wildcard XI player IDs.
2) If predictions are missing, use a fallback estimate (e.g., score-based proxy)
   instead of 0.
3) Add a warning when EV includes fallback values.

---

### 17) Simulation lineup builders ignore formation max caps

Locations:
- simulation/engine.py:609
- simulation/baseline.py:160

What happens:
- Lineup selection enforces minimums but does not cap positions (e.g., can pick
  6+ defenders).

Why it is a bug:
- Simulated lineups can be invalid under FPL rules, inflating xP and points.

Impact:
- Backtests may overstate strategy performance.

Detailed fix:
1) Enforce XI_MAX caps during lineup fill.
2) After selecting minima, only allow positions that have not reached their max.
3) Add validation checks to log/flag invalid formations in simulation outputs.

---

### 18) chance_of_playing=0 is coerced to 100% (availability leak)

Location:
- etl/transformers.py:164

What happens:
- `chance_playing` is set via `elem.get('chance_of_playing_next_round') or 100`.
- When the API returns 0, the `or` fallback turns it into 100.

Why it is a bug:
- Injured/suspended players with 0% chance are treated as fully available.

Impact:
- xP projections overestimate availability.
- Downstream predictions and EV metrics are inflated.

Detailed fix:
1) Replace the `or 100` fallback with an explicit `is None` check:
   - `chance = elem.get('chance_of_playing_next_round')`
   - `chance_playing = 100 if chance is None else int(chance)`
2) Add a test to ensure 0 stays 0 and None becomes 100.

---

### 19) Availability is applied twice in the heuristic ETL path

Locations:
- etl/pipeline.py:173
- etl/transformers.py:372

What happens:
- HeuristicAdapter already multiplies xP by availability.
- ProjectionTransformer multiplies by `chance_playing` again.

Why it is a bug:
- Players with 75% chance are effectively weighted by 0.75 * 0.75.

Impact:
- Systematic underestimation of xP for flagged players.

Detailed fix:
1) Decide a single place to apply availability:
   - Option A: Keep it in HeuristicAdapter and remove the adjustment in ProjectionTransformer.
   - Option B: Remove it from HeuristicAdapter and keep it in ProjectionTransformer.
2) Document the contract: predictions should be “raw” or “availability-adjusted,”
   then enforce consistently across pipelines.
3) Add a regression test for a player with chance=75.

---

### 20) Fixture lookup overwrites DGWs (multi-fixture weeks ignored)

Locations:
- etl/pipeline.py:107
- models/feature_engineering.py:118
- reports/fpl_report/fpl_core_predictor.py:525

What happens:
- Fixture lookups are stored as `(team_id, gw) -> fixture_info`.
- In a DGW, the later fixture overwrites the earlier one.

Why it is a bug:
- DGW expected points are undercounted.
- Fixture difficulty and Elo features reflect only one opponent.

Impact:
- Predictions and strategy logic under-value DGW players.

Detailed fix:
1) Store fixtures per (team_id, gw) as a list rather than a single dict.
2) For prediction features, aggregate multiple fixtures:
   - Use weighted averages for FDR/Elo.
   - Multiply base xP by fixture count or sum per-fixture predictions.
3) Update any callers that assume a single fixture object.

---

### 21) FeatureEngineer loads player history from a hardcoded season path

Location:
- models/feature_engineering.py:176

What happens:
- The history path is fixed to `data/2025-26/players`.

Why it is a bug:
- Running the pipeline for any other season fails to load history,
  resulting in empty features and fallback predictions.

Impact:
- Inference accuracy collapses outside the 2025-26 season.

Detailed fix:
1) Pass a `season` or `season_path` into FeatureEngineer.
2) Build the history directory using that season value.
3) Add a fallback to `config.yml` if no season is provided.

---

### 22) Inference pipeline defaults current_gw to 17 when no fixtures are finished

Location:
- models/inference.py:98

What happens:
- If there are no finished fixtures, `current_gw` is hardcoded to 17.

Why it is a bug:
- Early-season predictions are shifted by ~17 weeks, making horizons invalid.

Impact:
- Projections are generated for the wrong gameweeks.

Detailed fix:
1) Use a safer default:
   - `current_gw = 0` or `1` if no finished fixtures exist.
2) Optionally read the current GW from the FPL API bootstrap if available.
3) Log a warning when falling back to the default.

---

### 23) New inference pipeline is initialized but never used

Locations:
- reports/fpl_report/transfer_recommender.py:110
- reports/fpl_report/transfer_strategy.py:67

What happens:
- `FPLInferencePipeline` is created when `use_new_models=True`.
- No downstream logic actually calls it for predictions.

Why it is a bug:
- The “new models” flag has no effect; predictions still come from the older
  predictor implementations.

Impact:
- Users believe they are using new models when they are not.

Detailed fix:
1) Add a prediction path that uses the inference pipeline when available.
2) Define a consistent output format (predictions + cumulative) to match existing
   predictor contracts.
3) Add a log line indicating which predictor is used for each prediction call.

---

## Report Wiring (Verification)

Flow:
- generate_fpl_report.py calls TransferStrategyPlanner.generate_strategy() and
  passes the resulting dict into LaTeXReportGenerator.compile_report().
- reports/fpl_report/latex_generator.py reads multi_week_strategy and renders
  either a MIP dashboard (if optimal) or heuristic fallback.

Status:
- The MIP results are wired into the report path correctly. Any inconsistencies
  in the report are due to the solver issues described above, not wiring gaps.

Relevant entry points:
- generate_fpl_report.py:469
- reports/fpl_report/transfer_strategy.py:210
- reports/fpl_report/latex_generator.py:1853

## Notes

- The above items are functional correctness issues. Performance and modeling
  improvements (e.g., candidate pool heuristics, price prediction integration)
  are out of scope here.
