# **Architecture Plan: FPL Analytics Bot 2.0 (The "100x" Upgrade)**

Status: Draft  
Target: State-of-the-art Open Source Stack (2025/26 Standards)  
Goal: Transition from a Heuristic/Regression script collection to a professional-grade Data Warehouse / Ensemble ML / MIP Solver pipeline.

## **1\. Executive Summary: The Algorithmic Paradigm Shift**

The current iteration of the FPL bot operates on a "heuristic" paradigm: it calculates weighted scores based on simple rules of thumb (e.g., "form is 20% of the decision"). While this automates basic decision-making, it fundamentally fails to solve the **Knapsack Problem** inherent in Fantasy Premier League. FPL is not merely about picking the best players; it is a complex optimization challenge involving strict budget constraints, positional quotas, and temporal interdependencies (decisions made today constrain choices five weeks from now).

This plan proposes a complete architectural refactor to align with the "Computational Intelligence" standards observed in top-tier open-source repositories like OpenFPL and open-fpl-solver. We will move from intuition-based scripts to a mathematically rigorous three-layer stack:

1. **Data Layer (The Warehouse):** Moving from ephemeral API calls to a persistent SQL/Parquet warehouse (mirroring olbauday/FPL-Elo-Insights) to handle the massive influx of 2025/26 defensive metrics (CBIT).  
2. **Prediction Layer (The Oracle):** Replacing simple rolling-average regressions with **Position-Specific Ensembles** (XGBoost \+ Random Forest), capable of distinguishing between a high-xG forward and a high-BPS defender.  
3. **Decision Layer (The Solver):** Abandoning transfer heuristics for a **Mixed-Integer Programming (MIP)** solver. This utilizes Operations Research techniques to mathematically guarantee the optimal transfer strategy over a multi-week horizon, accounting for transfer hits, bank rollover, and double gameweeks.

## **2\. Audit of Current Architecture: Why It Fail**

| Component | Current Implementation | Critical Weaknesses & Technical Debt |
| :---- | :---- | :---- |
| **Data Ingestion** | Direct requests.get() calls to FPL API \+ pickle caching. | **Volatility:** Pickled objects are Python-version dependent and break easily. **Blind Spots:** The current pipeline ignores clearances, blocks, interceptions, and tackles (CBIT), which are critical for the 2025/26 BPS rules. **Static FDR:** Uses the FPL "1-5" difficulty scale, which falsely equates a match against 1st place to a match against 5th place. |
| **Modeling** | Simple GradientBoostingRegressor on rolling last-4-GW averages. | **Context Blindness:** A rolling average treats 5 goals against Ipswich the same as 5 goals against Man City. **Homogeneity:** A "One-size-fits-all" model fails to capture positional nuances (e.g., Saves are the primary predictor for GKs, but noise for FWDs). **Availability:** No robust handling of "75% flag" risks. |
| **Transfer Logic** | Weighted Sum Heuristic (score \= 0.2\*form \+ 0.3\*prediction). | **Myopic:** The "Horizon Effect"—the bot makes a move for GW1 without realizing it prevents a critical move in GW3. **Budget Inefficiency:** Heuristics cannot effectively solve "downgrade player A to upgrade player B" combinations simultaneously. **Hit Aversion:** The current logic cannot mathematically justify taking a \-4 hit, whereas a solver can prove the ROI. |
| **Chip Strategy** | Greedy algorithms (e.g., "Free Hit if \< 8 players"). | **Suboptimal:** Does not compare the global Expected Value (EV) of playing a chip now vs. holding it for a future Double Gameweek. |
| **Tech Stack** | Monolithic Python scripts, Pandas, direct LaTeX generation. | **Unscalable:** Adding a new feature (e.g., Understat xG) requires rewriting the entire fetch-predict-recommend chain. |

## **3\. Target Architecture: The "Best-of-Breed" Stack**

This architecture decouples the system into three autonomous layers. This allows you to swap out the prediction engine without breaking the solver, or update the data source without retraining the models.

### **3.1 High-Level Data Flow**

graph TD  
    subgraph "Layer 1: Data Engineering"  
        A\[FPL API\] & B\[Understat/FBref\] & C\[ClubElo\] \--\> D(ETL Pipeline)  
        D \--\>|Clean/Normalize/Join| E\[(Data Warehouse\\nSQLite/Parquet)\]  
    end

    subgraph "Layer 2: Predictive Modeling"  
        E \--\> F{Feature Engineering}  
        F \--\>|Positional Data| G\[Ensemble Trainers\]  
        G \--\>|Model Artifacts| H\[Inference Engine\]  
        H \--\> I\[Projections.csv\\n(PlayerID, GW, xP)\]  
    end

    subgraph "Layer 3: Prescriptive Analytics"  
        I \--\> J{MIP Solver\\n(sasoptpy \+ HiGHS)}  
        J \--\>|Optimize Horizon| K\[Transfer Plan\]  
        K \--\> L\[Report Generator\]  
        L \--\> M\[PDF Report\]  
    end

## **4\. Implementation Plan**

### **Phase 1: Data Infrastructure (The Foundation)**

**Objective:** Build a robust "Truth Source" that persists across runs and enables complex queries.

* **Architecture Shift:** Replace getters.py with an **ETL (Extract, Transform, Load)** pipeline.  
* **The "CBIT" Upgrade:**  
  * **Problem:** The 2025/26 season introduced points for Blocks and Tackles. Legacy datasets (like vaastav) lack this.  
  * **Solution:** Ingest match-level event data. We must compute defensive\_workrate metrics per player to accurately value defensive midfielders (CDMs) who were previously FPL-irrelevant.  
* **Dynamic Difficulty (Elo):**  
  * **Integration:** Ingest daily CSVs from ClubElo.com.  
  * **Transformation:** Map Club names to FPL Team IDs.  
  * **Math:** Calculate exact Win Probabilities. $P(Win) \= \\frac{1}{1 \+ 10^{(Elo\_{Opponent} \- Elo\_{Team})/400}}$.  
  * **Impact:** The model will now "know" that playing Man City (Elo \~2050) away is 4x harder than playing Southampton (Elo \~1600) at home.  
* **Entity Resolution:**  
  * Create mappings/player\_id\_map.csv to bridge the gap between "Bruno Fernandes" (FPL) and "B. Borges Fernandes" (Understat). This prevents data loss for players with naming inconsistencies.

### **Phase 2: Predictive Engine (The "xP" Layer)**

**Objective:** Generate highly accurate Expected Points (xP) using state-of-the-art Machine Learning.

* **Architecture Shift:** Move from a single model to **Position-Specific Ensembles** (inspired by OpenFPL).  
* **Why Ensembles?**  
  * **XGBoost (Gradient Boosting):** Excellent at capturing non-linear interactions (e.g., *Player is a Forward* AND *Opponent Defense is Weak* \= High Goal Probability).  
  * **Random Forest:** Excellent at reducing variance and preventing overfitting on noisy data.  
  * **Strategy:** Train both models for each position and average their outputs (Stacking).  
* **Feature Engineering by Position:**  
  * **GK Model:** Inputs: xG\_Against\_Team, Saves\_Per\_90, Clean\_Sheet\_Odds.  
  * **DEF Model:** Inputs: xGI (Goal Involvement), CBIT\_Score (New\!), Clean\_Sheet\_Odds.  
  * **MID/FWD Models:** Inputs: xG, xA, xGChain (Build-up play), Key\_Passes, Opponent\_xGA.  
* **Handling "xMins":**  
  * Use the chance\_of\_playing\_next\_round flag (0, 25, 50, 75, 100).  
  * **Logic:** $xP\_{final} \= xP\_{model} \\times (\\text{Chance} / 100\) \\times \\text{Substitution\_Risk\_Factor}$.

### **Phase 3: Prescriptive Analytics (The Solver)**

**Objective:** Solve the Multi-Period Optimization problem to generate the optimal transfer strategy.

* **Architecture Shift:** Replace TransferRecommender with a **Mixed-Integer Programming (MIP)** solver.  
* **Technical Stack:**  
  * **Modeling Language:** sasoptpy (Python wrapper for creating optimization models).  
  * **Solver Engine:** HiGHS (High-performance open-source linear optimization solver).  
* The Mathematical Formulation:  
  We define the problem as maximizing total weighted points over a horizon $H$ (e.g., 5-8 weeks).  
  $$ \\text{Maximize } Z \= \\sum\_{w=1}^{H} \\sum\_{p \\in Players} (xP\_{p,w} \\cdot \\text{lineup}\_{p,w} \\cdot \\gamma^{w-1}) \+ (\\text{FT\_Value} \\cdot \\text{SavedFT}\_w) \- \\text{TransferCost} $$  
  **Key Constraints:**  
  1. **Budget:** $\\sum \\text{Cost}\_{p} \\le \\text{Bank}\_w \+ \\text{SellPrice}\_{out}$.  
  2. **Squad Size:** Exactly 15 players.  
  3. **Team Limits:** Max 3 players from any single Premier League club.  
  4. **Formation:** Valid lineups only (e.g., Min 3 Defenders, Min 1 Forward).  
  5. **Transfer Continuity:** The squad in Week $W$ must equal the squad in Week $W-1$ plus transfers in minus transfers out.  
* **Capabilities:**  
  * **Multi-Period Planning:** The solver sees the future. It will hold a transfer in GW3 to enable a double-swap in GW4 without a hit.  
  * **Optimal Bench:** It selects the cheapest possible bench fodder that *might* play, maximizing the budget available for the starting XI.  
  * **Sensitivity Analysis:** By adjusting the FT\_Value (e.g., valuing a saved transfer at 2 points vs 4 points), we can toggle the bot between "Aggressive" and "Conservative" modes.

### **Phase 4: Strategy & Reporting**

**Objective:** Translate complex solver outputs into actionable, readable intelligence.

* **Architecture Shift:** From static data dumps to **Explainable AI**.  
* **New Reporting Features:**  
  * **"Why This Move?":** Explicitly calculate the ROI. *"Selling Salah for Palmer generates \+4.2 xP over the next 3 weeks, paying off the \-4 hit in GW2."*  
  * **Effective Ownership (EO):** Differentiate between "Blockers" (High EO players you own to protect rank) and "Differentials" (Low EO players you buy to gain rank).  
  * **Captaincy Matrix:** Instead of just picking the highest xP, simulate outcomes using Gamma distributions to show "Haul Probability." *Is it better to captain the safe Haaland (6.5 xP) or the risky Salah (6.0 xP but higher ceiling)?*

## **5\. Technical Requirements Checklist**

### **Core Libraries**

* **Data:** pandas, numpy, duckdb (for SQL-on-files).  
* **Machine Learning:** xgboost, scikit-learn (RandomForest, LinearRegression).  
* **Optimization:** sasoptpy (modeling), highspy (solver interface).  
* **Utilities:** requests, beautifulsoup4 (scraping).

### **Refactored File Structure**

/  
├── data/  
│   ├── raw/                 \# Raw JSON responses (archived by GW)  
│   ├── parquet/             \# High-performance storage for training data  
│   └── mappings/            \# CSVs linking FPL IDs to Understat/ClubElo  
├── etl/  
│   ├── fetchers.py          \# Scrapers  
│   └── pipeline.py          \# Cleaning & Joining logic  
├── models/  
│   ├── training/            \# Scripts to retrain models  
│   ├── artifacts/           \# Saved .json/.pkl model files (per position)  
│   └── predictor.py         \# Inference engine (loads models \-\> outputs projections)  
├── solver/  
│   ├── definitions.py       \# MIP Variable definitions  
│   ├── constraints.py       \# Budget, Team, Formation rules  
│   └── optimizer.py         \# Main solver loop  
├── reports/  
│   ├── latex\_templates/     \# .tex files  
│   └── visualizer.py        \# Matplotlib/Seaborn plot generators  
└── main.py                  \# CLI Orchestrator

## **6\. Migration Steps & Roadmap**

1. **Immediate Action (The Low Hanging Fruit):** Implement the **Solver** first. Even using your *current* heuristic predictions, the MIP solver will manage your budget and transfer planning infinitely better than the current weighted-sum scripts.  
2. **Data Warehouse (Parallel Track):** Begin scraping and archiving data *now*. You cannot train ML models without historical data. Start building the Parquet archive for the 2025/26 season.  
3. **ML Upgrade (Long Term):** Once you have \~10 GWs of clean data in your warehouse, train the new XGBoost ensembles to replace the rolling-average regression.  
4. **Reporting Polish:** Finally, update the LaTeX generator to visualize the solver's "Optimal Plan" (e.g., a Gantt chart of transfers over the next 5 weeks).


## **7\. Works Cited & Resources**

### **Key Open Source Projects**

1. OpenFPL-Scout-AI  
   Inspiration for position-specific Ensemble Machine Learning models.  
   https://github.com/elcaiseri/OpenFPL-Scout-AI  
2. open-fpl-solver (Solio Analytics)  
   The optimization framework and solver implementation guides.  
   https://github.com/solioanalytics/open-fpl-solver  
3. FPL-Elo-Insights  
   Reference for robust data warehousing and Elo integration.  
   https://github.com/olbauday/FPL-Elo-Insights

### **Data Sources**

4. ClubElo  
   Source for dynamic team strength and win probabilities.  
   http://clubelo.com/  
5. Understat  
   Source for Expected Goals (xG) and Expected Assists (xA).  
   https://understat.com/  
6. FBref  
   Comprehensive football statistics and player metrics.  
   https://fbref.com/en/

### **Core Libraries**

7. SAS Optimization (sasoptpy)  
   Modeling language for the Mixed-Integer Programming solver.  
   https://github.com/sassoftware/sasoptpy  
8. HiGHS Solver  
   High-performance open-source linear optimization engine.  
   https://highs.dev/
