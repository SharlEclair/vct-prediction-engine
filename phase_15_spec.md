# Phase 15: Doubly Robust Vetoes & Synergistic Drafts (v7.3)
**Objective:** Stabilize the Map Veto contextual bandit by implementing Doubly Robust (DR) estimators, and replace the linear Hungarian draft algorithm with a combinatorial engine that accounts for agent synergies.

## 1. Doubly Robust Map Veto Estimator
* **File Target:** `v5_simulation_engine.py` (specifically `MapVetoBandit`) and `veto_predictor.py`.
* **The Flaw:** The current Inverse Propensity Score (IPS) weighting creates unbounded variance if a map's pick propensity is near zero.
* **The Fix:** Implement a Doubly Robust (DR) estimator. 
  * Define a baseline expected map win rate (the direct outcome model $\hat{\mu}$). For simplicity, $\hat{\mu}$ can be the team's overall historical global win rate or a baseline `0.5`.
  * The DR estimator formula: 
    $\text{WinRate}_{DR} = \hat{\mu} + \frac{\text{Empirical Win Rate} - \hat{\mu}}{\text{Propensity Score} + \epsilon}$
  * This ensures that if a map has incredibly low propensity, the estimation smoothly decays back to the baseline $\hat{\mu}$ instead of exploding to infinity. Replace the IPS logic with this DR logic in the veto evaluation step.

## 2. Synergistic Combinatorial Draft Engine
* **File Target:** `v5_simulation_engine.py` (specifically the `HungarianAgentAssigner` class).
* **The Flaw:** The Hungarian algorithm (`linear_sum_assignment`) runs in $O(n^3)$ and enforces independent utility, making it impossible to model synergistic agent combos (e.g., pairing a Duelist with a specific Initiator).
* **The Fix:** * Remove `scipy.optimize.linear_sum_assignment`.
  * Rename the class to `SynergisticDraftEngine`.
  * **Synergy Heuristic:** Introduce a basic pairwise synergy dictionary or modifier (e.g., if the composition has both a Duelist and an Initiator, add a +10% utility synergy bonus; if it lacks a Sentinel, apply a -15% penalty).
  * **Combinatorial Search:** Since a roster is exactly 5 players, iterate through the standard meta-compositions (e.g., 2 Initiator/1 Duelist/1 Controller/1 Sentinel) and assign players greedily or via a quick permutation search that maximizes the *total* composition utility (Base Comfort + Synergy Bonus).