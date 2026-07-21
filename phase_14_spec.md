# Phase 14: Bayesian Skill Tracking (v7.2)
**Objective:** Replace sluggish Exponential Moving Averages (EMAs) with a dynamic, state-space Bayesian updating framework to instantly adapt to meta-shifts and quantify player uncertainty.

## 1. The Bayesian Tracker Class
* **Action:** In `feature_engineering.py` (or a new `utils/bayesian_tracker.py`), implement a `BayesianSkillTracker` class.
* **Math/Logic (1D Kalman Filter Update):**
  * Initialize players with a prior: $\mu_0$ (baseline expected stat, e.g., KPR = 0.75, ACS = 200) and $\sigma_0^2$ (initial uncertainty variance).
  * **Observation Update:** When a player plays a match, they generate an observation $y$ (their actual match KPR/ACS) with assumed measurement noise $\sigma_{obs}^2$.
    * Posterior Mean: $\mu_{new} = \frac{\mu_{old} \cdot \sigma_{obs}^2 + y \cdot \sigma_{old}^2}{\sigma_{old}^2 + \sigma_{obs}^2}$
    * Posterior Variance: $\sigma_{new}^2 = \frac{\sigma_{old}^2 \cdot \sigma_{obs}^2}{\sigma_{old}^2 + \sigma_{obs}^2}$
  * **Time Transition (Dynamics Factor):** Between matches, uncertainty grows due to meta-shifts or time passing. Apply a dynamics factor $\tau^2$:
    * $\sigma_{next}^2 = \sigma_{new}^2 + \tau^2$

## 2. Feature Engineering Overhaul
* **File Target:** `feature_engineering.py`
* **Action:** Remove the `compute_player_ema` functions entirely. 
* **Integration:** Modify the historical match processing loop to iterate strictly chronologically. For each match:
  1. Extract the *pre-match* $\mu$ and $\sigma$ for each player (these become the predictive features for this match).
  2. Run the Bayesian Observation Update using the match results.
  3. Apply the Time Transition $\tau^2$ for the next match.
* **Output:** The feature matrix (`X_features.csv`) must now contain columns like `player_kpr_mu`, `player_kpr_sigma`, `player_acs_mu`, and `player_acs_sigma` instead of EMA features.

## 3. Machine Learning Model Refactor
* **File Target:** `model_training.py`
* **Action:** Update the `feature_cols` list for the XGBoost regressor. Replace the old EMA columns (e.g., `prev_ema_kpr_alpha_0.1`, `prev_ema_kpr_alpha_0.4`) with the new `player_kpr_mu` and `player_kpr_sigma` features.
* **Integration:** Ensure the DFS slate prediction generation extracts the latest active $\mu$ and $\sigma$ states from the Bayesian Tracker ledger for the active players.