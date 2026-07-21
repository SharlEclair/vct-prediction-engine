# Phase 13: Variance Unleashed (v7.1)
**Objective:** Resolve the fatal statistical flaws in DFS ceiling generation by replacing the zero-sum Dirichlet distribution with a positive-covariance Copula framework, and removing artificial EV clamping.

## 1. Remove Erroneous Clamping & Winsorization
* **File Targets:** `model_training.py` and `feature_engineering.py`.
* **The Flaw:** GPP tournaments are won on the extreme right tail. Winsorizing kills per round (KPR) and clamping final DFS points between `15.0` and `80.0` truncates the model's ability to see "slate-breaking" performances.
* **The Fix:** 
  * Remove the aggressive Winsorization on target variables in `feature_engineering.py`.
  * In `model_training.py`, remove the `np.clip(..., 15.0, 80.0)` logic from the final points projection. Allow the XGBoost regressor to freely predict natural extreme outliers.

## 2. Replace Dirichlet with Copula-Based Covariance
* **File Target:** `v5_simulation_engine.py` (specifically `sample_deaths`, `sample_assists`, and the KDA distribution logic).
* **The Flaw:** The Dirichlet distribution dictates that $\text{Cov}(X_i, X_j) < 0$. If a Duelist gets many kills, the model forces the Initiator's assists to decrease. In Valorant, kills and assists share a strong *positive* correlation (synergy).
* **The Fix:** Implement a Copula-based approach using shared latent momentum.
  * **Step 1:** For each team in a simulated map, draw a shared "Team Momentum" scalar from a right-skewed distribution (e.g., Gumbel or Log-Normal).
  * **Step 2:** Define independent heavy-tailed marginal distributions for each player's Kills, Deaths, and Assists (e.g., Gamma distribution) based on their role and baseline ACS.
  * **Step 3:** Couple them. Use the shared Team Momentum scalar to shift the percentile (CDF) of the individual Gamma marginals upward or downward simultaneously. This ensures that when a team "pops off", multiple players can access the extreme right tail of their individual distributions simultaneously, creating mathematically sound, highly correlated DFS ceilings.