# Phase 1: Top-Down Infrastructure and Feature Engineering
**Objective:** Bypass the DAG entirely to establish a stable Expected Value baseline ($\mu_{TD}$) utilizing historical telemetry, Winsorization, Exponential Moving Averages, and Opponent Defensive Ratings. The final XGBoost regressor must outperform the naive Mean Absolute Error (MAE) baseline of 4.37.

## Task 1.1: Target Variable Generation
* **Data Ingestion:** Process `/v2/match/details` JSON telemetry.
* **Extraction:** Isolate raw target metrics: Kills, Deaths, Assists, and First Bloods for each player.
* **Transformation:** Transform absolute integers into rate metrics to decouple the data from match length variance. Focus primarily on Kills Per Round (KPR).
    * $KPR = \frac{Total Kills}{Rounds Played}$

## Task 1.2: Baseline Clipping (Winsorization)
* **Objective:** Mitigate extreme structural outliers (e.g., 13-0 sweeps or 18-16 overtimes).
* **Execution:** Apply statistical Winsorization to the historical KPR rate metrics.
* **Parameters:** Bound the data strictly between the 5th and 95th percentiles of the global historical dataset.

## Task 1.3: Exponential Moving Average (EMA) Construction
* **Objective:** Capture rapid meta-shifts and player form non-stationarity.
* **Formula:** Implement the recursive EMA calculation:
    $$EMA_{t} = \alpha \cdot X_{t} + (1 - \alpha) \cdot EMA_{t-1}$$
* **Execution:** Generate multiple temporal windows for the target metrics based on chronological match dates.
    * Slow Decay (Long-term form): $\alpha = 0.1$
    * Rapid Form Detection (Recent momentum): $\alpha = 0.4$

## Task 1.4: Opponent Defensive Rating (ODR) Matrix Generation
* **Objective:** Isolate true, schedule-adjusted defensive suppression capabilities.
* **Model:** Formulate a Ridge-penalized regression solver across a trailing 6-month dataset.
* **System of Equations:**
    $$Kills_{ij} = \mu_{league} + Offense_{i} - Defense_{j} + \epsilon_{ij}$$
* **Target Output:** Extract the $Defense_{j}$ parameter as a continuous scalar representing expected kills suppressed per round for every VCT team.

## Task 1.5: Regressor Training and Validation
* **Model:** Gradient Boosting framework (XGBoost).
* **Features:** Clipped KPR, $\alpha=0.1$ EMA, $\alpha=0.4$ EMA, and team ODR.
* **Target:** Predict the continuous Expected Value ($\mu_{TD}$) for every player.
* **Validation Constraint:** The model's Mean Absolute Error (MAE) must be evaluated against a holdout test set to ensure it conclusively defeats the 4.37 Naive Baseline.