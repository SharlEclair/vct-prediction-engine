# V5 Empirical Validation: The Backtesting Harness

## 1. Objective

To systematically evaluate the predictive accuracy of the V5 Simulation Engine against a strict hold-out dataset (2026 VCT matches) and benchmark its performance against naive historical heuristics. This directly answers the academic critique requiring out-of-sample validation.

## 2. Dataset Splitting

**Calibration Set:**

* All matches parsed from 2023 to 2025 serve as the historical ledger for the `HungarianAgentAssigner` and the `TemporalMapRegistry`.

**Hold-Out Set:**

* All matches occurring in 2026. The backtester will iterate through these matches, strictly using only data available prior to the match date to prevent data leakage.

## 3. Core Evaluation Metrics

The backtester must calculate and output three primary metrics:

### Map Veto Accuracy (Top-K Alignment)

* Compares the `MapVetoBandit`'s predicted Map 1, Map 2, and Decider against the actual played maps.

**Metric:**

* Percentage of actual played maps that appeared in the model's Top 3 predicted sequence.

### Match Winner Calibration (Log-Loss / Brier Score)

Evaluates the probabilistic confidence of the `SideConditionedMarkovSimulator`.

**Metric:**

Brier Score:

$$(Predicted_Prob - Actual_Outcome)^2$$

A lower score indicates better calibration (e.g., predicting an 80% win rate for a team that actually wins).

### Micro-Stats Error (Player Kill MAE)

Evaluates the accuracy of the Dirichlet kill-share constraints.

**Metric:**

* Mean Absolute Error (MAE) between a player's predicted kills and actual kills in the match.

**Baseline Benchmark:**

* Compare the V5 MAE against a "Naive Baseline" (simply guessing the player's historical average kills per map).

## 4. Execution Flow (`v5_backtester.py`)

1. Scan `data/raw/` for all matches containing "2026" in the date.
2. For each match, parse the actual outcome (winner, maps played, individual player kills).
3. Initialize the V5 engine, passing the exact `target_date` and `target_patch` of the historical match.
4. Run 500 Monte Carlo iterations (scaled down from 10,000 for backtesting speed).
5. Compare outputs, aggregate the errors, and generate a final `backtest_results_report.md`.
