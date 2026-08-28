# Project Context: VCT Fantasy Optimization & The V5 Engine

## 1. Project Overview

We are developing a production-grade Fantasy Esports optimization solver for the Valorant Champions Tour (VCT).

The objective is to output the mathematically optimal 6-man roster for a given gameweek.

### Fantasy League Rules (VFL Constraints)

* **Salary Cap:** 50.0 VP (Valorant Points) maximum budget.
* **Role Constraints:** Must draft exactly 1 Duelist, 1 Initiator, 1 Controller, 1 Sentinel, and 2 Flex (Wildcard) players.
* **Team Constraints:** Maximum of 2 players from any single real-world VCT team.
* **Scoring Multiplier:** The user designates exactly 1 player as the In-Game Leader (IGL), who receives a strictly $2\times$ multiplier on their Expected Value (EV).

## 2. Architectural History

### Version 4.3 (Top-Down Macro)

Initially, we used gradient boosting (CatBoost) with complex patch-decay matrices to predict match win probabilities.

**Failure reason:** A 68% win probability does not translate to discrete fantasy micro-events (kills, deaths, 13-0 sweep bonuses).

### Version 5.0 (Bottom-Up Micro Simulation)

We transitioned to a highly complex, sequential Monte Carlo simulation (Directed Acyclic Graph) to generate exact match events:

* **Map Veto:** Contextual Bandit (with Temporal Map Pool registry and Upper Bracket double-ban logic).
* **Agent Draft:** Simultaneous Hungarian Assignment Solver (optimizing utility based on Bayesian-smoothed ACS and historical pick rates).
* **Round Simulator:** Side-Conditioned Markov Chain (logistic round-win odds factoring in Attack/Defense side-bias, resetting at Round 13).
* **Player Stats:** Dirichlet Regression to distribute the exact simulated team kills among the 5 players (enforcing the summation constraint).

## 3. The Empirical Failure (The Error Cascade)

We built a strict backtester holding out 100 Tier 1 matches from the 2026 season.

The V5 Bottom-Up engine failed catastrophically due to the Error Cascade.

### Backtest Metrics

* **Map Veto Accuracy:** 18.3% (Missing the veto cascaded errors down the entire DAG).
* **Match Winner Brier Score:** 0.2565 (Worse than a coin-flip 0.2500 baseline).
* **Player Kill MAE:** 6.94 kills/map.
* **Naive Baseline MAE:** 4.37 kills/map (A naive heuristic guessing the player's career average heavily outperformed our state-of-the-art DAG).

## 4. The Goal: The Hybrid Micro Engine

We must pivot to a hybrid approach.

### Top-Down Micro

We need to predict a player's direct Expected Value (EV) via regression to establish a stable, highly accurate mean/baseline (beating the 4.37 MAE).

### Bottom-Up Monte Carlo

We **MUST** retain the Monte Carlo engine (despite its mean inaccuracy) because it natively captures covariance between teammates, game-theory ceilings (overtime matches), and 13-0 sweep bonuses.

These wide distribution bounds are critical for evaluating GPP (Guaranteed Prize Pool) upside.
