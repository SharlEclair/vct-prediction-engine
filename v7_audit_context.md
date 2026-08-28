# Simulation Engine v7: Critical Flaw Context & Ground Truth Data

## 1. Factorization Machine (FM) Draft Failure

Despite expanding the latent space to $k=16$ with Xavier initialization, the draft engine is generating tactically unviable compositions. In the recent PRX vs. KC simulation, both teams drafted exactly zero Controllers on Fracture, and KC drafted a heavy triple-Sentinel composition. In professional Valorant, Controllers are universally mandatory to slice up dangerous territory and provide vision denial.

* **Audit Directive:** Investigate why the FM's inner products $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ are failing to penalize missing mandatory roles. Determine if the training data matrix is too sparse for $k=16$ to organically learn these penalties. Propose a mathematically sound mechanism (e.g., role-boundary masking in the combinatorial search space or modifying the loss function) to enforce valid compositions without reverting to hardcoded scalar penalties.

## 2. Statistical Paradox: Mode-Mean Detachment in GEV Link

The simulation predicted a PRX map win rate of $50.2\%$ on Sunset, but the most frequent scoreline (the mode of the PMF) was 11-13 in favor of KC. This is a severe statistical contradiction.

* **Audit Directive:** Analyze the calibration of the Generalized Extreme Value (GEV) link function. While the centering constant $C=0.659$ successfully anchors equal-stat teams at $P(Z=0) = 0.5$, the shape parameter ($\xi$) is skewing the tail distribution of round wins so aggressively that the mode completely detaches from the expected value. Investigate the Softplus domain guard and the $\xi$ calibration logic to realign the Probability Mass Function (PMF).

## 3. Copula Over-Indexing on Opening Duels

The fantasy point projections exhibit a massive internal disparity: the model projects `something` at $5.56$ Expected Value (EV) points while his teammate `d4v41` lags at $1.25$.

* **Audit Directive:** Investigate the KDA Copula implementation and marginal distribution mappings. The copula appears to be severely over-indexing on the regularized `DuelDiff` prior ($+0.150$), funneling an unrealistic share of the team's total simulated kills to the primary entry fragger. The dependency structure is currently failing to accurately model the highly correlated, shared trade-fragging dynamics of a professional tactical shooter.

## 4. Real-World Ground Truth Misalignment (Match Outcome & Drafts)

The simulation predicted a $50.8\%$ series victory for Paper Rex, but the model failed to capture the actual map-specific structural advantages that played out in reality.

* **Historical Ground Truth:** In the actual Esports World Cup 2026 Group B Elimination match on July 4, 2026, Karmine Corp defeated Paper Rex 2-1. PRX won Sunset 13-11, while KC won Fracture 13-9 and Lotus 13-7.


* **Draft Ground Truth:** On Fracture, KC drafted Neon, Chamber, Viper, Fade, and Omen (utilizing two Controllers, zero zero-Controller comps). PRX drafted Raze, Viper, Waylay, Harbor, and Sova (also utilizing two Controllers).


* **Audit Directive:** Use this ground truth data to benchmark the draft engine's accuracy and the map-specific synergy weightings. The model completely missed the necessity of double-controller setups on Fracture and failed to predict KC's dominance on Lotus. Identify the blind spots in how map geometry and specific agent pairings are weighted in the continuous architecture.