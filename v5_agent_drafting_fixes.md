1. Correcting the Deterministic Drafting Bug

The initial V5 drafting sequence suffered from mathematical hyper-determinism due to an unscaled Softmax function and the absence of sample size regularization. A single historical match with an abnormally high Average Combat Score (ACS) on an off-meta agent would mathematically eclipse a player's standard comfort picks.

To resolve this, the Agent Composition Framework integrates three critical adjustments to the expected utility $U(a_i)$ of an agent prior to probabilistic sampling.

## 2. Bayesian Comfort Smoothing and Map Priors

Agent comfort is no longer a global flat average. It must respect the specific map geometry and penalize low-sample-size anomalies. We apply a Bayesian average to the player's historical ACS for candidate agent $a_i$ on map $M$:

$$
\text{Comfort}(a_i, M) = \frac{N_{a_i, M} \cdot \bar{X}_{a_i, M} + \alpha \cdot \bar{X}_{\text{global}}}{N_{a_i, M} + \alpha}
$$

Where:

- $N_{a_i, M}$ is the total matches played by the player on agent $a_i$ on map $M$.
- $\bar{X}_{a_i, M}$ is the empirical average ACS for that specific pairing.
- $\bar{X}_{\text{global}}$ is the player's baseline global ACS across all agents.
- $\alpha$ is the structural prior weight (e.g., $\alpha = 3.0$), which pulls low-sample outlier games back to the player's baseline.

## 3. Temperature-Scaled Softmax

To prevent the exponential function from collapsing the probability space into a single deterministic choice when faced with minor utility differences, the selection probabilities $P(a_i)$ are calculated using a Temperature-scaled Softmax function:

$$
P(a_i) = \frac{\exp(U(a_i) / T)}{\sum_j \exp(U(a_j) / T)}
$$

By setting $T \approx 20.0$ to $30.0$ (scaled to the relative variance of ACS), the distribution softens. This restores stochastic variance to the Monte Carlo simulation, allowing a player to occasionally pick a secondary comfort agent while still heavily favoring their primary main.