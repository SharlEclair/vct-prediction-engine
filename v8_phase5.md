# V8 Context: DRos Off-Policy Evaluation

## 1. The Zero-Day Cold-Start Contradiction
The purpose of the Patch Analyzer is to solve the Zero-Day Cold-Start Problem for Daily Fantasy Sports (DFS) Expected Value generation[cite: 1]. If the predictive pipeline requires historical post-patch data to dynamically adjust its weights, it loses its predictive edge before the market corrects[cite: 1]. 

Therefore, the differentiable graph must be trained offline against historical patch data, evaluating how accurately its predicted concept drift aligns with empirical reality[cite: 1]. Because treatments (patches) have already occurred, this is an Off-Policy Evaluation (OPE) task: estimating the Expected Value of a target policy (post-patch meta) using logged data from a behavior policy (pre-patch meta)[cite: 1].

## 2. Inverse Propensity Scoring (IPS) and Variance Explosion
Logged historical data in esports is subject to severe selection bias; rational professional teams stop picking heavily nerfed agents[cite: 1]. Standard OPE relies on Inverse Propensity Scoring (IPS) using the importance weight $w(x,a) = \frac{\pi_e(a|x)}{\pi_0(a|x)}$[cite: 1]. 

While unbiased, IPS estimators suffer from infinite variance when propensity scores are extreme (e.g., $\pi_0$ is very small), destroying the stability of gradients with massive likelihood ratios[cite: 1].

## 3. Doubly Robust Estimators with Optimistic Shrinkage (DRos)
To control variance while maintaining unbiasedness, the architecture mandates a Doubly Robust (DR) Estimator combined with Optimistic Shrinkage (DRos)[cite: 1].

The DRos estimator modifies the standard importance weight $w$ into a shrunk weight $w_\lambda$ controlled by a hyperparameter $\lambda \ge 0$[cite: 1]:
$$w_\lambda(x,a) = \frac{\lambda \cdot w(x,a)}{w(x,a)^2 + \lambda}$$

The full objective function for the policy value prediction becomes[cite: 1]:
$$\hat{V}_{DRos}(\pi_e; \lambda) = \frac{1}{n} \sum_{i=1}^n \left[ \sum_{a \in A} \pi_e(a|x_i)\hat{q}(x_i, a) + w_\lambda(x_i, a_i)(r_i - \hat{q}(x_i, a_i)) \right]$$

Where:
*   $\hat{q}(x,a)$ is the direct reward estimate (baseline expected Round Win Probability)[cite: 1].
*   $r_i$ is the empirical logged outcome[cite: 1].
*   $\lambda$ is the shrinkage hyperparameter controlling variance bounding[cite: 1]. 

By defining the total loss as the Mean Squared Error between the network's predicted EV under the simulated shock and the $\hat{V}_{DRos}$ empirical EV, the system isolates the causal impact of the patch without confounding variable interference[cite: 1].

## 4. Sequential Game Flow Modeling Requirement
Generating the baseline direct reward estimate $\hat{q}(x,a)$ requires capturing the deep temporal dependencies of tactical shooters (e.g., cascading economic collapses from consecutive round losses)[cite: 1]. The architecture should integrate an advanced sequence-to-sequence (Seq2Seq) model, such as an LSTM or Dynamic Bayesian Network, to act as the Direct Method reward predictor[cite: 1].