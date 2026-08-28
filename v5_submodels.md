## 2. Sub-Model 1: The Adversarial Map Veto Engine

The map veto is a zero-sum, adversarial decision-making process operating under incomplete information. Formulated as a multi-armed Contextual Bandit, the available map pool acts as the set of arms $\mathcal{A}$.

At each discrete step $t$, a context vector $x_t$ encapsulates the environmental state, merging historical map win-rates, macro team metrics, and patch distance penalties.

The algorithm seeks to learn a policy:

$$
\pi(a|x_t)
$$

to predict action $a \in \mathcal{A}$.

Using off-policy evaluation to estimate the shadow reward of unchosen maps, the model outputs a Map Probability Tensor $P(\mathcal{M})$, assigning a continuous probability likelihood to every possible sequence permutation.


---

## 3. Sub-Model 2: Generative Agent Composition Framework

Agent drafting is highly conditional and adversarial.

Rather than relying on static developer roles, V5 dynamically clusters functional roles via Jensen-Shannon Divergence (JSD) applied to co-occurrence matrices.

It then deploys an autoregressive, transformer-based framework (BERT) to treat drafting as a sequence generation problem.

The probability of the complete draft:

$$
S = (a_1, a_2, \dots, a_{10})
$$

conditional on the predicted Map $M$ and Patch Penalty Matrix $\Delta P$, is factored autoregressively:

$$
P(S|M, \Delta P) =
\prod_{i=1}^{10}
P(a_i|a_1, \dots, a_{i-1}, M, \Delta P, \Theta_{Team})
$$


---

## 4. Sub-Model 3: Discrete Match Round Score Simulation

Standard continuous regressions fail to predict exact discrete scorelines (e.g., 13-8).

Because teams compete for a shared pool of maximum rounds, their scoring rates are negatively correlated.

V5 implements a Bivariate Poisson Regression Model, where:

$$
X = X_1 + X_3
$$

and

$$
Y = X_2 + X_3
$$

with the covariance parameter $\lambda_3$ explicitly capturing this interdependence.

The estimated Poisson $\lambda$ rates inform a discrete-time Monte Carlo Markov Chain (MCMC) simulation initialized at state:

$$
S_0 = \{0, 0\}
$$

walking the game tree until a discrete terminal scoreline is achieved.


---

## 5. Sub-Model 4: Player Micro-Performance & Summation Constraint

Predicting exact micro-statistics for players requires enforcing strict structural integrity:

The predicted individual kills must sum exactly to the team total dictated by the simulated score.

V5 models this as compositional data using Dirichlet Regression, bounding outputs between 0 and 1 to predict the proportion vector:

$$
\mathbf{p} = (p_1, p_2, p_3, p_4, p_5)
$$

subject to the unit-sum constraint:

$$
\sum_{i=1}^{5} p_i = 1
$$

The parameters:

$$
\alpha = (\alpha_1, \dots, \alpha_5)
$$

act as structural priors informed by the predicted agent roles.


---

## 6. Mitigating the Error Cascade

To prevent a sequential cascade of inaccurate predictions, V5 uses Probabilistic Beam Search and Monte Carlo execution.

By retaining only the top $K$ most probable branches and simulating 10,000 parallel game instantiations, the engine generates probability-weighted Expected Values:

$$
EV(Player_X) =
\frac{1}{10000}
\sum_{n=1}^{10000}
F_n(Player_X)
$$