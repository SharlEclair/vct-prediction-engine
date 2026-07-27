# V8 Context: Copula-Based Synergistic Aggregation

## 1. Legacy System Vulnerability: The Independence Fallacy
The legacy patch analyzer aggregates individual ability shocks into a final Concept Drift Index using a Probabilistic Union formula[cite: 2]:
$$\text{Concept Drift Index} = 1.0 - \prod_{i} (1.0 - \min(\theta_i, 0.999))$$

This aggregation relies on the fundamental axiom of statistical independence, formally defined as $P(A \cap B) = P(A)P(B)$[cite: 1]. It assumes that a nerf to an agent's first ability ($\theta_1$) occurs entirely independently of a nerf to their second ability ($\theta_2$)[cite: 1]. 

In a tactical shooter, this independence assumption is structurally false[cite: 1]. Agent abilities are highly combinatorial and act as submodular or supermodular sets within tight mechanical loops[cite: 1].

### Synergistic Nerf Example:
Consider Jett's mechanical loop[cite: 1]: Jett deploys a vision-blocking smoke screen ("Cloudburst") and immediately uses her rapid movement dash ("Tailwind") into that smoke to break defensive crosshairs and secure space safely[cite: 1].
* If a patch reduces smoke duration **and** increases dash activation latency, these are not independent events[cite: 1].
* The reduced smoke duration makes the delayed dash exponentially more punishable by opponents[cite: 1].
* The combined nerf creates a compounding, synergistic shock that is strictly greater than the probabilistic union of its isolated parts[cite: 1].

Using the standard product rule of independent probabilities artificially suppresses the magnitude of multi-ability nerfs[cite: 1].

## 2. Mathematical Formulation: Archimedean Gumbel Copula
To resolve this fallacy, the aggregation layer must model the multivariate dependency between an agent's abilities using Copulas[cite: 1]. Copulas describe the dependence between random variables with uniform marginals, fundamentally rooted in Sklar's theorem[cite: 1].

We specifically utilize the **Archimedean Gumbel Copula**, which captures strong **upper-tail dependence**—the tendency for large values (or extreme shocks) in two or more variables to occur simultaneously[cite: 1].

The bivariate Gumbel copula is defined as[cite: 1]:
$$C(u, v; \theta) = \exp\left(-\left[(-\ln u)^\theta + (-\ln v)^\theta\right]^{1/\theta}\right)$$

Where:
* $u, v \in (0, 1)$ represent the survival probabilities ($1 - \theta_i$) or marginal shock probabilities of individual abilities[cite: 1].
* $\theta \in [1, \infty)$ is the learned dependence parameter[cite: 1].
* The upper-tail dependence coefficient follows directly as $\lambda_U = 2 - 2^{1/\theta}$[cite: 1].

### Behavior Across Dependence Parameters:
* When $\theta = 1$, $\lambda_U = 0$, and the copula collapses to represent completely independent events (equivalent to the current heuristic)[cite: 1].
* As the network learns that abilities possess high upper-tail dependence (synergy), it optimizes $\theta > 1$, mathematically amplifying the combined shock[cite: 1].

## 3. Multivariate Extension & Autograd Requirements
The generator function for Archimedean copulas extends to $d$ dimensions[cite: 1]:
$$\psi(t) = (-\ln t)^\theta, \quad \psi^{-1}(s) = \exp(-s^{1/\theta})$$
$$C(u_1, \dots, u_d; \theta) = \psi^{-1}\left(\sum_{i=1}^d \psi(u_i)\right) = \exp\left(-\left[\sum_{i=1}^d (-\ln u_i)^\theta\right]^{1/\theta}\right)$$

In `v8_copula_aggregation.py`, this expression must be implemented natively in PyTorch[cite: 1].
* Use numerical clamping ($\epsilon = 1e-7$) to prevent $\ln(0)$ or $0^{1/\theta}$ division-by-zero instability during autograd backward passes.
* Constrain $\theta \ge 1.0$ using a parameterized transformation: $\theta = 1.0 + \text{softplus}(\hat{\theta})$.
* Ensure gradients flow seamlessly through $C(\mathbf{u}; \theta)$ back to Phase 2 category embeddings and Phase 3 breakpoint outputs[cite: 1].