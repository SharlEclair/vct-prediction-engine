````markdown
# 1. Reframing the Objective Function

The agent draft for a 5-man roster is no longer modeled as an autoregressive sequence of independent player choices. It is formulated as a **Bipartite Matching / Linear Assignment Optimization Problem**.

Let $P$ be the set of 5 players on a roster, and $A$ be the global pool of available agents.

We construct a 2D Strategy Matrix $U$ of size $5 \times |A|$, where each cell $U_{i,j}$ represents the total structural utility of assigning Player $i$ to Agent $j$ on a specific Map $M$ and Patch:

$$
U_{i,j} = w_1 \cdot \text{NormalizedComfort}(i,j,M) + w_2 \cdot \text{HistoricalPickRate}(i,j,M)
$$

Where:

- $\text{NormalizedComfort}(i,j,M)$: The Bayesian-smoothed ACS of Player $i$ on Agent $j$ on Map $M$, normalized relative to the team's baseline.

- $\text{HistoricalPickRate}(i,j,M)$: The empirical frequency of this exact assignment:

$$
\text{HistoricalPickRate}(i,j,M) =
\frac{
\text{Matches Played by Player } i \text{ on Agent } j \text{ on Map } M
}{
\text{Total Matches Played by Player } i \text{ on Map } M + \epsilon
}
$$

**Weights:**

We explicitly anchor the model to behavior over capability by setting:

$$
w_1 = 0.3
$$

and

$$
w_2 = 0.7
$$


---

# 2. Constrained Linear Optimization (The Hungarian Solution)

To prevent role theft, the engine solves for the entire team composition simultaneously.

We maximize the global team strategy utility subject to strict tactical constraints:

$$
\max \sum_{i \in P} \sum_{j \in A} U_{i,j} \cdot x_{i,j}
$$

**Subject to:**

$$
\sum_{j \in A} x_{i,j} = 1
\quad \forall i \in P
\quad
(\text{Every player gets exactly 1 agent})
$$

$$
\sum_{i \in P} x_{i,j} \leq 1
\quad \forall j \in A
\quad
(\text{Agents cannot be duplicated on the same team})
$$

Where:

$$
x_{i,j} \in \{0,1\}
$$

This is solved instantly using:

```python
scipy.optimize.linear_sum_assignment
````

---

# 3. Injecting Controlled Variance (Stochastic Noise)

To maintain Monte Carlo exploration without breaking role structures, Gumbel-Max noise or scaled normal distribution variance is added directly to the Utility Matrix before the optimization step, rather than applying a wide Softmax after:

$$
\tilde{U}*{i,j} = U*{i,j} + \mathcal{N}(0,\sigma^2)
$$

By tuning $\sigma$ relative to historical pick rates, a player like Sato will naturally alternate between high-frequency assignments (Phoenix and Raze) but will possess a $0%$ mathematical probability of rolling a disruptive role theft (like stealing Omen).

```
```
