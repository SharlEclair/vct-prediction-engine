# Phase 3: The Copula Fusion Engine (Mathematical Combiner)
**Objective:** Synthesize the stable Top-Down XGBoost predictions ($\mu_{TD}$) with the Bottom-Up Monte Carlo covariance matrix ($\Sigma_{MC}$) using the Iman-Conover algorithm. This must perfectly preserve the XGBoost mean while imposing the structural correlations of the DAG.

## Task 3.1: Marginal Generation
* **Objective:** Define parametric marginal distributions for all 10 players on the server.
* **Execution:** For each player, generate a continuous probability distribution (e.g., Gamma or Normal) parameterized so that its expected value exactly equals the XGBoost $\mu_{TD}$ from Phase 1. 
* **Variance:** Apply a standard historical variance parameter to scale the distributions appropriately.

## Task 3.2: Independent Sampling
* **Execution:** Draw $N = 10000$ independent, uncorrelated samples from each of the 10 Top-Down marginal distributions.
* **Output:** This creates an uncorrelated $10000 \times 10$ matrix, denoted as $M$.

## Task 3.3: Cholesky Decomposition & Normal Scores
* **Execution:** 1. Compute the Van der Waerden normal scores for the columns of $M$ to create score matrix $S$.
    2. Uncorrelate $S$ by multiplying it by the inverse of its Cholesky decomposition, yielding orthogonal matrix $Z$.
    3. Compute the Cholesky decomposition of the target Phase 2 DAG correlation matrix ($\Sigma_{MC}$), denoted as $P$.
    4. Induce the target correlation: $Y = Z \cdot P$

## Task 3.4: Rank Reordering (Iman-Conover)
* **Execution:** Reorder the original independent samples in $M$ so that their rank order exactly matches the rank order of the columns in the correlated matrix $Y$.
* **Validation:** The new matrix must retain the exact marginal values of $M$ but exhibit a rank correlation practically identical to $\Sigma_{MC}$.

## Task 3.5: Metric Extraction
* **Execution:** From the final fused $10000 \times 10$ matrix, calculate three vital statistics for the downstream DFS Knapsack solver:
    * **Expected Value (Mean)** * **Floor (15th percentile)**
    * **Ceiling (85th percentile)**