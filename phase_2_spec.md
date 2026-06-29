# Phase 2: Bottom-Up Engine Pruning and Execution
**Objective:** Repurpose the legacy V5 Directed Acyclic Graph (DAG) not for absolute point-prediction truth, but strictly to generate the structural boundaries, heavy-tailed ceilings, and the in-game covariance matrix (the Copula).

## Task 2.1: DAG Execution (Monte Carlo)
* **Execution:** Run the V5 pipeline sequence for 10,000 Monte Carlo iterations for a target match.
* **Sequence:** Contextual Bandit Map Veto $\rightarrow$ Hungarian Agent Draft $\rightarrow$ Side-Conditioned Markov Round Simulator $\rightarrow$ Dirichlet Regression Kill Share.
* **Constraint:** Ensure the simulation environment captures extreme structural boundaries (e.g., 13-0 sweeps and overtime variance).

## Task 2.2: Simulation Matrix Extraction
* **Objective:** Structure the simulated outputs for statistical profiling.
* **Execution:** Aggregate the 10,000 discrete game states into a unified array.
* **Dimensions:** A $10,000 \times 10$ matrix representing the raw, unadjusted DFS fantasy points for all 10 players on the server across every iteration.

## Task 2.3: Correlation Profiling
* **Objective:** Quantify the zero-sum kill economy and tactical agent synergies.
* **Execution:** Calculate the $10 \times 10$ Spearman Rank Correlation matrix ($\Sigma_{MC}$) from the extracted simulation matrix.
* **Output:** This matrix must mathematically reflect the negative correlation between opponents (if Team A gets kills, Team B is dying) and the positive covariance between synergistic teammates (e.g., Initiator/Duelist trade fragging).