# Phase 4: Knapsack Solver Integration and Optimization
**Objective:** Ingest the fused probability distributions and execute a Mixed-Integer Linear Programming (MILP) solver to generate the optimal VFL lineup. The solver must maximize GPP tournament upside (Ceiling) while strictly adhering to salary cap and role constraints.

## Task 4.1: Array Ingestion and Setup
* **Objective:** Prepare the Phase 3 outputs for the optimization environment.
* **Execution:** Load the fused matrix and extracted metrics (EV, Floor, Ceiling) for the DFS slate into the solver (e.g., using the Python `pulp` library).
* **Decision Variables:** Define a binary decision variable x_i for drafting player i, and a secondary binary decision variable y_i for designating player i as the In-Game Leader (IGL).

## Task 4.2: Budget and Roster Constraints
* **Salary Cap:** Enforce the 50 VP maximum salary cap: Sum(salary_i * x_i) <= 50.0. (Assume mock salaries between 6.0 and 10.0 for the test).
* **Roster Cap:** Ensure a maximum of 2 players can be selected from any single real-world VCT team: Sum(x_i for i in Team T) <= 2.
* **Lineup Size:** The total lineup must consist of exactly 6 players.

## Task 4.3: VFL Role Constraints
* **Execution:** Enforce the strict positional requirements for the 6-man roster.
    * Sum(x_Duelist) = 1
    * Sum(x_Initiator) = 1
    * Sum(x_Controller) = 1
    * Sum(x_Sentinel) = 1
    * Sum(x_Flex) = 2

## Task 4.4: IGL Multiplier Logic
* **Constraint:** Exactly one drafted player must be designated as the IGL: Sum(y_i) = 1.
* **Dependency:** A player can only be the IGL if they are actually drafted: y_i <= x_i.
* **Bonus:** The selected IGL receives a 1.5x or 2.0x multiplier to their projected output in the objective function.

## Task 4.5: Objective Function Maximization
* **Execution:** Define the objective function to maximize the Ceiling (85th percentile) projection rather than the median EV to optimize for GPP tournament win equity.
* **Function:** Maximize Sum(Ceiling_i * x_i + Ceiling_i * y_i * IGL_Bonus).

## Task 4.6: Portfolio Simulation (Validation)
* **Execution:** Once the optimum lineup is generated, map it back against the 10,000 raw simulation iterations from Phase 3 to calculate the lineup's true aggregate ceiling and probability of hitting extreme tournament-winning scores.