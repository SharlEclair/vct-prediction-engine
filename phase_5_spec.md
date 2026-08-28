# Phase 5: Production Configuration & Dynamic Ingestion
**Objective:** Eliminate critical technical debt by extracting static engine parameters into a centralized configuration file (`config.yaml`) and refactoring the Phase 3 and Phase 4 solvers to ingest dynamic JSON payloads rather than internal hardcoded mock arrays.

## Task 5.1: Centralized Configuration File
* **Execution:** Create a `config.yaml` file at the root directory.
* **Contents:** Extract all hardcoded engine constants into this file, specifically:
    * `ROLE_ALPHA_WEIGHTS` (from `dag_simulation.py`: Duelist=3.5, Initiator=2.5, Controller=2.0, Sentinel=2.0, Flex=2.0)
    * `COMPETITIVE_MAP_POOL` (Ascent, Bind, Haven, Lotus, Sunset, Pearl, Fracture)
    * `BENCHMARKS` (Naive MAE = 4.37)
    * `DFS_CONSTRAINTS` (Salary Cap = 50.0, Lineup Size = 6, Max Per Team = 2)

## Task 5.2: Engine Refactoring for Configuration
* **Execution:** Update `dag_simulation.py`, `model_training.py`, and `knapsack_solver.py` to import and read their static parameters dynamically from `config.yaml` using the `PyYAML` library. Remove all hardcoded instances of these variables from the Python files.

## Task 5.3: Dynamic Slate Payload Ingestion
* **Objective:** Decouple the solvers from the phantom slate.
* **Execution:** 1. Create a sample `data/processed/current_slate.json` file. This file should contain a mock array of 10 players, including their real names, teams, designated VFL roles, and salaries.
    2. Refactor `copula_fusion.py` to read player names/teams from this JSON rather than generating a static list.
    3. Refactor `knapsack_solver.py` to pull the slate metadata (salaries, roles) directly from this JSON payload rather than relying on its internal mock generation logic.