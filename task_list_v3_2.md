# V3.2 Core Blueprint: Patch-Aware Matrix & Concept Drift Mitigation

## [x] Phase 3.2.1: Patch Telemetry & Registry
- [x] Update `predict_match.py` and `feature_engineering.py` match parsers to extract the `patch` string (e.g., "8.11") from the raw match JSON metadata. 
- [x] Create `patch_nerf_registry.json` in `./data/processed/`. This will store the manual Clustered Character Representation (CCR) distances for major agent nerfs (e.g., `{"9.02": {"Tejo": 0.8}}`).

## [x] Phase 3.2.2: Global Meta Distance Engine (`meta_engine.py`)
- [x] Create `meta_engine.py`. This module must scan all historical matches and calculate the global agent pick-rate probability distribution for every unique patch.
- [x] Compute the Jensen-Shannon Divergence (JSD) using `scipy.spatial.distance.jensenshannon` between all patches to create an $N \times N$ matrix (`patch_distance_matrix.json`) representing $\Delta P_{global}$.

## [x] Phase 3.2.3: Vectorized Composite Feature Engineering
- [x] Overwrite the rolling EMA calculations in `feature_engineering.py` and `predict_match.py`.
- [x] Replace pure time decay with the 2D Composite Weight formula:
  $W = e^{-\lambda \cdot \Delta t} \cdot [I_{agent} \cdot e^{-\gamma_1 \cdot \Delta P_{agent}} + (1 - I_{agent}) \cdot e^{-\gamma_2 \cdot \Delta P_{global}}]$
- [x] **Hyperparameters:** Set $\lambda = 0.02$ (Time), $\gamma_1 = 2.0$ (Agent Nerf Penalty), and $\gamma_2 = 0.5$ (Global Shift Penalty).
- [x] **Optimization Constraint:** Use vectorized Pandas/NumPy operations. Avoid row-by-row `.apply()` for performance.

## [x] Phase 3.2.4: CatBoost Concept Drift Integration (`model_pipeline.py`)
- [x] Refactor `model_pipeline.py`. When building the training set, compute the composite weight for *every* historical row relative to its target match date and patch.
- [x] Pass this calculated weight array directly into the `weight` parameter of `catboost.Pool()`.
- [x] Update CatBoost training parameters to use `cv` with `type='TimeSeries'` and add `early_stopping_rounds=50` to prevent overfitting on transient micro-drifts.