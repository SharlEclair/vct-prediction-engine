# V4 Core Blueprint: Autonomous Meta Engine & Ghost Nerf Detection

## [x] Phase 4.1: Dependency Installation & NLP Setup
- [x] Install `spacy` and `scikit-learn` in the virtual environment.
- [x] Download the `en_core_web_sm` spaCy model.

## [x] Phase 4.2: Data Ingestion & NLP Parsing (`patch_analyzer.py`)
- [x] Create `patch_analyzer.py`.
- [x] Implement API ingestion using `httpx` to fetch base agent and weapon states from `https://valorant-api.com/v1/agents` and `https://valorant-api.com/v1/weapons`.
- [x] Implement `parse_patch_deltas()` using `spaCy` and RegEx to extract numerical mechanical shifts (e.g., `>>>`) from Riot Patch notes text blocks.

## [x] Phase 4.3: Ghost Nerf Telemetry Engine
- [x] Scan the `./data/raw/` match JSON files to build a **Weapon Dependency Matrix** ($P(w|a)$).
- [x] For every agent, calculate the empirical probability they purchase a specific weapon (e.g., Operator, Outlaw, Frenzy) during buy rounds.

## [x] Phase 4.4: RBF Distance Calculation
- [x] Implement a Weighted Radial Basis Function (RBF) over a standardized Euclidean space (`StandardScaler`) to compute $\Delta P_{agent}$.
- [x] Implement the Ghost Nerf modifier: $\Delta P_{ghost}(a) = \sum P(w|a) \cdot \Delta D_{weapon}$.
- [x] Calculate the final penalty $\Delta P_{final} = \max(\Delta P_{agent}, \Delta P_{ghost})$ and output this dynamically to `./data/processed/automated_patch_nerf_registry.json`.

## [x] Phase 4.5: Pipeline Integration
- [x] Refactor `feature_engineering.py` and `predict_match.py` to ingest `automated_patch_nerf_registry.json` instead of the manual V3.2 registry.