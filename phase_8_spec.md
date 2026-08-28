# Phase 8: Pipeline Reconnection & Infrastructure Hardening
**Objective:** Eliminate the silent fallback cascades. Reconnect the XGBoost Expected Value predictions to the Copula Fusion engine and enforce absolute pathing across the entire repository to prevent directory-based execution failures.

## Task 8.1: XGBoost Output Serialization
* **Execution:** Modify `model_training.py`. 
* **Logic:** After the XGBoost model is trained and validated against the holdout set, it must generate expected value predictions (`mu_TD`) for the active players listed in `data/processed/current_slate.json`.
* **Output:** Serialize these exact predictions to a new file: `data/processed/xgb_predictions.json`. Structure it as a simple key-value dictionary `{"PlayerName": EV_Float}`.

## Task 8.2: Copula Fusion Ingestion & Fallback Removal
* **Execution:** Modify `copula_fusion.py`.
* **Logic:** Update the `get_top_down_predictions()` function. It must now load `xgb_predictions.json` to ingest the true machine learning expected values. 
* **Hardening:** Entirely delete the linear heuristic fallback (`base_ev = salary * 4.95`). If the XGBoost JSON is missing or a player is not found, the script must explicitly raise a `ValueError` rather than silently substituting a fake projection.

## Task 8.3: Universal Absolute Pathing
* **Execution:** Modify `utils.py`, `patch_ingestor.py`, `app.py`, `model_training.py`, and `copula_fusion.py`.
* **Logic:** Remove all instances of relative string paths (e.g., `"./data/processed/..."` or `"config.yaml"`).
* **Standard:** Implement `pathlib.Path` to anchor all file operations absolutely to the project root.
    * *Example:* `ROOT_DIR = Path(__file__).resolve().parent.parent` (adjusting depth as necessary per file location).