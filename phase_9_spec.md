# Phase 9: Live API & CSV Slate Ingestion

## Objective

Replace the mock `current_slate.json` pipeline with a live connection to the VFL REST API and a fallback CSV uploader, allowing the engine to ingest and solve real daily DFS slates dynamically.

## Task 9.1: Live API Sync Button

### Execution

Add a `st.button` in the `app.py` Command Center labeled:

> **Sync Live VFL Slate (API)**

### Logic

When clicked:

- Launch `vfl_scraper.py` as a subprocess.
- Read its output (`vfl_players_db.json`).
- Map the schema (converting integer roles to strings like `'Duelist'`).
- Overwrite `current_slate.json`.

---

## Task 9.2: Fallback Streamlit File Uploader

### Execution

Below the API button, add a `st.file_uploader` widget accepting `.csv` files.

### Logic

- Read the CSV into a Pandas DataFrame.
- Map standard DFS columns into the `current_slate.json` dictionary format.

---

## Task 9.3: Pipeline Reset & Autonomous Execution

### Execution

Once the new slate (via API or CSV) is saved to `current_slate.json`, the app must:

- Delete the old `data/processed/xgb_predictions.json` file.
- Clear the optimal lineup from `st.session_state`.

### Logic

Deleting the old predictions forces Phase 7's autonomous hook to re-trigger. The XGBoost model will spin up, map historical stats for the new 50+ players from the live tournament, generate fresh ML predictions, and pass them to the Knapsack solver.