# Phase 7: UI Feature Restoration & Autonomous Training
**Objective:** Restore the interactive legacy features (Meta Radar, Dynamic Budget Slider, 3-Transfer Advisor), wire them into the `v6` backend, and ensure the pipeline autonomously trains the XGBoost model if predictions are missing.

## Task 7.0: Autonomous Pipeline Execution
* **Execution:** In `app.py`, update the "Generate Optimal GPP Lineup" button logic. 
* **Logic:** Before calling `prepare_player_slate()`, check if `data/processed/xgb_predictions.json` exists. If it does not, use Python's `subprocess` module to automatically execute `model_training.py` to generate the predictions on the fly. 

## Task 7.1: Live Meta Radar Panel
* **Execution:** Above the main roster display in `app.py`, recreate the 3-column Meta Radar banner.
* **Logic:** Read the selected patch from the sidebar. Fetch the penalty data from `data/processed/automated_patch_nerf_registry.json`. Identify the top 3 most penalized agents.
* **UI:** Display the agent's name, their penalty score (e.g., `-0.26`), and a severity badge (`CRITICAL NERF` if penalty >= 0.05, else `MODERATE NERF`).

## Task 7.2: Dynamic Budget Slider Integration
* **Execution:** Restore the `st.slider` for "Available Fantasy Budget (VP)" (range: 35.0 to 60.0 VP, default: 50.0). Place this in the Command Center (Sidebar).
* **Backend Hook:** Refactor `knapsack_solver.py` (`solve_vfl_knapsack`) to accept `salary_cap` as a dynamic argument. Pass the Streamlit slider's value directly to the solver.

## Task 7.3: The 3-Transfer Advisor
* **Execution:** Recreate the interactive Transfer Advisor alongside the Optimal Roster.
* **Components:**
    * **Roster Multiselect:** `st.multiselect` loaded with the active slate from `current_slate.json`.
    * **Floating Bank:** A metric card showing the user's remaining budget (Slider Budget - Current Roster Cost). Turns red if negative.
    * **IGL Selectbox:** Dropdown to select the IGL from the drafted multiselect players.
    * **Calculate Trades Button:** Compare the user's multiselect roster against the `v6` Knapsack Optimal Roster. Generate visual UI cards (Green `IN` / Red `OUT`) detailing the exact player swaps required to bridge the gap.