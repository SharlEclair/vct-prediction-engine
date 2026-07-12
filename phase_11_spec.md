# Phase 11: Omni-Updater & Dynamic Event Whitelisting
**Objective:** Upgrade the UI dashboard to support dynamic VLR event whitelisting via user input, implement an incremental patch update pipeline, and create a master "Update All" orchestrator.

## 1. Dynamic VLR Whitelist (`scrapers/incremental_vlr_scraper.py`)
* **Action:** Upgrade the script to accept a `--whitelist` command-line argument using the `argparse` module.
* **Logic:** When parsing the VLR match results page, check the event name against the default Tier-1 regex. If it fails, check if any of the comma-separated whitelist strings exist within the event name (case-insensitive). 
* **Safety:** If the whitelist is empty, default strictly to the standard Tier-1 filter.

## 2. Master Orchestrator Upgrade (`run_pipeline.py`)
* **Action:** Upgrade `run_pipeline.py` to accept the `--whitelist` argument using `argparse`, and pass that argument down when it calls `scrapers/incremental_vlr_scraper.py`.
* **Execution Sequence:** The master pipeline must run exactly in this order:
  1. `scrapers/incremental_vlr_scraper.py --whitelist "..."` (Fetch new matches)
  2. `scrapers/vfl_scraper.py` (Fetch latest DFS pricing)
  3. `sync_map_pool.py` (Sync maps based on new matches)
  4. `scrapers/wiki_scraper.py` (Check for new patches)
  5. `scrapers/patch_ingestor.py` (Download new wikitext)
  6. `patch_analyzer.py` (Recalculate Concept Drift)
  7. `feature_engineering.py` (Rebuild ML features)
  8. `model_training.py` (Retrain XGBoost & Generate Projections)

## 3. UI Dashboard Controls (`app.py`)
* **Location:** Under the "System Administration" sidebar section.
* **Input Widget:** Add `whitelist_input = st.text_input("VLR Event Whitelist (comma-separated)", placeholder="e.g. Esports World Cup 2026")`.
* **Button 1 (Patch Update Only):** Add `st.button("🔄 Scrape Latest Patches & Rebuild Meta")`. When clicked, run a subprocess sequence of: `wiki_scraper`, `patch_ingestor`, `patch_analyzer`, `feature_engineering`, and `model_training`.
* **Button 2 (Master Update):** Rename the existing Full System Update button to `st.button("🚀 Master Update: Sync All Data & Retrain", type="primary")`. When clicked, it should call `python run_pipeline.py --whitelist "{whitelist_input}"`.