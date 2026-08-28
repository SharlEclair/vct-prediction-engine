# Phase 10: Structural Refactor & Incremental Upgrades
**Objective:** Eliminate tech debt, isolate utility and scraper modules into dedicated directories, securely resolve all import paths, and upgrade the scraping architecture to utilize smart, append-only incremental updates.

## 1. The Purge & Archive (Tech Debt Removal)
* **Action:** Create an `archive/` directory at the project root.
* **Targets to Archive:** * All obsolete engine scripts: `dag_simulation.py`, `covariance_profiler.py`, `model_pipeline.py`, `survival_analysis.py`, `main.py`, `api_client.py`.
  * All scratchpads, audits, and tests: Any file starting with `scratch_`, `audit_`, `test_`, or `verify_`.
  * All obsolete prototypes: `explore_api.py`, `explore_json.py`, `math_test.py`, `generate_and_compare.py`, `backtest_evaluation.py`, `v5_backtester.py`, `meta_engine.py`, `historical_scraper.py`.

## 2. Directory Reorganization (Modularization)
* **Action:** Create `utils/` and `scrapers/` directories at the project root.
* **Target `utils/`:** Move `utils.py`, `v4_parsing_skills.py`, and `v4_skills.py` into this folder.
* **Target `scrapers/`:** Move `vfl_scraper.py`, `vlr_scraper.py`, `wiki_scraper.py`, and `patch_ingestor.py` into this folder.
* **Import Resolution:** All files that import these moved modules must be updated. For example, `import utils` must become `from utils import utils` or `from utils.utils import X`.

## 3. Incremental VLR Scraper Upgrade
* **File:** Create `scrapers/incremental_vlr_scraper.py` (or upgrade the existing `vlr_scraper.py`).
* **Logic Requirements:**
  1. **State Check:** Scan the `data/raw/` directory to identify the most recent match date previously scraped.
  2. **Fetch:** Query the VLR match results pages. Stop pagination immediately once it reaches matches older than the latest local date.
  3. **Append:** Scrape only the new Tier-1 matches and append them to the `data/raw/` directory. Do *not* overwrite or re-scrape existing historical data.

## 4. UI Dashboard Integration
* **File:** `app.py`
* **Action:** In the "System Administration" sidebar block, add a new button: `st.button("📥 Scrape Latest VLR Matches (Incremental)")`.
* **Behavior:** Wrap the subprocess call to the incremental scraper in an `st.spinner`. Output the number of new matches added to the database in an `st.success` banner.