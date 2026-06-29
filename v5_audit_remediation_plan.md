# V5 Audit Remediation Plan

**Objective:** Systematically resolve every **[FAIL]** flagged in the V5 Architecture Audit to bring the codebase into 100% compliance before adding new features.

## Phase 1: Scraping & Data Ingestion Fixes

### WAF Bypass & Jitter

* Globally enforce `curl_cffi.requests.Session(impersonate="chrome")` across:

  * `vlr_scraper.py`
  * `historical_scraper.py`
  * `wiki_scraper.py`
  * `vfl_scraper.py`
* Standardize all `time.sleep()` calls to exactly:

  ```python id="ig8mrm"
  3.0 + random.uniform(0.5, 2.5)
  ```

### Tier 1 Strict Filtering

* Create a centralized `is_tier1_event(event_name)` utility function.
* Ensure the blacklist (Challengers, Ascension, GC, etc.) strictly overrides the whitelist globally.

### Data Consolidation

* Fix the tab URLs in the harvester to correctly hit:

  * `/match/tab/performance?match_id=X&game_id=Y`
  * `/match/tab/economy?match_id=X&game_id=Y`
* Instead of the base query parameters.

---

## Phase 2: Core Simulation Engine Fixes

### Simulation Date Passing

* Update `simulate_match(config)` to extract `target_date` and pass it down into `predict_veto()` and all subsequent sub-models.

### Map Veto Engine

* Update `predict_veto` to read the `ub_advantage` flag.
* If it is a Bo5 and a team holds the upper bracket advantage, immediately pop their opponent's 2 strongest maps from the pool before alternating picks.

### Agent Draft Clean-up

* Rename `AgentCompositionTransformer` to `HungarianAgentAssigner`.
* Purge all legacy Seq2Seq/Transformer terminology from the docstrings and comments.

### Round Simulator Overhaul

* Delete `BivariatePoissonMCMC`.
* Create:

  ```python id="rjwn2g"
  SideConditionedMarkovSimulator(team_a_stats, team_b_stats, map_name)
  ```
* The state tracker must include:

  * Score A
  * Score B
  * Economy Differential
  * Current Side (Atk/Def)
  * Round Number
* Implement the Round 13 side-swap (inverting the `side_advantage` modifier) and the 12-12 Overtime logic.

---

## Phase 3: UI/UX & Transfer Advisor Fixes

### Layout Topology

* Move the VFL database refresh button out of `st.sidebar` and into the **VFL Players** tab.
* Delete all `st.sidebar` calls.

### Transfer Advisor Liquidity Constraint

* Delete the UI slider for **Bank Balance**.
* Hardcode the calculation:

  ```python id="glk2gd"
  floating_bank = 50.0 - sum(player.cost for player in current_roster)
  ```
* Add a strict condition to the optimizer loops:

  ```python id="8sk5mb"
  if incoming_cost > outgoing_cost + floating_bank:
      continue
  ```

### IGL Multiplier Logic

* Update the fantasy solver so that if `forced_igl_name` is provided, the optimizer skips evaluating other players for the `2×` multiplier and permanently pins `is_igl=True` to the specified player.
