# V5 Simulation Engine: Full Architecture Audit Checklist

**Objective:** Verify that the codebase strictly adheres to the V5 Bottom-Up Micro Simulation paradigm, ensuring all legacy models (BLOPS, Seq2Seq drafting, naive Poisson) have been deprecated and replaced with the verified production methodologies.

## 1. Scraping & Data Ingestion (`scratch_harvest_all_t1.py` / `vlr_scraper.py`)

* [ ] **WAF Bypass:** Standard HTTP clients (`requests`, `httpx`) must be entirely replaced with `curl_cffi.requests.Session(impersonate="chrome")` to prevent TLS shadow-bans.
* [ ] **Anti-Friction Jitter:** Requests must use a non-linear sleep delay: `3.0 + random.uniform(0.5, 2.5)`.
* [ ] **Tier 1 Strict Filtering:** Tournaments must match the whitelist (Champions, Masters, Kickoff, Americas, EMEA, Pacific, CN) and strictly fail the blacklist (Challengers, Ascension, Game Changers, GC, Showmatch, Open Qualifier).
* [ ] **Data Consolidation:** The schema per match must successfully merge base HTML data, `/match/tab/performance`, and `/match/tab/economy` into a single JSON artifact.

## 2. Core Simulation Engine (`v5_simulation_engine.py`)

* [ ] **Global Player Telemetry:** Player historical metrics (ACS, Pick Rate) must be queried from a global ledger decoupled from their current team to ensure roster moves do not erase statistical history.
* [ ] **Temporal Map Registry:** The engine must possess a lookup table that restricts the active map pool (the 7 available maps) based on the specific patch or date of the match being simulated.
* [ ] **Map Veto Engine (Sub-Model 1):** Uses a Contextual Bandit framework. Must correctly implement the Upper Bracket Double-Ban advantage for Bo5 Grand Finals when flagged.
* [ ] **Agent Draft (Sub-Model 2):** Seq2Seq/Transformer logic must be completely removed. The draft must use the Simultaneous Hungarian Assignment Solver (`scipy.optimize.linear_sum_assignment`) on a utility matrix weighted as 30% Bayesian-Smoothed ACS + 70% Historical Pick Rate, with Gaussian noise injected for Monte Carlo variance.
* [ ] **Round Simulator (Sub-Model 3):** The naive Bivariate Poisson must be replaced with the Side-Conditioned Markov Chain. Transition probabilities must account for Attack/Defense side advantages, economy differentials, and explicitly trigger a side-swap inversion at Round 13.
* [ ] **Player Micro-Stats (Sub-Model 4):** Kill distributions must enforce the summation constraint strictly via Dirichlet Regression, ensuring total player kills perfectly match the simulated round score capacity.

## 3. UI/UX & Transfer Advisor (`app.py` / `fantasy_engine.py`)

* [ ] **Layout Topology:** "⚡ Open Simulation" must be the primary (first) tab. Global match settings must reside at the top of the "📊 Match Analysis" tab, completely clearing the sidebar.
* [ ] **Backtesting UI:** The Match Analysis tab must fetch and display the Actual historical match results side-by-side with the Predicted results.
* [ ] **Transfer Advisor Liquidity Constraint:** The system must dynamically calculate Floating Bank = `50.0 - sum(roster_cost)` and strictly enforce `Incoming Cost ≤ Outgoing Cost + Floating Bank`.
* [ ] **IGL Multiplier:** The UI must allow selecting 1 IGL, and the optimizer must strictly apply a `2×` Expected Value (EV) multiplier to that player.
* [ ] **Roster State Persistence:** The app must save to and load from `data/user_roster_state.json` to carry fantasy rosters across sessions.
