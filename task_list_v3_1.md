# V3.1 Core Blueprint: VFL Roster Optimization & Simulation Engine

## [x] Phase 3.1.1: Arbitrary Match Simulation & Global Patch Decay
- [x] Decouple `predict_match.py` from fixed historical match IDs to simulate an arbitrary match between any two selected teams using their latest form up to June 2026.
- [x] Align exponential patch decay ($W = e^{-0.02 \cdot t}$) relative to the current date (June 2026) to heavily prioritize recent performance vectors.

## [x] Phase 3.1.2: VFL Table Scraper → API-First Player Database ✅ COMPLETE (API Migration)
- [x] ~~Create `vfl_scraper.py` using `httpx` and `selectolax` to target HTML~~ → **SUPERSEDED**: Rewrote `vfl_scraper.py` to use the VFL REST API directly (no DOM parsing).
- [x] **Step A**: `GET /api/event/currentevent` → resolves active `event_id` (currently `9` = "VCT 2026: Masters London")
- [x] **Step B**: `GET /api/player/allplayers?eventId={id}` → 60-player roster JSON
- [x] Confirmed correct `playerRole` integer mapping: `0=Duelist, 1=Initiator, 2=Controller, 3=Sentinel`
- [x] Extracts: `player_name`, `vlr_team_id`, `team_name`, `team_short`, `role`, `price`, `gw_pts`, `tot_pts`, `ppg` (computed from history)
- [x] Cache envelope saved to `./data/processed/vfl_players_db.json` with `_meta` timestamp and player count.
- [x] Build a local database cache `./data/processed/vfl_players_db.json` and create a Streamlit sidebar button to re-trigger compilation when VCT 2026 Stage 2 updates drop.
- [x] **DB Shape**: 60 players, 12 teams (NRG, LEV, PRX, FUT, VIT, DRG, GE, XLG, FS, EDG, G2, TH), 5 per team
- [x] MILP optimizer confirmed `optimal` (78.06 pts, 50 VP, 6 players, IGL: Keiko)

## [x] Phase 3.1.3: Integer Linear Programming Roster Solver (`fantasy_engine.py`)
- [x] Implement a Bounded Knapsack or Mixed-Integer Linear Programming (MILP) solver mapped to these parameters:
  - **Hard Constraints**: 
    - Total Budget $\le$ 50 VP (Configure a soft parameter to favor leaving a 1-2 VP floating buffer).
    - Roster Size = Exactly 6 active players.
    - Role Multi-choice bounds: Exactly 1 Duelist, 1 Initiator, 1 Controller, 1 Sentinel, and 2 Wildcard slots (flex slots accepting any remaining player role).
    - Maximum of 2 players mapped to the same scraped VLR Team ID.
    - Tournament Survival: Cross-reference upcoming schedules to prioritize players viable to advance across long tournament phases.
  - **Reward Function Weights & Rules**:
    - Heavily weight players on teams projected by the core model to secure Map Wins (+1), 5-9 round margins (+1), or 10+ round sweeps (+2).
    - Drop-lowest rule: Formulate point estimations capping player scoring at their highest 2 maps per gameweek.
    - Build a severe penalty multiplier matrix preventing the selection of two players scheduled to face each other in a head-to-head matchup during the active gameweek.
    - Piecewise kill point mapping (-3 for 0 kills, -1 for 1-4, +1 at 10 kills, +1 per 5 kills thereafter).
    - Dynamic IGL Selector: Isolate the player in the optimal roster who yields the highest statistical point floor and apply a strict 2x multiplier to their output.

## [x] Phase 3.1.4: VFL Streamlit Hub Integration (`app.py`)
- [x] Build out a dedicated visual dashboard layout titled **"VFL Fantasy Manager Hub"**.
- [x] Render the **VCT 2026 Stage 2 Optimal Lineup** using visual role-labeled cards, player costs, and active VLR team identifiers.
- [x] Implement an interactive **"3-Transfer Advisor"** that reads an input user roster, references upcoming gameweek matchups via the arbitrary simulation engine, and highlights the 3 optimal trades to maximize score velocity.