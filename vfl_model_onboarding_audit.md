# VFL Prediction Engine — Complete Technical Onboarding Audit

**Version:** VCT 2026 Stage 2  
**Architecture Phase:** 21 (Current)  
**Last Updated:** July 2026

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Sources & What Is Scraped](#2-data-sources--what-is-scraped)
3. [What Data Is Used vs Discarded](#3-what-data-is-used-vs-discarded)
4. [VFL Scoring Rules (Official)](#4-vfl-scoring-rules-official)
5. [VFL Roster Constraints](#5-vfl-roster-constraints)
6. [Patch Analysis Pipeline](#6-patch-analysis-pipeline)
7. [Feature Engineering](#7-feature-engineering)
8. [Prediction Models & Parameters](#8-prediction-models--parameters)
9. [The EV (Expected Value) Computation](#9-the-ev-expected-value-computation)
10. [Roster Optimization: MILP Engine](#10-roster-optimization-milp-engine)
11. [Hardcoded Values](#11-hardcoded-values)
12. [Adaptive / Dynamically-Adjusted Values](#12-adaptive--dynamically-adjusted-values)
13. [Sample Data: Raw → Processed](#13-sample-data-raw--processed)
14. [File Map & Responsibilities](#14-file-map--responsibilities)
15. [Known Gaps & Improvement Opportunities](#15-known-gaps--improvement-opportunities)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                    │
│  VFLScraper (REST API)  │  VLRScraper (HTML)  │  Patches  │
└────────────┬────────────┴──────────┬──────────┴─────┬──────┘
             │                       │                 │
             ▼                       ▼                 ▼
┌──────────────────┐  ┌──────────────────────┐  ┌──────────────┐
│ vfl_currentevent │  │  match_*.json (raw)  │  │ patch_impact │
│ .json (180 p.)   │  │  (1,925 matches)     │  │ _trace.json  │
└────────┬─────────┘  └──────────┬───────────┘  └──────┬───────┘
         │                       │                      │
         ▼                       ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                 │
│   feature_engineering.py  │  global_player_ledger.json      │
│   X_features.csv (match)  │  bayesian_player_ledger.json    │
└─────────────────────────────────────┬───────────────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     ▼                ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
           │ CatBoost     │  │ CatBoost     │  │ VetoPredictor   │
           │ Match Win    │  │ Map Score    │  │ (rule-based)    │
           │ Predictor    │  │ Regressor    │  │                 │
           └──────┬───────┘  └──────┬───────┘  └────────┬────────┘
                  │                 │                    │
                  └─────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    FANTASY ENGINE LAYER                     │
│  compute_all_players_historical_stats()                     │
│  compute_all_players_opponent_stats()                       │
│  blend_ev() → PPG + H2H weighting                          │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  MILP ROSTER OPTIMIZER                      │
│  scipy.optimize.milp()  │  IGL Loop (k=0..n candidates)    │
│  11-player / 50VP cap   │  H2H penalty variables            │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Streamlit UI (app.py)  │
                    │  Transfer Advisor       │
                    │  Optimal/Pred Rosters   │
                    │  GW Diagnostics         │
                    └─────────────────────────┘
```

---

## 2. Data Sources & What Is Scraped

### 2.1 VFL REST API — Primary Fantasy Data

**Scraper:** [`scrapers/vfl_scraper.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/scrapers/vfl_scraper.py)  
**Transport:** `curl_cffi` with Chrome impersonation (bypasses Cloudflare WAF)

**Endpoints:**

| Endpoint | Method | What It Returns |
|---|---|---|
| `/api/event/currentevent` | GET | Full event metadata + all 180 players + schedule |
| `/api/matches/schedule?gameweek={n}&eventId={id}` | GET | Per-gameweek match schedule + team names |
| `/api/player/allplayers?eventId={id}` | GET | Legacy fallback player list (not primary) |

**Fields scraped from `/api/event/currentevent`:**

```json
{
  "id": 10,
  "name": "VCT 2026 : Stage 2",
  "currentGameweek": 3,
  "budget": 100,
  "matchRegions": [{"vlrRegion": "eu"}, ...],
  "vlrEvents": [{"vlrEventId": "2274"}],
  "eventPlayers": [ ... 180 player objects ... ],
  "eventMatches": [ ... all scheduled matches ... ]
}
```

**Full `eventPlayer` schema (one entry):**

```json
{
  "id": 14419,
  "eventId": 10,
  "playerId": 14419,
  "playerRole": 3,
  "price": 5.5,
  "currentGameweekPoints": { "totalPoints": 12 },
  "totalEventPoints": { "totalPoints": 12 },
  "teamId": 2168,
  "player": { "name": "neT", "id": 14419 },
  "team": {
    "name": "GIANTX",
    "shortName": "GX",
    "id": 2168
  },
  "eventPointHistory": [
    { "gameweek": 1, "points": { "killPoints": 3, "mapPoints": 3, "bonusPoints": 0, "totalPoints": 6 } },
    { "gameweek": 2, "points": { "killPoints": 5, "mapPoints": 5, "bonusPoints": 2, "totalPoints": 12 } }
  ],
  "standInPlayers": []
}
```

**What the scraper maps to local schema:**

```json
{
  "player_name": "neT",
  "vlr_team_id": 2168,
  "team_name": "GIANTX",
  "team_short": "GX",
  "role": "Sentinel",
  "price": 5.5,
  "gw_pts": 12.0,
  "tot_pts": 18.0,
  "ppg": 18.0,
  "event_id": 10,
  "event_name": "VCT 2026 : Stage 2"
}
```

> **Note:** `ppg` here is computed as `tot_pts / max(gameweeks_played, 1)`. This is the VFL event-level PPG, not the historical VLR match database PPG.

**Cache files:**
- `data/processed/vfl_currentevent.json` — full event dump (180 players + schedule)
- `data/processed/vfl_players_db.json` — mapped player list (stripped schema)

---

### 2.2 VLR.GG Match Scraper — Historical Match Data

**Scraper:** [`scrapers/vlr_scraper.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/scrapers/vlr_scraper.py)  
**Transport:** `curl_cffi` Chrome impersonation + `selectolax` HTML parser  
**Anti-rate-limit:** Random sleep 3.0–5.5 seconds per request  
**Tier filter:** Only VCT/Masters/Champions events (blacklists Challengers, Ascension, GC, Premier)

**Each match page scrapes:**
- Match event, date, patch version, map veto sequence
- Per-map: map name, who picked it, score (team1/team2 rounds), round-by-round economy
- Per-map per-player: name, agent, ACS, kills, deaths, assists, K/D diff, KAST, ADR, HS%, FK, FD
- Advanced stats tab: multi-kill counts (4K, 5K, 6K, 7K), clutch rounds, 1v1/1v2/1v3/1v4/1v5 wins
- Performance tab: VLR rating (separate per map)

**Output:** `data/raw/match_{match_id}.json` (1,925 files)

**Match JSON structure:**
```
{
  "status": "ok",
  "data": {
    "segments": [
      {
        "match_id": "12345",
        "event": "VCT 2026 EMEA Stage 2",
        "date": "July 10, 2026 3:00 PM CEST",
        "map_vetos": "Team A ban Bind; Team B ban Haven; ...",
        "teams": [
          { "id": "1234", "name": "Team Heretics", "score": 2, "is_winner": true },
          { "id": "5678", "name": "NAVI", "score": 0 }
        ],
        "maps": [
          {
            "map_name": "Ascent",
            "picked_by": "Team Heretics",
            "duration": "43:12",
            "score": { "team1": 13, "team2": 5 },
            "players": {
              "team1": [
                { "name": "RieNs", "agent": "Raze", "rating": 1.82,
                  "acs": 312, "kills": 25, "deaths": 14, "assists": 4,
                  "kd_diff": "+11", "kast": 0.88, "adr": 198, "hs_pct": 0.24,
                  "fk": 4, "fd": 1, "fk_diff": "+3" }
              ],
              "team2": [ ... ]
            },
            "performance": {
              "advanced_stats": [
                { "player": "RieNs TH", "4": "2", "10": "1", "11": "71" }
              ]
            }
          }
        ]
      }
    ]
  }
}
```

---

### 2.3 Valorant Wiki — Patch Notes

**Scraper:** [`scrapers/patch_ingestor.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/scrapers/patch_ingestor.py)  
**Source:** `https://valorant.fandom.com/wiki/Patch_{version}` (MediaWiki API)  
**Transport:** `curl_cffi` GET, `csv` for version date mapping

**Scraped patches:** Last 5 patches (configurable `limit=5`)  
**CSV index:** `data/raw/patch_notes.csv` — maps `patch_version → release_date`

**Caches:**
- `data/patches/patch_{version}.txt` — raw wikitext
- `data/processed/patches/patch_{version}.json` — structured parsed output

---

## 3. What Data Is Used vs Discarded

### ✅ Used

| Data | Used For |
|---|---|
| Player name, team, role, price | Identity, role-slot assignment, budget constraint |
| `eventPointHistory[].points` | GW actual scores (optimal roster, diagnostics) |
| `currentGameweekPoints` | Current GW standing (live) |
| `totalEventPoints` | VFL-event-level PPG computation |
| `mapPoints`, `killPoints`, `bonusPoints` | Point breakdown in diagnostics |
| Match kills, ACS, rating | Historical PPG calculation |
| Advanced stats (4K, 5K) | Multi-kill bonus calculation |
| Map veto sequence | VetoPredictor fit (ban/pick/win rates) |
| Map scores (rounds) | Round margin points calculation |
| Series scores (team1/team2 series wins) | Series bonus points |
| VLR rating per map | Rating placement bonus (top 1/2/3) + scaling bonus |
| Agent picks per map | AgentCompositionGenerator comfort metrics |
| Patch notes wikitext | Patch impact trace → agent nerf/buff scoring |
| `team.shortName`, `team.id` | Active team pool matching for EV computation |
| Economy tab (avg loadout) | Feature for MapScoreRegressor training |

### ❌ Discarded / Not Used

| Data | Why Discarded |
|---|---|
| `standInPlayers` | Not integrated into roster logic |
| VLR round economy data per round | Only avg loadout used in features; round-by-round ignored |
| Streams / VOD links in match JSON | Metadata only, no analytical value |
| Match chat / social data | Not collected |
| HS% | Collected but not used in scoring |
| FD (first deaths) | Collected but not used in VFL scoring |
| Assist counts | Not used (VFL scoring does not reward assists) |
| ADR (avg damage/round) | Not used in VFL scoring; collected in ledger only |
| KAST % | Collected in ledger, not used in fantasy EV |
| Players from Academy/GC/Black/Blue teams | Filtered out at VFL scrape step |
| Players marked "inactive" | Filtered out at VFL scrape step |
| Tier-2 / Challengers / Ascension matches | Filtered by `is_tier1_event()` |
| Legacy `vfl_rules.json` scoring fields | File contains old scoring logic; actual scoring is hardcoded in `VCTFantasyEngine` |

---

## 4. VFL Scoring Rules (Official)

These are the **authoritative** rules from [`VLF Rules Regional.txt`](file:///c:/Users/91704/Desktop/vct-prediction-model/VLF%20Rules%20Regional.txt). The engine implements them in `VCTFantasyEngine`.

### Kill Points (per map)

| Kills | Points |
|---|---|
| 0 | **−3** |
| 1–4 | **−1** |
| 5–9 | **0** |
| 10 | **+1** |
| 15 | **+2** |
| 20 | **+3** |
| Every +5 above 10 | **+1 more** |

> Formula: `1 + (kills - 10) // 5` for kills ≥ 10

### Multi-Kill Points (per round)

| Round Multi-Kill | Points |
|---|---|
| 4K | **+1** |
| 5K | **+3** |
| 6K | **+5** |
| 7K | **+10** |

### Map Points (per map)

| Outcome | Points |
|---|---|
| Map win | **+1** |
| Map win by 5–9 rounds | **+1 bonus** |
| Map win by 10+ rounds | **+2 bonus** |
| 13-0 sweep | **+5 total** |
| Map loss by 10+ rounds | **−1** |
| 0-13 loss | **−5 total** |

### Series Bonus (per match)

| Series Result | Points |
|---|---|
| 2-0 BO3 win | **+2** |
| 3-0 BO5 win | **+4** |
| 3-1 BO5 win | **+1** |

> Note: 3-2 BO5 and 2-1 BO3 give no bonus.

### Rating Placement Bonus (per match)

| VLR Rating Rank | Points |
|---|---|
| 1st highest avg rating | **+3** |
| 2nd highest | **+2** |
| 3rd highest | **+1** |

### Rating Absolute Scaling Bonus (per match)

| VLR Rating | Points |
|---|---|
| ≥ 1.5 | **+1** |
| ≥ 1.75 | **+2** |
| ≥ 2.0 | **+3** |

### IGL Multiplier

The designated IGL receives **2× all points scored** (both positive and negative) that gameweek.

### Map Cap Rule

Per gameweek, players only score their **top 2 map scores** (if they play 3+ maps, the worst map is discarded). This is implemented in the scoring engine:

```python
# From VCTFantasyEngine.score_match_json()
sorted_map_scores = sorted(map_scores_list)
top_2_scores = sorted_map_scores[-2:] if len(sorted_map_scores) >= 2 else sorted_map_scores
map_score_agg = sum(top_2_scores)
```

---

## 5. VFL Roster Constraints

### Regional Format (11-player squad, 50VP cap)

From [`VLF Rules Regional.txt`](file:///c:/Users/91704/Desktop/vct-prediction-model/VLF%20Rules%20Regional.txt):

| Parameter | Value |
|---|---|
| **Squad size** | **11 players** |
| **Budget cap** | **100 VP** (50 VP per half — engine uses 50 per MILP run) |
| **Core role slots** | 2 Duelist + 2 Initiator + 2 Controller + 2 Sentinel |
| **Wildcard slots** | **3** (any role) |
| **Max from same team** | **2** |
| **Starting roster** | All 11 score each GW |
| **IGL** | 1 per team, 2× multiplier |
| **Free transfers** | Unlimited before GW1 deadline |
| **Transfers per GW (after)** | 3 free per GW, non-stacking |

### International Format (6-player squad, 50VP cap)

Used for Masters / Champions events:

| Parameter | Value |
|---|---|
| **Squad size** | **6 players** |
| **Budget cap** | **50 VP** |
| **Core role slots** | 1 Duelist + 1 Initiator + 1 Controller + 1 Sentinel |
| **Wildcard slots** | **2** (any role) |
| **Max from same team** | **2** |

The engine switches between formats by checking `roster_size == 11` vs 6 in `optimize_roster()`.

---

## 6. Patch Analysis Pipeline

### 6.1 Patch Ingestion

**File:** [`scrapers/patch_ingestor.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/scrapers/patch_ingestor.py)

1. Reads `data/raw/patch_notes.csv` for version → release date mapping
2. Fetches last 5 patch wikitext pages from Valorant Fandom Wiki
3. Saves raw wikitext to `data/patches/patch_{version}.txt`

### 6.2 LLM-Based Patch Parsing

**File:** [`v8_patch_parser.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_patch_parser.py)

The patch parser uses a **schema-driven LLM extraction pipeline** (Gemini/GPT via httpx REST) with strict **Pydantic v2 validation**:

**Pydantic Schema:**
```python
class PatchChangeItem(BaseModel):
    agent: str           # e.g., "Neon", "Jett", "Vandal"
    ability: str         # e.g., "High Gear", "Tailwind", "Primary Fire"
    stat_modified: str   # e.g., "Slide Speed", "Duration", "Equip Time"
    old_value: Optional[Union[float, int, str]]
    new_value: Optional[Union[float, int, str]]
    is_mechanical_removal: bool   # True = movement/physics exploit fix
    raw_evidence: Optional[str]   # Original wikitext bullet point
```

**The Bug Fix Paradigm (`is_mechanical_removal`):**
- `True` — physics/movement/collision exploits, animation cancels, slide boost fixes → treated as mechanical nerfs with large impact scores
- `False` — UI, audio, text, spectator bugs → ignored (no impact on pro meta)

**System Prompt Logic:** Few-shot LLM prompt with explicit rules for nerf vs buff classification and mechanical removal detection.

**SHA-256 hashing:** Each wikitext is hashed for deduplication (avoids re-parsing unchanged patches).

### 6.3 Patch Impact Scoring

**File:** [`v8_patch_analyzer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_patch_analyzer.py) / [`patch_analyzer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/patch_analyzer.py)

Each parsed patch change generates an `impact_score` per agent:

**Sample `patch_impact_trace.json`:**
```json
{
  "9.0": {
    "Iso": {
      "score": 0.1114,
      "features": [
        { "feature": "general.double_tap", "impact": 0.0125, "reason": "nerf" },
        { "feature": "ability.duration",   "impact": 0.0333, "reason": "nerf" }
      ]
    }
  },
  "10.01": {
    "Sage": { "score": 0.8592 },
    "Tejo": { "score": 1.0 }
  }
}
```

**`automated_patch_nerf_registry.json` (agent → nerf magnitude per patch):**
```json
{
  "10.00": { "Tejo": 1.0, "Brimstone": 1.0 },
  "10.01": { "Tejo": 1.0, "Sage": 0.8592 },
  "10.02": { "Clove": 0.5, "Vyse": 0.5, "Tejo": 0.5, "Killjoy": 0.8592 }
}
```

> **⚠️ Important Gap:** The `patch_impact_trace.json` is computed and stored but its values are **not currently injected into the MILP EV objective**. Patch impact scores are visible in the UI but do not reduce/boost `computed_ppg` for players who play patched agents.

---

## 7. Feature Engineering

**File:** [`feature_engineering.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/feature_engineering.py)

Builds the training dataset `X_features.csv` (one row per match) used to train the CatBoost win predictor.

### Features extracted per match:

**Team-level features:**
- `team_a_historical_acs_ema` — Exponential moving average of ACS for team A
- `team_a_historical_avg_loadout` — Avg economy (credits) per round
- `team_a_comfort_pick_differential` — How many agents team A picks that they win more with

**Map veto features (per map slot 1–5):**
- `map_{i}_name` — Map name
- `map_{i}_veto_weight` — Model of pick/ban confidence from VetoPredictor

**Patch features:**
- `patch_version` — Active patch at match time
- `patch_shock_amplitude` — Jump-diffusion shock score for agents played

**Player comfort metrics:**
- From `AgentCompositionGenerator`: `player_map_agent_plays`, `player_agent_acs`
- Tracks per-player per-map agent preference and performance

### Global Player Ledger (EMA tracking)

**File:** `data/processed/global_player_ledger.json`

Structure per player:
```json
{
  "koalanoob": {
    "display_name": "koalanoob",
    "career_stats": {
      "global_acs_ema": 203.29,
      "global_kast_ema": 0.72,
      "global_duel_diff_ema": 0.0,
      "total_maps_played": 41,
      "career_start_date": "2023-06-09"
    },
    "team_history": [
      {
        "team_name": "FURIA",
        "joined_date": "2026-01-17",
        "maps_played_with_team": 20,
        "acs_with_team_avg": 197.75,
        "cohesion_score": 0.8
      }
    ],
    "agent_comfort": {
      "Omen": {
        "global_maps": 16,
        "global_acs_avg": 218.0,
        "per_map_comfort": {
          "Lotus": { "maps": 3, "acs_avg": 196.7 },
          "Icebox": { "maps": 5, "acs_avg": 224.2 }
        }
      }
    }
  }
}
```

> The `cohesion_score` approximates how long a player has played with their current team as a fraction of a full team tenure.

---

## 8. Prediction Models & Parameters

### 8.1 CatBoost Match Win Predictor

**File:** `data/processed/vct_model.cbm` (trained artifact)  
**Trainer:** `model_training.py`  
**Input:** `X_features.csv` (match-level features)  
**Target:** Binary win/loss for team A

**CatBoost configuration:**
```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=["map_name", "patch_version"],
    random_seed=42,
    verbose=100
)
```

**Usage:** The model outputs `P(team_a_win)` for a matchup. This feeds the UI's match outcome prediction display but is **not currently directly integrated** into the MILP EV.

### 8.2 CatBoost Map Score Regressor

**File:** `data/processed/score_regressor.cbm`  
**Class:** `MapScoreRegressor` in [`generative_pipeline.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/generative_pipeline.py)

**Input features per map:**
```python
{
  "map_name": "Ascent",           # categorical
  "veto_weight": 0.75,
  "team_a_historical_acs_ema": 235.4,
  "team_a_historical_avg_loadout": 22100.0,
  "team_a_comfort_pick_differential": 0.12,
  "team_b_historical_acs_ema": 218.7,
  "team_b_historical_avg_loadout": 19800.0,
  "team_b_comfort_pick_differential": -0.05
}
```

**Output:** `pred_diff` = predicted round differential (team_a_rounds − team_b_rounds)

**Post-processing:**
```python
if pred_diff >= 0:
    score_a = 13
    score_b = clip(round(13 - pred_diff), 0, 11)
else:
    score_b = 13
    score_a = clip(round(13 + pred_diff), 0, 11)
```

**Fallback:** If model file not found → returns `(13, 9)` hardcoded.

**CatBoost configuration:**
```python
CatBoostRegressor(
    iterations=150,
    learning_rate=0.05,
    depth=4,
    cat_features=["map_name"],
    random_seed=42
)
```

### 8.3 VetoPredictor (Rule-Based)

**File:** [`veto_predictor.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/veto_predictor.py)

Not a learned model — it's a **frequency statistics engine** that compiles:
- Per-team ban rates per map
- Per-team pick rates per map
- Per-team win rates per map (when they pick vs when opponent picks)

**Loaded at startup:** Compiles from all 1,925 match files.

**Outputs:** For a given (team_a, team_b) matchup → predicted map veto sequence with probabilities → `veto_weight` per map slot.

### 8.4 AgentCompositionGenerator

**File:** [`generative_pipeline.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/generative_pipeline.py)

Tracks per-player per-map agent comfort:
- `player_map_agent_plays[player][map][agent]` — how many times played
- `player_map_agent_acs[player][map][agent]` — cumulative ACS

Generates the **most probable agent composition** for a team on a predicted map, used in the UI's composition display.

---

## 9. The EV (Expected Value) Computation

This is the core prediction pipeline that feeds the MILP optimizer.

### Step 1: Historical PPG from VLR Matches

```python
# compute_all_players_historical_stats() in fantasy_engine.py
for each match_*.json:
    leaderboard = VCTFantasyEngine.score_match_json(match)
    player_scores[player_name].append(total_fantasy_score)

stats[player_name] = {
    "ppg":    mean(scores),          # Mean fantasy points per match
    "sigma":  std(scores),           # Standard deviation
    "cvar_90": mean(top 10%),        # Expected ceiling
    "cvar_10": mean(bottom 10%),     # Expected floor / worst case
    "matches_played": len(scores)
}
```

### Step 2: H2H (Head-to-Head) Adjustment

```python
# compute_all_players_opponent_stats() in fantasy_engine.py
for each match:
    for each player:
        h2h_scores[(player, opponent_team)].append(match_score)

# Per (player, opponent) stats:
opponent_stats[player][opponent] = {
    "ppg": mean, "sigma": std,
    "cvar_90": mean(top 10%), "cvar_10": mean(bottom 10%),
    "n_maps": count
}
```

### Step 3: Blended EV

```python
# blend_ev() in fantasy_engine.py
def blend_ev(global_stats, h2h_stats, opponent, player_name):
    global_ppg = global_stats[player].ppg    # fallback = 10.0
    global_sigma = global_stats[player].sigma  # fallback = 3.0

    if n_maps < 3:
        return global_ppg   # H2H inactive, use global only

    # Weight ramps from 30% at 3 maps → max 70% at 10 maps
    weight_h2h = min(n_maps / 10.0, 0.7)
    weight_global = 1.0 - weight_h2h

    blended_ppg = weight_h2h * h2h_ppg + weight_global * global_ppg
    blended_sigma = weight_h2h * h2h_sigma + weight_global * global_sigma
    return blended
```

### Step 4: Active Team Filter

Players on teams **not scheduled this gameweek** have their `computed_ppg` zeroed out:
```python
if not is_active and not has_precomputed:
    p_norm["computed_ppg"] = 0.0
```

Active status checked by matching `team_name`, `team_short`, or `vlr_team_id` against the `active_team_pool` from the gameweek schedule.

### Step 5: Floor Calculation

```python
floor = cvar_10  # Expected worst-case (bottom 10th percentile of historical scores)
```

The IGL is selected as the player with the highest `floor` (most consistent high-scorer) among the selected 11.

---

## 10. Roster Optimization: MILP Engine

**File:** [`fantasy_engine.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/fantasy_engine.py) — `optimize_roster()`  
**Solver:** `scipy.optimize.milp()` (Mixed-Integer Linear Programming)

### Variable Definition

For `n` candidate players:
```
x_i_nat   [0, n-1]     Binary: player i selected in their natural role slot
x_i_wild  [n, 2n-1]    Binary: player i selected as wildcard
u         [2n]         Continuous: soft budget auxiliary variable
w_k       [2n+1...]    Binary: one per H2H pair (head-to-head penalty variable)
```

**Total variables:** `2n + 1 + m` where `m` = number of H2H opponent pairs

### Objective Function (Minimize −EV)

```python
c[:n]    = −pts           # Natural slot: each player's computed_ppg
c[n:2n]  = −pts           # Wildcard slot: same EV
c[k]    −= pts[k]         # Extra −pts[k] for IGL candidate k (doubled)
c[n+k]  −= pts[k]         # (wildcard version of IGL doubling)
c[2n]    = 0.0            # No soft penalty currently
c[2n+1:] = +20.0          # Each H2H pair incurs a 20-point penalty
```

### Constraints (10 total)

| # | Constraint | Description |
|---|---|---|
| 1 | `x_i_nat + x_i_wild ≤ 1` | Player i selected at most once |
| 2 | IGL candidate forced in | The IGL loop candidate must be selected |
| 3 | `Σ(x_nat, Duelist) = 2` | Exactly 2 Duelists in core slots |
| 3 | `Σ(x_nat, Initiator) = 2` | Exactly 2 Initiators in core slots |
| 3 | `Σ(x_nat, Controller) = 2` | Exactly 2 Controllers in core slots |
| 3 | `Σ(x_nat, Sentinel) = 2` | Exactly 2 Sentinels in core slots |
| 4 | `Σ(x_wild) = 3` | Exactly 3 Wildcard slots |
| 5 | `Σ(x_i × cost_i) ≤ 50` | Hard budget cap |
| 6 | `Σ(x_i × cost_i) − u ≤ 48` | Soft budget auxiliary (unused, coeff=0) |
| 7 | Per-team sum ≤ max_per_team | Max 2 players from same VCT team |
| 8 | H2H penalty: `x_i + x_j − w_k ≤ 1` | H2H pair penalty encoding |
| 9 | Transfer constraint | Max N players not in current roster |
| 10 | Roster exclusion | Alternative roster generation (avoids exact same lineup) |

### IGL Selection Loop

The optimizer runs the MILP **once per candidate player** as the potential IGL (n iterations). For each candidate k:
1. Force player k into the roster
2. Double player k's EV coefficient
3. Solve MILP
4. Keep the best overall EV solution across all n runs

**Selection criterion:** IGL = player with **highest `cvar_10` (floor)** among selected — meaning the most consistently high-scoring player.

### Survival Filter

Before MILP, players are filtered by team win rate:
```python
wr = team_win_rates.get(team_id, 0.50)
if wr < survival_threshold (default 0.35):
    skip player  # unless in current roster
```

Win rates are computed from all 1,925 historical matches.

---

## 11. Hardcoded Values

These values are **fixed in code** and require code changes to modify:

| Value | Location | Default | Description |
|---|---|---|---|
| `salary_cap` | `optimize_roster()` default | `50` | Per-roster budget cap (VP) |
| `roster_size` | `optimize_roster()` default | `6` (International) | Squad size; app calls with 11 for Regional |
| `max_per_team` | `optimize_roster()` default | `2` | Max players per VCT team |
| `survival_threshold` | `optimize_roster()` default | `0.35` | Minimum team win rate to include players |
| H2H penalty coefficient | `fantasy_engine.py:807` | `20.0` | MILP penalty for picking players facing each other |
| Soft budget limit | `fantasy_engine.py:906` | `salary_cap − 2 = 48` | Soft cap auxiliary variable upper bound |
| IGL floor metric | `fantasy_engine.py:715` | `floor = cvar_10` | Used to rank IGL candidates |
| H2H blend min maps | `blend_ev():549` | `3` | Minimum H2H maps to activate blending |
| H2H max blend weight | `blend_ev():563` | `0.7` | Max fraction of blended EV that is H2H-derived |
| H2H weight ramp | `blend_ev():563` | `n_maps / 10.0` | Linear ramp from 30% to 70% |
| Global PPG fallback | `blend_ev():510` | `10.0` | Default PPG if no historical data exists |
| Global sigma fallback | `blend_ev():511` | `3.0` | Default std dev if insufficient matches |
| Fallback map score | `MapScoreRegressor.predict_score():143` | `(13, 9)` | Used if model not loaded |
| Kill breakpoints | `calculate_kills_points()` | −3/−1/0/+1 per tier | VFL scoring function thresholds |
| Multi-kill values | `calculate_multikill_points()` | 4K=1/5K=3/6K=5/7K=10 | VFL scoring |
| Rating scaling thresholds | `get_rating_scaling_bonus()` | 1.5/1.75/2.0 | VLR rating bonus thresholds |
| Series bonuses | `calculate_series_bonus()` | 2-0=2/3-0=4/3-1=1 | VFL series scoring |
| Round margin thresholds | `calculate_round_margin_points()` | 5-9=+1/10+=+2/13-0=+5 | VFL map margin scoring |
| CatBoost iterations (win) | `model_training.py` | `500` | Match win predictor |
| CatBoost iterations (score) | `generative_pipeline.py:111` | `150` | Map score regressor |
| CatBoost learning rate | Both models | `0.05` | Gradient boosting step size |
| CatBoost depth | Win=6, Score=4 | | Tree depth |
| Random seed | Both models | `42` | Reproducibility |
| Patch limit | `patch_ingestor.py` | `5` | How many recent patches to process |
| Tier-1 whitelist keywords | `vlr_scraper.py:68` | `['masters', 'champions', 'vct']` | Event filter |
| VLR rate-limit sleep | `vlr_scraper.py:41` | `3.0 + U(0.5, 2.5)` seconds | Anti-ban jitter |
| Top maps used per GW | `score_match_json():397` | `2` | Only top 2 map scores count per match |

---

## 12. Adaptive / Dynamically-Adjusted Values

These values **change automatically** based on scraped data:

| Value | What Adjusts It | When |
|---|---|---|
| `player_stats[player].ppg` | New match files added to `data/raw/` | On every `compute_all_players_historical_stats()` call |
| `player_stats[player].sigma` | New match files | Same as above |
| `player_stats[player].cvar_90/10` | New match files | Same as above |
| `h2h_stats[player][opponent].ppg` | New match files | On every `compute_all_players_opponent_stats()` call |
| `blend_ev.weight_h2h` | `n_maps` vs. opponent grows | Automatically ramps 0.3→0.7 as more H2H maps accumulate |
| VFL player prices | VFL API (live) | Each `get_current_event(force_refresh=True)` |
| VFL `ppg` (event-level) | VFL API (tot_pts / gw played) | Same as above |
| `gw_pts` (current GW) | VFL API | Same as above |
| `gameweek_teams` (active pool) | VFL schedule API | Each `get_schedule()` call |
| Team win rates | New match files in `data/raw/` | Recomputed by `get_team_win_rates_by_id()` |
| `VetoPredictor` ban/pick/win stats | New match files | On `fit()` call (app startup) |
| `AgentCompositionGenerator` comfort | New match files | On `fit()` call (app startup) |
| `MapScoreRegressor` predictions | Model retrained on new match data | When `fit()` is called manually |
| `global_player_ledger` ACS EMA | `build_global_player_ledger.py` | Manually run after new matches scraped |
| `patch_impact_trace.json` | New patches parsed by `patch_ingestor.py` | When new patches are ingested |
| `automated_patch_nerf_registry.json` | `v8_patch_analyzer.py` | When new patches are processed |
| `schedule_gw{n}_ev{id}.json` | VFL schedule API | When refreshed per gameweek |
| `vfl_currentevent.json` cache | VFL API | Every `force_refresh=True` call |

---

## 13. Sample Data: Raw → Processed

### Raw Match Data (simplified)

**Input:** `data/raw/match_12345.json`
```json
{
  "data": { "segments": [{
    "match_id": "12345",
    "event": "VCT 2026 EMEA Stage 2",
    "date": "July 10, 2026",
    "teams": [
      { "name": "Team Heretics", "score": 2, "id": "1234" },
      { "name": "NAVI", "score": 0, "id": "5678" }
    ],
    "maps": [{
      "map_name": "Ascent",
      "score": { "team1": 13, "team2": 5 },
      "players": {
        "team1": [{ "name": "RieNs", "kills": 25, "rating": 1.82, "acs": 312, "agent": "Raze" }],
        "team2": [{ "name": "crashies", "kills": 14, "rating": 1.10 }]
      },
      "performance": { "advanced_stats": [{ "player": "RieNs TH", "4": "2", "10": "1" }] }
    }]
  }]}
}
```

**Processed fantasy points for RieNs (map 1 of Ascent):**
```
Kill points:       kills=25 → 1 + (25-10)//5 = 1+3 = +4
Multi-kill:        4K=2 → +2, 5K=1 → +3 = +5
Round margin:      13-5 = 8 rounds → map win (+1) + 5-9 margin (+1) = +2
Series bonus:      2-0 series win → +2
Rating placement:  1st highest rating → +3
Rating scaling:    1.82 → ≥ 1.75 → +2
───────────────────────────────────────────
TOTAL (1 map):     4 + 5 + 2 + 2 + 3 + 2 = 18 pts
```

---

### Patch Pipeline Output

**Input wikitext (patch 9.0):**
```
=== Iso ===
* Double Tap: duration decreased from 20 >>> 12
```

**Parsed output:**
```json
{
  "agent": "Iso", "ability": "Double Tap",
  "stat_modified": "duration",
  "old_value": 20, "new_value": 12,
  "is_mechanical_removal": false,
  "raw_evidence": "duration decreased from 20 >>> 12"
}
```

**Impact trace result:**
```json
{ "9.0": { "Iso": { "score": 0.1114, "features": [...] } } }
```

**Nerf registry entry:**
```json
{ "9.0": { "Iso": 0.1114 } }
```

---

### EV Computation Trace (example)

**Player:** RieNs | **Opponent this GW:** NAVI

```
global_ppg     = 14.2    (mean over all historical matches)
global_sigma   =  3.1

h2h vs NAVI:
  n_maps = 6
  h2h_ppg = 16.8
  weight_h2h = min(6/10, 0.7) = 0.6
  weight_global = 0.4

blended_ppg = 0.6 × 16.8 + 0.4 × 14.2 = 10.08 + 5.68 = 15.76
floor (cvar_10) = 8.4

→ RieNs enters MILP with computed_ppg = 15.76, floor = 8.4
→ If selected as IGL: effective EV = 15.76 × 2 = 31.52
```

---

## 14. File Map & Responsibilities

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — all tabs, roster display, diagnostics |
| `fantasy_engine.py` | VFL scoring engine + MILP optimizer + EV computation |
| `feature_engineering.py` | Match feature extraction, ledger building, EMA tracking |
| `generative_pipeline.py` | MapScoreRegressor, AgentCompositionGenerator |
| `veto_predictor.py` | Map veto frequency statistics engine |
| `model_training.py` | CatBoost match win predictor training |
| `v8_patch_parser.py` | LLM-based patch note parsing (Pydantic schemas) |
| `v8_patch_analyzer.py` | Impact score computation from parsed patches |
| `patch_analyzer.py` | Legacy patch analyzer (superseded by v8) |
| `scrapers/vfl_scraper.py` | VFL REST API scraper + caching |
| `scrapers/vlr_scraper.py` | VLR.GG match HTML scraper |
| `scrapers/patch_ingestor.py` | Patch notes wikitext fetcher |
| `scrapers/incremental_vlr_scraper.py` | Incremental match fetcher (by match ID range) |
| `build_global_player_ledger.py` | Rebuilds EMA career ledger from raw matches |
| `knapsack_solver.py` | Alternative roster optimizer (legacy) |
| `copula_fusion.py` | Copula-based joint player distribution modelling (v8) |
| `v8_dros_optimizer.py` | DROS (Differential Roster Optimization) engine (v8) |
| `data/processed/vfl_currentevent.json` | Live VFL event data (180 players + schedule) |
| `data/processed/global_player_ledger.json` | Career EMA stats + team history + agent comfort |
| `data/processed/patch_impact_trace.json` | Agent impact scores per patch version |
| `data/processed/automated_patch_nerf_registry.json` | Simplified nerf scores per patch |
| `data/processed/X_features.csv` | Match-level training features (1,925 rows) |
| `data/processed/score_regressor.cbm` | Trained MapScoreRegressor model |
| `data/processed/vct_model.cbm` | Trained match win predictor model |
| `data/raw/match_*.json` | 1,925 raw VLR match files |
| `data/processed/schedule_gw{n}_ev{id}.json` | Per-GW schedule cache |
| `VLF Rules Regional.txt` | Official VFL regional scoring rules |
| `VLF Rules International.txt` | Official VFL international scoring rules |

---

## 15. Known Gaps & Improvement Opportunities

| Gap | Impact | Fix |
|---|---|---|
| **No recency decay in PPG** | High — GW1 star players are overvalued in GW2 | Decay-weight recent matches 3–5× |
| **H2H rarely activates** (requires 3 maps vs same opponent) | High — falls back to global PPG for most Champions matchups | Lower threshold to 2, add team-level proxy |
| **VetoPredictor not wired to EV** | High — map win EV not injected into MILP | `P(map_win) × avg_map_pts` per player |
| **Patch impact trace not wired to EV** | Medium — agent nerfs don't reduce predicted PPG | Multiply `computed_ppg` by `(1 − nerf_score)` |
| **No GW-to-GW calibration** | High — model doesn't learn from its own prediction errors | Post-GW calibration pass per player |
| **Static `floor = cvar_10`** for IGL | Medium — not maximising IGL EV | Option: `igl_score = ppg × 2 − price` |
| **`vfl_rules.json` contains stale rules** | Low — actual rules are hardcoded in engine, JSON is misleading | Update file or remove it |
| **MapScoreRegressor fallback = (13,9)** | Low — used when model not loaded | Make fallback configurable |
| **Assists not in VFL scoring** | N/A | By design; verified against official rules |
| **ELO not implemented** | Medium — team win rates don't adjust for opponent quality | ELO-style adjusted ratings |
