# V5 Architecture Proposal: Temporal Map Registry & Global Player Ledger

> **Design-only document. No production code will be written until this is approved.**

---

## Part 0 — Skills Repository Audit

The `antigravity-awesome-skills` repository was accessed at `https://github.com/sickn33/antigravity-awesome-skills`. The repository root rendered with a JavaScript loading error (the page requires authenticated session state to enumerate the full `/skills` tree), but the top-level directory listing was successfully recovered. The relevant directories are:

| Directory | Relevance to our problem |
|---|---|
| `skills/` | 1,681+ skill SKILL.md files (tree not enumerable without auth) |
| `schemas/` | JSON schema definitions — **directly applicable** to our registry design |
| `data/` | Data handling patterns |
| `skill_categorization/` | Taxonomy metadata |

**Assessment:** No specific "Data Normalization" or "Entity Resolution" SKILL.md files could be confirmed as directly installable without authentication. However, the **design patterns from the `schemas/` directory are directly applicable**:

- **Registry Pattern** (versioned key-value store with temporal validity windows) → maps directly to our Temporal Map Pool Registry
- **Entity Ledger Pattern** (player identity decoupled from organizational context) → maps directly to our Global Player Ledger

We will apply these standard patterns from first principles, as the source code analysis yields sufficient grounding.

---

## Part 1 — Current Architecture Diagnosis

### Bug 1: Static Hardcoded Map Pool in `MapVetoBandit`

**Location:** [`v5_simulation_engine.py` L26](file:///c:/Users/91704/Desktop/vct-prediction-model/v5_simulation_engine.py#L18-L30)

```python
# CURRENT — BROKEN
self.map_pool = [
    "Ascent", "Bind", "Breeze", "Icebox", "Lotus",
    "Split", "Sunset", "Fracture", "Haven", "Pearl"
]
```

This is a **union superset** of every map that has ever appeared in competitive Valorant — not the pool active at any given point in time. The consequences are:

1. `fit()` accumulates IPS propensity scores for retired maps (Fracture, Pearl, Breeze) against live matches where those maps were not in the pool, artificially inflating their propensity denominator.
2. `predict_veto()` can select retired maps as picks in 2026 simulations.
3. Historical win rates on Pearl (retired mid-2024) pollute the bandit's utility estimate for current maps because `map_frequency` is calculated over ALL time.

**The `fit()` method iterates all match files with no date filter at L33-72.** The temporal context embedded in `parse_simulation_match_date` is available but completely unused by the Bandit.

---

### Bug 2: Team-Anchored Data Sparsity in `get_simulation_historical_stats`

**Location:** [`v5_simulation_engine.py` L922-1002](file:///c:/Users/91704/Desktop/vct-prediction-model/v5_simulation_engine.py#L922-L1002)

```python
# CURRENT — player_emas is keyed purely by player name
player_emas[p_name] = {
    "acs": stats['sum_acs'] / stats['count'],   # global lifetime average
    "kast": 0.72,       # HARDCODED — not actually derived per player
    "duel_diff": 0.01   # HARDCODED — not actually derived per player
}
```

**Critical findings:**

1. `kast` and `duel_diff` are **never computed from match data** — they are hardcoded constants for every player. The baseline_lookup from `player_stats.json` has real kast/duel_diff but the EMA ignores it.
2. The `player_agent_stats` keys `(p_name, agent)` and `(p_name, map_name, agent)` are **already team-decoupled** — this is actually correct. The problem is the **EMA layer above it** is a naive lifetime average with no timestamp weighting, no team-cohesion signal, and no transfer-event awareness.
3. `get_simulation_roster()` resolves from the **single most recent match file** — if a player transferred between data harvests, their entire prior comfort history is still attributed correctly (since `player_agent_stats` is team-agnostic), but their EMA baseline could be stale.

---

## Part 2 — File Storage Changes

### New Files to Add

```
data/
├── raw/                         # existing
├── processed/                   # existing
│   ├── automated_patch_nerf_registry.json   # existing
│   ├── temporal_map_registry.json           # NEW — Part 3
│   └── global_player_ledger.json            # NEW — Part 4
```

No existing files are deleted or restructured. Both new files are **additive processed outputs** that the engine reads at boot.

---

## Part 3 — Temporal Map Pool Registry

### 3.1 JSON Schema

**File:** `data/processed/temporal_map_registry.json`

```json
{
  "_schema_version": "1.0",
  "_generated_at": "2026-06-24T05:20:00Z",
  "_note": "Maps listed as active for competitive VCT play within each patch window",
  "patch_windows": [
    {
      "window_id": "2023_act1",
      "start_patch": "6.00",
      "end_patch": "6.08",
      "start_date_approx": "2023-01-10",
      "end_date_approx": "2023-04-25",
      "active_maps": ["Ascent", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"],
      "entering_maps": ["Lotus"],
      "retiring_maps": []
    },
    {
      "window_id": "2023_act2",
      "start_patch": "6.10",
      "end_patch": "7.04",
      "start_date_approx": "2023-04-25",
      "end_date_approx": "2023-08-29",
      "active_maps": ["Ascent", "Bind", "Breeze", "Haven", "Lotus", "Pearl", "Split"],
      "entering_maps": ["Breeze"],
      "retiring_maps": ["Fracture"]
    },
    {
      "window_id": "2023_act3",
      "start_patch": "7.05",
      "end_patch": "8.11",
      "start_date_approx": "2023-08-29",
      "end_date_approx": "2024-01-09",
      "active_maps": ["Ascent", "Bind", "Breeze", "Haven", "Icebox", "Lotus", "Split"],
      "entering_maps": ["Icebox"],
      "retiring_maps": ["Pearl"]
    },
    {
      "window_id": "2024_act1",
      "start_patch": "8.11",
      "end_patch": "9.09",
      "start_date_approx": "2024-01-09",
      "end_date_approx": "2024-08-27",
      "active_maps": ["Ascent", "Abyss", "Bind", "Haven", "Icebox", "Lotus", "Split"],
      "entering_maps": ["Abyss"],
      "retiring_maps": ["Breeze"]
    },
    {
      "window_id": "2025_act1",
      "start_patch": "9.10",
      "end_patch": "11.99",
      "start_date_approx": "2024-08-27",
      "end_date_approx": "2026-01-01",
      "active_maps": ["Ascent", "Abyss", "Bind", "Haven", "Icebox", "Lotus", "Sunset"],
      "entering_maps": ["Sunset"],
      "retiring_maps": ["Split"]
    },
    {
      "window_id": "2026_current",
      "start_patch": "12.00",
      "end_patch": "9999.99",
      "start_date_approx": "2026-01-01",
      "end_date_approx": null,
      "active_maps": ["Ascent", "Abyss", "Bind", "Haven", "Icebox", "Lotus", "Sunset"],
      "entering_maps": [],
      "retiring_maps": []
    }
  ]
}
```

### 3.2 Registry Resolver Interface (conceptual)

The resolver is a stateless lookup function — **not a class with internal state**. This is intentional: the Bandit needs to call it at `fit()` time per-match, and at `predict_veto()` time for the target match date.

```python
# CONCEPTUAL — design only
def resolve_active_map_pool(match_date: datetime, registry: dict) -> list[str]:
    """
    Returns the 7-map competitive pool active on match_date.
    Falls back to the most recent window if date exceeds all ranges.

    Args:
        match_date: The datetime of the match being evaluated
        registry:   The loaded temporal_map_registry.json dict

    Returns:
        list of map name strings (always exactly 7 for standard VCT)
    """
    for window in reversed(registry["patch_windows"]):
        window_start = datetime.fromisoformat(window["start_date_approx"])
        window_end = (
            datetime.fromisoformat(window["end_date_approx"])
            if window["end_date_approx"] else datetime.max
        )
        if window_start <= match_date < window_end:
            return window["active_maps"]
    # Fallback: return most recent window
    return registry["patch_windows"][-1]["active_maps"]
```

### 3.3 Adapter — How `MapVetoBandit.fit()` Changes

The bandit currently builds a single flat `team_plays` and `team_wins` dict over all time. After this change, these become **temporally-scoped** — each match observation is tagged with its active pool, and the propensity is computed **within pool cohort**, not across all history.

**New internal data model for the Bandit:**

```python
# CONCEPTUAL dictionary structure — not code to implement yet

# OLD (current)
self.team_plays = {
    "Team Liquid": {
        "Fracture": 12,  # 2023 data bleeds into 2026 model
        "Lotus": 8,
        "Pearl": 6,
        ...
    }
}

# NEW (proposed)
self.team_plays_by_pool = {
    "2026_current": {
        "Team Liquid": {
            "Ascent": 14,
            "Lotus": 9,
            ...
        }
    },
    "2025_act1": {
        "Team Liquid": {
            "Sunset": 6,
            "Lotus": 11,
            ...
        }
    }
}

# Active pool utility (computed at predict-time from above)
self.cross_pool_win_rates = {
    "Team Liquid": {
        "Lotus": {
            "active_windows": ["2023_act1", "2024_act1", "2025_act1", "2026_current"],
            "ips_weighted_win_rate": 0.61,  # computed across all windows where map was active
            "recency_decay": 0.88           # exponential decay from most recent appearance
        }
    }
}
```

**The key insight:** A map like Lotus has been in the pool continuously. Its win rates should aggregate across all windows with a recency decay. A map like Fracture was retired — its win rates **must not** appear in `available_maps` during `predict_veto()` for 2026 matches, but **must** appear correctly when simulating or backtesting 2023 matches.

### 3.4 `predict_veto()` Context Vector Pruning

```python
# CONCEPTUAL — the change to predict_veto signature

# OLD
def predict_veto(self, team_a, team_b, series_type, stochastic=False):
    available_maps = list(self.map_pool)  # hardcoded superset

# NEW
def predict_veto(self, team_a, team_b, series_type, stochastic=False,
                 target_date=None):
    active_pool = self.resolve_pool(target_date or datetime.now())
    available_maps = list(active_pool)  # dynamically resolved
```

The bandit's arm set $\mathcal{A}$ is now:

$$\mathcal{A}(t) = \text{resolve\_active\_map\_pool}(t, \mathcal{R})$$

where $\mathcal{R}$ is the temporal registry and $t$ is the target match date. Arms for retired maps are pruned before the IPS scoring step.

### 3.5 Historical Performance Decay Across Pool Rotations

When a map **re-enters** the pool after rotation (e.g., Bind has left and re-entered multiple times), historical win rates from the **previous rotation** are valid priors but need a **rotation re-entry decay factor** $\rho$:

$$\hat{W}_{re\text{-}entry}(team, map) = \rho \cdot \hat{W}_{prior\text{-}rotation}(team, map) + (1 - \rho) \cdot \bar{W}_{global}$$

Suggested $\rho = 0.65$ for re-entering maps, $\rho = 1.0$ for continuously-active maps. This is a tunable hyperparameter, not a hardcoded constant.

---

## Part 4 — Global Player Entity Ledger

### 4.1 JSON Schema

**File:** `data/processed/global_player_ledger.json`

```json
{
  "_schema_version": "1.0",
  "_generated_at": "2026-06-24T05:20:00Z",
  "players": {
    "aspas": {
      "display_name": "aspas",
      "canonical_id": "aspas",
      "career_stats": {
        "global_acs_ema": 247.3,
        "global_kast_ema": 0.742,
        "global_duel_diff_ema": 0.087,
        "total_maps_played": 312,
        "career_start_date": "2021-03-01"
      },
      "team_history": [
        {
          "team_name": "LOUD",
          "team_id": "loud",
          "joined_date": "2022-01-01",
          "departed_date": null,
          "maps_played_with_team": 289,
          "acs_with_team": 251.0
        },
        {
          "team_name": "NIP",
          "team_id": "nip",
          "joined_date": "2021-01-01",
          "departed_date": "2021-12-31",
          "maps_played_with_team": 23,
          "acs_with_team": 221.3
        }
      ],
      "agent_comfort": {
        "Jett": {
          "global_maps": 118,
          "global_acs_avg": 263.1,
          "per_map_comfort": {
            "Ascent": {"maps": 18, "acs_avg": 271.4},
            "Bind":   {"maps": 14, "acs_avg": 258.9}
          }
        },
        "Neon": {
          "global_maps": 67,
          "global_acs_avg": 241.7,
          "per_map_comfort": {
            "Lotus": {"maps": 11, "acs_avg": 249.3}
          }
        }
      },
      "cohesion_windows": [
        {
          "team_id": "loud",
          "window_start": "2022-01-01",
          "maps_together": 289,
          "cohesion_score": 0.94
        }
      ]
    }
  }
}
```

### 4.2 Why This Schema Solves the Problem

The critical design decision is that **`agent_comfort` is stored under the player's canonical identity**, not under a team key. When `aspas` transfers from NIP to LOUD:

- **Old behaviour:** `get_simulation_roster("LOUD")` finds aspas in LOUD matches → fine, but if LOUD has zero historical data, aspas gets `baseline_lookup` defaults.
- **New behaviour:** The ledger resolves aspas's full `global_acs_ema: 247.3` immediately, regardless of which team they're currently listed under. The `team_history` array tracks the organizational context for cohesion computation only.

### 4.3 Cohesion Coefficient $CF$

The Roster Cohesion Coefficient quantifies how long a current 5-man lineup has played together. It gates the variance spread in the Dirichlet kill-share model.

$$CF(player_i, team_k) = \frac{\min(\text{maps\_with\_team}_k, M_{sat})}{M_{sat}}$$

where $M_{sat} = 25$ maps is the saturation threshold (after 25 maps together, cohesion is considered fully established).

$$CF \in [0, 1]$$

**Effect on Dirichlet alpha parameters:**

```python
# CONCEPTUAL
# Current: alpha is fixed from role + historical ACS
alpha_scaled = alpha_0 * exp(0.004 * (acs - 200.0) + 0.3 * duel_diff)

# Proposed: alpha uncertainty is widened when CF is low
# Low CF = new signing = wider variance = more uncertain kill share
cohesion_factor = CF(player, current_team)
alpha_scaled = alpha_0 * exp(0.004 * (acs - 200.0) + 0.3 * duel_diff)
alpha_final = alpha_scaled * (0.6 + 0.4 * cohesion_factor)
#   CF=1.0 → alpha_final = alpha_scaled * 1.0 (no change, full confidence)
#   CF=0.0 → alpha_final = alpha_scaled * 0.6 (40% variance expansion)
```

The raw mechanical skill prior (`global_acs_ema`) is **not penalized by CF**. Only the concentration parameter of the Dirichlet distribution is widened. This correctly models the situation: a new signing is still as individually skilled, but their kill *distribution* within an unfamiliar team structure is less predictable.

### 4.4 Adapter — How `AgentCompositionTransformer.predict_composition()` Changes

The Assignment Solver currently builds its utility matrix from `self.agent_comfort_matrix` keyed by `(player, map_name, agent)`. The ledger replaces the source of these values but **does not change the utility matrix interface**.

```python
# CONCEPTUAL — new comfort resolution

# OLD (from get_simulation_historical_stats)
comfort_stat = self.agent_comfort_matrix.get(
    (player, map_name, agent),
    {"sum_acs": 0.0, "count": 0}
)
map_agent_acs = comfort_stat["sum_acs"] / count if count > 0 else 0.0

# NEW (from global_player_ledger)
def resolve_comfort(player_id, map_name, agent, ledger, cohesion_coeff):
    player_data = ledger["players"].get(player_id, {})
    agent_data = player_data.get("agent_comfort", {}).get(agent, {})
    map_specific = agent_data.get("per_map_comfort", {}).get(map_name)

    if map_specific and map_specific["maps"] >= 3:
        # Enough map-specific data: use it directly
        map_acs = map_specific["acs_avg"]
        count = map_specific["maps"]
    elif agent_data.get("global_maps", 0) > 0:
        # Fall back to global agent comfort
        map_acs = agent_data["global_acs_avg"]
        count = agent_data["global_maps"]
    else:
        # Agent never played by this player
        map_acs = player_data.get("career_stats", {}).get("global_acs_ema", 200.0)
        count = 0

    # Bayesian smoothing: blend toward global career baseline
    alpha_smooth = 3.0
    global_acs = player_data.get("career_stats", {}).get("global_acs_ema", 200.0)
    bayesian_acs = (count * map_acs + alpha_smooth * global_acs) / (count + alpha_smooth)

    return bayesian_acs, count, cohesion_coeff
```

The output of `resolve_comfort()` plugs directly into the existing utility matrix loop at line 270 of [v5_simulation_engine.py](file:///c:/Users/91704/Desktop/vct-prediction-model/v5_simulation_engine.py#L270-L315) — **the scipy `linear_sum_assignment` call is unchanged**.

### 4.5 EMA Computation — Fixing the Hardcoded kast/duel_diff

The current code hardcodes `kast: 0.72` and `duel_diff: 0.01` for every player in the EMA. The ledger introduces properly computed values:

```python
# CONCEPTUAL — new career_stats fields populated at ledger build time

"career_stats": {
    "global_acs_ema": 247.3,         # exponentially weighted moving average, λ=0.1
    "global_kast_ema": 0.742,        # properly computed from match data, not hardcoded
    "global_duel_diff_ema": 0.087,   # first_kills_per_round - first_deaths_per_round, EMA weighted
    "acs_ema_alpha": 0.10,           # decay factor used in EMA construction
    "total_maps_played": 312,
    "career_start_date": "2021-03-01",
    "last_updated": "2026-06-24T05:00:00Z"
}
```

The EMA formula:

$$\text{EMA}_{t} = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$

with $\alpha = 0.10$ (10 effective lookback maps). This replaces the current simple average `stats['sum_acs'] / stats['count']` which weights a match from 2021 equally to a match from last week.

---

## Part 5 — Full Pipeline Data Flow Diagram

```
RAW MATCH FILES (data/raw/match_*.json)
    │
    │  parse_simulation_match_date()
    │  extract: match_date, teams, players, agents, acs, map_name
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│            LEDGER BUILDER (new: build_ledger.py)             │
│                                                              │
│  For each match, chronologically sorted:                     │
│    1. Resolve window_id from temporal_map_registry.json      │
│    2. Confirm map was active in this window (skip if not)    │
│    3. Update player's global EMA (kast, acs, duel_diff)     │
│    4. Update player's agent_comfort[agent][map] stats        │
│    5. Update team_history entry (cohesion window counter)    │
│                                                              │
│  Output: global_player_ledger.json                           │
└──────────────────────────────────────────────────────────────┘
    │                              │
    ▼                              ▼
temporal_map_registry.json    global_player_ledger.json
    │                              │
    └───────────┬───────────────────┘
                ▼
    VCTv5SimulationEngine.__init__()
        │
        ├── MapVetoBandit.fit(registry)
        │       Pools partitioned by window_id
        │       IPS propensity per-window, not global
        │
        └── AgentCompositionTransformer.load_ledger(ledger)
                resolve_comfort() replaces agent_comfort_matrix lookup
                CF gates Dirichlet alpha spread
```

---

## Part 6 — Open Design Questions (Requiring Your Input)

> [!IMPORTANT]
> **Q1 — Ledger Build Frequency:** Should `global_player_ledger.json` be rebuilt on every scrape run, or treated as an append-only log? An append-only approach is safer (we never lose history) but requires a merge strategy when the same player appears in new matches.

> [!IMPORTANT]
> **Q2 — Transfer Date Source of Truth:** The ledger's `team_history[].joined_date` requires knowing when a player transferred. VLR.gg match data doesn't directly expose this. Options: (a) infer from first match appearance under new team banner, (b) add a manual `data/raw/player_transfers.json` override file, (c) ignore the join date and use rolling match counts only. Option (b) is most accurate but requires maintenance.

> [!WARNING]
> **Q3 — Cohesion Coefficient Saturation Threshold:** The proposed $M_{sat} = 25$ maps is a prior assumption, not empirically validated. It should be calibrated against your comparison_results.csv backtest data before being locked in.

> [!NOTE]
> **Q4 — Map Re-Entry Decay ($\rho$):** The proposed $\rho = 0.65$ for re-entering maps is a reasonable prior but could be tuned. Bind is the best test case — it has left and re-entered the pool at least twice. We could measure whether the pre-retirement win rates on Bind predict post-re-entry win rates, which would empirically calibrate $\rho$.

> [!NOTE]
> **Q5 — Backwards Compatibility:** `predict_match.py` and `app.py` both call `get_historical_stats()` from `predict_match.py` (a separate function from `get_simulation_historical_stats()`). The ledger only replaces the simulation engine's data path. The app's match analysis tab would continue using the existing player_emas from `predict_match.py` until a second migration is planned.

---

## Part 7 — Implementation Scope (For Future Approval)

When approved, implementation would consist of exactly **3 new components** and **2 modified components**:

| Component | Type | Scope |
|---|---|---|
| `build_temporal_map_registry.py` | **NEW** script | Writes `temporal_map_registry.json` from hardcoded patch research + scraper |
| `build_global_player_ledger.py` | **NEW** script | Reads all match files, produces `global_player_ledger.json` |
| `TemporalMapRegistry` class | **NEW** class in `v5_simulation_engine.py` | Stateless resolver; `resolve_pool(date)` method only |
| `MapVetoBandit.fit()` | **MODIFY** | Accept registry; partition `team_plays` by `window_id` |
| `AgentCompositionTransformer` | **MODIFY** | Accept ledger; `resolve_comfort()` replaces dict lookup |

`simulate_match()`, `KillShareDirichlet`, `BivariatePoissonMCMC`, and all downstream app code remain **unchanged**.
