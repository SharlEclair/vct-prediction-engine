# V5 Engine Refinements: Patch Alignment, Outlier Removal, and Temporal Map Pools

## 1. Simulation Patch Override & UI Alignment

**Context:** VCT tournaments frequently play on older patches (e.g., playing on 9.02 when live is 9.04) to maintain competitive stability.

### Implementation

**UI Update (`app.py`):**

* In the "📊 Match Analysis" tab's configuration container, add a `st.selectbox` for **Target Simulation Patch**.

**Formatting:**

* Display options with release context, e.g., "Patch 9.04", "Patch 9.02", "Patch 8.11 (June 11, 2024)".

**Dynamic Default:**

* Use the `match target_date` to auto-calculate the default index, but allow user overrides.

**Backend Routing:**

* Pass the selected `target_patch` explicitly into the simulation configuration payload, dictating the NLP nerf penalties applied during the Hungarian Assignment draft.

## 2. Statistical Outlier Removal (Clipping)

**Context:** A player dropping 400 ACS in a 13-0 game against a disbanded team artificially skews their Bayesian smoothed average, breaking the draft utility matrix.

### Implementation

**Target Module:** `get_simulation_historical_stats` (in `v5_simulation_engine.py`).

**Methodology:**

* Implement 5th/95th percentile clipping on historical performance arrays before calculating the mean.

**Math:**

```python
import numpy as np

if len(acs_history) > 3:  # Only clip if we have enough sample size
    p5 = np.percentile(acs_history, 5)
    p95 = np.percentile(acs_history, 95)
    acs_history = np.clip(acs_history, p5, p95)
```

This removes the extreme variance of anomaly games while retaining the sample size count ($N$) required for the Bayesian prior weighting.

## 3. Strict Temporal Map Pool Enforcement

**Context:** The Map Veto Bandit cannot predict "Pearl" or "Fracture" for a match happening in 2026.

### Implementation

**Data Structure (`v5_simulation_engine.py`):**

* Create a chronological registry of map pools:

```python
TEMPORAL_MAP_POOLS = [
    {
        "end_date": "2023-09-08",
        "pool": ["Ascent", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
    },
    {
        "end_date": "2024-05-01",
        "pool": ["Ascent", "Bind", "Breeze", "Icebox", "Lotus", "Split", "Sunset"]
    },
    {
        "end_date": "2026-12-31",
        "pool": ["Ascent", "Bind", "Haven", "Icebox", "Lotus", "Abyss", "Sunset"]
    }
]
```

**Execution (`simulate_match` / `MapVetoBandit`):**

* Cross-reference the `target_date` against `TEMPORAL_MAP_POOLS` to retrieve the active 7-map pool.

**Arm Pruning:**

* Overwrite the bandit's available arms to strictly equal this active pool. Any historical win rates on out-of-rotation maps are ignored during the veto sequence.
