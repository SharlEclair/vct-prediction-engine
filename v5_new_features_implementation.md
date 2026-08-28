# V5 Engine Refinements: Patch Alignment, Outlier Removal, and Map Pools

**Objective:** Address edge cases in professional esports scheduling (patch delays) and statistical anomalies (blowout matches) to refine model accuracy.

## 1. Simulation Patch Override & UI Alignment

**Context:** VCT tournaments frequently play on older patches (e.g., playing on 9.02 when live is 9.04) to maintain competitive stability.

### Implementation

**UI Update (`app.py`):**

* In the Match Configuration section, add a dropdown for **Target Simulation Patch**.

**Dynamic Default:**

* The backend must calculate the standard patch based on the match date, but set it as the default index of the dropdown, allowing the user to manually override it to a previous patch.

**Labeling:**

* Format the dropdown options to include the release date for context:

  * `"Patch 8.11 (Released: June 11, 2024)"`

**Backend Routing:**

* Ensure the selected patch string from the UI overrides the date-inferred patch when passed into the simulation configuration payload, dictating the NLP nerf penalties applied in Sub-Model 2.

## 2. Statistical Outlier Removal (Clipping)

**Context:** A player dropping 400 ACS in a 13-0 game against a disbanded team artificially skews their Bayesian smoothed average, breaking the draft utility matrix.

### Implementation

**Target Module:**

* `get_simulation_historical_stats` (or equivalent data aggregation layer).

**Methodology:**

* Implement Interquartile Range (IQR) clipping or percentile capping on historical performance arrays before applying the Bayesian formula.

**Math:**

* For a player's historical ACS array on a specific agent/map, calculate the 5th and 95th percentiles.
* Floor all values below `P₅` to `P₅`, and cap all values above `P₉₅` to `P₉₅`.

This removes the extreme variance of anomaly games while retaining the sample size count (`N`) required for the Bayesian prior weighting.

## 3. Strict Temporal Map Pool Enforcement

**Context:** The Map Veto Bandit cannot output "Pearl" for a match happening in 2026.

### Implementation

**Data Structure (`constants.py` or similar):**

* Create a chronological registry of map pools:

```python
TEMPORAL_MAP_POOLS = [
    {
        "start_date": "2023-01-01",
        "end_date": "2023-09-08",
        "pool": ["Ascent", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
    },
    # ... up to current patch
]
```

**Execution (`v5_simulation_engine.py`):**

* Inside `MapVetoBandit.__init__` or `simulate_veto`, cross-reference the `reference_date` against `TEMPORAL_MAP_POOLS`.
* Intersect the historical IPS (Inverse Propensity Score) win-rate dictionaries to only include the 7 maps returned by the temporal registry.
* Any map outside the active pool must be mathematically excluded from the bandit's arm selection (`𝒜`) prior to generating the veto sequence.
