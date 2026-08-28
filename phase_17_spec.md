# Phase 17: UI v7 Integration & Visualization
**Objective:** Update `app.py` to expose the new v7 Bayesian and Stateful mechanics without removing any existing functionality, and eliminate all stale v5/v6/EMA terminology.

## 1. Stale Terminology Cleanup
* **Target:** `app.py`
* **Action:** Perform a comprehensive text-replacement:
  * "V5" / "v5" -> "v7" or "V7"
  * "v6" -> "v7"
  * "computes EMAs, and runs V5 Bottom-Up..." -> "resolves Bayesian skill states, and runs stateful economy rounds..."
  * "V5 Bottom-Up" -> "v7 Stateful Economy & Synergistic Draft"
  * "V5 Deep Simulation Analytics" -> "v7 Stateful Simulation Analytics"
  * Replace the key `"acs_ema"` with `"acs_mu"` in the feature extraction dictionaries for Match Analysis.

## 2. Expose Bayesian Skill Tracking (Player Database)
* **Target:** `app.py` (Tab 4: `📋 VFL Players`)
* **Action:** 
  * Currently, this tab loads basic VFL scraper data. 
  * Enhance this by loading `data/processed/bayesian_player_ledger.json`.
  * Merge the Bayesian data with the VFL data. Display new columns in the Streamlit dataframe: `KPR Expected (μ)`, `KPR Volatility (σ)`, `ACS Expected (μ)`, and `ACS Volatility (σ)`.
  * Add a Streamlit metric or banner above the table explaining that higher Volatility ($\sigma$) indicates high-variance players (better for GPPs).

## 3. Expose Synergistic Drafts & Stateful Economy
* **Target:** `app.py` (Tab 1: `⚡ Open Simulation` & Tab 2: `📊 Match Analysis`)
* **Action:**
  * **Synergy Badges:** Under the "Expected Agent Cards" section in the map-by-map analytics, dynamically display the Synergy modifiers used by the `SynergisticDraftEngine` (e.g., display a green badge `+10% Duelist/Initiator Synergy` or a red badge `-15% Missing Sentinel Penalty` if those apply to the generated composition).
  * **Economy Note:** Add a sub-header or `st.info` block in the simulation outputs that explicitly mentions the inclusion of Stateful Economy impacts (Loss bonuses, Eco rounds, and Survival penalties) driving the win probabilities.