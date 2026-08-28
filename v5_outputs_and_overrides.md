# 1. Statistical Aggregation & Confidence Metric Extraction

Because the V5 engine operates as a non-deterministic Monte Carlo framework, confidence metrics are not arbitrary heuristics — they are derived directly from the empirical probability distributions of the simulation outputs.

---

## Map Veto Confidence

The probability $P(M_k)$ of a map sequence or a specific map being selected is calculated as its frequency across all unpruned simulation branches:

$$
P(M_k) =
\frac{1}{N}
\sum_{n=1}^{N}
\mathbb{I}(\text{Map } M_k \text{ is played in iteration } n)
$$

---

## Map Score Confidence

The confidence of a predicted scoreline (e.g., 13-9) is the joint empirical probability mass function (PMF) output by the Bivariate Poisson MCMC execution.

---

## Player Performance Intervals (K/D/A/ACS)

Rather than displaying a flat average, player stats will be accompanied by a Confidence Interval.

Example:

- 80% mid-range interval
- Calculated using the 10th and 90th percentiles

The interval is calculated using the Dirichlet distribution outputs across all simulation runs.

---

# 2. The Map Override Mechanism

To allow for conditional **"What-If" analysis**, the UI introduces a stateful bypass.

If a user enables the Map Veto Override:

1. The MapVetoBandit sub-model is completely short-circuited.
2. The engine injects the user's explicit map choices directly into Sub-Model 2 (Agent Composition Framework).
3. Users can test compositions and performance on a deterministic playground.
4. Full stochastic integrity is maintained for downstream player projections.

---

# UI & Dashboard Architecture: Updates to `app.py`

This structure outlines how the Streamlit interface should visually present these deep-dive analytics without creating a wall of text.

```python
# app.py - Comprehensive Dashboard Visualization Layout

st.markdown("## 📊 V5 Deep Simulation Analytics")


# --- SECTION 1: ADVERSARIAL MAP VETO & OVERRIDE ---

st.markdown("### 🗺️ Map Veto Sequence & Probabilities")

use_override = st.checkbox("⚙️ Enable Manual Map Veto Override")


if use_override:

    selected_maps = st.multiselect(
        "Select Exact Maps to Force Run",
        options=[
            "Ascent",
            "Bind",
            "Breeze",
            "Fracture",
            "Icebox",
            "Lotus",
            "Split"
        ]
    )

    # Pass selected_maps into the config payload

else:

    # Display the predicted Veto Sequence from the Bandit model
    # Format:
    # [Pick/Ban] Team - Map Name (Confidence: XX%)

    st.dataframe(veto_summary_table)


st.markdown("---")


# --- SECTION 2: MAP-BY-MAP PREDICTIONS (TABS) ---

# Create dynamic tabs based on the simulated or overridden map sequence

map_tabs = st.tabs(
    [
        f"🗺️ Map {i+1}: {map_name}"
        for i, map_name in enumerate(final_maps)
    ]
)


for i, tab in enumerate(map_tabs):

    with tab:

        col_score, col_composition = st.columns([1, 2])


        with col_score:

            st.markdown("#### 🎯 Predicted Scoreline")

            # Display:
            # Team A 13 - 10 Team B
            # Most Likely Scoreline - Confidence: 14.2%

            st.metric(
                "Most Likely Score",
                f"{score_a} - {score_b}",
                f"Confidence: {score_confidence}%"
            )

            # Mini chart showing score distribution

            st.bar_chart(score_distribution_data)



        with col_composition:

            st.markdown("#### 🤖 Predicted Agent Compositions")

            # Render side-by-side table:
            # Player | Agent Pick | Agent Pick Confidence %

            st.table(composition_dataframe_for_map_i)



        st.markdown("#### 🔫 Player Performance Projections")

        # Comprehensive stats table with 80% confidence bounds
        #
        # Columns:
        # Player |
        # Expected Kills (10%-90% Range) |
        # Deaths |
        # Assists |
        # ACS |
        # VFL EV

        st.dataframe(player_performance_dataframe_for_map_i)