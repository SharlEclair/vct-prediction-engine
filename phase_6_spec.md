# Phase 6: The Presentation Layer (Streamlit Dashboard Overhaul)
**Objective:** Transform `app.py` into a production-ready, interactive DFS advisory dashboard. The UI must trigger the Phase 4 Knapsack solver on demand and visualize the optimal roster and its heavy-tailed tournament upside (The Engagement Hook).

## Task 6.1: The Command Center (Sidebar)
* **Execution:** Overhaul the Streamlit sidebar to remove all hardcoded mock inputs.
* **Components:**
    * `st.date_input`: For slate tracking (defaulting to today).
    * `st.selectbox`: The dynamic patch selector (implemented in Phase 5).
    * `st.select_slider`: For Monte Carlo simulation depth (Options: 1K, 5K, 10K iterations).
    * `st.button`: A primary action button labeled "Generate Optimal GPP Lineup" that triggers the Knapsack solver.

## Task 6.2: Roster Visualization (Main UI)
* **Execution:** When the solver successfully generates a lineup, display the 6-man roster using a clean, high-contrast visual layout (e.g., `st.columns`).
* **Components per Player:**
    * Fetch the static agent icon from the existing CDN URLs.
    * Display the Player Name, VFL Role, and Salary.
    * Visually badge or highlight the designated 1.5x In-Game Leader (IGL).

## Task 6.3: The Upside Hook (Metrics)
* **Execution:** Below or above the roster, create a highly visible metrics section using `st.metric` or lightweight charts.
* **Data Points:**
    * Total Salary Used (e.g., 47.1 / 50.0 VP).
    * Lineup Median EV.
    * **Tournament GPP Ceiling (85th Percentile)**: This is the primary metric to emphasize. Use deltas to explicitly show the massive gap between the median projection and the portfolio ceiling.