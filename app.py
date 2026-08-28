"""
app.py
------
VCT / VLF Daily Fantasy Sports (DFS) Predictive Engine & Differentiable Patch Analyzer.
Production Streamlit Application (v9/v10 Architecture).
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# Configure Streamlit Page
st.set_page_config(
    page_title="VCT / VLF DFS Predictive Engine",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

# Import Modern v8/v9/v10 Core Modules
from v9_fantasy_engine import generate_v9_optimal_roster, _normalize_price
from v9_milp_optimizer import execute_roster_optimization_milp, compute_sortino_igl_score, CANONICAL_ROLES
from v9_historical_stats import (
    compute_exponential_decay_stats,
    compute_bayesian_shrinkage_stats,
    compute_telemetry_zscores,
    apply_telemetry_modifiers,
    calculate_mastery_index,
    compute_mastery_inertia_buffer
)
from v9_map_scenario_simulation import (
    compute_map_margin_probabilities,
    compute_single_map_ev,
    compute_bo3_series_ev
)
from v9_h2h_and_calibration import (
    calculate_scaled_h2h_weight,
    compute_team_elo_proxy_multiplier,
    combine_h2h_prior_and_elo_proxy
)
from v9_multiperiod_horizon_optimizer import (
    optimize_multiperiod_horizon_plan,
    compute_team_survival_probabilities,
    generate_stochastic_ev_matrix
)

# Custom Styling (Dark Glassmorphic UI)
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4655 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin-bottom: 24px;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    .role-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .role-duelist { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .role-initiator { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
    .role-controller { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .role-sentinel { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="dashboard-title">⚔️ VCT / VLF PREDICTIVE & DFS ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">v9/v10 Architecture · Differentiable Patch Shocks · 2N MILP Knapsack Solver · Multi-Period Horizon Optimization</div>', unsafe_allow_html=True)

# Data Loading Helpers
@st.cache_data
def load_sample_player_database() -> List[Dict[str, Any]]:
    """Loads benchmark player database from data/sample/sample_players_db.json."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample", "sample_players_db.json")
    if os.path.exists(sample_path):
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "players" in data:
                    return data["players"]
        except Exception as e:
            logger.warning(f"Failed to load sample player db: {e}")
    
    # In-memory fallback
    teams = ["Sentinels", "Paper Rex", "Fnatic", "Team Liquid"]
    roles = ["Duelist", "Initiator", "Controller", "Sentinel"]
    players = []
    pid = 1
    for role in roles:
        for team in teams:
            players.append({
                "player_id": f"p_{pid}",
                "player_name": f"Player_{pid}",
                "name": f"Player_{pid}",
                "team": team,
                "role": role,
                "price": round(8.0 + (pid % 5) * 1.0, 1),
                "ppg": round(24.0 + (pid % 6) * 2.0, 1),
                "adr": 135.0 + (pid % 4) * 8.0,
                "kast": 0.74 + (pid % 3) * 0.03,
                "fd": 0.08 + (pid % 3) * 0.02,
                "std": 5.0 + (pid % 2) * 1.0,
                "scores_history": [26.0, 31.0, 24.5, 33.0]
            })
            pid += 1
    return players

player_pool = load_sample_player_database()

# Sidebar: Global Engine Settings
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    budget_cap_input = st.slider("Salary Budget Cap (VP)", min_value=60.0, max_value=120.0, value=100.0, step=0.5)
    sortino_igl_toggle = st.checkbox("Sortino Risk-Adjusted IGL", value=True, help="Optimizes IGL designation based on downside risk penalty.")
    sortino_tau_input = st.slider("Sortino Minimum Acceptable Return (Tau)", min_value=5.0, max_value=25.0, value=12.0, step=1.0)
    st.markdown("---")
    st.markdown("### 📊 Active Player Slate")
    st.info(f"Loaded {len(player_pool)} professional players across {len(set(p.get('team') for p in player_pool))} VCT teams.")

# Main Navigation Tabs
tab_opt, tab_sim, tab_horizon, tab_players = st.tabs([
    "🧠 2N MILP Roster Optimizer",
    "⚡ Map & Match Scenario Simulation",
    "🗓️ Multi-Period Horizon Roadmap",
    "📋 Player Database & Telemetry"
])

# ============================================================
# TAB 1: 2N MILP ROSTER OPTIMIZER
# ============================================================
with tab_opt:
    st.markdown("### 🧠 2N Binary Mixed-Integer Linear Programming Optimizer")
    st.markdown("Exact mathematical lineup solver maximizing $EV_{\\text{total}}$ under strict VLF constraints (11 players, 1 IGL, $2 \\le \\text{role} \\le 5$, $\\le 2$ per team).")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_solver = st.button("🚀 Solve Optimal Roster", type="primary", use_container_width=True)
        
    if run_solver or "last_roster_solution" in st.session_state:
        if run_solver:
            with st.spinner("Executing 2N MILP Knapsack Solver..."):
                sol = generate_v9_optimal_roster(
                    players=player_pool,
                    budget_cap=budget_cap_input,
                    use_risk_adjusted_igl=sortino_igl_toggle,
                    sortino_tau=sortino_tau_input
                )
                st.session_state["last_roster_solution"] = sol
        else:
            sol = st.session_state["last_roster_solution"]
            
        if sol.get("solver_status") == "optimal":
            st.success(f"✅ Optimal Lineup Found! Total EV: **{sol.get('total_ev', 0.0):.2f} pts** | Budget Spent: **{sol.get('total_cost', 0.0):.1f} / {budget_cap_input:.1f} VP**")
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">TOTAL EXPECTED VALUE</div><div class="metric-value">{sol.get("total_ev", 0.0):.2f}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">TOTAL SPENT (VP)</div><div class="metric-value">{sol.get("total_cost", 0.0):.1f}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">DESIGNATED IGL</div><div class="metric-value">{sol.get("igl_player", "None")}</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">ROSTER SIZE</div><div class="metric-value">{len(sol.get("optimal_roster", []))} / 11</div></div>', unsafe_allow_html=True)
                
            st.markdown("#### 📋 Selected 11-Player Roster")
            roster_df = pd.DataFrame(sol.get("optimal_roster", []))
            if not roster_df.empty:
                display_cols = [c for c in ["player_name", "team", "role", "price", "ev", "ppg", "adr", "kast", "fd"] if c in roster_df.columns]
                st.dataframe(roster_df[display_cols], use_container_width=True)
        else:
            st.error(f"Solver Error: {sol.get('solver_status', 'infeasible')}")

# ============================================================
# TAB 2: MAP & MATCH SCENARIO SIMULATION
# ============================================================
with tab_sim:
    st.markdown("### ⚡ Discrete Map Margin & Sweep Simulation")
    st.markdown("High-resolution Monte Carlo simulation of Best-of-3 series distributions, blowout bonuses, and map cap discounts.")
    
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        team_a_name = st.selectbox("Team A", list(set(p["team"] for p in player_pool)), index=0)
    with s_col2:
        team_b_name = st.selectbox("Team B", list(set(p["team"] for p in player_pool)), index=min(1, len(set(p["team"] for p in player_pool))-1))
    with s_col3:
        elo_diff = st.slider("Team A vs Team B Elo Delta", min_value=-300, max_value=300, value=50, step=10)
        
    if st.button("🎲 Simulate Bo3 Series", type="primary"):
        win_prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        sim_res = compute_bo3_series_ev(
            win_prob_a=win_prob_a,
            team_a_roster=[p for p in player_pool if p.get("team") == team_a_name][:5],
            team_b_roster=[p for p in player_pool if p.get("team") == team_b_name][:5]
        )
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"{team_a_name} 2-0 Sweep Prob", f"{sim_res.p_2_0 * 100:.1f}%")
        with c2:
            st.metric(f"3-Map Decider Prob", f"{sim_res.p_decider * 100:.1f}%")
        with c3:
            st.metric(f"{team_b_name} 0-2 Sweep Prob", f"{sim_res.p_0_2 * 100:.1f}%")
            
        st.info(f"Expected Map Count: **{sim_res.expected_maps:.2f} maps** | Applied Series EV Adjustment: **+{sim_res.series_ev_adjustment:.2f} pts**")

# ============================================================
# TAB 3: MULTI-PERIOD HORIZON ROADMAP
# ============================================================
with tab_horizon:
    st.markdown("### 🗓️ Multi-Period Horizon Optimizer")
    st.markdown("Stochastic dynamic programming for planning multi-gameweek transfer budgets and tournament survival paths.")
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        horizon_length = st.selectbox("Planning Horizon", [2, 3, 4], index=1)
    with h_col2:
        transfers_per_gw = st.selectbox("Free Transfers per Gameweek", [1, 2, 3], index=2)
        
    if st.button("📈 Compute Optimal Multi-Period Plan", type="primary"):
        with st.spinner("Solving Stochastic Horizon Plan..."):
            initial_roster = player_pool[:11]
            h_plan = optimize_multiperiod_horizon_plan(
                current_roster=initial_roster,
                player_pool=player_pool,
                horizon_weeks=horizon_length,
                free_transfers_per_gw=transfers_per_gw,
                budget_cap=budget_cap_input
            )
            st.success(f"Optimal {horizon_length}-Week Trajectory Calculated! Expected Cumulative Horizon EV: **{h_plan.cumulative_ev:.2f} pts**")
            
            for week_plan in h_plan.weekly_plans:
                with st.expander(f"🗓️ Gameweek {week_plan.week} Strategy", expanded=True):
                    st.write(f"**Expected Weekly EV:** {week_plan.expected_ev:.2f} pts | **Transfers Used:** {len(week_plan.transfers_in)}")
                    if week_plan.transfers_in:
                        for tin, tout in zip(week_plan.transfers_in, week_plan.transfers_out):
                            st.write(f"🔄 **Transfer:** IN `{tin}` ↔ OUT `{tout}`")
                    else:
                        st.write("🔒 *No transfers recommended this gameweek (Roll transfer bank).*")

# ============================================================
# TAB 4: PLAYER DATABASE & TELEMETRY
# ============================================================
with tab_players:
    st.markdown("### 📋 VFL Player Database & Statistical Profiler")
    st.markdown("Comprehensive view of active player telemetry and role-normalized performance metrics.")
    
    p_df = pd.DataFrame(player_pool)
    if not p_df.empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            role_filter = st.multiselect("Filter by Role", CANONICAL_ROLES, default=CANONICAL_ROLES)
        with f_col2:
            team_filter = st.multiselect("Filter by Team", list(set(p_df["team"])), default=list(set(p_df["team"])))
            
        filtered_df = p_df[p_df["role"].isin(role_filter) & p_df["team"].isin(team_filter)]
        st.dataframe(filtered_df, use_container_width=True)
