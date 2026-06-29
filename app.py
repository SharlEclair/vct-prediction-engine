import os
import re
import json
import glob
import logging
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from datetime import datetime

# Import local modules
import importlib
import veto_predictor
import generative_pipeline
import fantasy_engine
import predict_match
import vfl_scraper
import v5_simulation_engine

importlib.reload(veto_predictor)
importlib.reload(generative_pipeline)
importlib.reload(fantasy_engine)
importlib.reload(predict_match)
importlib.reload(vfl_scraper)
importlib.reload(v5_simulation_engine)

from veto_predictor import VCTMapVetoPredictor
from generative_pipeline import MapScoreRegressor, AgentCompositionGenerator
from fantasy_engine import VCTFantasyEngine, optimize_roster, suggest_transfers, generate_stage_2_baseline, get_team_win_rates_by_id
from predict_match import get_historical_stats, get_latest_roster, simulate_arbitrary_match
from vfl_scraper import VFLScraper
from v5_simulation_engine import VCTv5SimulationEngine

@st.cache_resource
def get_v5_simulation_engine():
    return VCTv5SimulationEngine()

def clean_html(html_str: str) -> str:
    """Strip leading/trailing whitespace from each line in a multiline HTML string
    to prevent the Markdown parser from identifying indentation as a code block."""
    if not html_str:
        return ""
    return "\n".join(line.strip() for line in html_str.splitlines())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
ROSTER_STATE_PATH = "./data/user_roster_state.json"

# ============================================================
# DATA LOADING HELPERS
# ============================================================

_, team_name_to_id = get_team_win_rates_by_id(RAW_DIR)
id_to_team_name = {v: k for k, v in team_name_to_id.items()}

@st.cache_data
def load_automated_registry():
    path = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
    if not os.path.exists(path):
        return "None", {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return "None", {}
        sorted_patches = sorted(list(data.keys()), key=lambda x: [int(i) if i.isdigit() else i for i in x.split('.')])
        if not sorted_patches:
            return "None", {}
        latest_patch = sorted_patches[-1]
        return latest_patch, data[latest_patch]
    except Exception as e:
        logger.error(f"Failed to load automated patch registry: {e}")
        return "None", {}

def load_roster_state():
    """Load saved roster state from disk. Returns (player_names, igl_name)."""
    if os.path.exists(ROSTER_STATE_PATH):
        try:
            with open(ROSTER_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state.get("players", []), state.get("igl", None)
        except Exception:
            pass
    return [], None

def save_roster_state(player_names: list, igl_name: str | None):
    """Persist roster state to disk."""
    os.makedirs(os.path.dirname(ROSTER_STATE_PATH), exist_ok=True)
    with open(ROSTER_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"players": player_names, "igl": igl_name, "saved_at": datetime.now().isoformat()}, f, indent=2)

def get_meta_penalty_badge(player_name, player_agent_stats, active_penalties):
    p_name_clean = player_name.lower().strip()
    primary_agent = None
    max_count = -1
    for (name, agent), info in player_agent_stats.items():
        if name.lower().strip() == p_name_clean:
            if info['count'] > max_count:
                max_count = info['count']
                primary_agent = agent
    if not primary_agent:
        return ""
    penalty = active_penalties.get(primary_agent, 0.0)
    if penalty > 0.10:
        return f'<span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(239, 68, 68, 0.15); color: #ef4444; margin-left: 8px;">⚠️ Meta Penalty: {penalty:.2f} ({primary_agent})</span>'
    return ""

@st.cache_resource
def load_cached_historical_data():
    return get_historical_stats(RAW_DIR)

# Agent Icons mapping
AGENT_ICONS = {
    "Jett": "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png",
    "Raze": "https://media.valorant-api.com/agents/f94c3b30-42be-e959-889c-5aa313dba261/displayicon.png",
    "Breach": "https://media.valorant-api.com/agents/5f8d3a7f-467b-97f3-062c-13acf203c006/displayicon.png",
    "Omen": "https://media.valorant-api.com/agents/8e253930-4c05-31dd-1b6c-968525494517/displayicon.png",
    "Brimstone": "https://media.valorant-api.com/agents/9f0d8ba9-4140-b941-57d3-a7ad57c6b417/displayicon.png",
    "Phoenix": "https://media.valorant-api.com/agents/eb93336a-449b-9c1b-0a54-a891f7921d69/displayicon.png",
    "Sage": "https://media.valorant-api.com/agents/569fdd95-4d10-43ab-ca70-79becc718b46/displayicon.png",
    "Sova": "https://media.valorant-api.com/agents/320b2a48-4d9b-a075-30f1-1f93a9b638fa/displayicon.png",
    "Viper": "https://media.valorant-api.com/agents/707eab51-4836-f488-046a-cda6bf494859/displayicon.png",
    "Cypher": "https://media.valorant-api.com/agents/117ed9e3-49f3-6512-3ccf-0cada7e3823b/displayicon.png",
    "Reyna": "https://media.valorant-api.com/agents/a3bfb853-43b2-7238-a4f1-ad90e9e46bcc/displayicon.png",
    "Killjoy": "https://media.valorant-api.com/agents/1e58de9c-4950-5125-93e9-a0aee9f98746/displayicon.png",
    "Astra": "https://media.valorant-api.com/agents/41fb69c1-4189-7b37-f117-bcaf1e96f1bf/displayicon.png",
    "KAY/O": "https://media.valorant-api.com/agents/601dbbe7-43ce-be57-2a40-4abd24953621/displayicon.png",
    "Chamber": "https://media.valorant-api.com/agents/22697a3d-45bf-8dd7-4fec-84a9e28c69d7/displayicon.png",
    "Neon": "https://media.valorant-api.com/agents/bb2a4828-46eb-8cd1-e765-15848195d751/displayicon.png",
    "Fade": "https://media.valorant-api.com/agents/dade69b4-4f5a-8528-247b-219e5a1facd6/displayicon.png",
    "Harbor": "https://media.valorant-api.com/agents/95b78ed7-4637-86d9-7e41-71ba8c293152/displayicon.png",
    "Gekko": "https://media.valorant-api.com/agents/e370fa57-4757-3604-3648-499e1f642d3f/displayicon.png",
    "Deadlock": "https://media.valorant-api.com/agents/cc8b64c8-4b25-4ff9-6e7f-37b4da43d235/displayicon.png",
    "Iso": "https://media.valorant-api.com/agents/0e38b510-41a8-5780-5e8f-568b2a4f2d6c/displayicon.png",
    "Clove": "https://media.valorant-api.com/agents/1dbf2edd-4729-0984-3115-daa5eed44993/displayicon.png",
    "Vyse": "https://media.valorant-api.com/agents/efba5359-4016-a1e5-7626-b1ae76895940/displayicon.png",
    "Skye": "https://media.valorant-api.com/agents/6f2a04ca-43e0-be17-7f36-b3908627744d/displayicon.png",
    "Yoru": "https://media.valorant-api.com/agents/7f94d92c-4234-0a36-9646-3a87eb8b5c89/displayicon.png",
    "Tejo": "https://media.valorant-api.com/agents/b444168c-4e35-8076-db47-ef9bf368f384/displayicon.png",
    "Miks": "https://media.valorant-api.com/agents/7c8a4701-4de6-9355-b254-e09bc2a34b72/displayicon.png",
    "Veto": "https://media.valorant-api.com/agents/92eeef5d-43b5-1d4a-8d03-b3927a09034b/displayicon.png",
    "Waylay": "https://media.valorant-api.com/agents/df1cb487-4902-002e-5c17-d28e83e78588/displayicon.png",
}

# ============================================================
# PAGE CONFIG AND GLOBAL THEME
# ============================================================
st.set_page_config(
    page_title="VCT Predictive Engine & Fantasy Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d0e12;
        color: #e2e8f0;
    }

    /* Hide sidebar toggle arrow when collapsed */
    [data-testid="collapsedControl"] { display: none; }

    .dashboard-title {
        background: linear-gradient(135deg, #ff4655 0%, #ff7676 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }

    .dashboard-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 20px;
        letter-spacing: 0.02em;
    }

    .glass-card {
        background: rgba(26, 29, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .winner-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 15px;
        border-radius: 8px;
        color: #4ade80;
        font-weight: 600;
        text-align: center;
    }

    .optimizer-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.03) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .transfer-in {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .transfer-out {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Actual vs Predicted comparison badges */
    .actual-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 99px;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.3);
        color: #4ade80;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .predicted-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 99px;
        background: rgba(168,85,247,0.12);
        border: 1px solid rgba(168,85,247,0.3);
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .match-config-panel {
        background: rgba(15, 17, 24, 0.8);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 22px;
    }

    div[data-testid="stDataFrame"] {
        background: rgba(26, 29, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15,17,24,0.6);
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 18px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.2) !important;
        color: #a78bfa !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="dashboard-title">VCT FANTASY & PREDICTIVE ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">V5 · Open Match Simulation · Fantasy Optimizer · Backtesting Visualizer</div>', unsafe_allow_html=True)

# ============================================================
# GLOBAL DATA INITIALIZATION (outside tabs for shared use)
# ============================================================

# Load match files
files = sorted(glob.glob(os.path.join(RAW_DIR, "match_*.json")))
matches_lookup = {}
match_options = []

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            content = json.load(file)
        if "data" not in content or "segments" not in content["data"] or not content["data"]["segments"]:
            continue
        seg = content["data"]["segments"][0]
        match_id = seg["match_id"]
        if len(seg.get("teams", [])) < 2:
            continue
        team_a = seg["teams"][0]["name"]
        team_b = seg["teams"][1]["name"]
        event = seg.get("event", {}).get("name", "Unknown Event")
        date_str = seg.get("date", "Unknown Date")
        display_name = f"{event}: {team_a} vs {team_b} ({date_str}) [ID: {match_id}]"
        matches_lookup[match_id] = {
            "filepath": f,
            "team_a": team_a,
            "team_b": team_b,
            "segment": seg,
            "display_name": display_name
        }
        match_options.append((match_id, display_name))
    except Exception:
        pass

# Load models
veto_pred = VCTMapVetoPredictor(RAW_DIR)
veto_pred.fit()
score_reg = MapScoreRegressor()
score_reg.load_model()
agent_comp = AgentCompositionGenerator(RAW_DIR)
agent_comp.fit()

clf_model_path = os.path.join(PROCESSED_DIR, "vct_model.cbm")
clf_model = None
if os.path.exists(clf_model_path):
    clf_model = CatBoostClassifier()
    clf_model.load_model(clf_model_path)

# Load VFL data
vfl_scraper_inst = VFLScraper()
vfl_players_data = vfl_scraper_inst.get_players()

vfl_rules = {"salary_cap": 50, "max_per_team": 2, "max_transfers_per_gameweek": 3}

# Load automated patch registry
latest_patch, active_penalties = load_automated_registry()

# Load historical data
player_emas, baseline_lookup, team_stats, player_global_stats, player_agent_stats = load_cached_historical_data()

# Apply meta penalties to VFL player database
for p in vfl_players_data:
    p_name = p["player_name"]
    p_name_clean = p_name.lower().strip()
    primary_agent = None
    max_count = -1
    for (name, agent), info in player_agent_stats.items():
        if name.lower().strip() == p_name_clean:
            if info['count'] > max_count:
                max_count = info['count']
                primary_agent = agent
    p["primary_agent"] = primary_agent
    p["meta_penalty"] = active_penalties.get(primary_agent, 0.0) if primary_agent else 0.0
    if p["meta_penalty"] > 0:
        p["ppg"] = p["ppg"] * (1.0 - p["meta_penalty"])

all_maps = sorted(list(veto_pred.map_pool))
all_teams = sorted(list(set(
    m["team_a"] for m in matches_lookup.values()
) | set(
    m["team_b"] for m in matches_lookup.values()
)))

# Load saved roster state for initializing session_state
_saved_players, _saved_igl = load_roster_state()
if "roster_state_loaded" not in st.session_state:
    st.session_state["roster_state_loaded"] = True
    st.session_state["saved_roster_names"] = _saved_players
    st.session_state["saved_igl_name"] = _saved_igl

# ============================================================
# MAIN TABS — Simulation first per v5_frontend_architecture.md
# ============================================================
tab_sim, tab_match, tab_optimizer, tab_vfl = st.tabs([
    "⚡ Open Simulation",
    "📊 Match Analysis",
    "🧠 Roster Optimizer",
    "📋 VFL Players"
])

# ============================================================
# TAB 1: OPEN SIMULATION
# ============================================================
with tab_sim:
    st.markdown("### ⚡ Open Match Simulation Engine")
    st.markdown(clean_html("""
        <div class="glass-card">
            <div class="metric-title">ARBITRARY MATCH SIMULATOR</div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
                Simulate any hypothetical VCT matchup using time-decay weighted historical data.
                The engine dynamically resolves rosters, computes EMAs, and runs V5 micro-simulation.
            </p>
        </div>
    """), unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_team_a = st.selectbox("Team A", all_teams, index=0, key="sim_team_a")
    with sim_col2:
        sim_team_b = st.selectbox("Team B", all_teams, index=min(1, len(all_teams)-1), key="sim_team_b")

    sim_col3, sim_col4, sim_col5, sim_col6 = st.columns(4)
    with sim_col3:
        sim_ref_date = st.date_input("Reference Date (for time-decay)", value=datetime(2026, 6, 22), key="sim_ref_date")
    with sim_col4:
        sim_series_type = st.selectbox("Series Format", ["Bo3", "Bo5"], index=0, key="sim_series_type")
    with sim_col5:
        sim_iterations = st.selectbox("Simulation Depth", [1000, 5000, 10000], index=1, key="sim_iterations")
    with sim_col6:
        sim_patch_select = st.selectbox("Target Simulation Patch", ["Patch 9.04", "Patch 9.02", "Patch 8.11 (June 11, 2024)"], index=1, key="sim_target_patch")
        patch_match = re.search(r'([0-9.]+)', sim_patch_select)
        sim_target_patch_val = patch_match.group(1) if patch_match else "9.02"

    # Map Veto Override Panel
    max_maps = 3 if sim_series_type == "Bo3" else 5
    with st.container():
        st.markdown(clean_html(f"""
            <div style="font-size: 0.78rem; font-weight: 700; color: #818cf8; text-transform: uppercase;
                        letter-spacing: 0.08em; margin-top: 10px; margin-bottom: 6px;">
                ⚙️ Map Veto Override Panel (Series Requires Exactly {max_maps} Maps)
            </div>
        """), unsafe_allow_html=True)
        enable_override = st.checkbox("Enable Manual Map Veto Override", value=False, key="enable_override_checkbox")
        if enable_override:
            override_maps = st.multiselect(
                f"Select Exact Maps to Force Run (Select exactly {max_maps} maps in order)",
                options=all_maps,
                default=[],
                max_selections=max_maps,
                key="override_maps_select"
            )
        else:
            override_maps = None

    if st.button("🚀 Run Simulation", key="btn_run_sim", type="primary"):
        if sim_team_a == sim_team_b:
            st.error("Please select two different teams.")
        elif enable_override and len(override_maps) != max_maps:
            st.error(f"Please select exactly {max_maps} override maps to simulate.")
        else:
            with st.spinner(f"Running V5 Bottom-Up Micro-Simulation ({sim_iterations:,} iterations) for {sim_team_a} vs {sim_team_b}..."):
                v5_engine = get_v5_simulation_engine()
                sim_target_datetime = datetime.combine(sim_ref_date, datetime.min.time()) if hasattr(sim_ref_date, 'year') else datetime.now()
                sim_result = v5_engine.simulate_match(
                    team_a=sim_team_a,
                    team_b=sim_team_b,
                    series_type=sim_series_type,
                    target_patch=sim_target_patch_val,
                    num_iterations=sim_iterations,
                    override_maps=override_maps if enable_override else None,
                    target_date=sim_target_datetime
                )

            win_prob_a = sim_result["win_prob_a"]
            win_prob_b = sim_result["win_prob_b"]
            sim_winner = sim_result["team_a"] if win_prob_a > win_prob_b else sim_result["team_b"]
            sim_loser  = sim_result["team_b"] if win_prob_a > win_prob_b else sim_result["team_a"]
            winner_prob = win_prob_a if win_prob_a > win_prob_b else win_prob_b
            loser_prob  = win_prob_b if win_prob_a > win_prob_b else win_prob_a
            winner_roster_key = "roster_a" if win_prob_a > win_prob_b else "roster_b"
            loser_roster_key  = "roster_b" if win_prob_a > win_prob_b else "roster_a"

            # Series Winner Hero Card
            st.markdown(clean_html(f"""
                <div style="background: linear-gradient(135deg, rgba(168,85,247,0.13) 0%, rgba(99,102,241,0.07) 100%);
                            border: 1px solid rgba(168,85,247,0.25); border-radius: 16px; padding: 28px 32px;
                            margin: 20px 0 24px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 0.72rem; font-weight: 700; color: #a78bfa; text-transform: uppercase;
                                        letter-spacing: 0.08em; margin-bottom: 6px;">🏆 Predicted Winner</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; line-height: 1.1;">{sim_winner}</div>
                            <div style="font-size: 1.6rem; font-weight: 700; color: #a78bfa; margin-top: 4px;">{winner_prob:.1%}</div>
                            <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px;">
                                Roster: {', '.join(sim_result.get(winner_roster_key, []))}
                            </div>
                        </div>
                        <div style="font-size: 2.5rem; color: #334155; padding: 0 24px; font-weight: 700;">VS</div>
                        <div style="text-align: center; flex: 1; opacity: 0.65;">
                            <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;
                                        letter-spacing: 0.08em; margin-bottom: 6px;">Runner-Up</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #94a3b8; line-height: 1.1;">{sim_loser}</div>
                            <div style="font-size: 1.6rem; font-weight: 700; color: #64748b; margin-top: 4px;">{loser_prob:.1%}</div>
                            <div style="font-size: 0.78rem; color: #475569; margin-top: 6px;">
                                Roster: {', '.join(sim_result.get(loser_roster_key, []))}
                            </div>
                        </div>
                    </div>
                </div>
            """), unsafe_allow_html=True)

            # Win probability gradient bar
            bar_pct_a = int(win_prob_a * 100)
            st.markdown(clean_html(f"""
                <div style="margin-bottom: 28px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.78rem;
                                color: #94a3b8; margin-bottom: 6px;">
                        <span>{sim_result['team_a']} ({win_prob_a:.1%})</span>
                        <span>{sim_result['team_b']} ({win_prob_b:.1%})</span>
                    </div>
                    <div style="background: rgba(239,68,68,0.2); border-radius: 99px; height: 10px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #a78bfa, #6366f1);
                                    width: {bar_pct_a}%; height: 100%; border-radius: 99px;"></div>
                    </div>
                </div>
            """), unsafe_allow_html=True)

            # Veto Sequence Card
            if enable_override and override_maps:
                st.markdown(clean_html(f"""
                    <div style="background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25);
                                border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #f59e0b;
                                    text-transform: uppercase; letter-spacing: 0.07em;">🗺️ Manual Override Active</div>
                        <div style="font-weight: 600; color: #f8fafc; margin-top: 4px;">
                            {' → '.join(override_maps)}
                        </div>
                    </div>
                """), unsafe_allow_html=True)
            elif "veto_confidences" in sim_result and sim_result["veto_confidences"]:
                st.markdown("#### 🗺️ Predicted Map Veto Sequence")
                step_colors = ["#6366f1", "#a78bfa", "#818cf8", "#4ade80", "#f59e0b"]
                veto_items_html = ""
                for vi, (action, conf) in enumerate(sim_result["veto_confidences"]):
                    color = step_colors[vi % len(step_colors)]
                    conf_pct = int(conf * 100)
                    veto_items_html += clean_html(f"""
                        <div style="display: flex; align-items: center; gap: 14px; padding: 10px 0;
                                    border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <div style="width: 26px; height: 26px; border-radius: 50%; background: {color}22;
                                        border: 1px solid {color}55; display: flex; align-items: center;
                                        justify-content: center; font-size: 0.72rem; font-weight: 700;
                                        color: {color}; flex-shrink: 0;">{vi+1}</div>
                            <div style="flex: 1; font-size: 0.88rem; color: #e2e8f0;">{action}</div>
                            <div style="text-align: right; min-width: 90px;">
                                <div style="font-size: 0.75rem; color: {color}; font-weight: 600;">{conf_pct}% conf.</div>
                                <div style="background: rgba(255,255,255,0.05); border-radius: 99px;
                                            height: 4px; margin-top: 3px; overflow: hidden;">
                                    <div style="background: {color}; width: {conf_pct}%; height: 100%; border-radius: 99px;"></div>
                                </div>
                            </div>
                        </div>
                    """)
                st.markdown(clean_html(f'<div class="glass-card" style="margin-bottom:24px;">{veto_items_html}</div>'),
                            unsafe_allow_html=True)

            # V5 Deep Simulation Analytics: Map-by-Map Tabs
            st.markdown(clean_html("""
                <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin: 28px 0 16px;">
                    📊 V5 Deep Simulation Analytics
                </div>
            """), unsafe_allow_html=True)

            final_maps = sim_result["predicted_maps"]
            map_tab_labels = []
            for i, map_name in enumerate(final_maps):
                d = sim_result["map_details"][map_name]
                suffix = f"  ·  {d['play_probability']}% played" if d["played"] else "  ·  (Decider)"
                map_tab_labels.append(f"Map {i+1}: {map_name}{suffix}")
            map_tabs = st.tabs(map_tab_labels)

            for idx, mtab in enumerate(map_tabs):
                map_name = final_maps[idx]
                details = sim_result["map_details"][map_name]
                with mtab:
                    if not details["played"]:
                        st.markdown(clean_html(f"""
                            <div style="background: rgba(100,116,139,0.08); border: 1px solid rgba(100,116,139,0.2);
                                        border-radius: 12px; padding: 28px; text-align: center; color: #64748b;">
                                <div style="font-size: 2rem; margin-bottom: 8px;">🏁</div>
                                <div style="font-weight: 600; font-size: 1rem;">'{map_name}' was not played in any simulation run.</div>
                                <div style="font-size: 0.85rem; margin-top: 4px;">Series settled before reaching this map.</div>
                            </div>
                        """), unsafe_allow_html=True)
                        continue

                    score_col, dist_col = st.columns([1, 2])
                    with score_col:
                        conf_val = details["score_confidence"]
                        conf_color = "#4ade80" if conf_val >= 40 else ("#f59e0b" if conf_val >= 20 else "#ef4444")
                        st.markdown(clean_html(f"""
                            <div class="glass-card" style="text-align: center; padding: 28px;">
                                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                                            text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 12px;">
                                    🎯 Most Probable Scoreline
                                </div>
                                <div style="font-size: 3rem; font-weight: 800; color: #f8fafc;
                                            letter-spacing: -2px; line-height: 1;">
                                    {details['most_probable_score']}
                                </div>
                                <div style="margin-top: 14px; display: inline-block; padding: 4px 16px;
                                            border-radius: 99px; background: {conf_color}22;
                                            border: 1px solid {conf_color}55;
                                            font-size: 0.8rem; font-weight: 700; color: {conf_color};">
                                    {conf_val}% confidence
                                </div>
                                <div style="margin-top: 10px; font-size: 0.78rem; color: #64748b;">
                                    Map play probability: {details['play_probability']}%
                                </div>
                            </div>
                        """), unsafe_allow_html=True)

                    with dist_col:
                        dist_data = details.get("score_distribution", {})
                        if dist_data:
                            st.markdown(clean_html("""
                                <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8;
                                            text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px;">
                                    📈 Score Distribution (Top 10 Outcomes)
                                </div>
                            """), unsafe_allow_html=True)
                            dist_df = pd.DataFrame(
                                list(dist_data.items()), columns=["Scoreline", "Frequency"]
                            ).sort_values("Frequency", ascending=False)
                            st.bar_chart(dist_df.set_index("Scoreline"), height=200, use_container_width=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Agent Compositions
                    st.markdown(clean_html(f"""
                        <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8;
                                    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 12px;">
                            🤖 Expected Agent Assignments — {map_name}
                        </div>
                    """), unsafe_allow_html=True)

                    comp_col_a, comp_col_b = st.columns(2)
                    player_agents = details.get("player_agents", {})

                    def _render_agent_card(player_name, info):
                        agent_name = info["agent"]
                        pick_pct = info["pick_probability"]
                        icon_url = AGENT_ICONS.get(agent_name, "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png")
                        v5_engine = get_v5_simulation_engine()
                        role = v5_engine.agent_transformer.agent_roles.get(agent_name, "Sentinel")
                        role_colors = {"Duelist": "#ef4444", "Controller": "#3b82f6", "Initiator": "#f59e0b", "Sentinel": "#10b981"}
                        role_color = role_colors.get(role, "#6366f1")
                        return clean_html(f"""
                            <div style="display: flex; align-items: center; justify-content: space-between;
                                        padding: 10px 14px; margin-bottom: 8px; border-radius: 10px;
                                        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <img src="{icon_url}" width="38" height="38"
                                         style="border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);"/>
                                    <div>
                                        <div style="font-weight: 600; font-size: 0.92rem; color: #f1f5f9;">{player_name}</div>
                                        <div style="font-size: 0.73rem; color: #64748b; margin-top: 1px;">{agent_name}</div>
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 0.7rem; font-weight: 700; padding: 2px 9px;
                                                 border-radius: 99px; background: {role_color}22;
                                                 color: {role_color}; border: 1px solid {role_color}44;">{role}</span>
                                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">{pick_pct}% pick rate</div>
                                </div>
                            </div>
                        """)

                    with comp_col_a:
                        cards_a = "".join(
                            _render_agent_card(p, player_agents[p])
                            for p in sim_result.get("roster_a", []) if p in player_agents
                        )
                        st.markdown(clean_html(f"""
                            <div style="margin-bottom: 6px; font-weight: 700; font-size: 0.95rem; color: #a78bfa;">{sim_team_a}</div>
                            <div class="glass-card">{cards_a or '<div style="color:#64748b;font-size:0.85rem;">No composition data.</div>'}</div>
                        """), unsafe_allow_html=True)

                    with comp_col_b:
                        cards_b = "".join(
                            _render_agent_card(p, player_agents[p])
                            for p in sim_result.get("roster_b", []) if p in player_agents
                        )
                        st.markdown(clean_html(f"""
                            <div style="margin-bottom: 6px; font-weight: 700; font-size: 0.95rem; color: #fbbf24;">{sim_team_b}</div>
                            <div class="glass-card">{cards_b or '<div style="color:#64748b;font-size:0.85rem;">No composition data.</div>'}</div>
                        """), unsafe_allow_html=True)

                    # Player Performance Table
                    st.markdown(clean_html("""
                        <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8;
                                    text-transform: uppercase; letter-spacing: 0.07em; margin: 24px 0 10px;">
                            🔫 Player Performance Projections — 80% Confidence Bounds (P10 – P90)
                        </div>
                    """), unsafe_allow_html=True)
                    perf_rows = details.get("player_stats", [])
                    if perf_rows:
                        perf_df = pd.DataFrame(perf_rows)
                        def _style_perf_row(row):
                            bg = "rgba(99,102,241,0.06)" if row["Team"] == sim_team_a else "rgba(245,158,11,0.06)"
                            return [f"background-color: {bg}"] * len(row)
                        st.dataframe(perf_df.style.apply(_style_perf_row, axis=1), use_container_width=True, hide_index=True)
                    else:
                        st.info("No performance data for this map.")

            # Series EV Projections
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 Series-Level VFL EV Projections (All Players)", expanded=False):
                if "projections" in sim_result:
                    proj_data = []
                    for p, ev in sim_result["projections"].items():
                        team = sim_team_a if p in sim_result.get("roster_a", []) else sim_team_b
                        proj_data.append({"Player": p, "Team": team, "Expected VFL Points (EV)": ev})
                    proj_df = pd.DataFrame(proj_data).sort_values("Expected VFL Points (EV)", ascending=False)
                    ev_max = proj_df["Expected VFL Points (EV)"].max() if len(proj_df) > 0 else 1
                    def _color_ev(val):
                        ratio = min(1.0, max(0.0, val / ev_max)) if ev_max > 0 else 0
                        g = int(74 + ratio * (222 - 74))
                        return f"color: rgb(74,{g},128)"
                    st.dataframe(
                        proj_df.style.map(_color_ev, subset=["Expected VFL Points (EV)"]),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("EV Projections unavailable.")

# ============================================================
# TAB 2: MATCH ANALYSIS (with inline config + Actual vs Predicted)
# ============================================================
with tab_match:

    # ── Match Configuration Control Panel (moved from sidebar) ──
    st.markdown(clean_html("""
        <div style="font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin-bottom: 10px;">
            ⚙️ Match Configuration
        </div>
    """), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="match-config-panel">', unsafe_allow_html=True)
        cfg_col1, cfg_col2 = st.columns([3, 1])
        with cfg_col1:
            selected_match_id = st.selectbox(
                "Select Target Match",
                options=[item[0] for item in match_options],
                format_func=lambda x: next(item[1] for item in match_options if item[0] == x),
                key="match_analysis_selector"
            )
        with cfg_col2:
            ma_series_type = st.selectbox(
                "Series Type",
                ["Bo3", "Bo5"],
                index=0,
                key="ma_series_type"
            )

        veto_override = st.checkbox("Override Map Veto Draft?", value=False, key="ma_veto_override")
        if veto_override:
            ma_maps_count = 3 if ma_series_type == "Bo3" else 5
            ma_custom_maps = []
            ma_custom_weights = {}
            ma_veto_cols = st.columns(ma_maps_count)
            for idx in range(ma_maps_count):
                with ma_veto_cols[idx]:
                    map_val = st.selectbox(f"Map {idx+1}", all_maps, index=min(idx, len(all_maps)-1), key=f"ma_custom_map_{idx}")
                    weight_val = st.select_slider(f"Weight (map {idx+1})", options=[-1, 0, 1],
                                                   value=0 if idx == (ma_maps_count-1) else (1 if idx % 2 == 0 else -1),
                                                   key=f"ma_custom_weight_{idx}")
                    ma_custom_maps.append(map_val)
                    ma_custom_weights[map_val] = weight_val
            predicted_veto = {
                "maps": ma_custom_maps,
                "veto_weights": ma_custom_weights,
                "veto_str": "Custom manual veto override"
            }
        else:
            predicted_veto = None  # will be computed below
        st.markdown('</div>', unsafe_allow_html=True)

    # Resolve selected match
    selected_match = matches_lookup[selected_match_id]
    ma_team_a = selected_match["team_a"]
    ma_team_b = selected_match["team_b"]
    segment = selected_match["segment"]

    if predicted_veto is None:
        predicted_veto = veto_pred.predict_veto(ma_team_a, ma_team_b, ma_series_type)

    # ── Actual vs Predicted Comparison Banner ──
    actual_maps_data = segment.get("maps", [])
    actual_team_a_score = segment["teams"][0].get("score", "?") if len(segment.get("teams", [])) > 0 else "?"
    actual_team_b_score = segment["teams"][1].get("score", "?") if len(segment.get("teams", [])) > 1 else "?"
    actual_winner_idx = None
    for i, t in enumerate(segment.get("teams", [])):
        if t.get("is_winner"):
            actual_winner_idx = i
            break
    actual_winner_name = segment["teams"][actual_winner_idx]["name"] if actual_winner_idx is not None else "N/A"
    actual_map_names = [m.get("map_name", "?") for m in actual_maps_data]

    # Model-side features
    def get_roster_features(roster):
        acs_list, kast_list, duel_list = [], [], []
        for p_name in roster:
            p_feat = player_emas.get(p_name)
            if p_feat is not None:
                acs_list.append(p_feat["acs"])
                kast_list.append(p_feat["kast"])
                duel_list.append(p_feat["duel_diff"])
            else:
                p_base = baseline_lookup.get(p_name, {"acs": 200.0, "kast": 0.70, "duel_diff": 0.0})
                acs_list.append(p_base["acs"])
                kast_list.append(p_base["kast"])
                duel_list.append(p_base["duel_diff"])
        return (
            sum(acs_list) / len(acs_list) if acs_list else 200.0,
            sum(kast_list) / len(kast_list) if kast_list else 0.70,
            sum(duel_list) / len(duel_list) if duel_list else 0.0
        )

    roster_a = []
    roster_b = []
    for map_data in actual_maps_data:
        for p in map_data.get('players', {}).get('team1', []):
            roster_a.append(p['name'])
        for p in map_data.get('players', {}).get('team2', []):
            roster_b.append(p['name'])
    roster_a = list(set(roster_a)) or get_latest_roster(ma_team_a, RAW_DIR)
    roster_b = list(set(roster_b)) or get_latest_roster(ma_team_b, RAW_DIR)

    ta_acs, ta_kast, ta_duel = get_roster_features(roster_a)
    tb_acs, tb_kast, tb_duel = get_roster_features(roster_b)
    ta_feat = team_stats.get(ma_team_a, {})
    tb_feat = team_stats.get(ma_team_b, {})
    ta_loadout = ta_feat.get("loadout", 20000.0)
    tb_loadout = tb_feat.get("loadout", 20000.0)

    # Run V5 engine
    v5_engine = get_v5_simulation_engine()
    with st.spinner("Running V5 Bottom-Up Micro-Simulation (2,000 iterations)..."):
        v5_res = v5_engine.simulate_match(ma_team_a, ma_team_b, ma_series_type, target_patch="9.02", num_iterations=2000)
    win_prob_a = v5_res["win_prob_a"]
    win_prob_b = v5_res["win_prob_b"]
    predicted_winner = ma_team_a if win_prob_a > win_prob_b else ma_team_b

    # ── Actual vs Predicted Side-by-Side ──
    st.markdown("### 📊 Actual vs. Predicted — Series Overview")
    avp_col_actual, avp_col_divider, avp_col_predicted = st.columns([5, 1, 5])

    with avp_col_actual:
        st.markdown(clean_html(f"""
            <div class="glass-card" style="border-color: rgba(34,197,94,0.2);">
                <div class="metric-title" style="color: #4ade80; margin-bottom: 14px;">
                    ✅ ACTUAL RESULT
                </div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 14px;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">{ma_team_a}</div>
                        <div style="font-size: 3rem; font-weight: 800; color: #f8fafc; line-height: 1;">{actual_team_a_score}</div>
                    </div>
                    <div style="font-size: 1.4rem; color: #334155; font-weight: 700;">—</div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">{ma_team_b}</div>
                        <div style="font-size: 3rem; font-weight: 800; color: #f8fafc; line-height: 1;">{actual_team_b_score}</div>
                    </div>
                </div>
                <div style="text-align: center; padding: 8px 16px; border-radius: 8px; background: rgba(34,197,94,0.1);
                            border: 1px solid rgba(34,197,94,0.2); color: #4ade80; font-weight: 700; font-size: 0.95rem;">
                    🏆 Winner: {actual_winner_name}
                </div>
                <div style="margin-top: 14px;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">Maps Played</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                        {''.join(f'<span style="padding: 3px 10px; border-radius: 99px; background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.2); color: #4ade80; font-size: 0.78rem; font-weight: 600;">{mn}</span>' for mn in actual_map_names)}
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

    with avp_col_divider:
        st.markdown('<div style="height: 100%; display: flex; align-items: center; justify-content: center; color: #334155; font-size: 1.4rem; font-weight: 700; padding-top: 60px;">VS</div>', unsafe_allow_html=True)

    with avp_col_predicted:
        pred_a_score_str = f"~{round(win_prob_a * (3 if ma_series_type == 'Bo5' else 2))}"
        pred_b_score_str = f"~{round(win_prob_b * (3 if ma_series_type == 'Bo5' else 2))}"
        pred_map_names = predicted_veto.get("maps", [])
        st.markdown(clean_html(f"""
            <div class="glass-card" style="border-color: rgba(168,85,247,0.2);">
                <div class="metric-title" style="color: #a78bfa; margin-bottom: 14px;">
                    🔮 ENGINE PREDICTION
                </div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 14px;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">{ma_team_a}</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #a78bfa; line-height: 1;">{win_prob_a:.0%}</div>
                    </div>
                    <div style="font-size: 1.4rem; color: #334155; font-weight: 700;">—</div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">{ma_team_b}</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #f59e0b; line-height: 1;">{win_prob_b:.0%}</div>
                    </div>
                </div>
                <div style="text-align: center; padding: 8px 16px; border-radius: 8px; background: rgba(168,85,247,0.1);
                            border: 1px solid rgba(168,85,247,0.2); color: #a78bfa; font-weight: 700; font-size: 0.95rem;">
                    🔮 Predicted: {predicted_winner}
                </div>
                <div style="margin-top: 14px;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">Predicted Map Pool</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                        {''.join(f'<span style="padding: 3px 10px; border-radius: 99px; background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.2); color: #a78bfa; font-size: 0.78rem; font-weight: 600;">{mn}</span>' for mn in pred_map_names)}
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

    # ── Accuracy Callout ──
    winner_correct = actual_winner_name.strip().lower() == predicted_winner.strip().lower()
    accuracy_msg = "✅ Winner Prediction: CORRECT" if winner_correct else "❌ Winner Prediction: INCORRECT"
    accuracy_color = "#4ade80" if winner_correct else "#ef4444"
    map_overlap = len(set(actual_map_names) & set(pred_map_names))
    st.markdown(clean_html(f"""
        <div style="display: flex; gap: 14px; margin-bottom: 24px;">
            <div style="flex: 1; padding: 12px 18px; border-radius: 10px; background: {accuracy_color}11;
                        border: 1px solid {accuracy_color}33; text-align: center; font-weight: 700; color: {accuracy_color}; font-size: 0.9rem;">
                {accuracy_msg}
            </div>
            <div style="flex: 1; padding: 12px 18px; border-radius: 10px; background: rgba(56,189,248,0.08);
                        border: 1px solid rgba(56,189,248,0.2); text-align: center; font-weight: 600; color: #38bdf8; font-size: 0.9rem;">
                🗺️ Map Overlap: {map_overlap}/{len(actual_map_names)} maps correctly predicted
            </div>
        </div>
    """), unsafe_allow_html=True)

    # ── Series Winner Projection ──
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(clean_html(f"""
            <div class="glass-card">
                <div class="metric-title">SERIES WIN PROBABILITY</div>
                <div style="display: flex; justify-content: space-between; gap: 24px; margin-top: 18px;">
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #a78bfa;">{ma_team_a}</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">{win_prob_a:.1%}</div>
                        <div style="background: rgba(167, 139, 250, 0.15); border-radius: 99px; height: 8px; overflow: hidden; margin-top: 6px;">
                            <div style="background: #a78bfa; width: {win_prob_a * 100}%; height: 100%; border-radius: 99px;"></div>
                        </div>
                    </div>
                    <div style="width: 1px; background: rgba(255, 255, 255, 0.05); align-self: stretch;"></div>
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #f59e0b;">{ma_team_b}</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">{win_prob_b:.1%}</div>
                        <div style="background: rgba(245, 158, 11, 0.15); border-radius: 99px; height: 8px; overflow: hidden; margin-top: 6px;">
                            <div style="background: #f59e0b; width: {win_prob_b * 100}%; height: 100%; border-radius: 99px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

    with col2:
        winner_name = ma_team_a if win_prob_a > win_prob_b else ma_team_b
        win_conf = win_prob_a if win_prob_a > win_prob_b else win_prob_b
        st.markdown(clean_html(f"""
            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="metric-title">PREDICTED WINNER</div>
                    <div class="winner-box" style="margin-top: 15px;">
                        🏆 {winner_name} ({win_conf:.1%})
                    </div>
                </div>
                <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 10px;">
                    {predicted_veto['veto_str']}
                </div>
            </div>
        """), unsafe_allow_html=True)

    # ── Projected Map Scores ──
    st.markdown("### Projected Map Scores")
    cols_maps = st.columns(len(predicted_veto["maps"]))
    for idx, m_name in enumerate(predicted_veto["maps"]):
        with cols_maps[idx]:
            team_a_features = {"acs_ema": ta_acs, "avg_loadout": ta_loadout, "comfort_diff": 0.0}
            team_b_features = {"acs_ema": tb_acs, "avg_loadout": tb_loadout, "comfort_diff": 0.0}
            veto_w = predicted_veto["veto_weights"].get(m_name, 0)
            rounds_a, rounds_b = score_reg.predict_score(team_a_features, team_b_features, m_name, veto_w)

            picker = "Decider"
            if veto_w == 1:
                picker = f"Picked by {ma_team_a}"
            elif veto_w == -1:
                picker = f"Picked by {ma_team_b}"

            # Check actual score if this map was played
            actual_score_str = ""
            for am in actual_maps_data:
                if am.get("map_name") == m_name:
                    sc = am.get("score", {})
                    actual_score_str = f"Actual: {sc.get('team1','?')}-{sc.get('team2','?')}"
                    break

            st.markdown(clean_html(f"""
                <div class="glass-card" style="text-align: center; padding: 20px 14px;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">MAP {idx+1}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin: 6px 0;">{m_name}</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #ff4655; margin: 8px 0; letter-spacing: -1px;">{rounds_a} - {rounds_b}</div>
                    <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 500;">{picker}</div>
                    {f'<div style="margin-top: 8px; font-size: 0.78rem; color: #4ade80; font-weight: 600;">{actual_score_str}</div>' if actual_score_str else ''}
                </div>
            """), unsafe_allow_html=True)

    # ── Predicted Agent Compositions with Actual Side-by-Side ──
    st.markdown("### Projected vs. Actual Agent Compositions")
    tab_maps = st.tabs([f"Map {i+1}: {name}" for i, name in enumerate(predicted_veto["maps"])])

    # Build actual compositions lookup
    actual_comp_lookup = {}
    for am in actual_maps_data:
        mname = am.get("map_name")
        if mname:
            actual_comp_lookup[mname] = {
                "team1": [(p["name"], p.get("agent", "?")) for p in am.get("players", {}).get("team1", [])],
                "team2": [(p["name"], p.get("agent", "?")) for p in am.get("players", {}).get("team2", [])],
            }

    for idx, m_name in enumerate(predicted_veto["maps"]):
        with tab_maps[idx]:
            comp_a_map = agent_comp.predict_composition(ma_team_a, m_name)
            comp_b_map = agent_comp.predict_composition(ma_team_b, m_name)
            actual_this_map = actual_comp_lookup.get(m_name, {})

            col_la, col_lb = st.columns(2)

            with col_la:
                st.markdown(f"#### {ma_team_a}")
                # Predicted
                st.markdown('<span class="predicted-badge">🔮 Predicted</span>', unsafe_allow_html=True)
                cards_a_html = ""
                for p_name, details in comp_a_map.items():
                    agent = details["agent"]
                    role = details["role"]
                    icon = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png")
                    cards_a_html += f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.02); padding-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <img src="{icon}" width="34" height="34" style="border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);"/>
                                <div>
                                    <div style="font-weight: 600; font-size: 0.93rem; color: #f8fafc;">{p_name}</div>
                                    <div style="font-size: 0.73rem; color: #94a3b8;">{agent}</div>
                                </div>
                            </div>
                            <span style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05); color: #38bdf8;">{role}</span>
                        </div>
                    """
                st.markdown(clean_html(f'<div class="glass-card">{cards_a_html}</div>'), unsafe_allow_html=True)

                # Actual (if available)
                if actual_this_map.get("team1"):
                    st.markdown('<span class="actual-badge">✅ Actual</span>', unsafe_allow_html=True)
                    actual_a_html = ""
                    for p_name, agent in actual_this_map["team1"]:
                        icon = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png")
                        actual_a_html += f"""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                                <img src="{icon}" width="30" height="30" style="border-radius: 4px; border: 1px solid rgba(34,197,94,0.3);"/>
                                <div style="font-size: 0.88rem; color: #e2e8f0; font-weight: 500;">{p_name} <span style="color:#64748b; font-size:0.75rem;">· {agent}</span></div>
                            </div>
                        """
                    st.markdown(clean_html(f'<div class="glass-card" style="border-color: rgba(34,197,94,0.2);">{actual_a_html}</div>'), unsafe_allow_html=True)

            with col_lb:
                st.markdown(f"#### {ma_team_b}")
                st.markdown('<span class="predicted-badge">🔮 Predicted</span>', unsafe_allow_html=True)
                cards_b_html = ""
                for p_name, details in comp_b_map.items():
                    agent = details["agent"]
                    role = details["role"]
                    icon = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png")
                    cards_b_html += f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.02); padding-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <img src="{icon}" width="34" height="34" style="border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);"/>
                                <div>
                                    <div style="font-weight: 600; font-size: 0.93rem; color: #f8fafc;">{p_name}</div>
                                    <div style="font-size: 0.73rem; color: #94a3b8;">{agent}</div>
                                </div>
                            </div>
                            <span style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05); color: #38bdf8;">{role}</span>
                        </div>
                    """
                st.markdown(clean_html(f'<div class="glass-card">{cards_b_html}</div>'), unsafe_allow_html=True)

                if actual_this_map.get("team2"):
                    st.markdown('<span class="actual-badge">✅ Actual</span>', unsafe_allow_html=True)
                    actual_b_html = ""
                    for p_name, agent in actual_this_map["team2"]:
                        icon = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png")
                        actual_b_html += f"""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                                <img src="{icon}" width="30" height="30" style="border-radius: 4px; border: 1px solid rgba(34,197,94,0.3);"/>
                                <div style="font-size: 0.88rem; color: #e2e8f0; font-weight: 500;">{p_name} <span style="color:#64748b; font-size:0.75rem;">· {agent}</span></div>
                            </div>
                        """
                    st.markdown(clean_html(f'<div class="glass-card" style="border-color: rgba(34,197,94,0.2);">{actual_b_html}</div>'), unsafe_allow_html=True)

    # ── Fantasy Leaderboard ──
    st.markdown("### Valorant Fantasy League Leaderboard")
    fantasy_eng = VCTFantasyEngine()
    filepath = selected_match["filepath"]
    leaderboard = fantasy_eng.score_match_json(filepath)
    if leaderboard:
        lead_df = pd.DataFrame(leaderboard)
        lead_df = lead_df.rename(columns={
            "player": "Player Name",
            "team": "VCT Team",
            "avg_rating": "VLR Rating",
            "map_score_agg": "Map Points Agg (Top 2)",
            "series_bonus": "Series Bonus",
            "rating_placement_bonus": "Placement Bonus",
            "rating_scaling_bonus": "Rating Scaling Bonus",
            "total_score": "Total Fantasy Score"
        })
        display_df = lead_df.drop(columns=["map_scores"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Leaderboard scores currently unavailable for this match.")

    st.markdown("### V5 Projected Fantasy Points (Expected Value)")
    if "projections" in v5_res:
        proj_data = []
        for p, ev in v5_res["projections"].items():
            team = ma_team_a if p in v5_res["roster_a"] else ma_team_b
            proj_data.append({"Player": p, "Team": team, "Expected Value (EV) Points": ev})
        proj_df = pd.DataFrame(proj_data).sort_values("Expected Value (EV) Points", ascending=False)
        st.dataframe(proj_df, use_container_width=True, hide_index=True)
    else:
        st.info("EV Projections unavailable.")

# ============================================================
# TAB 3: ROSTER OPTIMIZER
# ============================================================
with tab_optimizer:
    st.markdown("### 🧠 VFL Fantasy Manager Hub")

    # Live Meta Radar
    st.markdown(f"#### 📡 Live Meta Radar (Patch {latest_patch})")
    if active_penalties:
        sorted_penalties = sorted(active_penalties.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_penalties[:3]
        cols = st.columns(3)
        for idx, (agent_name, score) in enumerate(top_3):
            with cols[idx]:
                if score >= 0.5:
                    color = "#ef4444"
                    severity = "CRITICAL NERF"
                else:
                    color = "#f97316"
                    severity = "MODERATE NERF"
                icon_url = AGENT_ICONS.get(agent_name, "https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png")
                st.markdown(clean_html(f"""
                    <div style="background: rgba(26, 29, 36, 0.6); border-left: 5px solid {color}; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                        <img src="{icon_url}" width="42" height="42" style="border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);"/>
                        <div>
                            <div style="font-size: 0.75rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 0.05em;">{severity}</div>
                            <div style="font-weight: 600; font-size: 1.1rem; color: #f8fafc;">{agent_name}</div>
                            <div style="font-weight: 700; font-size: 1.4rem; color: {color}; margin-top: 2px;">-{score:.2f}</div>
                        </div>
                    </div>
                """), unsafe_allow_html=True)
    else:
        st.info("No active patch nerfs registered in the automated registry.")

    # Budget slider
    salary_cap = st.slider("Available Fantasy Budget (VP)", min_value=35.0, max_value=60.0, value=50.0, step=0.5, key="global_salary_cap")

    col_roster, col_transfer = st.columns([3, 2])

    with col_roster:
        st.markdown("#### 🏆 VCT 2026 Stage 2 Optimal Roster")
        st.markdown(clean_html("""
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">
                Mathematically optimized using MILP to maximize projected points under strict VFL constraints.
            </p>
        """), unsafe_allow_html=True)

        with st.spinner("Computing optimal Stage 2 baseline roster..."):
            baseline_result = optimize_roster(vfl_players_data, salary_cap=salary_cap, survival_threshold=0.35)

        if baseline_result["solver_status"] == "optimal":
            st.markdown(clean_html(f"""
                <div class="optimizer-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; padding: 4px 12px; border-radius: 12px; background: rgba(34, 197, 94, 0.15); color: #4ade80;">
                                optimal
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.8rem; color: #94a3b8;">Projected Score</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc;">{baseline_result['projected_points']} pts</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 20px; margin-top: 10px;">
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Total Cost</span>
                            <div style="font-weight: 600;">{baseline_result['total_cost']} VP</div>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Buffer Float</span>
                            <div style="font-weight: 600; color: #4ade80;">{salary_cap - baseline_result['total_cost']} VP</div>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Active IGL</span>
                            <div style="font-weight: 600; color: #38bdf8;">👑 {baseline_result['igl_player']}</div>
                        </div>
                    </div>
                </div>
            """), unsafe_allow_html=True)

            for idx, p in enumerate(baseline_result["optimal_roster"]):
                role_emoji = {"Duelist": "⚔️", "Controller": "🌀", "Initiator": "🔍", "Sentinel": "🛡️", "Flex": "🔄"}.get(p["role"], "🎮")
                igl_badge = '<span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; margin-left: 8px;">👑 IGL (2x Multiplier)</span>' if p["is_igl"] else ""
                wc_badge = '<span style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); color: #facc15; margin-left: 8px;">Wildcard</span>' if p["is_wildcard"] else f'<span style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); color: #a78bfa; margin-left: 8px;">{p["role"]}</span>'
                penalty_badge = get_meta_penalty_badge(p["player_name"], player_agent_stats, active_penalties)

                st.markdown(clean_html(f"""
                    <div class="glass-card" style="padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="font-size: 1.3rem; font-weight: 700; color: #6366f1;">#{idx+1}</span>
                            <div>
                                <div style="font-weight: 600; font-size: 1.0rem; display: flex; align-items: center;">
                                    {p['player_name']} {igl_badge} {penalty_badge}
                                </div>
                                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                                    Role: {role_emoji} {wc_badge} · Team: <span style="color: #4ade80;">{id_to_team_name.get(p['vlr_team_id'], f"ID: {p['vlr_team_id']}")}</span>
                                </div>
                             </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 600; color: #f8fafc;">{p['price']} VP</div>
                            <div style="font-size: 0.75rem; color: #4ade80;">{p['ppg']:.1f} projected pts</div>
                        </div>
                    </div>
                """), unsafe_allow_html=True)
        else:
            st.error("Roster Optimizer was unable to calculate an optimal starting lineup. Try updating the database.")

    # ── Transfer Advisor with Persistence ──
    with col_transfer:
        st.markdown("#### 🔄 3-Transfer Advisor")
        st.markdown(clean_html("""
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">
                Enter your current fantasy roster. Bank balance is auto-calculated from the VP cap.
            </p>
        """), unsafe_allow_html=True)

        player_names_list = sorted([p["player_name"] for p in vfl_players_data])

        # Determine default selections: prefer saved state
        saved_names = st.session_state.get("saved_roster_names", [])
        valid_saved = [n for n in saved_names if n in player_names_list]
        if len(valid_saved) == 6:
            default_selections = valid_saved
        else:
            # Fallback defaults
            fallback_defaults = [n for n in ["aspas", "zekken", "wo0t", "Derke", "Leo", "something"] if n in player_names_list]
            if len(fallback_defaults) < 6:
                fallback_defaults = player_names_list[:6]
            default_selections = fallback_defaults[:6]

        current_roster_names = st.multiselect(
            "Select Your Current 6 Players",
            player_names_list,
            default=default_selections,
            key="transfer_current_roster_new"
        )

        # Auto floating-bank calculation (v5_fantasy_state_management.md §1)
        current_roster_objs = []
        for name in current_roster_names:
            for p in vfl_players_data:
                if p["player_name"] == name:
                    current_roster_objs.append(p)
                    break

        current_roster_value = sum(p["price"] for p in current_roster_objs)
        floating_bank = salary_cap - current_roster_value

        st.markdown(clean_html(f"""
            <div style="display: flex; gap: 12px; margin: 10px 0 14px;">
                <div style="flex: 1; padding: 10px 14px; border-radius: 10px; background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2); text-align: center;">
                    <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Roster Cost</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc;">{current_roster_value} VP</div>
                </div>
                <div style="flex: 1; padding: 10px 14px; border-radius: 10px; background: {'rgba(34,197,94,0.08)' if floating_bank >= 0 else 'rgba(239,68,68,0.08)'}; border: 1px solid {'rgba(34,197,94,0.2)' if floating_bank >= 0 else 'rgba(239,68,68,0.2)'}; text-align: center;">
                    <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Floating Bank</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {'#4ade80' if floating_bank >= 0 else '#ef4444'};">{floating_bank:.1f} VP</div>
                </div>
            </div>
        """), unsafe_allow_html=True)

        # IGL Designation (v5_fantasy_state_management.md §2)
        forced_igl_name = None
        if len(current_roster_names) > 0:
            igl_options = ["Auto-Detect (Highest Floor)"] + current_roster_names
            saved_igl = st.session_state.get("saved_igl_name")
            igl_default_idx = 0
            if saved_igl in igl_options:
                igl_default_idx = igl_options.index(saved_igl)
            selected_igl_opt = st.selectbox(
                "👑 Designate IGL (2× EV Multiplier)",
                igl_options,
                index=igl_default_idx,
                key="igl_selector_v5"
            )
            if selected_igl_opt != "Auto-Detect (Highest Floor)":
                forced_igl_name = selected_igl_opt

            # Show IGL EV preview if IGL is selected
            if forced_igl_name:
                igl_obj = next((p for p in vfl_players_data if p["player_name"] == forced_igl_name), None)
                if igl_obj:
                    non_igl_ev = sum(p["ppg"] for p in current_roster_objs if p["player_name"] != forced_igl_name)
                    total_ev = non_igl_ev + 2.0 * igl_obj["ppg"]
                    st.markdown(clean_html(f"""
                        <div style="padding: 8px 14px; border-radius: 8px; background: rgba(56,189,248,0.08);
                                    border: 1px solid rgba(56,189,248,0.2); font-size: 0.85rem; color: #38bdf8; margin-bottom: 10px;">
                            👑 IGL: <b>{forced_igl_name}</b> ({igl_obj['ppg']:.1f} pts × 2) — Total Roster EV: <b>{total_ev:.1f} pts</b>
                        </div>
                    """), unsafe_allow_html=True)
        else:
            st.info("Select players to enable IGL selection.")

        # Save / Load Roster Persistence (v5_fantasy_state_management.md §3)
        persist_col1, persist_col2 = st.columns(2)
        with persist_col1:
            if st.button("💾 Save Current Roster", key="btn_save_roster", use_container_width=True):
                if len(current_roster_names) == 6:
                    save_roster_state(current_roster_names, forced_igl_name)
                    st.session_state["saved_roster_names"] = current_roster_names
                    st.session_state["saved_igl_name"] = forced_igl_name
                    st.success("✅ Roster saved to disk!")
                else:
                    st.warning("Select exactly 6 players to save.")
        with persist_col2:
            roster_exists = os.path.exists(ROSTER_STATE_PATH)
            if st.button("📂 Load Saved Roster", key="btn_load_roster", disabled=not roster_exists, use_container_width=True):
                loaded_names, loaded_igl = load_roster_state()
                st.session_state["saved_roster_names"] = loaded_names
                st.session_state["saved_igl_name"] = loaded_igl
                st.rerun()

        st.markdown("---")

        # Transfer calculation
        if st.button("🔮 Calculate Optimal Trades", key="btn_suggest_transfers_new", type="primary", use_container_width=True):
            if len(current_roster_names) != 6:
                st.error("Please select exactly 6 players currently in your roster.")
            else:
                with st.spinner("Analyzing transfer combinations..."):
                    transfer_result = suggest_transfers(
                        current_roster_objs,
                        vfl_players_data,
                        remaining_bank_balance=float(max(0.0, floating_bank)),
                        forced_igl_name=forced_igl_name
                    )

                if transfer_result["solver_status"] == "optimal":
                    recs = transfer_result.get("recommendations", [])
                    if recs:
                        tabs = st.tabs([f"Option {idx+1} (+{rec['projected_gain']:.1f} pts)" for idx, rec in enumerate(recs)])
                        for idx, rec in enumerate(recs):
                            with tabs[idx]:
                                if rec["projected_gain"] > 0:
                                    st.markdown(clean_html(f"""
                                        <div class="optimizer-card" style="background: rgba(34, 197, 94, 0.05); border-color: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 8px; border: 1px solid; margin-bottom: 15px;">
                                            <div style="font-weight: 700; font-size: 1.1rem; color: #4ade80;">
                                                📈 Projected Score Velocity: +{rec['projected_gain']:.1f} pts
                                            </div>
                                        </div>
                                    """), unsafe_allow_html=True)

                                    st.markdown("**Suggested Swaps (Max 3 Trades):**")

                                    for p in rec["transfers_out"]:
                                        p_name = p["player_name"]
                                        p_name_clean = p_name.lower().strip()
                                        primary_agent = None
                                        max_count = -1
                                        for (name, agent), info in player_agent_stats.items():
                                            if name.lower().strip() == p_name_clean:
                                                if info['count'] > max_count:
                                                    max_count = info['count']
                                                    primary_agent = agent
                                        reason = "Transfer Reason: Budget optimization and roster rebalancing to maximize score velocity."
                                        if primary_agent:
                                            penalty = active_penalties.get(primary_agent, 0.0)
                                            if penalty > 0.10:
                                                reason = f"Transfer Reason: Player's primary agent ({primary_agent}) suffered a {penalty:.2f} Ghost/Meta Nerf."
                                        st.markdown(clean_html(f"""
                                            <div class="transfer-out" style="padding: 10px 14px; margin-bottom: 8px; border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.05);">
                                                <span style="color: #ef4444; font-weight: 700;">OUT ⬇</span>
                                                <span style="margin-left: 12px; font-weight: 600;">{p['player_name']}</span>
                                                <span style="color: #94a3b8; font-size: 0.8rem;"> · Cost: {p['price']} VP · PPG: {p['ppg']:.1f}</span>
                                                <div style="font-size: 0.8rem; color: #ef4444; margin-top: 4px; font-style: italic;">{reason}</div>
                                            </div>
                                        """), unsafe_allow_html=True)

                                    for p in rec["transfers_in"]:
                                        igl_tag = " 👑" if p["is_igl"] else ""
                                        p_name = p["player_name"]
                                        p_name_clean = p_name.lower().strip()
                                        primary_agent = None
                                        max_count = -1
                                        for (name, agent), info in player_agent_stats.items():
                                            if name.lower().strip() == p_name_clean:
                                                if info['count'] > max_count:
                                                    max_count = info['count']
                                                    primary_agent = agent
                                        if primary_agent:
                                            reason = f"Transfer Reason: High-performing meta asset on comfort agent ({primary_agent}) with 0.00 penalty."
                                        else:
                                            reason = "Transfer Reason: Optimal target pickup to maximize projected score under budget cap."
                                        st.markdown(clean_html(f"""
                                            <div class="transfer-in" style="padding: 10px 14px; margin-bottom: 8px; border-left: 4px solid #4ade80; background: rgba(74, 222, 128, 0.05);">
                                                <span style="color: #4ade80; font-weight: 700;">IN ⬆</span>
                                                <span style="margin-left: 12px; font-weight: 600;">{p['player_name']}{igl_tag}</span>
                                                <span style="color: #94a3b8; font-size: 0.8rem;"> · Cost: {p['price']} VP · PPG: {p['ppg']:.1f}</span>
                                                <div style="font-size: 0.8rem; color: #4ade80; margin-top: 4px; font-style: italic;">{reason}</div>
                                            </div>
                                        """), unsafe_allow_html=True)

                                    st.markdown(clean_html(f"""
                                        <div style="margin-top: 10px; color: #94a3b8; font-size: 0.8rem; text-align: right;">
                                            New Total Cost: <b>{rec['new_total_cost']} VP</b> | New Projected Points: <b>{rec['new_projected_points']:.1f} pts</b>
                                        </div>
                                    """), unsafe_allow_html=True)
                                else:
                                    st.success("✅ Your current roster is already optimally positioned! No transfers recommended.")
                    else:
                        st.success("✅ Your current roster is already optimally positioned! No transfers recommended.")
                else:
                    st.error(f"Solver Error: {transfer_result['solver_status']}. Try adjusting the inputs.")

# ============================================================
# TAB 4: VFL PLAYERS DATABASE
# ============================================================
with tab_vfl:
    st.markdown("### 📋 VFL Player Database")
    
    col_db1, col_db2 = st.columns([3, 1])
    with col_db2:
        if st.button("🔄 Update VFL Database", key="btn_update_vfl_db", use_container_width=True):
            with st.spinner("Executing scraper and rebuilding JSON registry..."):
                vfl_players_data_refreshed = vfl_scraper_inst.scrape_player_stats()
            st.success(f"Rebuilt VFL Database Cache with {len(vfl_players_data_refreshed)} players!")

    if vfl_players_data:
        vfl_df = pd.DataFrame(vfl_players_data)

        if not vfl_df.empty:
            if 'cost' not in vfl_df.columns and 'price' in vfl_df.columns:
                vfl_df['cost'] = vfl_df['price']
            if 'avg_points' not in vfl_df.columns and 'ppg' in vfl_df.columns:
                vfl_df['avg_points'] = vfl_df['ppg']
            if 'team' not in vfl_df.columns and 'vlr_team_id' in vfl_df.columns:
                vfl_df['team'] = vfl_df['vlr_team_id'].map(id_to_team_name).fillna(vfl_df['vlr_team_id'])
            elif 'team' in vfl_df.columns:
                vfl_df['team'] = vfl_df['team'].map(id_to_team_name).fillna(vfl_df['team'])
            if 'total_points' not in vfl_df.columns and 'tot_pts' in vfl_df.columns:
                vfl_df['total_points'] = vfl_df['tot_pts']
            if 'ownership_pct' not in vfl_df.columns:
                vfl_df['ownership_pct'] = 0.0

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Players", len(vfl_df))
        with m_col2:
            st.metric("Avg Cost", f"{vfl_df['cost'].mean():,.1f} VP")
        with m_col3:
            st.metric("Avg PPG", f"{vfl_df['avg_points'].mean():.1f}")
        with m_col4:
            st.metric("Teams", vfl_df['team'].nunique())

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            team_filter = st.multiselect("Filter by Team", sorted(vfl_df['team'].unique()), key="vfl_team_filter")
        with filter_col2:
            role_filter = st.multiselect("Filter by Role", sorted(vfl_df['role'].unique()), key="vfl_role_filter")

        filtered_df = vfl_df.copy()
        if team_filter:
            filtered_df = filtered_df[filtered_df['team'].isin(team_filter)]
        if role_filter:
            filtered_df = filtered_df[filtered_df['role'].isin(role_filter)]

        filtered_df = filtered_df.sort_values('avg_points', ascending=False)

        display_vfl_df = filtered_df.copy()
        display_vfl_df['cost'] = display_vfl_df['cost'].apply(lambda x: f"{x} VP")
        display_vfl_df['ownership_pct'] = display_vfl_df['ownership_pct'].apply(lambda x: f"{x:.1f}%")

        display_vfl_df = display_vfl_df.rename(columns={
            "player_name": "Player",
            "team": "Team",
            "role": "Role",
            "cost": "Cost",
            "total_points": "Total Points",
            "avg_points": "Avg PPG",
            "ownership_pct": "Ownership %"
        })

        st.dataframe(display_vfl_df, use_container_width=True, hide_index=True)
    else:
        st.info("No VFL data available. Click '🔄 Update VFL Database' in the sidebar to load data.")
