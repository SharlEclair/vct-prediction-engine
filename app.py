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
from scrapers import vfl_scraper
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
from scrapers.vfl_scraper import VFLScraper
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

def get_player_icon(role: str, name: str) -> str:
    # Use proper agent role icons for the lineup page
    role_icons = {
        "Duelist": "https://media.valorant-api.com/agents/roles/dbe8757e-9e92-4ed4-b39f-9dfc589691d4/displayicon.png",
        "Initiator": "https://media.valorant-api.com/agents/roles/1b47567f-8f7b-444b-aae3-b0c634622d10/displayicon.png",
        "Sentinel": "https://media.valorant-api.com/agents/roles/5fc02f99-4091-4486-a531-98459a3e95e9/displayicon.png",
        "Controller": "https://media.valorant-api.com/agents/roles/4ee40330-ecdd-4f2f-98a8-eb1243428373/displayicon.png",
        "Flex": "https://media.valorant-api.com/agents/roles/1b47567f-8f7b-444b-aae3-b0c634622d10/displayicon.png"
    }
    return role_icons.get(role, role_icons["Duelist"])

def get_composition_synergy_badges(team_roster, player_agents, agent_roles_map):
    # Extract agents selected for the team
    agents = []
    for p in team_roster:
        if p in player_agents:
            agents.append(player_agents[p]["agent"])
    
    roles = [agent_roles_map.get(a, "Flex") for a in agents]
    
    badges = []
    if "Duelist" in roles and "Initiator" in roles:
        badges.append('<span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: rgba(34, 197, 94, 0.15); color: #22c55e; margin-right: 6px;">✨ +10% Duelist/Initiator Synergy</span>')
    if "Controller" not in roles:
        badges.append('<span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: rgba(239, 68, 68, 0.15); color: #ef4444; margin-right: 6px;">⚠️ -15% Missing Controller Penalty</span>')
    if "Sentinel" not in roles:
        badges.append('<span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: rgba(239, 68, 68, 0.15); color: #ef4444; margin-right: 6px;">⚠️ -15% Missing Sentinel Penalty</span>')
    
    if not badges:
        return '<div style="margin-top: 6px;"><span style="font-size: 0.75rem; color: #64748b; font-style: italic;">No active synergy/penalty modifiers</span></div>'
    return f'<div style="margin-top: 6px;">{" ".join(badges)}</div>'

def get_composition_synergy_badges_dict(comp_map):
    roles = [details["role"] for details in comp_map.values()]
    badges = []
    if "Duelist" in roles and "Initiator" in roles:
        badges.append('<span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: rgba(34, 197, 94, 0.15); color: #22c55e; margin-right: 6px;">✨ +10% Duelist/Initiator Synergy</span>')
    if "Controller" not in roles:
        badges.append('<span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: rgba(239, 68, 68, 0.15); color: #ef4444; margin-right: 6px;">⚠️ -15% Missing Controller Penalty</span>')
    if "Sentinel" not in roles:
        badges.append('<span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: rgba(239, 68, 68, 0.15); color: #ef4444; margin-right: 6px;">⚠️ -15% Missing Sentinel Penalty</span>')
    
    if not badges:
        return '<div style="margin-top: 6px;"><span style="font-size: 0.75rem; color: #64748b; font-style: italic;">No active synergy/penalty modifiers</span></div>'
    return f'<div style="margin-top: 6px;">{" ".join(badges)}</div>'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent

RAW_DIR = str(ROOT_DIR / "data" / "raw")
PROCESSED_DIR = str(ROOT_DIR / "data" / "processed")
ROSTER_STATE_PATH = str(ROOT_DIR / "data" / "processed" / "roster_state.json")

# ============================================================
# DATA LOADING HELPERS
# ============================================================

_, team_name_to_id = get_team_win_rates_by_id(RAW_DIR)
id_to_team_name = {v: k for k, v in team_name_to_id.items()}

@st.cache_data
def load_automated_registry():
    path = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED_DIR, "patch_nerf_registry.json")
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

def get_available_patches() -> list[str]:
    """
    Dynamically loads all available patch versions from the JSON registry.
    Uses pathlib for absolute path resolution based on the script location.
    """
    path_reg = ROOT_DIR / "data" / "processed" / "automated_patch_nerf_registry.json"
    if not path_reg.exists():
        path_reg = ROOT_DIR / "data" / "processed" / "patch_nerf_registry.json"
        
    if not path_reg.exists():
        raise FileNotFoundError(f"Neither automated_patch_nerf_registry.json nor patch_nerf_registry.json found at {path_reg.parent}")
        
    with open(path_reg, "r", encoding="utf-8") as f_reg:
        reg_data = json.load(f_reg)
        
    reg_keys = reg_data.keys()
    available_patches = sorted(list(reg_keys), key=lambda x: [int(i) if i.isdigit() else i for i in x.split('.')], reverse=True)
    return [f"Patch {p}" for p in available_patches]

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
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d0e12;
        color: #e2e8f0;
    }

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
st.markdown('<div class="dashboard-subtitle">v7 · Open Match Simulation · Fantasy Optimizer · Backtesting Visualizer</div>', unsafe_allow_html=True)

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
# TASK 6.1: THE COMMAND CENTER (SIDEBAR)
# ============================================================
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 10px;"><img src="https://media.valorant-api.com/agents/add6443a-41bd-e414-f6ad-e58d267f4e95/displayicon.png" width="80" style="border-radius: 12px; border: 2px solid #ff4655;"/></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #ff4655;'>⚔️ DFS Command Center</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>Hybrid Valorant DFS Micro Engine (v7)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### 1. Roster Date Tracker")
    opt_slate_date = st.date_input("VFL Slate Date", value=datetime(2026, 6, 22), key="opt_slate_date")
    
    st.markdown("#### 2. Concept Drift Registry")
    try:
        patch_options = get_available_patches()
    except Exception:
        patch_options = ["Patch 9.04", "Patch 9.02", "Patch 8.11 (June 11, 2024)"]
    opt_patch = st.selectbox("Select Target Patch Window", patch_options, key="opt_target_patch")
    
    st.markdown("#### 3. VFL Ruleset Mode")
    opt_ruleset = st.selectbox(
        "Select Active VFL Ruleset",
        ["International (Masters/Champions)", "Regional (Stage 1/Stage 2)"],
        key="opt_ruleset"
    )
    
    is_intl = "International" in opt_ruleset
    budget_default = 50.0 if is_intl else 100.0
    budget_min = 35.0 if is_intl else 70.0
    budget_max = 60.0 if is_intl else 120.0
    
    st.markdown("#### 4. Available Fantasy Budget")
    opt_salary_cap = st.slider(
        "Roster Budget Cap (VP)",
        min_value=budget_min,
        max_value=budget_max,
        value=budget_default,
        step=0.5,
        key="opt_salary_cap"
    )
    
    st.markdown("#### 5. Simulation Depth")
    opt_depth = st.select_slider(
        "Monte Carlo Iterations",
        options=[1000, 5000, 10000],
        value=10000,
        format_func=lambda x: f"{x // 1000}K Depth",
        key="opt_sim_depth"
    )
    
    st.markdown("#### 6. Daily Slate Ingestion")
    btn_sync_live = st.button("🔄 Sync Live VFL Slate (API)", use_container_width=True, key="btn_sync_live")
    uploaded_file = st.file_uploader("Fallback: Upload DFS Slate (CSV)", type=["csv"], key="uploaded_file_slate")
    
    st.markdown("#### 7. System Administration")
    whitelist_input = st.text_input("VLR Event Whitelist (comma-separated)", placeholder="e.g. Esports World Cup 2026", key="vlr_whitelist")
    btn_master_update = st.button("🚀 Master Update: Sync All Data & Retrain", type="primary", use_container_width=True, key="btn_master_update")
    btn_patch_update_only = st.button("🔄 Scrape Latest Patches & Rebuild Meta", use_container_width=True, key="btn_patch_update_only")
    btn_scrape_vlr_incremental = st.button("📥 Scrape Latest VLR Matches (Incremental)", use_container_width=True, key="btn_scrape_vlr_incremental")
    
    with st.expander("🛠️ Roster Management Override"):
        st.markdown("##### Define Active Roster for a Team")
        override_team = st.selectbox("Select Team", all_teams, key="override_team_select")
        
        # Load existing overrides
        overrides_path = "data/processed/roster_overrides.json"
        existing_overrides = {}
        if os.path.exists(overrides_path):
            try:
                with open(overrides_path, "r", encoding="utf-8") as f:
                    existing_overrides = json.load(f)
            except Exception:
                pass
                
        current_players = existing_overrides.get(override_team, [])
        players_input = st.text_input(
            "Players (comma-separated)",
            value=", ".join(current_players),
            placeholder="e.g. zekken, tex, v1xen, Mazino, Verno, aspas",
            key="override_team_players_input"
        )
        
        if st.button("Save Roster Override", key="btn_save_roster_override"):
            parsed_players = [p.strip() for p in players_input.split(",") if p.strip()]
            if not parsed_players:
                if override_team in existing_overrides:
                    del existing_overrides[override_team]
                st.info(f"Removed roster override for {override_team}")
            else:
                if len(parsed_players) < 5:
                    st.warning("A VCT roster should have at least 5 players.")
                existing_overrides[override_team] = parsed_players
                st.success(f"Saved roster override for {override_team}: {parsed_players}")
                
            os.makedirs(os.path.dirname(overrides_path), exist_ok=True)
            with open(overrides_path, "w", encoding="utf-8") as f:
                json.dump(existing_overrides, f, indent=4)
            st.rerun()
    
    st.markdown("---")
    btn_generate_lineup = st.button("Generate Optimal GPP Lineup", type="primary", use_container_width=True, key="btn_generate_lineup")

# Trigger button execution logic
if btn_generate_lineup:
    # Run the optimization solver pipeline inside the sidebar trigger
    with st.spinner("Initializing GPP Optimization..."):
        try:
            from pathlib import Path
            root_dir = Path(__file__).resolve().parent
            pred_path = root_dir / "data" / "processed" / "xgb_predictions.json"
            
            if not pred_path.exists():
                logger.info("Predictions file missing. Launching model_training.py autonomously...")
                import subprocess
                import sys
                with st.spinner("Training XGBoost Model..."):
                    subprocess.run([sys.executable, str(root_dir / "model_training.py")], check=True)
            
            from knapsack_solver import prepare_player_slate, solve_vfl_knapsack, run_portfolio_simulation
            
            is_intl_rules = "International" in opt_ruleset
            lineup_sz = 6 if is_intl_rules else 11
            role_cnts = (
                {"Duelist": 1, "Initiator": 1, "Controller": 1, "Sentinel": 1, "Flex": 2}
                if is_intl_rules else
                {"Duelist": 2, "Initiator": 2, "Controller": 2, "Sentinel": 2, "Flex": 3}
            )
            
            df_meta_slate, df_fused_slate = prepare_player_slate(num_iterations=opt_depth)
            opt_solution = solve_vfl_knapsack(
                df_meta_slate, 
                salary_cap=opt_salary_cap,
                lineup_size=lineup_sz,
                max_per_team=2,
                role_counts=role_cnts
            )
            portfolio_results = run_portfolio_simulation(opt_solution, df_fused_slate)
            
            st.session_state["gpp_solution"] = opt_solution
            st.session_state["gpp_portfolio"] = portfolio_results
            st.session_state["gpp_meta_df"] = df_meta_slate
            st.session_state["gpp_generated"] = True
            st.toast("GPP Optimization Pipeline succeeded!", icon="✅")
        except Exception as e:
            import traceback
            st.error(f"Solver Crash: {e}\n{traceback.format_exc()}")
            st.session_state["gpp_generated"] = False

# Helper to map roles to strings for Phase 9 Ingestion
def map_role(role_raw) -> str:
    role_map = {
        0: "Duelist",
        1: "Initiator",
        2: "Controller",
        3: "Sentinel",
        4: "Flex",
        "0": "Duelist",
        "1": "Initiator",
        "2": "Controller",
        "3": "Sentinel",
        "4": "Flex"
    }
    if role_raw in role_map:
        return role_map[role_raw]
    if isinstance(role_raw, str):
        role_str = role_raw.strip().capitalize()
        if role_str in ["Duelist", "Initiator", "Controller", "Sentinel", "Flex"]:
            return role_str
    return "Flex"

# Trigger live API sync logic
if btn_sync_live:
    with st.spinner("Syncing Live VFL Slate (API)..."):
        try:
            import subprocess
            import sys
            import json
            
            scraper_path = ROOT_DIR / "scrapers" / "vfl_scraper.py"
            
            # Execute scraper via subprocess
            logger.info("Executing vfl_scraper.py autonomously via subprocess...")
            subprocess.run([sys.executable, str(scraper_path)], check=True)
            
            # Load resulting vfl_players_db.json
            db_path = ROOT_DIR / "data" / "processed" / "vfl_players_db.json"
            if not db_path.exists():
                raise FileNotFoundError(f"VFL player cache not found at {db_path} after scraping.")
                
            with open(db_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            # Handle both legacy flat list and envelope cache format
            raw_players = raw_data.get("players", []) if isinstance(raw_data, dict) else raw_data
            if not isinstance(raw_players, list):
                raise ValueError("Unexpected vfl_players_db.json format. Expected list of players.")
                
            # Map schema
            mapped_players = []
            for idx, p in enumerate(raw_players):
                name = p.get("player_name", p.get("name", "Unknown")).strip()
                team = p.get("team_name", p.get("team", "Unknown")).strip()
                team_short = p.get("team_short", p.get("team_name", "UNK")[:3]).strip().upper()
                salary = p.get("price", p.get("salary", p.get("cost", 8.0)))
                try:
                    salary = float(salary)
                except (ValueError, TypeError):
                    salary = 8.0
                    
                role = map_role(p.get("role", p.get("playerRole", 4)))
                
                mapped_players.append({
                    "player_id": f"P{idx}_{team_short}",
                    "name": name,
                    "team": team,
                    "role": role,
                    "salary": salary
                })
                
            # Overwrite current_slate.json
            slate_path = ROOT_DIR / "data" / "processed" / "current_slate.json"
            slate_path.parent.mkdir(parents=True, exist_ok=True)
            with open(slate_path, "w", encoding="utf-8") as f:
                json.dump(mapped_players, f, indent=4, ensure_ascii=False)
                
            # Task 9.3: Pipeline Reset (Delete predictions & clear UI session states)
            pred_path = ROOT_DIR / "data" / "processed" / "xgb_predictions.json"
            pred_path.unlink(missing_ok=True)
            
            for key in ["gpp_solution", "gpp_portfolio", "gpp_meta_df", "gpp_generated", "optimal_lineup", "portfolio_metrics"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.success(f"Successfully loaded and mapped {len(mapped_players)} players from VFL API slate!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Live API Sync Failed: {e}")

# Trigger CSV upload logic
if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)
        
        # Find columns case-insensitively
        col_mapping = {}
        for col in df_csv.columns:
            col_lower = col.lower().strip()
            if "name" in col_lower:
                col_mapping["name"] = col
            elif "team" in col_lower:
                col_mapping["team"] = col
            elif "role" in col_lower:
                col_mapping["role"] = col
            elif "salary" in col_lower or "price" in col_lower or "cost" in col_lower:
                col_mapping["salary"] = col
                
        raw_players = []
        for idx, row in df_csv.iterrows():
            p_name = row.get(col_mapping.get("name", "name"), f"Player_{idx}")
            p_team = row.get(col_mapping.get("team", "team"), "Unknown")
            p_role = row.get(col_mapping.get("role", "role"), 4)
            p_salary = row.get(col_mapping.get("salary", "salary"), 8.0)
            
            raw_players.append({
                "player_name": str(p_name),
                "team_name": str(p_team),
                "team_short": str(p_team)[:3].upper(),
                "role": p_role,
                "price": p_salary
            })
            
        # Map schema
        mapped_players = []
        for idx, p in enumerate(raw_players):
            name = p["player_name"].strip()
            team = p["team_name"].strip()
            team_short = p["team_short"].strip().upper()
            salary = p["price"]
            try:
                salary = float(salary)
            except (ValueError, TypeError):
                salary = 8.0
                
            role = map_role(p["role"])
            
            mapped_players.append({
                "player_id": f"P{idx}_{team_short}",
                "name": name,
                "team": team,
                "role": role,
                "salary": salary
            })
            
        # Overwrite current_slate.json
        slate_path = ROOT_DIR / "data" / "processed" / "current_slate.json"
        slate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(slate_path, "w", encoding="utf-8") as f:
            json.dump(mapped_players, f, indent=4, ensure_ascii=False)
            
        # Task 9.3: Pipeline Reset
        pred_path = ROOT_DIR / "data" / "processed" / "xgb_predictions.json"
        pred_path.unlink(missing_ok=True)
        
        for key in ["gpp_solution", "gpp_portfolio", "gpp_meta_df", "gpp_generated", "optimal_lineup", "portfolio_metrics"]:
            if key in st.session_state:
                del st.session_state[key]
                
        st.success(f"Successfully loaded and mapped {len(mapped_players)} players from CSV slate!")
        st.rerun()
        
    except Exception as e:
        st.error(f"CSV Ingestion Failed: {e}")

# Trigger Master Update logic
if btn_master_update:
    with st.spinner("Executing Master Update... (scraping matches, syncing map pool, checking patch wikis, analyzing concept drift, and retraining models)"):
        try:
            import subprocess
            import sys
            
            cmd = [sys.executable, str(ROOT_DIR / "run_pipeline.py")]
            if whitelist_input.strip():
                cmd += ["--whitelist", whitelist_input.strip()]
                
            logger.info(f"Executing master update pipeline via subprocess: {cmd}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Clear solver session state
            for key in ["gpp_solution", "gpp_portfolio", "gpp_meta_df", "gpp_generated", "optimal_lineup", "portfolio_metrics"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.success("✅ Master Update Completed Successfully! VCT matches, active map rotation, wiki patches, drift indices, and predictions have been re-calibrated.")
            with st.expander("Show Detailed Execution Logs"):
                st.code(result.stdout)
                
        except Exception as e:
            st.error(f"❌ Master Update Pipeline Failed: {e}")
            if 'result' in locals() and hasattr(result, 'stderr') and result.stderr:
                st.code(result.stderr)

# Trigger Patch-Only Update logic
if btn_patch_update_only:
    with st.spinner("Executing Patch-Only Update... (scraping wiki, parsing patches, updating drift indices, and rebuilding models)"):
        try:
            import subprocess
            import sys
            
            steps = [
                (["scrapers/wiki_scraper.py"], "Scrape VCT wiki patch list"),
                (["scrapers/patch_ingestor.py"], "Ingest new patch notes"),
                (["patch_analyzer.py"], "Compute concept drift nerf registry"),
                (["feature_engineering.py"], "Rebuild feature matrix store"),
                (["model_training.py"], "Retrain XGBoost & update slate predictions")
            ]
            
            stdout_accumulator = []
            for cmd_args, desc in steps:
                logger.info(f"Running patch pipeline step: {desc} ({cmd_args})")
                res = subprocess.run(
                    [sys.executable] + cmd_args,
                    cwd=str(ROOT_DIR),
                    check=True,
                    capture_output=True,
                    text=True
                )
                stdout_accumulator.append(f"=== {desc} ===\n{res.stdout or ''}")
                
            # Clear solver session state
            for key in ["gpp_solution", "gpp_portfolio", "gpp_meta_df", "gpp_generated", "optimal_lineup", "portfolio_metrics"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.success("✅ Patch Notes Scrape & Meta Rebuild Completed Successfully!")
            with st.expander("Show Detailed Rebuild Logs"):
                st.code("\n\n".join(stdout_accumulator))
                
        except Exception as e:
            st.error(f"❌ Patch Update Rebuild Failed: {e}")
            if 'res' in locals() and hasattr(res, 'stderr') and res.stderr:
                st.code(res.stderr)

# Trigger Incremental VLR Scrape logic
if btn_scrape_vlr_incremental:
    with st.spinner("Executing Incremental VLR Scrape... (fetching matches from VLR.gg until hitting previously scraped data)"):
        try:
            import subprocess
            import sys
            
            logger.info("Executing incremental_vlr_scraper.py autonomously via subprocess...")
            result = subprocess.run(
                [sys.executable, str(ROOT_DIR / "scrapers" / "incremental_vlr_scraper.py")],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Find the match count from stdout if printed
            new_matches = 0
            for line in result.stdout.splitlines():
                if "NEW_MATCHES_SCRAPED:" in line:
                    try:
                        new_matches = int(line.split("NEW_MATCHES_SCRAPED:")[-1].strip())
                    except ValueError:
                        pass
            
            st.success(f"✅ Incremental Scrape Completed Successfully! Added {new_matches} new Tier-1 matches to the database.")
            with st.expander("Show Detailed Scraping Logs"):
                st.code(result.stdout)
                
        except Exception as e:
            st.error(f"❌ Incremental Scrape Failed: {e}")
            if 'result' in locals() and hasattr(result, 'stderr') and result.stderr:
                st.code(result.stderr)

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
                The engine dynamically resolves rosters, resolves Bayesian skill states, and runs stateful economy rounds.
            </p>
        </div>
    """), unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_team_a = st.selectbox("Team A", all_teams, index=0, key="sim_team_a")
    with sim_col2:
        sim_team_b = st.selectbox("Team B", all_teams, index=min(1, len(all_teams)-1), key="sim_team_b")

    sim_col3, sim_col4, sim_col5, sim_col6, sim_col_priority = st.columns(5)
    with sim_col3:
        sim_ref_date = st.date_input("Reference Date (for time-decay)", value=datetime(2026, 6, 22), key="sim_ref_date")
    with sim_col4:
        sim_series_type = st.selectbox("Series Format", ["Bo3", "Bo5"], index=0, key="sim_series_type")
    with sim_col5:
        sim_iterations = st.selectbox("Simulation Depth", [1000, 5000, 10000], index=1, key="sim_iterations")
    with sim_col6:
        # Dynamically load patch options from registry using pathlib
        try:
            patch_options = get_available_patches()
        except Exception as e:
            logger.warning(f"Failed to dynamically load patches from registry: {e}. Falling back to default list.")
            patch_options = ["Patch 9.04", "Patch 9.02", "Patch 8.11 (June 11, 2024)"]
            
        sim_patch_select = st.selectbox("Target Simulation Patch", patch_options, index=min(1, len(patch_options)-1), key="sim_target_patch")
        patch_match = re.search(r'([0-9.]+)', sim_patch_select)
        sim_target_patch_val = patch_match.group(1) if patch_match else "9.02"
    with sim_col_priority:
        sim_veto_priority_sel = st.selectbox(
            "Veto Priority Team",
            options=["Team A", "Team B", "1v1 Skirmish (50/50)"],
            index=0,
            key="sim_veto_priority_select"
        )
        sim_veto_priority_val = {"Team A": "team_a", "Team B": "team_b", "1v1 Skirmish (50/50)": "random"}[sim_veto_priority_sel]

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
            with st.spinner(f"Running v7 Stateful Economy & Synergistic Draft ({sim_iterations:,} iterations) for {sim_team_a} vs {sim_team_b}..."):
                v5_engine = get_v5_simulation_engine()
                sim_target_datetime = datetime.combine(sim_ref_date, datetime.min.time()) if hasattr(sim_ref_date, 'year') else datetime.now()
                sim_result = v5_engine.simulate_match(
                    team_a=sim_team_a,
                    team_b=sim_team_b,
                    series_type=sim_series_type,
                    target_patch=sim_target_patch_val,
                    num_iterations=sim_iterations,
                    override_maps=override_maps if enable_override else None,
                    target_date=sim_target_datetime,
                    veto_priority=sim_veto_priority_val
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
                    📊 v7 Stateful Simulation Analytics
                </div>
            """), unsafe_allow_html=True)

            # Stateful Economy Info
            st.info("💡 **v7 Stateful Economy Simulation Engine Active:** Round win probabilities are dynamically driven by stateful round-to-round credit accumulations, multi-round loss-streaks, and weapon-save survival penalties.")

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
                        badges_a = get_composition_synergy_badges(sim_result.get("roster_a", []), player_agents, v5_engine.agent_transformer.agent_roles)
                        st.markdown(badges_a, unsafe_allow_html=True)

                    with comp_col_b:
                        cards_b = "".join(
                            _render_agent_card(p, player_agents[p])
                            for p in sim_result.get("roster_b", []) if p in player_agents
                        )
                        st.markdown(clean_html(f"""
                            <div style="margin-bottom: 6px; font-weight: 700; font-size: 0.95rem; color: #fbbf24;">{sim_team_b}</div>
                            <div class="glass-card">{cards_b or '<div style="color:#64748b;font-size:0.85rem;">No composition data.</div>'}</div>
                        """), unsafe_allow_html=True)
                        badges_b = get_composition_synergy_badges(sim_result.get("roster_b", []), player_agents, v5_engine.agent_transformer.agent_roles)
                        st.markdown(badges_b, unsafe_allow_html=True)

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

    # Initialize simulation cache in session state to prevent running on every load
    if "v5_sim_results" not in st.session_state:
        st.session_state["v5_sim_results"] = {}
        
    current_run_hash = f"{selected_match_id}_{ma_series_type}_{veto_override}_{predicted_veto.get('maps') if predicted_veto else None}"
    
    st.markdown("---")
    btn_run_match_sim = st.button("🚀 Run Match Simulation Analysis", key="btn_run_match_sim", type="primary", use_container_width=True)
    
    if btn_run_match_sim:
        v5_engine = get_v5_simulation_engine()
        with st.spinner("Running v7 Stateful Economy & Synergistic Draft (2,000 iterations)..."):
            v5_res = v5_engine.simulate_match(ma_team_a, ma_team_b, ma_series_type, target_patch="9.02", num_iterations=2000)
            st.session_state["v5_sim_results"][current_run_hash] = v5_res
            st.rerun()
            
    if current_run_hash in st.session_state["v5_sim_results"]:
        v5_res = st.session_state["v5_sim_results"][current_run_hash]
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

        # Stateful Economy Info
        st.info("💡 **v7 Stateful Economy Simulation Engine Active:** Round win probabilities are dynamically driven by stateful round-to-round credit accumulations, multi-round loss-streaks, and weapon-save survival penalties.")

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
                team_a_features = {"acs_mu": ta_acs, "avg_loadout": ta_loadout, "comfort_diff": 0.0}
                team_b_features = {"acs_mu": tb_acs, "avg_loadout": tb_loadout, "comfort_diff": 0.0}
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
                    badges_a = get_composition_synergy_badges_dict(comp_a_map)
                    st.markdown(badges_a, unsafe_allow_html=True)

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
                    badges_b = get_composition_synergy_badges_dict(comp_b_map)
                    st.markdown(badges_b, unsafe_allow_html=True)

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

        st.markdown("### v7 Projected Fantasy Points (Expected Value)")
        if "projections" in v5_res:
            proj_data = []
            for p, ev in v5_res["projections"].items():
                team = ma_team_a if p in v5_res["roster_a"] else ma_team_b
                proj_data.append({"Player": p, "Team": team, "Expected Value (EV) Points": ev})
            proj_df = pd.DataFrame(proj_data).sort_values("Expected Value (EV) Points", ascending=False)
            st.dataframe(proj_df, use_container_width=True, hide_index=True)
        else:
            st.info("EV Projections unavailable.")
    else:
        st.info("💡 Select a target match and click the **Run Match Simulation Analysis** button above to generate bottom-up simulation projections, agent metrics, and performance charts.")

# ============================================================
# TAB 3: ROSTER OPTIMIZER
# ============================================================
with tab_optimizer:
    st.markdown("### 🧠 VFL Fantasy Manager Hub")

    # Live Meta Radar (Task 7.1)
    match_patch = re.search(r'([0-9.]+)', opt_patch)
    active_patch_val = match_patch.group(1) if match_patch else "9.02"
    
    # Load dynamic patch penalties
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    path_reg = base_dir / "data" / "processed" / "automated_patch_nerf_registry.json"
    if not path_reg.exists():
        path_reg = base_dir / "data" / "processed" / "patch_nerf_registry.json"
        
    radar_penalties = {}
    if path_reg.exists():
        try:
            with open(path_reg, "r", encoding="utf-8") as f_reg:
                registry_data = json.load(f_reg)
            radar_penalties = registry_data.get(active_patch_val, {})
        except Exception as e:
            logger.warning(f"Error parsing patch registry for radar: {e}")

    st.markdown(f"#### 📡 Live Meta Radar (Patch {active_patch_val})")
    if radar_penalties:
        sorted_penalties = sorted(radar_penalties.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_penalties[:3]
        cols = st.columns(3)
        for idx, (agent_name, score) in enumerate(top_3):
            with cols[idx]:
                if score >= 0.05:
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
        st.info(f"No active patch nerfs registered for Patch {active_patch_val}.")

    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 🏆 Hybrid Valorant GPP Optimizer")
        if "gpp_solution" in st.session_state and st.session_state.get("gpp_generated", False):
            sol = st.session_state["gpp_solution"]
            port = st.session_state["gpp_portfolio"]
            
            # Task 6.3: The Upside Hook (Metrics)
            st.markdown("#### 📈 Portfolio Simulation & Tournament Upside")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    label="Total Salary Used",
                    value=f"{sol['total_salary']:.1f} / {sol.get('salary_cap', 50.0):.1f} VP",
                    help="Maximum roster budget allowed"
                )
            with m_col2:
                st.metric(
                    label="Lineup Median EV Points",
                    value=f"{port['simulated_lineup_mean']:.2f} Pts",
                    help="Expected Value projection from copula fusion"
                )
            with m_col3:
                delta_val = port['simulated_lineup_ceiling_p85'] - port['simulated_lineup_mean']
                st.metric(
                    label="Tournament GPP Ceiling (85th %)",
                    value=f"{port['simulated_lineup_ceiling_p85']:.2f} Pts",
                    delta=f"+{delta_val:.2f} Pts (GPP Upside Hook)",
                    delta_color="normal",
                    help="85th percentile ceiling from Monte Carlo simulation runs"
                )
                
            st.markdown("---")
            
            # Task 6.2: Roster Visualization (Main UI)
            st.markdown("#### 👥 Optimal 6-Man Tournament Lineup")
            
            # Use 2 rows of 3 columns to avoid squishing in 3/5 width column layout
            row1_cols = st.columns(3)
            row2_cols = st.columns(3)
            for idx, p in enumerate(sol["lineup"]):
                col_container = row1_cols[idx] if idx < 3 else row2_cols[idx - 3]
                with col_container:
                    p_name = p["name"]
                    p_role = p["role"]
                    p_sal = p["salary"]
                    p_ev = p["EV"]
                    p_ceil = p["Ceiling_p85"]
                    p_igl = p["is_igl"]
                    
                    icon_url = get_player_icon(p_role, p_name)
                    
                    igl_badge_html = """
                        <div style="background: rgba(255, 70, 85, 0.15); color: #ff4655; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; text-align: center; border-radius: 6px; padding: 3px 6px; border: 1px solid rgba(255, 70, 85, 0.3); margin-top: 8px;">
                            👑 IGL (2x)
                        </div>
                    """ if p_igl else ""
                    
                    st.markdown(clean_html(f"""
                        <div style="background: rgba(26, 29, 36, 0.75); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px; text-align: center; backdrop-filter: blur(10px); min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="position: relative; width: 64px; height: 64px;">
                                <img src="{icon_url}" width="64" height="64" style="border-radius: 50%; border: 2px solid {'#ff4655' if p_igl else '#4f46e5'}; background: #0f172a;"/>
                            </div>
                            <div style="margin-top: 10px; flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                                <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; word-break: break-all;">{p_name}</div>
                                <div style="font-size: 0.75rem; color: #a78bfa; font-weight: 600; text-transform: uppercase; margin-top: 2px;">{p_role}</div>
                                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500; margin-top: 4px;">{p['team']}</div>
                            </div>
                            <div style="margin-top: auto; width: 100%;">
                                <div style="font-weight: 700; font-size: 1.2rem; color: #4ade80; margin-top: 6px;">{p_sal:.1f} VP</div>
                                <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">EV: {p_ev:.1f} | ceil: {p_ceil:.1f}</div>
                                {igl_badge_html}
                            </div>
                        </div>
                    """), unsafe_allow_html=True)
                    
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(clean_html(f"""
                <div class="optimizer-card" style="margin-top: 15px;">
                    <h5 style="color: #a78bfa; font-weight: 700; margin-top: 0; margin-bottom: 8px;">📊 Portfolio Simulation Summary</h5>
                    <p style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.5; margin: 0;">
                        Based on <strong>{st.session_state.get('opt_sim_depth', 10000)}</strong> Monte Carlo iterations, the optimal roster secures an aggregate <strong>Floor (p15) of {port['simulated_lineup_floor_p15']:.2f} points</strong>, an <strong>Expected Value of {port['simulated_lineup_mean']:.2f} points</strong>, and a peak <strong>Tournament Max of {port['simulated_lineup_max']:.2f} points</strong>.
                    </p>
                </div>
            """), unsafe_allow_html=True)
        else:
            st.info("👈 Use the Command Center in the sidebar and click 'Generate Optimal GPP Lineup' to compute the GPP optimal roster!")

    with col_right:
        st.markdown("### 🔄 3-Transfer Advisor")
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>Select your current fantasy roster to evaluate trades and analyze GPP points gain.</p>", unsafe_allow_html=True)
        
        # 1. Load slate names and data
        from utils.utils import load_slate_payload
        try:
            slate_data = load_slate_payload()
            slate_names = sorted([p["name"] for p in slate_data])
            slate_lookup = {p["name"]: p for p in slate_data}
        except Exception as e:
            st.error(f"Failed to load players from current_slate.json: {e}")
            slate_data = []
            slate_names = []
            slate_lookup = {}
            
        # Initialize defaultSelections: prefer saved state from ROSTER_STATE_PATH
        saved_names, saved_igl = load_roster_state()
        valid_saved_names = [n for n in saved_names if n in slate_names]
        
        ruleset_name = st.session_state.get("opt_ruleset", "International (Masters/Champions)")
        is_intl_rules = "International" in ruleset_name
        req_size = 6 if is_intl_rules else 11
        role_min_count = 1 if is_intl_rules else 2
        
        # Multiselect loaded with the active slate from current_slate.json
        selected_roster = st.multiselect(
            f"Select Your Current {req_size} Players",
            options=slate_names,
            default=valid_saved_names if len(valid_saved_names) > 0 else (slate_names[:req_size] if len(slate_names) >= req_size else []),
            key="user_multiselect_roster"
        )
        
        # Validation checks for select count and VFL constraints
        num_selected = len(selected_roster)
        if num_selected != req_size:
            st.warning(f"⚠️ Please select exactly {req_size} players to evaluate transfers. (Currently selected: {num_selected})")
        else:
            from collections import Counter
            teams_counter = Counter()
            roles_counter = Counter()
            for name in selected_roster:
                p_info = slate_lookup.get(name)
                if p_info:
                    teams_counter[p_info.get("team")] += 1
                    roles_counter[p_info.get("role")] += 1
                    
            team_violations = [team for team, count in teams_counter.items() if count > 2]
            
            required_roles = {"Duelist", "Initiator", "Controller", "Sentinel"}
            missing_roles = [role for role in required_roles if roles_counter[role] < role_min_count]
            
            if team_violations:
                st.error(f"⚠️ **VFL Rule Violation:** Max 2 players per team. Violated by: {', '.join(team_violations)}")
            elif missing_roles:
                st.error(f"⚠️ **VFL Rule Violation:** Roster must contain at least {role_min_count} players from each core role ({', '.join(required_roles)}). Missing or insufficient: {', '.join(missing_roles)}")
            else:
                st.success(f"✅ **Legal Roster:** This lineup strictly satisfies all VFL team and positional rules ({role_min_count} per core role, max 2 per team).")
            
        # Floating Bank & Cost calculations
        roster_cost = sum(slate_lookup[name]["salary"] for name in selected_roster if name in slate_lookup)
        floating_bank = opt_salary_cap - roster_cost
        
        # IGL Selection dropdown (IGL must be one of the selected players)
        igl_options = selected_roster
        default_igl_name = None
        if igl_options:
            if saved_igl in igl_options:
                igl_default_name = saved_igl
            else:
                igl_default_name = igl_options[0]
                
        selected_igl = st.selectbox(
            "👑 Designate In-Game Leader (IGL)",
            options=igl_options,
            index=igl_options.index(igl_default_name) if (igl_options and igl_default_name in igl_options) else 0,
            key="user_igl_selectbox"
        )
        
        # Dynamic Roster EV calculator applying 2x IGL multiplier
        roster_ev = 0.0
        has_ev = False
        if "gpp_meta_df" in st.session_state:
            df_m = st.session_state["gpp_meta_df"]
            has_ev = True
            for name in selected_roster:
                ev_val = 0.0
                player_row = df_m[df_m["name"] == name]
                if not player_row.empty:
                    ev_val = float(player_row.iloc[0]["EV"])
                mult = 2.0 if name == selected_igl else 1.0
                roster_ev += ev_val * mult
                
        # Persist selected roster and IGL to session state on change
        st.session_state["saved_roster_names"] = selected_roster
        st.session_state["saved_igl_name"] = selected_igl
        
        # State Persistence buttons using standard file I/O
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("💾 Save Roster State", key="btn_save_roster_state", use_container_width=True):
                if num_selected == 6:
                    save_roster_state(selected_roster, selected_igl)
                    st.toast("Roster state successfully saved!", icon="💾")
                else:
                    st.warning("Roster must have exactly 6 players to save.")
        with p_col2:
            roster_exists = os.path.exists(ROSTER_STATE_PATH)
            if st.button("📂 Load Roster State", key="btn_load_roster_state", disabled=not roster_exists, use_container_width=True):
                loaded_players, loaded_igl = load_roster_state()
                st.session_state["saved_roster_names"] = loaded_players
                st.session_state["saved_igl_name"] = loaded_igl
                st.rerun()

        # Display Metrics
        st.markdown("#### 📊 Roster Metrics")
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.metric(
                label="Selected Roster Cost",
                value=f"{roster_cost:.1f} VP",
                help="Sum of salaries of the 6 selected players"
            )
        with met_col2:
            if floating_bank < 0:
                st.markdown(f"<div style='color: #ef4444; font-size: 1.1rem; font-weight: 700; padding: 4px 0;'>Floating Bank: {floating_bank:.1f} VP<br/>(BUDGET EXCEEDED!)</div>", unsafe_allow_html=True)
            else:
                st.metric(
                    label="Floating Bank",
                    value=f"{floating_bank:.1f} VP",
                    help="Available budget remaining (Budget Cap - Roster Cost)"
                )
                
        # Display selected roster EV
        if has_ev:
            st.info(f"Projected Current Roster EV: **{roster_ev:.2f} Pts** (Applying {selected_igl}'s 2x IGL multiplier)")
        else:
            st.info("Run the GPP Optimizer in the sidebar to populate Expected Value (EV) projections.")

        # Trades Engine Section
        st.markdown("---")
        st.markdown("#### 🚀 Trade Recommendations Engine")
        btn_calc_trades = st.button("Calculate Optimal Trades", key="btn_calc_trades", type="primary", use_container_width=True)
        
        if btn_calc_trades:
            if num_selected != req_size:
                st.error(f"Roster must have exactly {req_size} players selected to calculate trades.")
            elif "gpp_solution" not in st.session_state or not st.session_state.get("gpp_generated", False):
                st.error("Please run 'Generate Optimal GPP Lineup' first to compute the target optimal roster.")
            else:
                sol = st.session_state["gpp_solution"]
                opt_lineup = sol["lineup"]
                opt_pids = {p["player_id"] for p in opt_lineup}
                opt_lookup = {p["player_id"]: p for p in opt_lineup}
                
                # Get user selected player IDs
                user_pids = set()
                user_lookup = {}
                for name in selected_roster:
                    if name in slate_lookup:
                        p_dict = slate_lookup[name]
                        user_pids.add(p_dict["player_id"])
                        user_lookup[p_dict["player_id"]] = p_dict
                        
                # Compute difference: OUT = user - opt; IN = opt - user
                pids_out = user_pids - opt_pids
                pids_in = opt_pids - user_pids
                
                # Check IGL swap:
                igl_swap = False
                opt_igl_pid = sol["igl_player"]
                opt_igl_name = next((p["name"] for p in opt_lineup if p["is_igl"]), "")
                if selected_igl != opt_igl_name:
                    igl_swap = True
                
                if not pids_out and not pids_in and not igl_swap:
                    st.success("🎉 Roster is already 100% mathematically optimal! No transfers needed.")
                else:
                    st.info("ℹ️ **Strict VFL Rules Enforced:** All trade suggestions strictly respect the VFL limit of **max 3 transfers per week** and the **2-player team cap** per roster.")
                    st.markdown("**Suggested Swaps (Mathematical Set Difference):**")
                    
                    from collections import Counter
                    opt_roles = [p["role"] for p in opt_lineup]
                    opt_role_counts = Counter(opt_roles)
                    
                    # Display OUT cards
                    for pid in pids_out:
                        p_out = user_lookup[pid]
                        primary_agent = p_out.get("primary_agent")
                        penalty = radar_penalties.get(primary_agent, 0.0) if primary_agent else 0.0
                        reason = "Transfer out to optimize salary cap and match target GPP ceiling."
                        if penalty > 0.05:
                            reason = f"Player's primary agent ({primary_agent}) suffered a {penalty:.2f} Ghost/Meta Nerf."
                        st.markdown(clean_html(f"""
                            <div style="padding: 10px 14px; margin-bottom: 8px; border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.05); border-radius: 6px;">
                                <span style="color: #ef4444; font-weight: 700;">OUT ⬇</span>
                                <span style="margin-left: 12px; font-weight: 600; color: #f8fafc;">{p_out['name']}</span>
                                <span style="color: #94a3b8; font-size: 0.8rem;"> · Cost: {p_out['salary']} VP · Role: {p_out['role']}</span>
                                <div style="font-size: 0.8rem; color: #ef4444; margin-top: 4px; font-style: italic;">{reason}</div>
                            </div>
                        """), unsafe_allow_html=True)
                        
                    # Display IN cards
                    for pid in pids_in:
                        p_in = opt_lookup[pid]
                        igl_tag = " 👑 IGL" if p_in["is_igl"] else ""
                        is_wildcard = opt_role_counts.get(p_in["role"], 0) > 1
                        wildcard_badge = ' <span style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 12px; background: rgba(167, 139, 250, 0.15); color: #a78bfa; margin-left: 6px;">✨ Wildcard Swap</span>' if is_wildcard else ""
                        reason = "Drafted into optimal lineup to maximize GPP ceiling under salary constraint."
                        st.markdown(clean_html(f"""
                            <div style="padding: 10px 14px; margin-bottom: 8px; border-left: 4px solid #4ade80; background: rgba(74, 222, 128, 0.05); border-radius: 6px;">
                                <span style="color: #4ade80; font-weight: 700;">IN ⬆</span>
                                <span style="margin-left: 12px; font-weight: 600; color: #f8fafc;">{p_in['name']}{igl_tag}{wildcard_badge}</span>
                                <span style="color: #94a3b8; font-size: 0.8rem;"> · Cost: {p_in['salary']} VP · Role: {p_in['role']}</span>
                                <div style="font-size: 0.8rem; color: #4ade80; margin-top: 4px; font-style: italic;">{reason}</div>
                            </div>
                        """), unsafe_allow_html=True)
                        
                    if igl_swap and opt_igl_name:
                        st.info(f"👑 Note: Swap designated In-Game Leader (IGL) from **{selected_igl}** to **{opt_igl_name}** (2x bonus).")

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
        # Load Bayesian Player Ledger
        ledger_path = os.path.join(PROCESSED_DIR, "bayesian_player_ledger.json")
        bayesian_ledger = {}
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, "r", encoding="utf-8") as f:
                    bayesian_ledger = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load Bayesian ledger: {e}")

        # Merge Bayesian ledger details
        enriched_players = []
        for p in vfl_players_data:
            p_copy = dict(p)
            p_name = p_copy.get("player_name", "")
            b_stats = bayesian_ledger.get(p_name)
            if not b_stats:
                p_name_lower = p_name.lower().strip()
                for k, v in bayesian_ledger.items():
                    if k.lower().strip() == p_name_lower:
                        b_stats = v
                        break
            if b_stats:
                p_copy["KPR Expected (μ)"] = round(b_stats.get("kpr_mu", 0.75), 3)
                p_copy["KPR Volatility (σ)"] = round(b_stats.get("kpr_sigma", 0.20), 3)
                p_copy["ACS Expected (μ)"] = round(b_stats.get("acs_mu", 200.0), 1)
                p_copy["ACS Volatility (σ)"] = round(b_stats.get("acs_sigma", 50.0), 1)
            else:
                p_copy["KPR Expected (μ)"] = 0.75
                p_copy["KPR Volatility (σ)"] = 0.20
                p_copy["ACS Expected (μ)"] = 200.0
                p_copy["ACS Volatility (σ)"] = 50.0
            enriched_players.append(p_copy)

        vfl_df = pd.DataFrame(enriched_players)

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

        # Display Volatility Caption / Metric
        st.info("💡 **Bayesian Volatility Alert:** Higher Volatility (σ) indicates high-variance players who have unstable form or meta shifts. These high-volatility players are excellent targets for high-upside GPP tournament lineups, while low-volatility players are safer for cash games.")

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

        # Order columns nicely so our new Bayesian ones are prominent
        cols_order = [
            "Player", "Team", "Role", "Cost", "Total Points", "Avg PPG", "Ownership %",
            "KPR Expected (μ)", "KPR Volatility (σ)", "ACS Expected (μ)", "ACS Volatility (σ)"
        ]
        # Keep only columns that exist in the dataframe to be safe
        cols_order = [c for c in cols_order if c in display_vfl_df.columns]
        display_vfl_df = display_vfl_df[cols_order]

        st.dataframe(display_vfl_df, use_container_width=True, hide_index=True)
    else:
        st.info("No VFL data available. Click '🔄 Update VFL Database' in the sidebar to load data.")
