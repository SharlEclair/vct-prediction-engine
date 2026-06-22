import os
import json
import glob
import logging
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from datetime import datetime

# Import local modules
from veto_predictor import VCTMapVetoPredictor
from generative_pipeline import MapScoreRegressor, AgentCompositionGenerator
from fantasy_engine import VCTFantasyEngine, optimize_roster, suggest_transfers, generate_stage_2_baseline, get_team_win_rates_by_id
from predict_match import get_historical_stats, get_latest_roster, simulate_arbitrary_match
from vfl_scraper import VFLScraper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

# Load team ID mapping
try:
    _, team_name_to_id = get_team_win_rates_by_id(RAW_DIR)
    id_to_team_name = {v: k for k, v in team_name_to_id.items()}
except Exception:
    id_to_team_name = {}

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
        # Sort versions properly (e.g. 9.02, 10.00, etc.)
        sorted_patches = sorted(list(data.keys()), key=lambda x: [int(i) if i.isdigit() else i for i in x.split('.')])
        if not sorted_patches:
            return "None", {}
        latest_patch = sorted_patches[-1]
        return latest_patch, data[latest_patch]
    except Exception as e:
        logger.error(f"Failed to load automated patch registry: {e}")
        return "None", {}

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

# Cache historical statistics to speed up dashboard loading
@st.cache_resource
def load_cached_historical_data():
    return get_historical_stats(RAW_DIR)

# Agent Icons mapping
AGENT_ICONS = {
    "Jett": "https://media.valorant-api.com/agents/add6443a-41da-e1c3-d774-4598a6c7e2ca/displayicon.png",
    "Raze": "https://media.valorant-api.com/agents/f7aeec33-4615-a78d-6b58-5e35024b3e30/displayicon.png",
    "Breach": "https://media.valorant-api.com/agents/5f8d3a7f-467b-97f3-062c-13db7f307e32/displayicon.png",
    "Omen": "https://media.valorant-api.com/agents/8e2535ab-4c40-07a4-4b1a-7b3b7e329840/displayicon.png",
    "Brimstone": "https://media.valorant-api.com/agents/9f0f7139-4456-04ce-774b-14bc73a5d641/displayicon.png",
    "Phoenix": "https://media.valorant-api.com/agents/117ed9e3-49f3-6512-3ccf-00a68e7d0ace/displayicon.png",
    "Sage": "https://media.valorant-api.com/agents/569fdd95-4d10-43ab-ca70-79becc718b46/displayicon.png",
    "Sova": "https://media.valorant-api.com/agents/ded3520f-4064-ae4b-b6cd-db07307a5d90/displayicon.png",
    "Viper": "https://media.valorant-api.com/agents/707e2116-4f48-434a-2052-ed87db41663f/displayicon.png",
    "Cypher": "https://media.valorant-api.com/agents/115d613b-4f08-593c-2c54-398a773ccb3b/displayicon.png",
    "Reyna": "https://media.valorant-api.com/agents/a3593731-4771-54f3-e5d0-a7b4109b457e/displayicon.png",
    "Killjoy": "https://media.valorant-api.com/agents/1e58de9c-4950-5125-93e9-a0aee9f98746/displayicon.png",
    "Astra": "https://media.valorant-api.com/agents/41fb69c1-4189-7b37-f117-bcaf1e96f101/displayicon.png",
    "KAY/O": "https://media.valorant-api.com/agents/601dbbe7-43ce-be57-2a40-4b24fc6d9006/displayicon.png",
    "Chamber": "https://media.valorant-api.com/agents/22697a3d-44bf-8dd7-4f4f-4a90a3245040/displayicon.png",
    "Neon": "https://media.valorant-api.com/agents/bb2a2828-4bc5-47a3-af3e-408c697b41a7/displayicon.png",
    "Fade": "https://media.valorant-api.com/agents/dade69b4-4f5a-8528-247b-219e5a1facd6/displayicon.png",
    "Harbor": "https://media.valorant-api.com/agents/95b58a7a-46db-986d-ab41-41abb40d1b34/displayicon.png",
    "Gekko": "https://media.valorant-api.com/agents/e370fa57-4757-3604-361d-76b3615c5025/displayicon.png",
    "Deadlock": "https://media.valorant-api.com/agents/cc8b3908-4b4b-a3a9-dda1-37ab35519918/displayicon.png",
    "Iso": "https://media.valorant-api.com/agents/1dbf2eff-4395-95c3-b69d-29b1cc7a86b6/displayicon.png",
    "Clove": "https://media.valorant-api.com/agents/0e38b6e5-4cda-caee-312d-2f9e436b43b6/displayicon.png",
    "Vyse": "https://media.valorant-api.com/agents/608298b4-4b56-c782-b70d-f0ba29b4e727/displayicon.png"
}

# 1. Page Configuration and Theme Injection
st.set_page_config(
    page_title="VCT Predictive Engine & Fantasy Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Dark Theme Styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d0e12;
        color: #e2e8f0;
    }
    
    /* Header gradient styling */
    .dashboard-title {
        background: linear-gradient(135deg, #ff4655 0%, #ff7676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Sleek card container */
    .glass-card {
        background: rgba(26, 29, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }
    
    /* Metric label */
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* Winner indicator */
    .winner-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 15px;
        border-radius: 8px;
        color: #4ade80;
        font-weight: 600;
        text-align: center;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .team-badge {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    /* Optimizer result card */
    .optimizer-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.03) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* Transfer card */
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
    
    /* Simulation pulse */
    .sim-result {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(168, 85, 247, 0.03) 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="dashboard-title">VCT FANTASY & PREDICTIVE DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Version 3.1: Open Match Simulation, VFL Roster Optimizer & Transfer Advisor</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

# 2. Ingest Match Files
files = sorted(glob.glob(os.path.join(RAW_DIR, "match_*.json")))
matches_lookup = {}
match_options = []

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            content = json.load(file)
        if "data" not in content or "segments" not in content["data"] or not content["data"]["segments"]:
            continue
        segment = content["data"]["segments"][0]
        match_id = segment["match_id"]
        team_a = segment["teams"][0]["name"]
        team_b = segment["teams"][1]["name"]
        event = segment.get("event", {}).get("name", "Unknown Event")
        date_str = segment.get("date", "Unknown Date")
        
        display_name = f"{event}: {team_a} vs {team_b} ({date_str}) [ID: {match_id}]"
        matches_lookup[match_id] = {
            "filepath": f,
            "team_a": team_a,
            "team_b": team_b,
            "segment": segment,
            "display_name": display_name
        }
        match_options.append((match_id, display_name))
    except Exception as e:
        pass

# Sidebar: Match Settings
st.sidebar.markdown("### Match Settings")
selected_match_id = st.sidebar.selectbox(
    "Select Target Match ID",
    options=[item[0] for item in match_options],
    format_func=lambda x: next(item[1] for item in match_options if item[0] == x)
)

selected_match = matches_lookup[selected_match_id]
team_a = selected_match["team_a"]
team_b = selected_match["team_b"]
segment = selected_match["segment"]

# Load Veto Predictor
veto_pred = VCTMapVetoPredictor(RAW_DIR)
veto_pred.fit()

# Load Score Regressor and Agent Comp models
score_reg = MapScoreRegressor()
score_reg.load_model()
agent_comp = AgentCompositionGenerator(RAW_DIR)
agent_comp.fit()

# Load Classification Model
clf_model_path = os.path.join(PROCESSED_DIR, "vct_model.cbm")
clf_model = None
if os.path.exists(clf_model_path):
    clf_model = CatBoostClassifier()
    clf_model.load_model(clf_model_path)

# Veto Mode selection
st.sidebar.markdown("### Map Veto Config")
veto_override = st.sidebar.checkbox("Override Map Veto Draft?", value=False)
series_type = st.sidebar.selectbox("Series Type", ["Bo3", "Bo5"], index=1 if "grand final" in selected_match["display_name"].lower() else 0)

# Load maps list
all_maps = sorted(list(veto_pred.map_pool))

if veto_override:
    st.sidebar.markdown("#### Custom Veto Sequence")
    maps_count = 3 if series_type == "Bo3" else 5
    custom_maps = []
    custom_weights = {}
    
    for idx in range(maps_count):
        map_val = st.sidebar.selectbox(
            f"Map {idx+1} Selection",
            all_maps,
            index=min(idx, len(all_maps) - 1),
            key=f"custom_map_{idx}"
        )
        weight_val = st.sidebar.slider(
            f"Map {idx+1} Pick weight for {team_a}",
            min_value=-1,
            max_value=1,
            value=0 if idx == (maps_count-1) else (1 if idx % 2 == 0 else -1),
            step=1,
            key=f"custom_weight_{idx}"
        )
        custom_maps.append(map_val)
        custom_weights[map_val] = weight_val
        
    predicted_veto = {
        "maps": custom_maps,
        "veto_weights": custom_weights,
        "veto_str": "Custom manual veto draft override"
    }
else:
    predicted_veto = veto_pred.predict_veto(team_a, team_b, series_type)

# Sidebar: VFL Fantasy Manager Hub
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 VFL Fantasy Manager Hub")
vfl_scraper = VFLScraper()

if st.sidebar.button("🔄 Update VFL Database", key="btn_update_vfl_db"):
    with st.spinner("Executing scraper and rebuilding JSON registry..."):
        vfl_players_data = vfl_scraper.scrape_player_stats()
    st.sidebar.success(f"Rebuilt VFL Database Cache with {len(vfl_players_data)} players!")
else:
    vfl_players_data = vfl_scraper.get_players()

# Mock vfl rules for backward compatibility in match scoring tab if needed
vfl_rules = {
    "salary_cap": 50,
    "max_per_team": 2,
    "max_transfers_per_gameweek": 3
}

# ============================================================
# MAIN CONTENT — TABS
# ============================================================

# Load automated patch nerf registry on startup
latest_patch, active_penalties = load_automated_registry()

# Globally load historical data for all tabs
player_emas, baseline_lookup, team_stats, player_global_stats, player_agent_stats = load_cached_historical_data()

# Apply meta penalties and enrich players database
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
        # Scale the player's PPG dynamically based on the penalty severity
        p["ppg"] = p["ppg"] * (1.0 - p["meta_penalty"])

tab_match, tab_sim, tab_optimizer, tab_vfl = st.tabs([
    "📊 Match Analysis",
    "⚡ Open Simulation",
    "🧠 Roster Optimizer",
    "📋 VFL Players"
])

# ============================================================
# TAB 1: MATCH ANALYSIS (original dashboard content)
# ============================================================
with tab_match:
    # Run classification inference to get match winner probability
    # (Using globally loaded data)
    
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
    
    # Rosters
    roster_a = []
    roster_b = []
    for map_data in segment.get('maps', []):
        for p in map_data.get('players', {}).get('team1', []):
            roster_a.append(p['name'])
        for p in map_data.get('players', {}).get('team2', []):
            roster_b.append(p['name'])
    roster_a = list(set(roster_a))
    roster_b = list(set(roster_b))
    
    if not roster_a:
        roster_a = get_latest_roster(team_a, RAW_DIR)
    if not roster_b:
        roster_b = get_latest_roster(team_b, RAW_DIR)
    
    ta_acs, ta_kast, ta_duel = get_roster_features(roster_a)
    tb_acs, tb_kast, tb_duel = get_roster_features(roster_b)
    
    ta_feat = team_stats.get(team_a, {})
    tb_feat = team_stats.get(team_b, {})
    
    ta_loadout = ta_feat.get("loadout", 20000.0)
    ta_clutch = ta_feat.get("clutch_rate", 0.05)
    ta_thrifty = ta_feat.get("thrifty_rate", 0.02)
    ta_flawless = ta_feat.get("flawless_rate", 0.05)
    
    tb_loadout = tb_feat.get("loadout", 20000.0)
    tb_clutch = tb_feat.get("clutch_rate", 0.05)
    tb_thrifty = tb_feat.get("thrifty_rate", 0.02)
    tb_flawless = tb_feat.get("flawless_rate", 0.05)
    
    # Comfort picked diff
    map_comfort_diffs_a = []
    map_comfort_diffs_b = []
    for map_data in segment.get('maps', []):
        map_diffs_a = []
        map_diffs_b = []
        for team_key in ['team1', 'team2']:
            for p in map_data.get('players', {}).get(team_key, []):
                p_name = p['name']
                agent = p.get('agent')
                if not agent:
                    continue
                p_glob = player_global_stats.get(p_name, {'sum_acs': 0, 'count': 0})
                prior_global_acs = p_glob['sum_acs'] / p_glob['count'] if p_glob['count'] > 0 else baseline_lookup.get(p_name, {}).get("acs", 200.0)
                p_agent = player_agent_stats.get((p_name, agent), {'sum_acs': 0, 'count': 0})
                prior_agent_acs = p_agent['sum_acs'] / p_agent['count'] if p_agent['count'] > 0 else prior_global_acs
                diff = prior_agent_acs - prior_global_acs
                if team_key == 'team1':
                    map_diffs_a.append(diff)
                else:
                    map_diffs_b.append(diff)
        if map_diffs_a:
            map_comfort_diffs_a.append(sum(map_diffs_a) / len(map_diffs_a))
        if map_diffs_b:
            map_comfort_diffs_b.append(sum(map_diffs_b) / len(map_diffs_b))
    comfort_a = sum(map_comfort_diffs_a) / len(map_comfort_diffs_a) if map_comfort_diffs_a else 0.0
    comfort_b = sum(map_comfort_diffs_b) / len(map_comfort_diffs_b) if map_comfort_diffs_b else 0.0
    
    # Compositions counts
    agent_roles = agent_comp.agent_roles
    map_roles_a = []
    map_roles_b = []
    for map_data in segment.get('maps', []):
        role_counts_a = {'Duelist': 0, 'Controller': 0, 'Initiator': 0, 'Sentinel': 0}
        role_counts_b = {'Duelist': 0, 'Controller': 0, 'Initiator': 0, 'Sentinel': 0}
        for team_key in ['team1', 'team2']:
            for p in map_data.get('players', {}).get(team_key, []):
                agent = p.get('agent')
                role = agent_roles.get(agent, 'Sentinel')
                if team_key == 'team1':
                    role_counts_a[role] = role_counts_a.get(role, 0) + 1
                else:
                    role_counts_b[role] = role_counts_b.get(role, 0) + 1
        map_roles_a.append(role_counts_a)
        map_roles_b.append(role_counts_b)
    
    def avg_role_counts(map_roles_list):
        res = {'Duelist': 0.0, 'Controller': 0.0, 'Initiator': 0.0, 'Sentinel': 0.0}
        if not map_roles_list:
            return res
        for roles in map_roles_list:
            for k in res:
                res[k] += roles.get(k, 0)
        for k in res:
            res[k] /= len(map_roles_list)
        return res
    
    comp_a = avg_role_counts(map_roles_a)
    comp_b = avg_role_counts(map_roles_b)
    
    # Build feature map features dynamically
    map_features = {}
    for idx in range(5):
        map_key_name = f"map_{idx+1}_name"
        map_key_veto = f"map_{idx+1}_veto_weight"
        if idx < len(predicted_veto["maps"]):
            m_name = predicted_veto["maps"][idx]
            map_features[map_key_name] = m_name
            map_features[map_key_veto] = predicted_veto["veto_weights"].get(m_name, 0)
        else:
            map_features[map_key_name] = "None"
            map_features[map_key_veto] = 0
    
    row = {
        "team_a_name": team_a,
        "team_b_name": team_b,
        "team_a_historical_acs_ema": ta_acs,
        "team_a_historical_kast_ema": ta_kast,
        "team_a_historical_duel_diff": ta_duel,
        "team_a_historical_avg_loadout": ta_loadout,
        "team_a_historical_clutch_rate": ta_clutch,
        "team_a_historical_thrifty_rate": ta_thrifty,
        "team_a_historical_flawless_rate": ta_flawless,
        "team_a_comfort_pick_differential": comfort_a,
        "team_a_duelist_count": comp_a.get('Duelist', 0.0),
        "team_a_controller_count": comp_a.get('Controller', 0.0),
        "team_a_initiator_count": comp_a.get('Initiator', 0.0),
        "team_a_sentinel_count": comp_a.get('Sentinel', 0.0),
        "team_b_historical_acs_ema": tb_acs,
        "team_b_historical_kast_ema": tb_kast,
        "team_b_historical_duel_diff": tb_duel,
        "team_b_historical_avg_loadout": tb_loadout,
        "team_b_historical_clutch_rate": tb_clutch,
        "team_b_historical_thrifty_rate": tb_thrifty,
        "team_b_historical_flawless_rate": tb_flawless,
        "team_b_comfort_pick_differential": comfort_b,
        "team_b_duelist_count": comp_b.get('Duelist', 0.0),
        "team_b_controller_count": comp_b.get('Controller', 0.0),
        "team_b_initiator_count": comp_b.get('Initiator', 0.0),
        "team_b_sentinel_count": comp_b.get('Sentinel', 0.0),
        **map_features
    }
    
    X_inf = pd.DataFrame([row])
    if clf_model is not None:
        X_inf = X_inf[clf_model.feature_names_]
        for col in ['team_a_name', 'team_b_name', 'map_1_name', 'map_2_name', 'map_3_name', 'map_4_name', 'map_5_name']:
            if col in X_inf.columns:
                X_inf[col] = X_inf[col].astype(str).fillna('None')
        probs = clf_model.predict_proba(X_inf)[0]
        win_prob_a = probs[1]
        win_prob_b = probs[0]
    else:
        win_prob_a = 0.50
        win_prob_b = 0.50
    
    # Match Winner Projection Card
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">SERIES WINNER PROJECTION</div>', unsafe_allow_html=True)
        
        col_ta, col_tb = st.columns(2)
        with col_ta:
            st.markdown(f"#### {team_a}")
            st.subheader(f"{win_prob_a:.1%}")
            st.progress(float(win_prob_a))
        with col_tb:
            st.markdown(f"#### {team_b}")
            st.subheader(f"{win_prob_b:.1%}")
            st.progress(float(win_prob_b))
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">PREDICTED WINNER</div>', unsafe_allow_html=True)
        
        winner_name = team_a if win_prob_a > win_prob_b else team_b
        win_conf = win_prob_a if win_prob_a > win_prob_b else win_prob_b
        
        st.markdown(f'<div class="winner-box" style="margin-top: 15px;">🏆 {winner_name} ({win_conf:.1%})</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 10px;'>Map vetoes: {predicted_veto['veto_str']}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Projected Map Scores Grid
    st.markdown("### Projected Map Scores")
    cols_maps = st.columns(len(predicted_veto["maps"]))
    
    for idx, m_name in enumerate(predicted_veto["maps"]):
        with cols_maps[idx]:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            team_a_features = {"acs_ema": ta_acs, "avg_loadout": ta_loadout, "comfort_diff": comfort_a}
            team_b_features = {"acs_ema": tb_acs, "avg_loadout": tb_loadout, "comfort_diff": comfort_b}
            veto_w = predicted_veto["veto_weights"].get(m_name, 0)
            
            rounds_a, rounds_b = score_reg.predict_score(team_a_features, team_b_features, m_name, veto_w)
            
            st.markdown(f"**MAP {idx+1}**")
            st.subheader(m_name)
            st.markdown(f"### {rounds_a} - {rounds_b}")
            
            picker = "Decider"
            if veto_w == 1:
                picker = f"Picked by {team_a}"
            elif veto_w == -1:
                picker = f"Picked by {team_b}"
                
            st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>{picker}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Predicted Agent Compositions Grid
    st.markdown("### Projected Agent Compositions")
    tab_maps = st.tabs([f"Map {i+1}: {name}" for i, name in enumerate(predicted_veto["maps"])])
    
    for idx, m_name in enumerate(predicted_veto["maps"]):
        with tab_maps[idx]:
            comp_a_map = agent_comp.predict_composition(team_a, m_name)
            comp_b_map = agent_comp.predict_composition(team_b, m_name)
            
            col_la, col_lb = st.columns(2)
            
            with col_la:
                st.markdown(f"#### {team_a} Projected Lineup")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                for p_name, details in comp_a_map.items():
                    agent = details["agent"]
                    role = details["role"]
                    icon = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/add6443a-41da-e1c3-d774-4598a6c7e2ca/displayicon.png")
                    
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.02); padding-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <img src="{icon}" width="36" height="36" style="border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);"/>
                                <div>
                                    <div style="font-weight: 600; font-size: 0.95rem;">{p_name}</div>
                                    <div style="font-size: 0.75rem; color: #94a3b8;">{agent}</div>
                                </div>
                            </div>
                            <span style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05); color: #38bdf8;">{role}</span>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_lb:
                st.markdown(f"#### {team_b} Projected Lineup")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                for p_name, details in comp_b_map.items():
                    agent = details["agent"]
                    role = details["role"]
                    icon = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/add6443a-41da-e1c3-d774-4598a6c7e2ca/displayicon.png")
                    
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.02); padding-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <img src="{icon}" width="36" height="36" style="border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);"/>
                                <div>
                                    <div style="font-weight: 600; font-size: 0.95rem;">{p_name}</div>
                                    <div style="font-size: 0.75rem; color: #94a3b8;">{agent}</div>
                                </div>
                            </div>
                            <span style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05); color: #38bdf8;">{role}</span>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Fantasy Leaderboard
    st.markdown("### Valorant Fantasy League Leaderboard")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
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
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Leaderboard scores currently unavailable for this match.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2: OPEN SIMULATION
# ============================================================
with tab_sim:
    st.markdown("### ⚡ Open Match Simulation Engine")
    st.markdown("""
        <div class="glass-card">
            <div class="metric-title">ARBITRARY MATCH SIMULATOR</div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
                Simulate any hypothetical VCT matchup using time-decay weighted historical data.
                The engine dynamically resolves rosters, computes EMAs, and runs CatBoost inference.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Collect all known team names from match data
    all_teams = sorted(list(set(
        m["team_a"] for m in matches_lookup.values()
    ) | set(
        m["team_b"] for m in matches_lookup.values()
    )))
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_team_a = st.selectbox("Team A", all_teams, index=0, key="sim_team_a")
    with sim_col2:
        sim_team_b = st.selectbox("Team B", all_teams, index=min(1, len(all_teams)-1), key="sim_team_b")
    
    sim_col3, sim_col4 = st.columns(2)
    with sim_col3:
        sim_ref_date = st.date_input("Reference Date (for time-decay)", value=datetime(2026, 6, 22), key="sim_ref_date")
    with sim_col4:
        sim_map_pool = st.multiselect("Map Pool Override (optional)", all_maps, default=[], key="sim_map_pool")
    
    if st.button("🚀 Run Simulation", key="btn_run_sim", type="primary"):
        if sim_team_a == sim_team_b:
            st.error("Please select two different teams.")
        else:
            with st.spinner(f"Simulating {sim_team_a} vs {sim_team_b}..."):
                sim_result = simulate_arbitrary_match(
                    team_a_name=sim_team_a,
                    team_b_name=sim_team_b,
                    reference_date=datetime.combine(sim_ref_date, datetime.min.time()),
                    map_pool_override=sim_map_pool if sim_map_pool else None
                )
            
            # Display results
            st.markdown('<div class="sim-result">', unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"#### {sim_result['team_a']}")
                st.metric("Win Probability", f"{sim_result['win_prob_a']:.1%}")
                st.progress(float(sim_result['win_prob_a']))
                st.caption(f"Roster: {', '.join(sim_result.get('roster_a', []))}")
            with res_col2:
                st.markdown(f"#### {sim_result['team_b']}")
                st.metric("Win Probability", f"{sim_result['win_prob_b']:.1%}")
                st.progress(float(sim_result['win_prob_b']))
                st.caption(f"Roster: {', '.join(sim_result.get('roster_b', []))}")
            
            # Winner announcement
            sim_winner = sim_result['team_a'] if sim_result['win_prob_a'] > sim_result['win_prob_b'] else sim_result['team_b']
            sim_conf = max(sim_result['win_prob_a'], sim_result['win_prob_b'])
            st.markdown(f'<div class="winner-box">🏆 Predicted Winner: {sim_winner} ({sim_conf:.1%})</div>', unsafe_allow_html=True)
            
            # Predicted maps
            if sim_result.get('predicted_maps'):
                st.markdown("**Predicted Map Sequence:** " + " → ".join(sim_result['predicted_maps']))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature vector expander
            with st.expander("📊 Full Feature Vector"):
                fv = sim_result.get('feature_vector', {})
                fv_df = pd.DataFrame([fv])
                st.dataframe(fv_df, use_container_width=True)

# ============================================================
# TAB 3: ROSTER OPTIMIZER
# ============================================================
with tab_optimizer:
    st.markdown("### 🧠 VFL Fantasy Manager Hub")
    
    # Task 2: Live Meta Radar Component
    st.markdown(f"#### 📡 Live Meta Radar (Patch {latest_patch})")
    if active_penalties:
        # Sort penalties descending
        sorted_penalties = sorted(active_penalties.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_penalties[:3]
        
        cols = st.columns(3)
        for idx, (agent_name, score) in enumerate(top_3):
            with cols[idx]:
                if score >= 0.5:
                    color = "#ef4444" # red
                    severity = "CRITICAL NERF"
                else:
                    color = "#f97316" # orange
                    severity = "MODERATE NERF"
                
                icon_url = AGENT_ICONS.get(agent_name, "https://media.valorant-api.com/agents/add6443a-41da-e1c3-d774-4598a6c7e2ca/displayicon.png")
                st.markdown(f"""
                    <div style="background: rgba(26, 29, 36, 0.6); border-left: 5px solid {color}; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                        <img src="{icon_url}" width="42" height="42" style="border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);"/>
                        <div>
                            <div style="font-size: 0.75rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 0.05em;">{severity}</div>
                            <div style="font-weight: 600; font-size: 1.1rem; color: #f8fafc;">{agent_name}</div>
                            <div style="font-weight: 700; font-size: 1.4rem; color: {color}; margin-top: 2px;">-{score:.2f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No active patch nerfs registered in the automated registry.")

    # Grid: Stage 2 Optimal Roster and Transfer Advisor
    col_roster, col_transfer = st.columns([3, 2])
    
    with col_roster:
        st.markdown("#### 🏆 VCT 2026 Stage 2 Optimal Roster")
        st.markdown("""
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">
                Mathematically optimized using Mixed-Integer Linear Programming (MILP) to maximize projected points under strict VFL constraints.
            </p>
        """, unsafe_allow_html=True)
        
        with st.spinner("Computing optimal Stage 2 baseline roster..."):
            baseline_result = generate_stage_2_baseline(vfl_players_data)
            
        if baseline_result["solver_status"] == "optimal":
            # Show summary stats card
            st.markdown(f"""
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
                            <div style="font-weight: 600; color: #4ade80;">{50 - baseline_result['total_cost']} VP</div>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Active IGL</span>
                            <div style="font-weight: 600; color: #38bdf8;">👑 {baseline_result['igl_player']}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show the selected players in the optimal lineup
            for idx, p in enumerate(baseline_result["optimal_roster"]):
                role_emoji = {"Duelist": "⚔️", "Controller": "🌀", "Initiator": "🔍", "Sentinel": "🛡️", "Flex": "🔄"}.get(p["role"], "🎮")
                igl_badge = '<span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; margin-left: 8px;">👑 IGL (2x Multiplier)</span>' if p["is_igl"] else ""
                wc_badge = '<span style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); color: #facc15; margin-left: 8px;">Wildcard</span>' if p["is_wildcard"] else f'<span style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 2px 8px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); color: #a78bfa; margin-left: 8px;">{p["role"]}</span>'
                penalty_badge = get_meta_penalty_badge(p["player_name"], player_agent_stats, active_penalties)
                
                st.markdown(f"""
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
                """, unsafe_allow_html=True)
        else:
            st.error("Roster Optimizer was unable to calculate an optimal starting lineup. Try updating the database.")
            
    with col_transfer:
        st.markdown("#### 3-Transfer Advisor")
        st.markdown("""
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">
                Enter your current fantasy roster to identify the top 3 optimal player trades to maximize score velocity.
            </p>
        """, unsafe_allow_html=True)
        
        player_names_list = sorted([p["player_name"] for p in vfl_players_data])
        # Find some default players present in the registry to auto-select
        default_selections = []
        for name in ["something", "aspas", "zekken", "wo0t", "Derke", "Leo"]:
            if name in player_names_list:
                default_selections.append(name)
        if len(default_selections) < 6:
            default_selections = player_names_list[:6]
            
        current_roster_names = st.multiselect(
            "Select Your Current 6 Players",
            player_names_list,
            default=default_selections[:6],
            key="transfer_current_roster_new"
        )
        
        if st.button("🔮 Calculate Optimal Trades", key="btn_suggest_transfers_new", type="primary"):
            if len(current_roster_names) != 6:
                st.error("Please select exactly 6 players currently in your roster.")
            else:
                current_roster_objs = []
                for name in current_roster_names:
                    for p in vfl_players_data:
                        if p["player_name"] == name:
                            current_roster_objs.append(p)
                            break
                            
                with st.spinner("Analyzing transfer combinations..."):
                    transfer_result = suggest_transfers(current_roster_objs, vfl_players_data)
                    
                if transfer_result["solver_status"] == "optimal":
                    if transfer_result["projected_gain"] > 0:
                        st.markdown(f"""
                            <div class="optimizer-card" style="background: rgba(34, 197, 94, 0.05); border-color: rgba(34, 197, 94, 0.2);">
                                <div style="font-weight: 700; font-size: 1.1rem; color: #4ade80;">
                                    📈 Projected Score Velocity: +{transfer_result['projected_gain']:.1f} pts
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Suggested Swaps (Max 3 Trades):**")
                        
                        for p in transfer_result["transfers_out"]:
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
                                    
                            st.markdown(f"""
                                <div class="transfer-out" style="padding: 10px 14px; margin-bottom: 8px;">
                                    <span style="color: #ef4444; font-weight: 700;">OUT ⬇</span>
                                    <span style="margin-left: 12px; font-weight: 600;">{p['player_name']}</span>
                                    <span style="color: #94a3b8; font-size: 0.8rem;"> · Cost: {p['price']} VP · PPG: {p['ppg']:.1f}</span>
                                    <div style="font-size: 0.8rem; color: #ef4444; margin-top: 4px; font-style: italic;">{reason}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        for p in transfer_result["transfers_in"]:
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
                                
                            st.markdown(f"""
                                <div class="transfer-in" style="padding: 10px 14px; margin-bottom: 8px;">
                                    <span style="color: #4ade80; font-weight: 700;">IN ⬆</span>
                                    <span style="margin-left: 12px; font-weight: 600;">{p['player_name']}{igl_tag}</span>
                                    <span style="color: #94a3b8; font-size: 0.8rem;"> · Cost: {p['price']} VP · PPG: {p['ppg']:.1f}</span>
                                    <div style="font-size: 0.8rem; color: #4ade80; margin-top: 4px; font-style: italic;">{reason}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style="margin-top: 10px; color: #94a3b8; font-size: 0.8rem; text-align: right;">
                                New Total Cost: <b>{transfer_result['new_total_cost']} VP</b> | New Projected Points: <b>{transfer_result['new_projected_points']:.1f} pts</b>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Your current roster is already optimally positioned! No transfers recommended.")
                else:
                    st.error(f"Solver Error: {transfer_result['solver_status']}. Try adjusting the inputs.")

# ============================================================
# TAB 4: VFL PLAYERS DATABASE
# ============================================================
with tab_vfl:
    st.markdown("### 📋 VFL Player Database")
    
    if vfl_players_data:
        vfl_df = pd.DataFrame(vfl_players_data)
        
        # Align DataFrame fields from VFL scraper structure to dashboard expectations
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
        
        # Summary metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Players", len(vfl_df))
        with m_col2:
            st.metric("Avg Cost", f"${vfl_df['cost'].mean():,.0f}")
        with m_col3:
            st.metric("Avg Points", f"{vfl_df['avg_points'].mean():.1f}")
        with m_col4:
            st.metric("Teams", vfl_df['team'].nunique())
        
        # Filters
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
        
        # Sort by avg points descending
        filtered_df = filtered_df.sort_values('avg_points', ascending=False)
        
        # Format cost column
        display_vfl_df = filtered_df.copy()
        display_vfl_df['cost'] = display_vfl_df['cost'].apply(lambda x: f"${x:,}")
        display_vfl_df['ownership_pct'] = display_vfl_df['ownership_pct'].apply(lambda x: f"{x:.1f}%")
        
        display_vfl_df = display_vfl_df.rename(columns={
            "player_name": "Player",
            "team": "Team",
            "role": "Role",
            "cost": "Cost",
            "total_points": "Total Points",
            "avg_points": "Avg Points",
            "ownership_pct": "Ownership %"
        })
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.dataframe(display_vfl_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No VFL data available. Click '🔄 Scrape VFL Player Stats' in the sidebar to load data.")
