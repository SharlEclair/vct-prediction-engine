import os
import json
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("feature_engineering")

RAW_DIR = os.path.join(".", "data", "raw")
PROCESSED_DIR = os.path.join(".", "data", "processed")

# Load patch years from patch_notes.csv globally to infer missing years
PATCH_YEARS = {}
csv_path = os.path.join(RAW_DIR, "patch_notes.csv")
if os.path.exists(csv_path):
    try:
        df_patches = pd.read_csv(csv_path)
        for _, row in df_patches.iterrows():
            version = str(row['patch_version']).strip().lower()
            if version.startswith('v'):
                version = version[1:]
            date_str_val = str(row['release_date'])
            match_yr = re.search(r'\b(20\d{2})\b', date_str_val)
            if match_yr:
                PATCH_YEARS[version] = int(match_yr.group(1))
    except Exception as e:
        logger.error(f"Failed to load patch notes for year mapping: {e}")

def parse_match_date(date_str: str) -> datetime:
    """Parses date string from VLR API match segments.
    Supports formats with or without weekdays and years (e.g. including 'Friday, April 24 6:00 PM AEST Patch 12.06').
    """
    # 1. Extract patch if present
    patch_version = None
    patch_match = re.search(r'Patch\s+([0-9.]+)', date_str)
    if patch_match:
        patch_version = patch_match.group(1).strip()
        
    # 2. Extract 4-digit year or infer from patch
    year = None
    year_match = re.search(r'\b(20\d{2})\b', date_str)
    if year_match:
        year = int(year_match.group(1))
    elif patch_version:
        year = PATCH_YEARS.get(patch_version.lower())
        
    if year is None:
        year = 2026 # Default fallback
        
    # 3. Clean and parse Month, Day, Time, AM/PM
    clean_str = date_str.split(" Patch ")[0]
    clean_str = re.sub(r'\s+[A-Z]{3,4}$', '', clean_str).strip()
    clean_str = re.sub(r'^[A-Za-z]+,\s*', '', clean_str).strip()
    
    month_day_match = re.search(r'^([A-Za-z]+)\s+(\d+)', clean_str)
    if not month_day_match:
        raise ValueError(f"Could not parse Month/Day from: {clean_str}")
        
    month = month_day_match.group(1)
    day = int(month_day_match.group(2))
    
    time_match = re.search(r'(\d+:\d+)\s+([AP]M)', clean_str)
    if not time_match:
        raise ValueError(f"Could not parse Time from: {clean_str}")
        
    time_str = time_match.group(1)
    ampm = time_match.group(2)
    
    normalized_date_str = f"{month} {day}, {year} {time_str} {ampm}"
    return datetime.strptime(normalized_date_str, "%B %d, %Y %I:%M %p")

def match_team(token_team: str, team_a: str, team_b: str) -> int:
    """Matches team names from raw strings to Team A (1) or Team B (-1). Returns 0 if unmatched."""
    token_team = token_team.lower().strip()
    ta = team_a.lower().strip()
    tb = team_b.lower().strip()
    
    if token_team in ta or ta in token_team:
        return 1
    if token_team in tb or tb in token_team:
        return -1
        
    if len(token_team) >= 3:
        prefix = token_team[:3]
        if ta.startswith(prefix):
            return 1
        if tb.startswith(prefix):
            return -1
            
    # Check initials e.g. 'PRX' -> 'Paper Rex'
    def get_initials(name: str) -> str:
        return "".join(word[0] for word in name.split() if word)
        
    ta_init = get_initials(ta)
    tb_init = get_initials(tb)
    if token_team == ta_init:
        return 1
    if token_team == tb_init:
        return -1
        
    if "prx" in token_team and "paper rex" in ta:
        return 1
    if "prx" in token_team and "paper rex" in tb:
        return -1
        
    return 0

def parse_vetos(map_vetos_str: str, team_a_name: str, team_b_name: str) -> dict:
    """Parses map veto strings to map map_name -> picked weight (1 = Team A, -1 = Team B, 0 = Decider)."""
    if not map_vetos_str:
        return {}
    tokens = [t.strip() for t in map_vetos_str.split(";") if t.strip()]
    veto_weights = {}
    
    for token in tokens:
        if "pick" in token:
            parts = token.split(" pick ")
            if len(parts) == 2:
                team_part, map_part = parts
                weight = match_team(team_part, team_a_name, team_b_name)
                map_name = map_part.strip()
                veto_weights[map_name] = weight
        elif "remains" in token:
            parts = token.split(" remains")
            if len(parts) >= 1:
                map_name = parts[0].strip()
                veto_weights[map_name] = 0
                
    return veto_weights

def parse_map_economy(econ_list: list, team_a_name: str, team_b_name: str) -> tuple[float, float]:
    """Extracts average loadout value in dollars for Team A and Team B from a map's economy list."""
    team_a_loadouts = []
    team_b_loadouts = []
    
    if not econ_list:
        return 0.0, 0.0
        
    for row in econ_list:
        team_id_raw = row.get('0', '')
        weight = match_team(team_id_raw, team_a_name, team_b_name)
        
        row_loadouts = []
        for k, v in row.items():
            if k == '0' or not k.isdigit():
                continue
            # Parse loadout e.g. "4 (0)" -> 4
            val_str = v.split("(")[0].strip()
            try:
                val = float(val_str)
                row_loadouts.append(val * 1000.0)  # Convert thousands to dollars
            except ValueError:
                pass
                
        if weight == 1:
            team_a_loadouts.extend(row_loadouts)
        elif weight == -1:
            team_b_loadouts.extend(row_loadouts)
            
    avg_a = sum(team_a_loadouts) / len(team_a_loadouts) if team_a_loadouts else 0.0
    avg_b = sum(team_b_loadouts) / len(team_b_loadouts) if team_b_loadouts else 0.0
    
    return avg_a, avg_b

def load_raw_matches() -> list[dict]:
    """Loads all match JSON files from RAW_DIR and sorts them chronologically."""
    files = glob.glob(os.path.join(RAW_DIR, "match_*.json"))
    matches = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            content = json.load(file)
            segment = content["data"]["segments"][0]
            # Parse datetime
            segment["timestamp"] = parse_match_date(segment["date"])
            # Extract team names
            segment["team_a"] = segment["teams"][0]["name"]
            segment["team_b"] = segment["teams"][1]["name"]
            matches.append(segment)
            
    matches.sort(key=lambda x: x["timestamp"])
    return matches

def build_feature_store():
    """Builds and orchestrates the point-in-time features DataFrame."""
    logger.info("Loading raw datasets...")
    matches = load_raw_matches()
    
    # Load player stats baseline lookup
    player_stats_path = os.path.join(RAW_DIR, "player_stats.json")
    with open(player_stats_path, "r", encoding="utf-8") as f:
        player_stats_baseline = json.load(f)["data"]["segments"]
        
    baseline_lookup = {}
    for ps in player_stats_baseline:
        p_name = ps["player"]
        acs_b = float(ps.get("average_combat_score", 200.0))
        kast_str = ps.get("kill_assists_survived_traded", "70%")
        kast_b = float(kast_str.replace("%", "")) / 100.0 if "%" in kast_str else 0.70
        fk_per_r = float(ps.get("first_kills_per_round", 0.0))
        fd_per_r = float(ps.get("first_deaths_per_round", 0.0))
        baseline_lookup[p_name] = {"acs": acs_b, "kast": kast_b, "duel_diff": fk_per_r - fd_per_r}

    logger.info("Extracting individual player-match performance records...")
    player_performances = []
    for m in matches:
        match_id = m['match_id']
        ts = m['timestamp']
        
        # Player map performance tracking
        player_map_stats = {}
        for map_data in m['maps']:
            rounds_count = len(map_data['rounds'])
            if rounds_count == 0:
                score = map_data.get('score', {})
                rounds_count = int(score.get('team1', 0)) + int(score.get('team2', 0))
                if rounds_count == 0:
                    rounds_count = 24
                    
            for team_key in ['team1', 'team2']:
                for p in map_data['players'][team_key]:
                    p_name = p['name']
                    acs_val = float(p['acs']) if (p.get('acs') and str(p['acs']).isdigit()) else 0.0
                    kast_str = p.get('kast', '')
                    kast_val = float(kast_str.replace('%', '')) / 100.0 if (kast_str and '%' in kast_str) else 0.70
                    fk_val = float(p['fk']) if (p.get('fk') and str(p['fk']).isdigit()) else 0.0
                    fd_val = float(p['fd']) if (p.get('fd') and str(p['fd']).isdigit()) else 0.0
                    
                    if p_name not in player_map_stats:
                        player_map_stats[p_name] = []
                    player_map_stats[p_name].append({
                        'acs': acs_val,
                        'kast': kast_val,
                        'fk': fk_val,
                        'fd': fd_val,
                        'rounds': rounds_count
                    })
                    
        for p_name, stats_list in player_map_stats.items():
            avg_acs = sum(s['acs'] for s in stats_list) / len(stats_list)
            avg_kast = sum(s['kast'] for s in stats_list) / len(stats_list)
            total_fk = sum(s['fk'] for s in stats_list)
            total_fd = sum(s['fd'] for s in stats_list)
            total_rounds = sum(s['rounds'] for s in stats_list)
            
            fk_per_round = total_fk / total_rounds if total_rounds > 0 else 0.0
            fd_per_round = total_fd / total_rounds if total_rounds > 0 else 0.0
            
            player_performances.append({
                'player': p_name,
                'match_id': match_id,
                'timestamp': ts,
                'acs': avg_acs,
                'kast': avg_kast,
                'duel_diff': fk_per_round - fd_per_round
            })
            
    df_player_perf = pd.DataFrame(player_performances)
    df_player_perf = df_player_perf.sort_values(by=["player", "timestamp"])
    
    # Calculate rolling player EMAs (span=3 to align with short match windows)
    df_player_perf["acs_ema"] = df_player_perf.groupby("player")["acs"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    df_player_perf["kast_ema"] = df_player_perf.groupby("player")["kast"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    df_player_perf["duel_diff_ema"] = df_player_perf.groupby("player")["duel_diff"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    
    # Strictly shift(1) to avoid temporal leakage
    df_player_perf["acs_ema_shifted"] = df_player_perf.groupby("player")["acs_ema"].shift(1)
    df_player_perf["kast_ema_shifted"] = df_player_perf.groupby("player")["kast_ema"].shift(1)
    df_player_perf["duel_diff_ema_shifted"] = df_player_perf.groupby("player")["duel_diff_ema"].shift(1)
    
    # Fill missing priors with baseline stats
    def fill_baseline_acs(row):
        if pd.isna(row["acs_ema_shifted"]):
            return baseline_lookup.get(row["player"], {}).get("acs", 200.0)
        return row["acs_ema_shifted"]
        
    def fill_baseline_kast(row):
        if pd.isna(row["kast_ema_shifted"]):
            return baseline_lookup.get(row["player"], {}).get("kast", 0.70)
        return row["kast_ema_shifted"]
        
    def fill_baseline_duel(row):
        if pd.isna(row["duel_diff_ema_shifted"]):
            return baseline_lookup.get(row["player"], {}).get("duel_diff", 0.0)
        return row["duel_diff_ema_shifted"]
        
    df_player_perf["acs_ema_shifted"] = df_player_perf.apply(fill_baseline_acs, axis=1)
    df_player_perf["kast_ema_shifted"] = df_player_perf.apply(fill_baseline_kast, axis=1)
    df_player_perf["duel_diff_ema_shifted"] = df_player_perf.apply(fill_baseline_duel, axis=1)
    
    # Create player feature lookup mapping
    player_features_lookup = df_player_perf.set_index(["player", "match_id"])[
        ["acs_ema_shifted", "kast_ema_shifted", "duel_diff_ema_shifted"]
    ].to_dict(orient="index")
    
    logger.info("Extracting team-match economy management metrics...")
    team_performances = []
    for m in matches:
        match_id = m['match_id']
        ts = m['timestamp']
        team_a_name = m['team_a']
        team_b_name = m['team_b']
        
        map_loadouts_a = []
        map_loadouts_b = []
        for map_data in m['maps']:
            avg_a, avg_b = parse_map_economy(map_data.get('economy', []), team_a_name, team_b_name)
            if avg_a > 0:
                map_loadouts_a.append(avg_a)
            if avg_b > 0:
                map_loadouts_b.append(avg_b)
                
        match_loadout_a = sum(map_loadouts_a) / len(map_loadouts_a) if map_loadouts_a else 20000.0
        match_loadout_b = sum(map_loadouts_b) / len(map_loadouts_b) if map_loadouts_b else 20000.0
        
        team_performances.append({
            'team': team_a_name,
            'match_id': match_id,
            'timestamp': ts,
            'loadout': match_loadout_a
        })
        team_performances.append({
            'team': team_b_name,
            'match_id': match_id,
            'timestamp': ts,
            'loadout': match_loadout_b
        })
        
    df_team_perf = pd.DataFrame(team_performances)
    df_team_perf = df_team_perf.sort_values(by=["team", "timestamp"])
    
    # Compute rolling loadout average (window=3, min_periods=1)
    df_team_perf["loadout_roll"] = df_team_perf.groupby("team")["loadout"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df_team_perf["loadout_roll_shifted"] = df_team_perf.groupby("team")["loadout_roll"].shift(1)
    # Fill first-match NaN values with average loadout baseline ($20,000)
    df_team_perf["loadout_roll_shifted"] = df_team_perf["loadout_roll_shifted"].fillna(20000.0)
    
    team_features_lookup = df_team_perf.set_index(["team", "match_id"])["loadout_roll_shifted"].to_dict()
    
    logger.info("Constructing master dataset...")
    master_rows = []
    
    for m in matches:
        match_id = m['match_id']
        ts = m['timestamp']
        team_a_name = m['team_a']
        team_b_name = m['team_b']
        
        # Parse map vetoes
        veto_weights = parse_vetos(m.get('map_vetos', ''), team_a_name, team_b_name)
        
        # Identify the maps played in the match and populate names/veto weights
        maps_played = m.get('maps', [])
        map_features = {}
        for idx in range(3):
            map_key_name = f"map_{idx+1}_name"
            map_key_veto = f"map_{idx+1}_veto_weight"
            
            if idx < len(maps_played):
                m_name = maps_played[idx]["map_name"]
                map_features[map_key_name] = m_name
                map_features[map_key_veto] = veto_weights.get(m_name, 0)
            else:
                map_features[map_key_name] = "None"
                map_features[map_key_veto] = 0
                
        # Identify rosters played in the match (from map players lists)
        roster_a = set()
        roster_b = set()
        for map_data in maps_played:
            for p in map_data['players']['team1']:
                roster_a.add(p['name'])
            for p in map_data['players']['team2']:
                roster_b.add(p['name'])
                
        roster_a = list(roster_a)
        roster_b = list(roster_b)
        
        # Function to compute aggregate player-level prior EMAs
        def get_roster_features(roster):
            acs_list, kast_list, duel_list = [], [], []
            for p_name in roster:
                p_feat = player_features_lookup.get((p_name, match_id))
                if p_feat is not None:
                    acs_list.append(p_feat["acs_ema_shifted"])
                    kast_list.append(p_feat["kast_ema_shifted"])
                    duel_list.append(p_feat["duel_diff_ema_shifted"])
                else:
                    # Fallback to player statistics baseline
                    p_base = baseline_lookup.get(p_name, {"acs": 200.0, "kast": 0.70, "duel_diff": 0.0})
                    acs_list.append(p_base["acs"])
                    kast_list.append(p_base["kast"])
                    duel_list.append(p_base["duel_diff"])
            return (
                sum(acs_list) / len(acs_list) if acs_list else 200.0,
                sum(kast_list) / len(kast_list) if kast_list else 0.70,
                sum(duel_list) / len(duel_list) if duel_list else 0.0
            )
            
        ta_acs, ta_kast, ta_duel = get_roster_features(roster_a)
        tb_acs, tb_kast, tb_duel = get_roster_features(roster_b)
        
        # Look up team economy features
        ta_loadout = team_features_lookup.get((team_a_name, match_id), 20000.0)
        tb_loadout = team_features_lookup.get((team_b_name, match_id), 20000.0)
        
        # Target variable column
        y_target = 1 if m["teams"][0]["is_winner"] else 0
        
        row = {
            "match_id": match_id,
            "timestamp": ts,
            "team_a_name": team_a_name,
            "team_b_name": team_b_name,
            "team_a_historical_acs_ema": ta_acs,
            "team_a_historical_kast_ema": ta_kast,
            "team_a_historical_duel_diff": ta_duel,
            "team_a_historical_avg_loadout": ta_loadout,
            "team_b_historical_acs_ema": tb_acs,
            "team_b_historical_kast_ema": tb_kast,
            "team_b_historical_duel_diff": tb_duel,
            "team_b_historical_avg_loadout": tb_loadout,
            **map_features,
            "y_target": y_target
        }
        master_rows.append(row)
        
    df_master = pd.DataFrame(master_rows)
    # Ensure sorted chronologically by timestamp
    df_master = df_master.sort_values(by="timestamp").reset_index(drop=True)
    
    # Export datasets
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    y_target_df = df_master[["match_id", "y_target"]]
    X_features_df = df_master.drop(columns=["y_target"])
    
    features_path = os.path.join(PROCESSED_DIR, "X_features.csv")
    target_path = os.path.join(PROCESSED_DIR, "y_target.csv")
    
    X_features_df.to_csv(features_path, index=False, encoding="utf-8")
    y_target_df.to_csv(target_path, index=False, encoding="utf-8")
    
    logger.info(f"Features exported to {features_path} (Shape: {X_features_df.shape})")
    logger.info(f"Target exported to {target_path} (Shape: {y_target_df.shape})")
    
    return X_features_df.shape, y_target_df.shape

if __name__ == "__main__":
    build_feature_store()
