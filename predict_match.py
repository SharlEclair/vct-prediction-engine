import os
import json
import glob
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from catboost import CatBoostClassifier
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predict_match")

RAW_DIR = os.path.join(".", "data", "raw")
PROCESSED_DIR = os.path.join(".", "data", "processed")
TARGET_MATCH_ID = "670471"  # Legacy constant, kept for backward compatibility

# Dynamic time-decay constants
DECAY_LAMBDA = 0.02
# Use system time if in June 2026, otherwise anchor to June 22, 2026
system_now = datetime.now()
if system_now.year == 2026 and system_now.month == 6:
    REFERENCE_DATE = system_now
else:
    REFERENCE_DATE = datetime(2026, 6, 22)

# --- Date Parser Helpers ---
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
    patch_version = None
    patch_match = re.search(r'Patch\s+([0-9.]+)', date_str)
    if patch_match:
        patch_version = patch_match.group(1).strip()
        
    year = None
    year_match = re.search(r'\b(20\d{2})\b', date_str)
    if year_match:
        year = int(year_match.group(1))
    elif patch_version:
        year = PATCH_YEARS.get(patch_version.lower())
        
    if year is None:
        year = 2026
        
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

# --- Scraper Veto & Economy Helpers ---
def match_team(token_team: str, team_a: str, team_b: str) -> int:
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
    if "lev" in token_team and "leviatán" in ta:
        return 1
    if "lev" in token_team and "leviatán" in tb:
        return -1
        
    return 0

def parse_vetos(map_vetos_str: str, team_a_name: str, team_b_name: str) -> dict:
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
            val_str = v.split("(")[0].strip()
            try:
                val = float(val_str)
                row_loadouts.append(val * 1000.0)
            except ValueError:
                pass
                
        if weight == 1:
            team_a_loadouts.extend(row_loadouts)
        elif weight == -1:
            team_b_loadouts.extend(row_loadouts)
            
    avg_a = sum(team_a_loadouts) / len(team_a_loadouts) if team_a_loadouts else 0.0
    avg_b = sum(team_b_loadouts) / len(team_b_loadouts) if team_b_loadouts else 0.0
    
    return avg_a, avg_b

def parse_econ_cell_wins(val_str: str) -> int:
    """Extracts wins from an economy cell (e.g. '4 (2)' -> 2, '2' -> 2)."""
    if not val_str:
        return 0
    val_str = str(val_str).strip()
    match = re.search(r'\((\d+)\)', val_str)
    if match:
        return int(match.group(1))
    digit_match = re.search(r'\d+', val_str)
    if digit_match:
        return int(digit_match.group())
    return 0

# --- Historical Stats Calculation ---
def get_historical_stats(raw_dir: str, exclude_match_ids: list = None, reference_date: datetime = None):
    logger.info("Computing latest rolling player EMAs and team economy averages...")
    if exclude_match_ids is None:
        exclude_match_ids = []
    if reference_date is None:
        reference_date = REFERENCE_DATE
    
    exclude_set = set(str(mid) for mid in exclude_match_ids)
    
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    matches = []
    for f in files:
        # Check if any excluded match ID appears in the filename
        skip = False
        for eid in exclude_set:
            if eid in f:
                skip = True
                break
        if skip:
            continue
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                segment = content["data"]["segments"][0]
                if str(segment.get("match_id")) in exclude_set:
                    continue
                segment["timestamp"] = parse_match_date(segment["date"])
                segment["team_a"] = segment["teams"][0]["name"]
                segment["team_b"] = segment["teams"][1]["name"]
                matches.append(segment)
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
            
    matches.sort(key=lambda x: x["timestamp"])
    
    # Load player stats baseline lookup
    player_stats_path = os.path.join(raw_dir, "player_stats.json")
    baseline_lookup = {}
    if os.path.exists(player_stats_path):
        with open(player_stats_path, "r", encoding="utf-8") as f:
            player_stats_baseline = json.load(f)["data"]["segments"]
            for ps in player_stats_baseline:
                p_name = ps["player"]
                acs_b = float(ps.get("average_combat_score", 200.0))
                kast_str = ps.get("kill_assists_survived_traded", "70%")
                kast_b = float(kast_str.replace("%", "")) / 100.0 if "%" in kast_str else 0.70
                fk_per_r = float(ps.get("first_kills_per_round", 0.0))
                fd_per_r = float(ps.get("first_deaths_per_round", 0.0))
                baseline_lookup[p_name] = {"acs": acs_b, "kast": kast_b, "duel_diff": fk_per_r - fd_per_r}

    # Player global and agent stats dictionaries
    player_global_stats = {}
    player_agent_stats = {}

    player_performances = []
    for m in matches:
        match_id = m['match_id']
        ts = m['timestamp']
        team_a_name = m['team_a']
        team_b_name = m['team_b']
        
        player_map_stats = {}
        for map_data in m['maps']:
            rounds_count = len(map_data['rounds'])
            if rounds_count == 0:
                score = map_data.get('score', {})
                rounds_count = int(score.get('team1', 0)) + int(score.get('team2', 0))
                if rounds_count == 0:
                    rounds_count = 24
                    
            for team_key in ['team1', 'team2']:
                for p in map_data['players'].get(team_key, []):
                    p_name = p['name']
                    agent = p.get('agent', '')
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

                    # Update player agent & global stats for comfort picks
                    if acs_val > 0:
                        if p_name not in player_global_stats:
                            player_global_stats[p_name] = {'sum_acs': 0, 'count': 0}
                        player_global_stats[p_name]['sum_acs'] += acs_val
                        player_global_stats[p_name]['count'] += 1
                        
                        if agent:
                            if (p_name, agent) not in player_agent_stats:
                                player_agent_stats[(p_name, agent)] = {'sum_acs': 0, 'count': 0}
                            player_agent_stats[(p_name, agent)]['sum_acs'] += acs_val
                            player_agent_stats[(p_name, agent)]['count'] += 1
                    
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
    # Filter out excluded match IDs
    if exclude_set:
        df_player_perf = df_player_perf[~df_player_perf["match_id"].astype(str).isin(exclude_set)]
    df_player_perf = df_player_perf.sort_values(by=["player", "timestamp"])
    
    # Apply exponential time-decay weights based on reference_date
    if not df_player_perf.empty:
        df_player_perf["decay_weight"] = df_player_perf["timestamp"].apply(
            lambda ts: np.exp(-DECAY_LAMBDA * max((reference_date - ts).days, 0))
        )
    else:
        df_player_perf["decay_weight"] = 1.0
    
    # Calculate time-decay weighted EMA
    df_player_perf["acs_weighted"] = df_player_perf["acs"] * df_player_perf["decay_weight"]
    df_player_perf["kast_weighted"] = df_player_perf["kast"] * df_player_perf["decay_weight"]
    df_player_perf["duel_diff_weighted"] = df_player_perf["duel_diff"] * df_player_perf["decay_weight"]
    
    df_player_perf["acs_ema"] = df_player_perf.groupby("player")["acs_weighted"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    df_player_perf["kast_ema"] = df_player_perf.groupby("player")["kast_weighted"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    df_player_perf["duel_diff_ema"] = df_player_perf.groupby("player")["duel_diff_weighted"].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    
    # Normalize EMAs back by dividing by decay weight to restore scale
    df_player_perf["acs_ema"] = df_player_perf["acs_ema"] / df_player_perf["decay_weight"].replace(0, 1)
    df_player_perf["kast_ema"] = df_player_perf["kast_ema"] / df_player_perf["decay_weight"].replace(0, 1)
    df_player_perf["duel_diff_ema"] = df_player_perf["duel_diff_ema"] / df_player_perf["decay_weight"].replace(0, 1)
    
    # Extract latest computed EMA row for each player
    latest_player_rows = df_player_perf.sort_values('timestamp').groupby('player').last()
    player_emas = {}
    for p_name, row in latest_player_rows.iterrows():
        player_emas[p_name] = {
            "acs": row["acs_ema"],
            "kast": row["kast_ema"],
            "duel_diff": row["duel_diff_ema"]
        }

    # Extract team economy and round closeness
    team_performances = []
    for m in matches:
        ts = m['timestamp']
        team_a_name = m['team_a']
        team_b_name = m['team_b']
        
        map_loadouts_a = []
        map_loadouts_b = []
        
        total_rounds = 0
        total_clutches_a = 0
        total_clutches_b = 0
        total_thrifty_a = 0
        total_thrifty_b = 0
        total_flawless_a = 0
        total_flawless_b = 0
        
        for map_data in m.get('maps', []):
            rounds_count = len(map_data.get('rounds', []))
            if rounds_count == 0:
                score = map_data.get('score', {})
                rounds_count = int(score.get('team1', 0)) + int(score.get('team2', 0))
                if rounds_count == 0:
                    rounds_count = 24
            total_rounds += rounds_count
            
            avg_a, avg_b = parse_map_economy(map_data.get('economy', []), team_a_name, team_b_name)
            if avg_a > 0:
                map_loadouts_a.append(avg_a)
            if avg_b > 0:
                map_loadouts_b.append(avg_b)
                
            # Clutch wins
            adv_stats = map_data.get('performance', {}).get('advanced_stats', [])
            for p_adv in adv_stats:
                p_name = p_adv['player']
                weight = match_team(p_name, team_a_name, team_b_name)
                clutches = 0
                for k in ['5', '6', '7', '8', '9']:
                    val = p_adv.get(k, '')
                    if val and str(val).isdigit():
                        clutches += int(val)
                if weight == 1:
                    total_clutches_a += clutches
                elif weight == -1:
                    total_clutches_b += clutches
                    
            # Thrifty wins
            econ_rows = map_data.get('economy', [])
            for econ_row in econ_rows:
                team_raw = econ_row.get('0', '')
                weight = match_team(team_raw, team_a_name, team_b_name)
                thrifty = parse_econ_cell_wins(econ_row.get('2')) + parse_econ_cell_wins(econ_row.get('3'))
                if weight == 1:
                    total_thrifty_a += thrifty
                elif weight == -1:
                    total_thrifty_b += thrifty
                    
            # Flawless wins
            score_dict = map_data.get('score', {})
            win_a = int(score_dict.get('team1', 0))
            win_b = int(score_dict.get('team2', 0))
            total_flawless_a += int(win_a * 0.15)
            total_flawless_b += int(win_b * 0.15)
            
        match_loadout_a = sum(map_loadouts_a) / len(map_loadouts_a) if map_loadouts_a else 20000.0
        match_loadout_b = sum(map_loadouts_b) / len(map_loadouts_b) if map_loadouts_b else 20000.0
        
        clutch_rate_a = total_clutches_a / total_rounds if total_rounds > 0 else 0.0
        clutch_rate_b = total_clutches_b / total_rounds if total_rounds > 0 else 0.0
        thrifty_rate_a = total_thrifty_a / total_rounds if total_rounds > 0 else 0.0
        thrifty_rate_b = total_thrifty_b / total_rounds if total_rounds > 0 else 0.0
        flawless_rate_a = total_flawless_a / total_rounds if total_rounds > 0 else 0.0
        flawless_rate_b = total_flawless_b / total_rounds if total_rounds > 0 else 0.0
        
        team_performances.append({
            'team': team_a_name,
            'match_id': match_id,
            'timestamp': ts,
            'loadout': match_loadout_a,
            'clutch_rate': clutch_rate_a,
            'thrifty_rate': thrifty_rate_a,
            'flawless_rate': flawless_rate_a
        })
        team_performances.append({
            'team': team_b_name,
            'match_id': match_id,
            'timestamp': ts,
            'loadout': match_loadout_b,
            'clutch_rate': clutch_rate_b,
            'thrifty_rate': thrifty_rate_b,
            'flawless_rate': flawless_rate_b
        })
        
    df_team_perf = pd.DataFrame(team_performances)
    if exclude_set:
        df_team_perf = df_team_perf[~df_team_perf["match_id"].astype(str).isin(exclude_set)]
    df_team_perf = df_team_perf.sort_values(by=["team", "timestamp"])
    
    # Apply time-decay to team performance metrics
    if not df_team_perf.empty:
        df_team_perf["decay_weight"] = df_team_perf["timestamp"].apply(
            lambda ts: np.exp(-DECAY_LAMBDA * max((reference_date - ts).days, 0))
        )
        for col in ['loadout', 'clutch_rate', 'thrifty_rate', 'flawless_rate']:
            df_team_perf[col] = df_team_perf[col] * df_team_perf["decay_weight"]
    
    for col in ['loadout', 'clutch_rate', 'thrifty_rate', 'flawless_rate']:
        df_team_perf[f"{col}_roll"] = df_team_perf.groupby("team")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
    latest_team_rows = df_team_perf.groupby("team").last()
    team_stats = {}
    for team, row in latest_team_rows.iterrows():
        team_stats[team] = {
            "loadout": row["loadout_roll"],
            "clutch_rate": row["clutch_rate_roll"],
            "thrifty_rate": row["thrifty_rate_roll"],
            "flawless_rate": row["flawless_rate_roll"]
        }
        
    return player_emas, baseline_lookup, team_stats, player_global_stats, player_agent_stats

def get_latest_roster(team_name: str, raw_dir: str, exclude_match_ids: list = None) -> list[str]:
    """Finds the most recent roster for this team from historical matches."""
    if exclude_match_ids is None:
        exclude_match_ids = []
    exclude_set = set(str(mid) for mid in exclude_match_ids)
    
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    matches_with_team = []
    for f in files:
        skip = False
        for eid in exclude_set:
            if eid in f:
                skip = True
                break
        if skip:
            continue
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                segment = content["data"]["segments"][0]
                if str(segment.get("match_id")) in exclude_set:
                    continue
                ta = segment["teams"][0]["name"]
                tb = segment["teams"][1]["name"]
                ts = parse_match_date(segment["date"])
                weight = match_team(team_name, ta, tb)
                if weight != 0:
                    matches_with_team.append((ts, segment, ta, tb, weight))
        except Exception as e:
            pass
            
    if not matches_with_team:
        return []
        
    matches_with_team.sort(key=lambda x: x[0], reverse=True)
    latest_segment = matches_with_team[0][1]
    weight = matches_with_team[0][4]
    
    team_key = 'team1' if weight == 1 else 'team2'
    
    roster = set()
    for map_data in latest_segment.get('maps', []):
        for p in map_data.get('players', {}).get(team_key, []):
            roster.add(p['name'])
            
    return list(roster)

# --- Arbitrary Match Simulation ---
def simulate_arbitrary_match(
    team_a_name: str,
    team_b_name: str,
    patch_override: str = None,
    map_pool_override: list = None,
    reference_date: datetime = None,
    exclude_match_ids: list = None
) -> dict:
    """
    Simulate an arbitrary match between any two teams using the latest
    historical data up to the reference_date. Fully decoupled from
    fixed match IDs.
    
    Args:
        team_a_name: Name of team A (e.g., "Paper Rex")
        team_b_name: Name of team B (e.g., "LEVIATÁN")
        patch_override: Optional patch version string to filter data
        map_pool_override: Optional list of map names to restrict veto predictions
        reference_date: Date to use for time-decay (defaults to 2026-06-22)
        exclude_match_ids: List of match IDs to exclude from historical data
    
    Returns:
        dict with team_a, team_b, win_prob_a, win_prob_b, predicted_maps, feature_vector
    """
    if reference_date is None:
        reference_date = REFERENCE_DATE
    if exclude_match_ids is None:
        exclude_match_ids = []
    
    logger.info(f"Simulating arbitrary match: {team_a_name} vs {team_b_name}")
    logger.info(f"Reference date: {reference_date.isoformat()}, Decay λ={DECAY_LAMBDA}")
    
    # 1. Load historical database with time-decay weighting
    player_emas, baseline_lookup, team_stats, player_global_stats, player_agent_stats = get_historical_stats(
        RAW_DIR,
        exclude_match_ids=exclude_match_ids,
        reference_date=reference_date
    )
    
    # 2. Dynamically resolve rosters from latest historical appearances
    roster_a = get_latest_roster(team_a_name, RAW_DIR, exclude_match_ids=exclude_match_ids)
    roster_b = get_latest_roster(team_b_name, RAW_DIR, exclude_match_ids=exclude_match_ids)
    
    if not roster_a:
        logger.warning(f"Could not find historical roster for {team_a_name}. Using empty roster.")
    if not roster_b:
        logger.warning(f"Could not find historical roster for {team_b_name}. Using empty roster.")
    
    logger.info(f"Roster A ({team_a_name}): {roster_a}")
    logger.info(f"Roster B ({team_b_name}): {roster_b}")
    
    # 3. Calculate aggregate player features
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
    
    ta_acs, ta_kast, ta_duel = get_roster_features(roster_a)
    tb_acs, tb_kast, tb_duel = get_roster_features(roster_b)
    
    # 4. Team economy stats
    ta_feat = team_stats.get(team_a_name, {})
    tb_feat = team_stats.get(team_b_name, {})
    
    ta_loadout = ta_feat.get("loadout", 20000.0)
    ta_clutch = ta_feat.get("clutch_rate", 0.05)
    ta_thrifty = ta_feat.get("thrifty_rate", 0.02)
    ta_flawless = ta_feat.get("flawless_rate", 0.05)
    
    tb_loadout = tb_feat.get("loadout", 20000.0)
    tb_clutch = tb_feat.get("clutch_rate", 0.05)
    tb_thrifty = tb_feat.get("thrifty_rate", 0.02)
    tb_flawless = tb_feat.get("flawless_rate", 0.05)
    
    # 5. Comfort pick differentials from historical data
    comfort_a = 0.0
    comfort_b = 0.0
    for p_name in roster_a:
        p_glob = player_global_stats.get(p_name, {'sum_acs': 0, 'count': 0})
        prior_global_acs = p_glob['sum_acs'] / p_glob['count'] if p_glob['count'] > 0 else baseline_lookup.get(p_name, {}).get("acs", 200.0)
        # Average across all agent appearances for this player
        agent_diffs = []
        for (pn, agent), stats in player_agent_stats.items():
            if pn == p_name and stats['count'] > 0:
                agent_acs = stats['sum_acs'] / stats['count']
                agent_diffs.append(agent_acs - prior_global_acs)
        if agent_diffs:
            comfort_a += max(agent_diffs)  # Best comfort agent
    if roster_a:
        comfort_a /= len(roster_a)
    
    for p_name in roster_b:
        p_glob = player_global_stats.get(p_name, {'sum_acs': 0, 'count': 0})
        prior_global_acs = p_glob['sum_acs'] / p_glob['count'] if p_glob['count'] > 0 else baseline_lookup.get(p_name, {}).get("acs", 200.0)
        agent_diffs = []
        for (pn, agent), stats in player_agent_stats.items():
            if pn == p_name and stats['count'] > 0:
                agent_acs = stats['sum_acs'] / stats['count']
                agent_diffs.append(agent_acs - prior_global_acs)
        if agent_diffs:
            comfort_b += max(agent_diffs)
    if roster_b:
        comfort_b /= len(roster_b)
    
    # 6. Predict map veto sequence
    try:
        from veto_predictor import VCTMapVetoPredictor
        veto_pred = VCTMapVetoPredictor(RAW_DIR)
        veto_pred.fit()
        if map_pool_override:
            veto_pred.map_pool = set(map_pool_override)
        predicted_veto = veto_pred.predict_veto(team_a_name, team_b_name, "Bo3")
    except Exception as e:
        logger.warning(f"Veto predictor failed: {e}. Using neutral defaults.")
        default_maps = map_pool_override or ["Ascent", "Bind", "Haven"]
        predicted_veto = {
            "maps": default_maps[:3],
            "veto_weights": {m: 0 for m in default_maps[:3]},
            "veto_str": "Default (veto predictor unavailable)"
        }
    
    # 7. Agent composition role counts (use defaults since we don't have match-specific data)
    comp_a = {'Duelist': 1.0, 'Controller': 1.0, 'Initiator': 2.0, 'Sentinel': 1.0}
    comp_b = {'Duelist': 1.0, 'Controller': 1.0, 'Initiator': 2.0, 'Sentinel': 1.0}
    
    # 8. Build feature vector
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
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
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
    
    X_inference = pd.DataFrame([row])
    
    # 9. Load persisted CatBoost model and run inference
    model_path = os.path.join(PROCESSED_DIR, "vct_model.cbm")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Returning 50/50 probabilities.")
        return {
            "team_a": team_a_name,
            "team_b": team_b_name,
            "win_prob_a": 0.50,
            "win_prob_b": 0.50,
            "predicted_maps": predicted_veto["maps"],
            "veto_str": predicted_veto.get("veto_str", ""),
            "feature_vector": row,
            "roster_a": roster_a,
            "roster_b": roster_b
        }
    
    logger.info(f"Loading trained CatBoost model from {model_path}...")
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Reorder features to match model training order
    try:
        X_inference = X_inference[model.feature_names_]
    except KeyError as e:
        logger.warning(f"Feature mismatch: {e}. Adding missing columns with defaults.")
        for feat in model.feature_names_:
            if feat not in X_inference.columns:
                X_inference[feat] = 0
        X_inference = X_inference[model.feature_names_]
    
    # Ensure categorical columns are clean
    cat_cols = [c for c in X_inference.columns if 'name' in c]
    for col in cat_cols:
        X_inference[col] = X_inference[col].astype(str).fillna('None')
    
    logger.info("Inference Feature Vector:")
    logger.info(X_inference.to_dict(orient='records')[0])
    
    # 10. Run prediction
    probs = model.predict_proba(X_inference)[0]
    win_prob_a = probs[1]
    win_prob_b = probs[0]
    
    result = {
        "team_a": team_a_name,
        "team_b": team_b_name,
        "win_prob_a": float(win_prob_a),
        "win_prob_b": float(win_prob_b),
        "predicted_maps": predicted_veto["maps"],
        "veto_str": predicted_veto.get("veto_str", ""),
        "feature_vector": row,
        "roster_a": roster_a,
        "roster_b": roster_b
    }
    
    logger.info(f"Simulation Result: {team_a_name} {win_prob_a:.2%} vs {team_b_name} {win_prob_b:.2%}")
    return result


# --- Legacy Predict Match (backward-compatible wrapper) ---
def predict_grand_final():
    logger.info("Starting real-time Grand Final prediction orchestrator...")
    
    # 1. Load match details
    match_file = os.path.join(RAW_DIR, f"match_{TARGET_MATCH_ID}.json")
    if not os.path.exists(match_file):
        raise FileNotFoundError(f"Grand Final match file not found at {match_file}")
        
    with open(match_file, "r", encoding="utf-8") as f:
        match_data = json.load(f)
        
    segment = match_data["data"]["segments"][0]
    team_a_name = segment["teams"][0]["name"]
    team_b_name = segment["teams"][1]["name"]
    
    logger.info(f"Target Grand Final Match: {team_a_name} vs {team_b_name} (ID: {TARGET_MATCH_ID})")
    
    # 2. Roster Identification
    roster_a = set()
    roster_b = set()
    for map_data in segment.get('maps', []):
        for p in map_data.get('players', {}).get('team1', []):
            roster_a.add(p['name'])
        for p in map_data.get('players', {}).get('team2', []):
            roster_b.add(p['name'])
            
    roster_a = list(roster_a)
    roster_b = list(roster_b)
    
    # Fallback to historical rosters if match is pending/empty
    if not roster_a:
        logger.warning(f"Roster for {team_a_name} is empty in the match details. Finding active roster from history...")
        roster_a = get_latest_roster(team_a_name, RAW_DIR)
    if not roster_b:
        logger.warning(f"Roster for {team_b_name} is empty in the match details. Finding active roster from history...")
        roster_b = get_latest_roster(team_b_name, RAW_DIR)
        
    logger.info(f"Roster A ({team_a_name}): {roster_a}")
    logger.info(f"Roster B ({team_b_name}): {roster_b}")
    
    # 3. Load historical database stats
    player_emas, baseline_lookup, team_stats, player_global_stats, player_agent_stats = get_historical_stats(RAW_DIR)
    
    # Calculate aggregate player feature values
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
        
    ta_acs, ta_kast, ta_duel = get_roster_features(roster_a)
    tb_acs, tb_kast, tb_duel = get_roster_features(roster_b)
    
    # Look up team economy and closeness stats
    ta_feat = team_stats.get(team_a_name, {})
    tb_feat = team_stats.get(team_b_name, {})
    
    ta_loadout = ta_feat.get("loadout", 20000.0)
    ta_clutch = ta_feat.get("clutch_rate", 0.05)
    ta_thrifty = ta_feat.get("thrifty_rate", 0.02)
    ta_flawless = ta_feat.get("flawless_rate", 0.05)
    
    tb_loadout = tb_feat.get("loadout", 20000.0)
    tb_clutch = tb_feat.get("clutch_rate", 0.05)
    tb_thrifty = tb_feat.get("thrifty_rate", 0.02)
    tb_flawless = tb_feat.get("flawless_rate", 0.05)

    # Compute comfort pick differential for active rosters in the target match
    map_comfort_diffs_a = []
    map_comfort_diffs_b = []
    
    for map_data in segment.get('maps', []):
        map_diffs_a = []
        map_diffs_b = []
        for team_key in ['team1', 'team2']:
            for p in map_data.get('players', {}).get(team_key, []):
                p_name = p['name']
                agent = p['agent']
                
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

    # Load agent roles mapping and compute compositions
    agent_roles_path = os.path.join(RAW_DIR, "agent_roles.json")
    agent_roles = {}
    if os.path.exists(agent_roles_path):
        try:
            with open(agent_roles_path, "r", encoding="utf-8") as f:
                agent_roles = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load agent roles: {e}")

    map_roles_a = []
    map_roles_b = []
    for map_data in segment.get('maps', []):
        role_counts_a = {'Duelist': 0, 'Controller': 0, 'Initiator': 0, 'Sentinel': 0}
        role_counts_b = {'Duelist': 0, 'Controller': 0, 'Initiator': 0, 'Sentinel': 0}
        for team_key in ['team1', 'team2']:
            for p in map_data.get('players', {}).get(team_key, []):
                p_name = p['name']
                agent = p['agent']
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
    
    # 4. Map Vetoes Handling
    map_vetos_str = segment.get('map_vetos', '')
    veto_weights = {}
    if map_vetos_str:
        logger.info(f"Parsing match map vetoes: '{map_vetos_str}'")
        veto_weights = parse_vetos(map_vetos_str, team_a_name, team_b_name)
    else:
        logger.warning("Map vetoes not populated in JSON. Defaulting map veto weights to 0 (neutral).")
        
    maps_played = segment.get('maps', [])
    map_features = {}
    for idx in range(5):
        map_key_name = f"map_{idx+1}_name"
        map_key_veto = f"map_{idx+1}_veto_weight"
        
        if idx < len(maps_played):
            m_name = maps_played[idx].get("map_name", "None")
            map_features[map_key_name] = m_name
            map_features[map_key_veto] = veto_weights.get(m_name, 0) if map_vetos_str else 0
        else:
            map_features[map_key_name] = "None"
            map_features[map_key_veto] = 0
            
    # 5. Build X_inference DataFrame
    row = {
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
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
    
    X_inference = pd.DataFrame([row])
    
    # 6. Load Persisted CatBoost Model
    model_path = os.path.join(PROCESSED_DIR, "vct_model.cbm")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run model_pipeline.py first.")
        
    logger.info(f"Loading trained CatBoost model from {model_path}...")
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Reorder X_inference to match model training feature order
    X_inference = X_inference[model.feature_names_]
    
    # Ensure categorical columns are clean string type and filled
    cat_features = ['team_a_name', 'team_b_name', 'map_1_name', 'map_2_name', 'map_3_name']
    for col in cat_features:
        X_inference[col] = X_inference[col].astype(str).fillna('None')
        
    logger.info("Inference Feature Vector:")
    logger.info(X_inference.to_dict(orient='records')[0])
    
    # 7. Run Predict Proba
    probs = model.predict_proba(X_inference)[0]
    win_prob_team_a = probs[1]
    win_prob_team_b = probs[0]
    
    print("\n" + "="*60)
    print("VALORANT MASTERS LONDON GRAND FINAL INFERENCE RESULTS")
    print("="*60)
    print(f"Match: {team_a_name} vs {team_b_name} (ID: {TARGET_MATCH_ID})")
    print(f"Predicted Probability of {team_a_name} Winning: {win_prob_team_a:.2%}")
    print(f"Predicted Probability of {team_b_name} Winning: {win_prob_team_b:.2%}")
    print("="*60 + "\n")
    
    # 8. SHAP Explainability Plot
    logger.info("Generating SHAP explanation for the Grand Final prediction...")
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_inference)
    
    shap_plot_path = os.path.join(PROCESSED_DIR, "prx_vs_lev_shap.png")
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], show=False)
    plt.title(f"SHAP Waterfall Plot for Grand Final: {team_a_name} vs {team_b_name}", fontsize=12, pad=20)
    plt.tight_layout()
    
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"SHAP waterfall explanation saved to: {shap_plot_path}")
    
    return {
        "team_a": team_a_name,
        "team_b": team_b_name,
        "win_prob_team_a": win_prob_team_a,
        "win_prob_team_b": win_prob_team_b,
        "shap_plot": shap_plot_path
    }

if __name__ == "__main__":
    predict_grand_final()
