import os
import json
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

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

def is_strict_team_match(target: str, candidate: str) -> bool:
    target = target.lower().strip()
    candidate = candidate.lower().strip()
    
    if target == candidate:
        return True
        
    suffixes = ["academy", "gc", "game changers", "black", "blue"]
    target_has_suffix = any(s in target for s in suffixes)
    candidate_has_suffix = any(s in candidate for s in suffixes)
    
    if target_has_suffix != candidate_has_suffix:
        return False
        
    if target in candidate or candidate in target:
        return True
        
    def get_initials(name: str) -> str:
        return "".join(word[0] for word in name.split() if word)
        
    t_init = get_initials(target)
    c_init = get_initials(candidate)
    if t_init == candidate or c_init == target:
        return True
        
    if "prx" in target and "paper rex" in candidate:
        return True
    if "lev" in target and "leviatán" in candidate:
        return True
        
    return False

def match_team(token_team: str, team_a: str, team_b: str) -> int:
    if is_strict_team_match(token_team, team_a):
        return 1
    if is_strict_team_match(token_team, team_b):
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

def load_raw_matches() -> list[dict]:
    """Loads all match JSON files from RAW_DIR and sorts them chronologically."""
    files = glob.glob(os.path.join(RAW_DIR, "match_*.json"))
    matches = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            content = json.load(file)
            
            raw_segments = content.get("data", {}).get("segments", [])
            if not raw_segments:
                continue
                
            if "team1" in raw_segments[0]:
                # New schema: list of map segments
                m_id = os.path.basename(f).replace("match_", "").replace(".json", "")
                team_a = raw_segments[0]["team1"]
                team_b = raw_segments[0]["team2"]
                
                maps = []
                for raw_map in raw_segments:
                    rounds = raw_map.get("round_history", [])
                    score = {
                        "team1": sum(1 for r in rounds if r.get("winner") == team_a),
                        "team2": sum(1 for r in rounds if r.get("winner") == team_b)
                    }
                    
                    players = {"team1": [], "team2": []}
                    for p in raw_map.get("players", []):
                        if p.get("team") == team_a:
                            players["team1"].append(p)
                        else:
                            players["team2"].append(p)
                            
                    map_name_raw = raw_map.get("map", "Unknown")
                    map_name = re.split(r'\s+(?:PICK|DECIDER|BAN)\b', map_name_raw)[0].strip()
                    
                    maps.append({
                        "map_name": map_name,
                        "rounds": rounds,
                        "score": score,
                        "players": players,
                        "economy": raw_map.get("economy", []),
                        "performance": raw_map.get("performance", {})
                    })
                    
                # Determine winner based on map wins
                map_wins_a = sum(1 for m in maps if m["score"]["team1"] > m["score"]["team2"])
                map_wins_b = len(maps) - map_wins_a
                is_winner_a = map_wins_a > map_wins_b
                
                date_str = raw_segments[0]["date"]
                ts = parse_match_date(date_str)
                
                patch = raw_segments[0].get("patch")
                if not patch or patch == "Unknown":
                    patch_match = re.search(r'Patch\s+([0-9.]+)', date_str)
                    patch = patch_match.group(1).strip() if patch_match else None
                    
                segment = {
                    "match_id": m_id,
                    "timestamp": ts,
                    "patch": patch,
                    "team_a": team_a,
                    "team_b": team_b,
                    "teams": [
                        {"name": team_a, "is_winner": is_winner_a},
                        {"name": team_b, "is_winner": not is_winner_a}
                    ],
                    "maps": maps,
                    "map_vetos": "; ".join(raw_segments[0].get("vetoes", []))
                }
            else:
                # Old schema: single match segment with maps nested
                segment = raw_segments[0]
                segment["timestamp"] = parse_match_date(segment["date"])
                patch_match = re.search(r'Patch\s+([0-9.]+)', segment["date"])
                segment["patch"] = patch_match.group(1).strip() if patch_match else None
                segment["team_a"] = segment["teams"][0]["name"]
                segment["team_b"] = segment["teams"][1]["name"]
                
            matches.append(segment)
            
    matches.sort(key=lambda x: x["timestamp"])
    
    # Forward fill missing patches
    last_patch = None
    for m in matches:
        if m.get("patch") is None:
            m["patch"] = last_patch
        else:
            last_patch = m["patch"]
            
    # Backward fill missing patches (fallback)
    last_patch = None
    for m in reversed(matches):
        if m.get("patch") is None:
            m["patch"] = last_patch
        else:
            last_patch = m["patch"]
            
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
        
        kast_val = ps.get("kill_assists_survived_traded")
        if isinstance(kast_val, str) and "%" in kast_val:
            kast_b = float(kast_val.replace("%", "")) / 100.0
        elif kast_val is not None and kast_val != "":
            try:
                kast_b = float(kast_val)
                if kast_b > 1.0:
                    kast_b = kast_b / 100.0
            except (ValueError, TypeError):
                kast_b = 0.70
        else:
            kast_b = 0.70
            
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
                    # ACS
                    acs_val = p.get('acs')
                    try:
                        acs_val = float(acs_val) if acs_val is not None and acs_val != "" else 0.0
                    except (ValueError, TypeError):
                        acs_val = 0.0
                        
                    # KAST
                    kast_val = p.get('kast')
                    if isinstance(kast_val, str) and '%' in kast_val:
                        kast_val = float(kast_val.replace('%', '')) / 100.0
                    elif kast_val is not None and kast_val != "":
                        try:
                            kast_val = float(kast_val)
                            if kast_val > 1.0:
                                kast_val = kast_val / 100.0
                        except (ValueError, TypeError):
                            kast_val = 0.70
                    else:
                        kast_val = 0.70
                        
                    # FK
                    fk_val = p.get('fk')
                    try:
                        fk_val = float(fk_val) if fk_val is not None and fk_val != "" else 0.0
                    except (ValueError, TypeError):
                        fk_val = 0.0
                        
                    # FD
                    fd_val = p.get('fd')
                    try:
                        fd_val = float(fd_val) if fd_val is not None and fd_val != "" else 0.0
                    except (ValueError, TypeError):
                        fd_val = 0.0
                        
                    # Kills
                    kills_val = p.get('kills')
                    try:
                        kills_val = float(kills_val) if kills_val is not None and kills_val != "" else 0.0
                    except (ValueError, TypeError):
                        kills_val = 0.0
                    
                    if p_name not in player_map_stats:
                        player_map_stats[p_name] = []
                    player_map_stats[p_name].append({
                        'acs': acs_val,
                        'kast': kast_val,
                        'fk': fk_val,
                        'fd': fd_val,
                        'kills': kills_val,
                        'rounds': rounds_count,
                        'agent': p.get('agent', '')
                    })
                    
        for p_name, stats_list in player_map_stats.items():
            avg_acs = sum(s['acs'] for s in stats_list) / len(stats_list)
            avg_kast = sum(s['kast'] for s in stats_list) / len(stats_list)
            total_fk = sum(s['fk'] for s in stats_list)
            total_fd = sum(s['fd'] for s in stats_list)
            total_kills = sum(s['kills'] for s in stats_list)
            total_rounds = sum(s['rounds'] for s in stats_list)
            
            fk_per_round = total_fk / total_rounds if total_rounds > 0 else 0.0
            fd_per_round = total_fd / total_rounds if total_rounds > 0 else 0.0
            avg_kpr = total_kills / total_rounds if total_rounds > 0 else 0.0
            
            from collections import Counter
            agent_counts = Counter(s['agent'] for s in stats_list if s.get('agent'))
            most_common_agent = agent_counts.most_common(1)[0][0] if agent_counts else ""
            
            player_performances.append({
                'player': p_name,
                'match_id': match_id,
                'timestamp': ts,
                'patch': m.get('patch', ''),
                'agent': most_common_agent,
                'acs': avg_acs,
                'kast': avg_kast,
                'duel_diff': fk_per_round - fd_per_round,
                'kpr': avg_kpr
            })
            
    df_player_perf = pd.DataFrame(player_performances)
    df_player_perf = df_player_perf.sort_values(by=["player", "timestamp"])
    
    # Initialize trackers
    tracker_kpr = BayesianSkillTracker(mu_prior=0.75, var_prior=0.04, var_obs=0.0225, tau_base_var=0.0025 / 30.0, lambda_val=0.001)
    tracker_acs = BayesianSkillTracker(mu_prior=200.0, var_prior=2500.0, var_obs=1600.0, tau_base_var=100.0 / 30.0, lambda_val=40.0)
    tracker_kast = BayesianSkillTracker(mu_prior=0.70, var_prior=0.01, var_obs=0.01, tau_base_var=0.0005 / 30.0, lambda_val=0.0002)
    tracker_duel = BayesianSkillTracker(mu_prior=0.0, var_prior=0.01, var_obs=0.01, tau_base_var=0.0005 / 30.0, lambda_val=0.0002)
    
    player_features_lookup = {}
    
    # Loop matches chronologically to calculate pre-match player state and update trackers
    for m in matches:
        match_id = m['match_id']
        
        # 1. Fetch pre-match state for players active in this match
        roster_a = set()
        roster_b = set()
        for map_data in m.get('maps', []):
            for p in map_data['players']['team1']:
                roster_a.add(p['name'])
            for p in map_data['players']['team2']:
                roster_b.add(p['name'])
                
        all_players = roster_a.union(roster_b)
        
        for p_name in all_players:
            kpr_mu, kpr_sigma = tracker_kpr.get_state(p_name)
            acs_mu, acs_sigma = tracker_acs.get_state(p_name)
            kast_mu, kast_sigma = tracker_kast.get_state(p_name)
            duel_mu, duel_sigma = tracker_duel.get_state(p_name)
            
            player_features_lookup[(p_name, match_id)] = {
                "kpr_mu": kpr_mu,
                "kpr_sigma": kpr_sigma,
                "acs_mu": acs_mu,
                "acs_sigma": acs_sigma,
                "kast_mu": kast_mu,
                "duel_mu": duel_mu
            }
            
        # 2. Update state with match results
        dt_datetime = m.get('timestamp')
        patch_str = None
        for map_data in m.get('maps', []):
            p_patch = map_data.get('patch')
            if p_patch:
                patch_str = p_patch
                break

        match_perf = df_player_perf[df_player_perf["match_id"] == match_id]
        for _, row in match_perf.iterrows():
            p_name = row["player"]
            y_kpr = float(row.get("kpr", 0.75))
            y_acs = float(row.get("acs", 200.0))
            y_kast = float(row.get("kast", 0.70))
            y_duel = float(row.get("duel_diff", 0.0))
            
            tracker_kpr.update(p_name, y_kpr, dt_datetime, patch_str)
            tracker_acs.update(p_name, y_acs, dt_datetime, patch_str)
            tracker_kast.update(p_name, y_kast, dt_datetime, patch_str)
            tracker_duel.update(p_name, y_duel, dt_datetime, patch_str)
            
    # Save the latest post-match states of the trackers to the Bayesian player ledger
    bayesian_ledger = {}
    for p_name in set(tracker_kpr.player_states.keys()).union(tracker_acs.player_states.keys()):
        k_mu, k_sigma = tracker_kpr.get_state(p_name)
        a_mu, a_sigma = tracker_acs.get_state(p_name)
        bayesian_ledger[p_name] = {
            "kpr_mu": k_mu,
            "kpr_sigma": k_sigma,
            "acs_mu": a_mu,
            "acs_sigma": a_sigma
        }
        
    bayesian_ledger_path = os.path.join(PROCESSED_DIR, "bayesian_player_ledger.json")
    with open(bayesian_ledger_path, "w", encoding="utf-8") as f:
        json.dump(bayesian_ledger, f, indent=4)
    logger.info("Saved latest active player states to bayesian_player_ledger.json")
    
    logger.info("Extracting team-match economy, comfort pick, and round closeness metrics...")
    
    # Load agent roles mapping
    agent_roles_path = os.path.join(RAW_DIR, "agent_roles.json")
    agent_roles = {}
    if os.path.exists(agent_roles_path):
        try:
            with open(agent_roles_path, "r", encoding="utf-8") as f:
                agent_roles = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load agent roles: {e}")

    # Track player global and agent ACS sums/counts for comfort pick differential
    player_global_stats = {}
    player_agent_stats = {}
    
    # Comfort pick diff and agent compositions lookups
    comfort_pick_lookup = {}
    agent_composition_lookup = {}
    
    # Pass 1: Chronological calculations for comfort pick, agent composition, and match-level team closeness
    team_performances = []
    
    for m in matches:
        match_id = m['match_id']
        ts = m['timestamp']
        team_a_name = m['team_a']
        team_b_name = m['team_b']
        
        # A. Comfort Pick Differential (strictly point-in-time)
        map_comfort_diffs_a = []
        map_comfort_diffs_b = []
        
        for map_data in m.get('maps', []):
            map_diffs_a = []
            map_diffs_b = []
            for team_key in ['team1', 'team2']:
                for p in map_data.get('players', {}).get(team_key, []):
                    p_name = p['name']
                    agent = p['agent']
                    acs = float(p['acs']) if (p.get('acs') and str(p['acs']).isdigit()) else 0.0
                    
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
                
        comfort_pick_lookup[match_id] = {
            'team_a': sum(map_comfort_diffs_a) / len(map_comfort_diffs_a) if map_comfort_diffs_a else 0.0,
            'team_b': sum(map_comfort_diffs_b) / len(map_comfort_diffs_b) if map_comfort_diffs_b else 0.0
        }
        
        # B. Agent Composition Tensors (Counts of roles)
        map_roles_a = []
        map_roles_b = []
        
        for map_data in m.get('maps', []):
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
            
        agent_composition_lookup[match_id] = {
            'team_a': avg_role_counts(map_roles_a),
            'team_b': avg_role_counts(map_roles_b)
        }
        
        # Update running stats for players after calculating features
        for map_data in m.get('maps', []):
            for team_key in ['team1', 'team2']:
                for p in map_data.get('players', {}).get(team_key, []):
                    p_name = p['name']
                    agent = p['agent']
                    acs = float(p['acs']) if (p.get('acs') and str(p['acs']).isdigit()) else 0.0
                    if acs > 0:
                        if p_name not in player_global_stats:
                            player_global_stats[p_name] = {'sum_acs': 0, 'count': 0}
                        player_global_stats[p_name]['sum_acs'] += acs
                        player_global_stats[p_name]['count'] += 1
                        
                        if (p_name, agent) not in player_agent_stats:
                            player_agent_stats[(p_name, agent)] = {'sum_acs': 0, 'count': 0}
                        player_agent_stats[(p_name, agent)]['sum_acs'] += acs
                        player_agent_stats[(p_name, agent)]['count'] += 1

        # C. Match-level metrics: clutch, thrifty, flawless wins
        total_rounds = 0
        total_clutches_a = 0
        total_clutches_b = 0
        total_thrifty_a = 0
        total_thrifty_b = 0
        total_flawless_a = 0
        total_flawless_b = 0
        
        map_loadouts_a = []
        map_loadouts_b = []
        
        for map_data in m.get('maps', []):
            rounds_count = len(map_data.get('rounds', []))
            if rounds_count == 0:
                score = map_data.get('score', {})
                rounds_count = int(score.get('team1', 0)) + int(score.get('team2', 0))
                if rounds_count == 0:
                    rounds_count = 24
            
            total_rounds += rounds_count
            
            # 1. Economy
            avg_a, avg_b = parse_map_economy(map_data.get('economy', []), team_a_name, team_b_name)
            if avg_a > 0:
                map_loadouts_a.append(avg_a)
            if avg_b > 0:
                map_loadouts_b.append(avg_b)
                
            # 2. Clutch wins
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
                    
            # 3. Thrifty wins
            econ_rows = map_data.get('economy', [])
            for econ_row in econ_rows:
                team_raw = econ_row.get('0', '')
                weight = match_team(team_raw, team_a_name, team_b_name)
                thrifty = parse_econ_cell_wins(econ_row.get('2')) + parse_econ_cell_wins(econ_row.get('3'))
                if weight == 1:
                    total_thrifty_a += thrifty
                elif weight == -1:
                    total_thrifty_b += thrifty
                    
            # 4. Flawless wins (proxy based on score wins * 0.15)
            score_dict = map_data.get('score', {})
            win_a = int(score_dict.get('team1', 0))
            win_b = int(score_dict.get('team2', 0))
            total_flawless_a += int(win_a * 0.15)
            total_flawless_b += int(win_b * 0.15)
            
        match_loadout_a = sum(map_loadouts_a) / len(map_loadouts_a) if map_loadouts_a else 20000.0
        match_loadout_b = sum(map_loadouts_b) / len(map_loadouts_b) if map_loadouts_b else 20000.0
        
        # Calculate rates per round
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
        
    # Pass 2: Calculate rolling averages and shift(1)
    df_team_perf = pd.DataFrame(team_performances)
    df_team_perf = df_team_perf.sort_values(by=["team", "timestamp"])
    
    # Compute rolling averages
    for col in ['loadout', 'clutch_rate', 'thrifty_rate', 'flawless_rate']:
        df_team_perf[f"{col}_roll"] = df_team_perf.groupby("team")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_team_perf[f"{col}_roll_shifted"] = df_team_perf.groupby("team")[f"{col}_roll"].shift(1)
        
    # Fill missing values
    df_team_perf["loadout_roll_shifted"] = df_team_perf["loadout_roll_shifted"].fillna(20000.0)
    df_team_perf["clutch_rate_roll_shifted"] = df_team_perf["clutch_rate_roll_shifted"].fillna(0.05)
    df_team_perf["thrifty_rate_roll_shifted"] = df_team_perf["thrifty_rate_roll_shifted"].fillna(0.02)
    df_team_perf["flawless_rate_roll_shifted"] = df_team_perf["flawless_rate_roll_shifted"].fillna(0.05)
    
    # Convert back to lookups
    team_features_lookup = df_team_perf.set_index(["team", "match_id"])[
        ["loadout_roll_shifted", "clutch_rate_roll_shifted", "thrifty_rate_roll_shifted", "flawless_rate_roll_shifted"]
    ].to_dict(orient="index")
    
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
        for idx in range(5):
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
        
        # Function to compute aggregate player-level prior Bayesian features
        def get_roster_bayesian_features(roster):
            kpr_mu_list, kpr_sigma_list = [], []
            acs_mu_list, acs_sigma_list = [], []
            kast_mu_list, duel_mu_list = [], []
            
            for p_name in roster:
                p_feat = player_features_lookup.get((p_name, match_id))
                if p_feat is not None:
                    kpr_mu_list.append(p_feat["kpr_mu"])
                    kpr_sigma_list.append(p_feat["kpr_sigma"])
                    acs_mu_list.append(p_feat["acs_mu"])
                    acs_sigma_list.append(p_feat["acs_sigma"])
                    kast_mu_list.append(p_feat["kast_mu"])
                    duel_mu_list.append(p_feat["duel_mu"])
                else:
                    p_base = baseline_lookup.get(p_name, {"acs": 200.0, "kast": 0.70, "duel_diff": 0.0})
                    kpr_mu_list.append(0.75)
                    kpr_sigma_list.append(0.20)
                    acs_mu_list.append(p_base["acs"])
                    acs_sigma_list.append(50.0)
                    kast_mu_list.append(p_base["kast"])
                    duel_mu_list.append(p_base["duel_diff"])
                    
            return (
                sum(kpr_mu_list) / len(kpr_mu_list) if kpr_mu_list else 0.75,
                sum(kpr_sigma_list) / len(kpr_sigma_list) if kpr_sigma_list else 0.20,
                sum(acs_mu_list) / len(acs_mu_list) if acs_mu_list else 200.0,
                sum(acs_sigma_list) / len(acs_sigma_list) if acs_sigma_list else 50.0,
                sum(kast_mu_list) / len(kast_mu_list) if kast_mu_list else 0.70,
                sum(duel_mu_list) / len(duel_mu_list) if duel_mu_list else 0.0
            )
            
        ta_kpr_mu, ta_kpr_sigma, ta_acs_mu, ta_acs_sigma, ta_kast_mu, ta_duel_mu = get_roster_bayesian_features(roster_a)
        tb_kpr_mu, tb_kpr_sigma, tb_acs_mu, tb_acs_sigma, tb_kast_mu, tb_duel_mu = get_roster_bayesian_features(roster_b)
        
        # Look up team economy and round closeness features
        ta_feat = team_features_lookup.get((team_a_name, match_id), {})
        tb_feat = team_features_lookup.get((team_b_name, match_id), {})
        
        ta_loadout = ta_feat.get("loadout_roll_shifted", 20000.0)
        ta_clutch = ta_feat.get("clutch_rate_roll_shifted", 0.05)
        ta_thrifty = ta_feat.get("thrifty_rate_roll_shifted", 0.02)
        ta_flawless = ta_feat.get("flawless_rate_roll_shifted", 0.05)
        
        tb_loadout = tb_feat.get("loadout_roll_shifted", 20000.0)
        tb_clutch = tb_feat.get("clutch_rate_roll_shifted", 0.05)
        tb_thrifty = tb_feat.get("thrifty_rate_roll_shifted", 0.02)
        tb_flawless = tb_feat.get("flawless_rate_roll_shifted", 0.05)
        
        # Look up agent comfort differentials
        comfort_a = comfort_pick_lookup.get(match_id, {}).get('team_a', 0.0)
        comfort_b = comfort_pick_lookup.get(match_id, {}).get('team_b', 0.0)
        
        # Look up agent compositions (role counts)
        comp_a = agent_composition_lookup.get(match_id, {}).get('team_a', {'Duelist': 0.0, 'Controller': 0.0, 'Initiator': 0.0, 'Sentinel': 0.0})
        comp_b = agent_composition_lookup.get(match_id, {}).get('team_b', {'Duelist': 0.0, 'Controller': 0.0, 'Initiator': 0.0, 'Sentinel': 0.0})
        
        # Target variable column
        y_target = 1 if m["teams"][0]["is_winner"] else 0
        
        row = {
            "match_id": match_id,
            "timestamp": ts,
            "team_a_name": team_a_name,
            "team_b_name": team_b_name,
            "team_a_kpr_mu": ta_kpr_mu,
            "team_a_kpr_sigma": ta_kpr_sigma,
            "team_a_acs_mu": ta_acs_mu,
            "team_a_acs_sigma": ta_acs_sigma,
            "team_a_historical_acs_ema": ta_acs_mu,
            "team_a_historical_kast_ema": ta_kast_mu,
            "team_a_historical_duel_diff": ta_duel_mu,
            "team_a_historical_avg_loadout": ta_loadout,
            "team_a_historical_clutch_rate": ta_clutch,
            "team_a_historical_thrifty_rate": ta_thrifty,
            "team_a_historical_flawless_rate": ta_flawless,
            "team_a_comfort_pick_differential": comfort_a,
            "team_a_duelist_count": comp_a.get('Duelist', 0.0),
            "team_a_controller_count": comp_a.get('Controller', 0.0),
            "team_a_initiator_count": comp_a.get('Initiator', 0.0),
            "team_a_sentinel_count": comp_a.get('Sentinel', 0.0),
            "team_b_kpr_mu": tb_kpr_mu,
            "team_b_kpr_sigma": tb_kpr_sigma,
            "team_b_acs_mu": tb_acs_mu,
            "team_b_acs_sigma": tb_acs_sigma,
            "team_b_historical_acs_ema": tb_acs_mu,
            "team_b_historical_kast_ema": tb_kast_mu,
            "team_b_historical_duel_diff": tb_duel_mu,
            "team_b_historical_avg_loadout": tb_loadout,
            "team_b_historical_clutch_rate": tb_clutch,
            "team_b_historical_thrifty_rate": tb_thrifty,
            "team_b_historical_flawless_rate": tb_flawless,
            "team_b_comfort_pick_differential": comfort_b,
            "team_b_duelist_count": comp_b.get('Duelist', 0.0),
            "team_b_controller_count": comp_b.get('Controller', 0.0),
            "team_b_initiator_count": comp_b.get('Initiator', 0.0),
            "team_b_sentinel_count": comp_b.get('Sentinel', 0.0),
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


class BayesianSkillTracker:
    def __init__(self, mu_prior: float, var_prior: float, var_obs: float, tau_base_var: float, lambda_val: float):
        self.mu_prior = mu_prior
        self.var_prior = var_prior
        self.var_obs = var_obs
        self.tau_base_var = tau_base_var # Variance drift rate per day
        self.lambda_val = lambda_val     # Patch transition variance scalar
        self.player_states = {}
        self.player_last_time = {}
        self.player_last_patch = {}

    def get_state(self, player_id: str) -> Tuple[float, float]:
        """Returns (mu, std_dev) for the player."""
        if player_id not in self.player_states:
            self.player_states[player_id] = (self.mu_prior, self.var_prior)
        mu, var = self.player_states[player_id]
        return float(mu), float(np.sqrt(var))

    def update(self, player_id: str, y: float, dt_datetime=None, patch_str: str = None):
        """Observation update followed by continuous-time and patch-aware temporal transition update."""
        if player_id not in self.player_states:
            self.player_states[player_id] = (self.mu_prior, self.var_prior)
            
        mu_old, var_old = self.player_states[player_id]
        
        # 1. Observation Update
        denom = var_old + self.var_obs
        mu_new = (mu_old * self.var_obs + y * var_old) / denom
        var_new = (var_old * self.var_obs) / denom
        
        # 2. Continuous-time & Patch-aware Time Transition Update
        delta_t_days = 0.0
        if player_id in self.player_last_time and dt_datetime is not None:
            delta_time = dt_datetime - self.player_last_time[player_id]
            delta_t_days = max(0.0, delta_time.total_seconds() / 86400.0)
            
        patch_severity = 0.0
        if player_id in self.player_last_patch and patch_str is not None:
            last_p = self.player_last_patch[player_id]
            if last_p and patch_str and last_p != patch_str:
                patch_severity = 1.0
                
        var_next = var_new + (self.tau_base_var * delta_t_days) + (self.lambda_val * patch_severity)
        
        self.player_states[player_id] = (mu_new, var_next)
        if dt_datetime is not None:
            self.player_last_time[player_id] = dt_datetime
        if patch_str:
            self.player_last_patch[player_id] = patch_str


def compute_bayesian_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Ensure dataframe is sorted by timestamp and player
    if "match_timestamp" in df.columns:
        df = df.sort_values(by=["player_id", "match_timestamp"])
    df.reset_index(drop=True, inplace=True)
    
    # Initialize trackers with daily variance drift rates and patch severity scalars
    tracker_kpr = BayesianSkillTracker(mu_prior=0.75, var_prior=0.04, var_obs=0.0225, tau_base_var=0.0025 / 30.0, lambda_val=0.001)
    tracker_acs = BayesianSkillTracker(mu_prior=200.0, var_prior=2500.0, var_obs=1600.0, tau_base_var=100.0 / 30.0, lambda_val=40.0)
    
    kpr_mus, kpr_sigmas = [], []
    acs_mus, acs_sigmas = [], []
    
    for idx, row in df.iterrows():
        p_id = str(row.get("player_id", row.get("player", "Unknown")))
        
        y_kpr = float(row.get("kpr", 0.75))
        y_acs = float(row.get("acs", 200.0))
        
        # Parse match timestamp
        ts = row.get("match_timestamp")
        if isinstance(ts, str):
            try:
                dt_datetime = datetime.fromisoformat(ts)
            except ValueError:
                dt_datetime = datetime.now()
        elif hasattr(ts, "to_pydatetime"):
            dt_datetime = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            dt_datetime = ts
        else:
            dt_datetime = datetime.now()
            
        patch_str = str(row.get("patch_version", ""))
        
        tracker_kpr.update(p_id, y_kpr, dt_datetime, patch_str)
        tracker_acs.update(p_id, y_acs, dt_datetime, patch_str)
        
        kpr_mu, kpr_sigma = tracker_kpr.get_state(p_id)
        acs_mu, acs_sigma = tracker_acs.get_state(p_id)
        
        kpr_mus.append(kpr_mu)
        kpr_sigmas.append(kpr_sigma)
        acs_mus.append(acs_mu)
        acs_sigmas.append(acs_sigma)
        
    df["player_kpr_mu"] = kpr_mus
    df["player_kpr_sigma"] = kpr_sigmas
    df["player_acs_mu"] = acs_mus
    df["player_acs_sigma"] = acs_sigmas
    
    logger.info("Task 1.3 complete: Computed Bayesian features.")
    return df


def generate_odr_matrix(
    df: pd.DataFrame, 
    target_col: str = "kpr", 
    alpha_ridge: float = 1.0
) -> Dict[str, float]:
    from typing import Dict
    from sklearn.linear_model import Ridge
    
    all_teams = sorted(list(set(df["team_name"]).union(set(df["opponent_team_name"]))))
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    num_teams = len(all_teams)
    
    num_samples = len(df)
    X = np.zeros((num_samples, 2 * num_teams))
    y = df[target_col].values
    
    for row_idx, (_, row) in enumerate(df.iterrows()):
        off_idx = team_to_idx[row["team_name"]]
        def_idx = team_to_idx[row["opponent_team_name"]]
        
        X[row_idx, off_idx] = 1.0                # Offense_i
        X[row_idx, num_teams + def_idx] = -1.0     # -Defense_j
        
    ridge = Ridge(alpha=alpha_ridge, fit_intercept=True)
    ridge.fit(X, y)
    
    mu_league = ridge.intercept_
    def_coefs = ridge.coef_[num_teams:] # Coefficients corresponding to -Defense_j
    
    odr_matrix = {team: float(def_coefs[idx]) for team, idx in team_to_idx.items()}
    
    logger.info("Task 1.4 complete: ODR Matrix solved for %d teams across %d observations. Baseline mu_league = %.4f", 
                num_teams, num_samples, mu_league)
    for team, odr in odr_matrix.items():
        logger.debug("Team: %-15s ODR (Defensive Suppression): %+.4f KPR", team, odr)
        
    return odr_matrix


def attach_odr_features(df: pd.DataFrame, odr_matrix: Dict[str, float]) -> pd.DataFrame:
    from typing import Dict
    df = df.copy()
    df["opponent_odr"] = df["opponent_team_name"].map(odr_matrix).fillna(0.0)
    return df


if __name__ == "__main__":
    build_feature_store()
