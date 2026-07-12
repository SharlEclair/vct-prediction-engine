import os
import json
import glob
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_match_data")

def clean_numeric(val):
    if val is None or str(val).strip() == "" or str(val).strip().lower() == "null":
        return None
    val_str = str(val).strip()
    if "%" in val_str:
        try:
            return round(float(val_str.replace("%", "")) / 100.0, 4)
        except ValueError:
            return None
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

def clean_player_dict(p: dict):
    # Standard stats mapping
    stat_keys = ["kills", "deaths", "assists", "acs", "rating", "adr", "hs_pct", "kast", "fk", "fd", "average_combat_score"]
    for k in stat_keys:
        if k in p:
            p[k] = clean_numeric(p[k])

def clean_single_file(fpath: str):
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_segments = data.get("data", {}).get("segments", [])
    if not raw_segments:
        return False
        
    is_new_schema = "team1" in raw_segments[0]
    modified = False
    
    if is_new_schema:
        team_a = raw_segments[0].get("team1")
        team_b = raw_segments[0].get("team2")
        
        for raw_map in raw_segments:
            # 1. Clean mixed map field
            map_field = raw_map.get("map", "")
            if map_field and ("PICK" in map_field or "DECIDER" in map_field or ":" in map_field):
                match = re.match(r'^([A-Za-z0-9\s\-\_]+?)\s+(PICK|DECIDER|BAN)?\s*([0-9\:]+)?$', map_field)
                if match:
                    raw_map["map"] = match.group(1).strip()
                    raw_map["picked_by"] = match.group(2) or ""
                    raw_map["duration"] = match.group(3) or ""
                    modified = True
            
            # 2. Clean empty rounds padding
            rounds = raw_map.get("round_history", [])
            clean_rounds = [r for r in rounds if r.get("winner") != "" and r.get("side") != ""]
            if len(clean_rounds) != len(rounds):
                raw_map["round_history"] = clean_rounds
                modified = True
                
            # 3. Clean winner contradictions
            t1_wins = sum(1 for r in clean_rounds if r.get("winner") == team_a)
            t2_wins = sum(1 for r in clean_rounds if r.get("winner") == team_b)
            if max(t1_wins, t2_wins) >= 13:
                expected_winner = team_a if t1_wins > t2_wins else team_b
                if raw_map.get("winner") != expected_winner:
                    raw_map["winner"] = expected_winner
                    modified = True
            
            # 4. Clean players stats
            players = raw_map.get("players", [])
            for p in players:
                clean_player_dict(p)
                modified = True
                
            # 5. Clean patch Unknown
            if raw_map.get("patch") == "Unknown":
                raw_map["patch"] = None
                modified = True
                
    else:
        # Old schema
        segment = raw_segments[0]
        teams = segment.get("teams", [])
        if len(teams) == 2:
            team_a = teams[0].get("name")
            team_b = teams[1].get("name")
            
            # Check maps list
            maps = segment.get("maps", [])
            map_wins_a = 0
            map_wins_b = 0
            
            for m in maps:
                # 1. Clean rounds padding
                rounds = m.get("rounds", [])
                clean_rounds = [r for r in rounds if r.get("winner") != "" and r.get("side") != ""]
                if len(clean_rounds) != len(rounds):
                    m["rounds"] = clean_rounds
                    modified = True
                    
                # 2. Re-tally map scores
                t1_wins = sum(1 for r in clean_rounds if r.get("winner") == "team1")
                t2_wins = sum(1 for r in clean_rounds if r.get("winner") == "team2")
                m["score"] = {"team1": t1_wins, "team2": t2_wins}
                
                if t1_wins > t2_wins:
                    map_wins_a += 1
                elif t2_wins > t1_wins:
                    map_wins_b += 1
                    
                # 3. Clean players stats
                players = m.get("players", {})
                for p in players.get("team1", []) + players.get("team2", []):
                    clean_player_dict(p)
                    modified = True
                    
            # 4. Clean series winner contradictions
            is_winner_a = map_wins_a > map_wins_b
            if teams[0].get("is_winner") != is_winner_a:
                teams[0]["is_winner"] = is_winner_a
                teams[1]["is_winner"] = not is_winner_a
                modified = True
                
            # 5. Clean patch Unknown
            if segment.get("patch") == "Unknown":
                segment["patch"] = None
                modified = True
                
    if modified:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    return False

def clean_all_matches():
    workspace_dir = "C:/Users/91704/Desktop/vct-prediction-model"
    files = glob.glob(os.path.join(workspace_dir, "data/raw/match_*.json"))
    logger.info(f"Loaded {len(files)} raw matches for cleanup...")
    
    updated_count = 0
    for f in files:
        if clean_single_file(f):
            updated_count += 1
            
    logger.info(f"Cleanup complete! Updated {updated_count} files.")

if __name__ == "__main__":
    clean_all_matches()
