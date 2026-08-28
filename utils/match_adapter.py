import re
import logging
from datetime import datetime

logger = logging.getLogger("match_adapter")

ADAPTER_VERSION = "1.0"

def parse_match_date(date_val) -> datetime:
    """Helper to parse raw date strings or timestamps into a naive datetime object."""
    if isinstance(date_val, datetime):
        return date_val.replace(tzinfo=None)
    if not date_val:
        return datetime(2025, 1, 1)
        
    date_str = str(date_val).strip()
    
    # ISO UTC format (e.g. 2026-07-26T11:00:00Z)
    if "T" in date_str:
        clean_iso = date_str.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(clean_iso).replace(tzinfo=None)
        except ValueError:
            pass
            
    # Remove Patch info if present
    date_clean = re.sub(r'\s*Patch\s+[0-9.]+', '', date_str, flags=re.IGNORECASE).strip()
    
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%A, %B %d %I:%M %p",
        "%A, %B %d, %Y",
        "%B %d, %Y"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_clean, fmt)
            if dt.year == 1900:
                # Infer year if missing
                dt = dt.replace(year=2026 if "2026" in date_str else 2025)
            return dt
        except ValueError:
            pass
            
    return datetime(2025, 1, 1)


def validate_normalized_match(match: dict) -> dict:
    """
    Defensive validation layer for normalized match dictionaries.
    Ensures required fields are present and well-formed.
    """
    required_top_keys = ["match_id", "date", "teams", "score", "maps", "_adapter"]
    for key in required_top_keys:
        if key not in match:
            raise ValueError(f"Normalized match missing required key: '{key}'")
            
    if not isinstance(match["teams"], dict) or "team1" not in match["teams"] or "team2" not in match["teams"]:
        raise ValueError("Normalized match missing valid 'teams' dictionary")
        
    if not isinstance(match["maps"], list):
        raise ValueError("Normalized match 'maps' must be a list")
        
    for idx, m in enumerate(match["maps"]):
        if not isinstance(m, dict):
            raise ValueError(f"Map index {idx} is not a valid dictionary")
        for m_key in ["map_id", "map_name", "winner", "score"]:
            if m_key not in m:
                raise ValueError(f"Map index {idx} ({m.get('map_name')}) missing required key '{m_key}'")
                
    # Add top-level convenience aliases for downstream consumers
    t1 = match["teams"]["team1"]
    t2 = match["teams"]["team2"]
    score = match.get("score", {})
    match.setdefault("team_a", t1)
    match.setdefault("team_b", t2)
    match.setdefault("score_a", score.get("team1_score", score.get(t1, 0)))
    match.setdefault("score_b", score.get("team2_score", score.get(t2, 0)))
    
    # Derive top-level match winner from map wins
    if not match.get("winner"):
        sa = match["score_a"]
        sb = match["score_b"]
        if isinstance(sa, int) and isinstance(sb, int):
            if sa > sb:
                match["winner"] = t1
            elif sb > sa:
                match["winner"] = t2
            else:
                match["winner"] = ""
        else:
            match["winner"] = ""
    
    # Add timestamp alias
    match.setdefault("timestamp", match.get("date"))
    
    return match


def normalize_v1(content: dict) -> dict:
    """Normalizes Schema v1.0 match content into canonical structure."""
    ovw = content.get("overview", {})
    meta = ovw.get("metadata", {})
    
    match_id = str(content.get("match_id", "unknown"))
    date_dt = parse_match_date(meta.get("date"))
    patch = meta.get("patch") or ""
    
    t1 = meta.get("teams", {}).get("team1", "Team 1")
    t2 = meta.get("teams", {}).get("team2", "Team 2")
    
    score_meta = meta.get("score", {})
    t1_s = score_meta.get("team1_score", 0)
    t2_s = score_meta.get("team2_score", 0)
    score_dict = {
        "team1_score": t1_s,
        "team2_score": t2_s,
        t1: score_meta.get(t1, t1_s),
        t2: score_meta.get(t2, t2_s)
    }
    
    maps_list = []
    for seg in ovw.get("segments", []):
        m_id = seg.get("map_id", "unknown")
        m_name = seg.get("map_name", "Unknown")
        winner = seg.get("winner", "")
        
        m_score = seg.get("score", {})
        ms_1 = m_score.get("team1_score", 0)
        ms_2 = m_score.get("team2_score", 0)
        map_score_dict = {
            "team1_score": ms_1,
            "team2_score": ms_2,
            t1: m_score.get(t1, ms_1),
            t2: m_score.get(t2, ms_2)
        }
        
        maps_list.append({
            "map_id": m_id,
            "map_name": m_name,
            "winner": winner,
            "score": map_score_dict,
            "players": seg.get("players", []),
            "round_history": seg.get("round_history", []),
            "vetoes": seg.get("vetoes", []),
            "composition": seg.get("composition", {})
        })
        
    return {
        "match_id": match_id,
        "date": date_dt,
        "patch": patch,
        "teams": {
            "team1": t1,
            "team2": t2
        },
        "score": score_dict,
        "maps": maps_list,
        "performance": content.get("performance"),
        "economy": content.get("economy"),
        "_adapter": {
            "version": ADAPTER_VERSION,
            "source_schema": "1.0"
        }
    }


def normalize_gen2(content: dict) -> dict:
    """Normalizes Gen 2 (Intermediate Scraper) match content into canonical structure."""
    raw_segments = content.get("data", {}).get("segments", [])
    if not raw_segments and isinstance(content.get("segments"), list):
        raw_segments = content.get("segments")
        
    seg0 = raw_segments[0]
    t1 = seg0.get("team1", "Team 1")
    t2 = seg0.get("team2", "Team 2")
    
    raw_date = seg0.get("date", "")
    date_dt = parse_match_date(raw_date)
    patch_match = re.search(r'Patch\s+([0-9.]+)', str(raw_date))
    patch = patch_match.group(1).strip() if patch_match else ""
    
    match_id = str(seg0.get("match_id", "unknown"))
    
    maps_list = []
    t1_map_wins = 0
    t2_map_wins = 0
    
    for raw_map in raw_segments:
        m_name_raw = raw_map.get("map", "Unknown")
        m_name = re.split(r'\s+(?:PICK|DECIDER|BAN)\b', str(m_name_raw))[0].strip()
        m_id = re.sub(r'[^a-z0-9_]', '', m_name.lower().replace(' ', '_'))
        
        rounds = raw_map.get("round_history", [])
        w = raw_map.get("winner", "")
        
        s1 = sum(1 for r in rounds if r.get("winner") == t1)
        s2 = sum(1 for r in rounds if r.get("winner") == t2)
        if s1 == 0 and s2 == 0:
            if w == t1:
                s1, s2 = 13, 9
            elif w == t2:
                s1, s2 = 9, 13
                
        if s1 > s2:
            t1_map_wins += 1
            map_winner = t1
        elif s2 > s1:
            t2_map_wins += 1
            map_winner = t2
        else:
            map_winner = w
            
        maps_list.append({
            "map_id": m_id,
            "map_name": m_name,
            "winner": map_winner,
            "score": {
                "team1_score": s1,
                "team2_score": s2,
                t1: s1,
                t2: s2
            },
            "players": raw_map.get("players", []),
            "round_history": rounds,
            "vetoes": raw_map.get("vetoes", []),
            "composition": raw_map.get("composition", {})
        })
        
    match_score = {
        "team1_score": t1_map_wins,
        "team2_score": t2_map_wins,
        t1: t1_map_wins,
        t2: t2_map_wins
    }
    
    return {
        "match_id": match_id,
        "date": date_dt,
        "patch": patch,
        "teams": {
            "team1": t1,
            "team2": t2
        },
        "score": match_score,
        "maps": maps_list,
        "performance": None,
        "economy": None,
        "_adapter": {
            "version": ADAPTER_VERSION,
            "source_schema": "gen2"
        }
    }


def normalize_gen1(content: dict) -> dict:
    """Normalizes Gen 1 (Legacy API) match content into canonical structure."""
    seg = content["data"]["segments"][0]
    
    teams_list = seg.get("teams", [])
    if len(teams_list) >= 2:
        t1 = teams_list[0].get("name", "Team 1")
        t2 = teams_list[1].get("name", "Team 2")
        t1_s = int(teams_list[0].get("score") or 0)
        t2_s = int(teams_list[1].get("score") or 0)
    else:
        t1 = seg.get("team1", "Team 1")
        t2 = seg.get("team2", "Team 2")
        t1_s = 0
        t2_s = 0
        
    date_dt = parse_match_date(seg.get("date"))
    patch_match = re.search(r'Patch\s+([0-9.]+)', str(seg.get("date", "")))
    patch = patch_match.group(1).strip() if patch_match else ""
    
    match_id = str(seg.get("match_id") or seg.get("id") or "unknown")
    
    # Parse map_vetos string if present
    vetoes = []
    map_vetos_str = seg.get("map_vetos", "")
    if map_vetos_str:
        vetoes = [t.strip() for t in map_vetos_str.split(";") if t.strip()]
    elif "vetoes" in seg and isinstance(seg["vetoes"], list):
        vetoes = [str(v).strip() for v in seg["vetoes"] if v]
        
    maps_list = []
    for raw_m in seg.get("maps", []):
        m_name = raw_m.get("map_name") or raw_m.get("map") or "Unknown"
        m_id = re.sub(r'[^a-z0-9_]', '', m_name.lower().replace(' ', '_'))
        
        m_score = raw_m.get("score", {})
        ms1 = int(m_score.get("team1") or m_score.get(t1) or 0)
        ms2 = int(m_score.get("team2") or m_score.get(t2) or 0)
        m_winner = t1 if ms1 > ms2 else (t2 if ms2 > ms1 else raw_m.get("winner", ""))
        
        maps_list.append({
            "map_id": m_id,
            "map_name": m_name,
            "winner": m_winner,
            "score": {
                "team1_score": ms1,
                "team2_score": ms2,
                t1: ms1,
                t2: ms2
            },
            "players": raw_m.get("players", []),
            "round_history": raw_m.get("round_history", []),
            "vetoes": vetoes,
            "composition": raw_m.get("composition", {})
        })
        
    return {
        "match_id": match_id,
        "date": date_dt,
        "patch": patch,
        "teams": {
            "team1": t1,
            "team2": t2
        },
        "score": {
            "team1_score": t1_s,
            "team2_score": t2_s,
            t1: t1_s,
            t2: t2_s
        },
        "maps": maps_list,
        "performance": None,
        "economy": None,
        "_adapter": {
            "version": ADAPTER_VERSION,
            "source_schema": "gen1"
        }
    }


def normalize_match(content: dict) -> dict:
    """
    Main Adapter API: Ingests raw JSON content from Schema v1.0, Gen 2, or Gen 1
    and returns a defensively validated, canonical normalized match dictionary.
    """
    if not isinstance(content, dict):
        raise ValueError("Match content must be a valid dictionary")
        
    if content.get("schema_version") == "1.0":
        normalized = normalize_v1(content)
    elif "data" in content and "segments" in content["data"]:
        raw_segments = content["data"]["segments"]
        if raw_segments and isinstance(raw_segments, list) and "team1" in raw_segments[0]:
            normalized = normalize_gen2(content)
        else:
            normalized = normalize_gen1(content)
    elif "overview" in content:
        normalized = normalize_v1(content)
    else:
        raise ValueError("Unknown match schema format")
        
    return validate_normalized_match(normalized)
