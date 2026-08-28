import os
import json
import logging

logger = logging.getLogger("team_registry")

# Static mapping for standard VCT team aliases and VLR team IDs
KNOWN_TEAMS = {
    "Paper Rex": {"vlr_id": 624, "name": "Paper Rex", "aliases": ["PRX", "Paper Rex"]},
    "LEVIATÁN": {"vlr_id": 2359, "name": "LEVIATÁN", "aliases": ["LEV", "Leviatán", "Leviatan", "LEVIATAN"]},
    "Sentinels": {"vlr_id": 2, "name": "Sentinels", "aliases": ["SEN", "Sentinels"]},
    "Fnatic": {"vlr_id": 2593, "name": "Fnatic", "aliases": ["FNC", "Fnatic"]},
    "Team Liquid": {"vlr_id": 474, "name": "Team Liquid", "aliases": ["TL", "Liquid", "Team Liquid"]},
    "Gen.G": {"vlr_id": 17, "name": "Gen.G", "aliases": ["GEN", "Gen.G Esports", "Gen.G"]},
    "DRX": {"vlr_id": 8185, "name": "DRX", "aliases": ["DRX", "Vision Strikers"]},
    "BBL Esports": {"vlr_id": 397, "name": "BBL Esports", "aliases": ["BBL", "BBL Esports"]},
    "Natus Vincere": {"vlr_id": 4915, "name": "Natus Vincere", "aliases": ["NAVI", "NAVi", "Natus Vincere"]},
    "Team Heretics": {"vlr_id": 1001, "name": "Team Heretics", "aliases": ["TH", "Heretics", "Team Heretics"]},
    "100 Thieves": {"vlr_id": 120, "name": "100 Thieves", "aliases": ["100T", "100 Thieves"]},
    "Cloud9": {"vlr_id": 188, "name": "Cloud9", "aliases": ["C9", "Cloud9"]},
    "NRG": {"vlr_id": 1034, "name": "NRG", "aliases": ["NRG", "NRG Esports"]},
    "LOUD": {"vlr_id": 6961, "name": "LOUD", "aliases": ["LOUD"]},
    "Kru Esports": {"vlr_id": 2355, "name": "Kru Esports", "aliases": ["KRU", "KRÜ Esports", "Kru Esports"]},
    "FURIA": {"vlr_id": 2406, "name": "FURIA", "aliases": ["FUR", "Furia", "FURIA Esports"]},
    "T1": {"vlr_id": 14, "name": "T1", "aliases": ["T1"]},
    "ZETA DIVISION": {"vlr_id": 5448, "name": "ZETA DIVISION", "aliases": ["ZETA", "ZETA DIVISION"]},
    "Team Secret": {"vlr_id": 6199, "name": "Team Secret", "aliases": ["TS", "Secret", "Team Secret"]},
    "Talon Esports": {"vlr_id": 8304, "name": "Talon Esports", "aliases": ["TLN", "Talon", "Talon Esports"]},
    "EDward Gaming": {"vlr_id": 1120, "name": "EDward Gaming", "aliases": ["EDG", "EDward Gaming"]},
    "FunPlus Phoenix": {"vlr_id": 11328, "name": "FunPlus Phoenix", "aliases": ["FPX", "FunPlus Phoenix"]},
    "Trace Esports": {"vlr_id": 12684, "name": "Trace Esports", "aliases": ["TE", "Trace Esports"]}
}

def resolve_team_info(team_name: str) -> dict:
    """
    Resolves a team name or alias to canonical info containing:
    { "vlr_id": int | None, "name": str, "aliases": list[str] }
    """
    if not team_name:
        return {"vlr_id": None, "name": "Unknown", "aliases": []}
        
    cleaned = team_name.strip()
    
    # 1. Exact name lookup
    if cleaned in KNOWN_TEAMS:
        return KNOWN_TEAMS[cleaned]
        
    # 2. Alias lookup
    for t_info in KNOWN_TEAMS.values():
        if any(alias.lower() == cleaned.lower() for alias in t_info["aliases"]):
            return t_info
            
    # 3. Substring fallback
    for t_info in KNOWN_TEAMS.values():
        if t_info["name"].lower() in cleaned.lower() or cleaned.lower() in t_info["name"].lower():
            return t_info
            
    return {"vlr_id": None, "name": cleaned, "aliases": [cleaned]}

def resolve_team_id(team_name: str) -> int | None:
    """Returns VLR numeric ID for a team name, or None if unknown."""
    return resolve_team_info(team_name).get("vlr_id")
