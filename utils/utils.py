"""
Utility Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 5).

Provides centralized, error-handled loading functions for YAML configuration and JSON slate payloads.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Safely load engine parameters from central YAML configuration file.
    
    Args:
        config_path (str): Path to config.yaml (default 'config.yaml').
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    config_p = Path(config_path)
    if not config_p.is_absolute():
        config_p = root_dir / config_p
        
    if not config_p.exists():
        logger.error("Config file not found at %s. Returning empty config.", config_p)
        raise FileNotFoundError(f"Configuration file not found: {config_p}")
        
    try:
        with open(config_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.debug("Successfully loaded config from %s", config_p)
        return config
    except Exception as e:
        logger.error("Failed to parse config file %s: %s", config_p, e)
        raise e


def load_slate_payload(slate_path: str = "data/processed/current_slate.json") -> List[Dict[str, Any]]:
    """
    Safely load active DFS slate player metadata from JSON payload.
    
    Args:
        slate_path (str): Path to current_slate.json.
        
    Returns:
        List[Dict[str, Any]]: List of player metadata dictionaries.
    """
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    slate_p = Path(slate_path)
    if not slate_p.is_absolute():
        slate_p = root_dir / slate_p
        
    if not slate_p.exists():
        logger.error("Slate payload file not found at %s.", slate_p)
        raise FileNotFoundError(f"Slate payload file not found: {slate_p}")
        
    try:
        with open(slate_p, "r", encoding="utf-8") as f:
            slate = json.load(f)
        logger.debug("Successfully loaded %d player records from %s", len(slate), slate_p)
        return slate
    except Exception as e:
        logger.error("Failed to parse slate payload file %s: %s", slate_p, e)
        raise e


# Centrally managed Team-to-Region mapping
REGION_MAP = {
    "Americas": ["100 Thieves", "Cloud9", "Evil Geniuses", "FURIA", "KRÜ Esports", "LEVIATÁN", "LOUD", "MIBR", "NRG", "Sentinels", "G2 Esports"],
    "EMEA": ["BBL Esports", "FNATIC", "FUT Esports", "GIANTX", "Karmine Corp", "Natus Vincere", "Team Heretics", "Team Liquid", "Team Vitality", "Gentle Mates"],
    "Pacific": ["DetonatioN FocusMe", "DRX", "Gen.G", "Global Esports", "Paper Rex", "Rex Regum Qeon", "T1", "Team Secret", "ZETA DIVISION", "Talon Esports", "Bleed Esports"],
    "China": ["All Gamers", "Bilibili Gaming", "EDward Gaming", "FunPlus Phoenix", "JD Gaming", "Nova Esports", "Trace Esports", "Titan Esports Club", "TyLoo", "Dragon Ranger Gaming", "Wolves Esports"]
}

def get_region_for_team(team_name: str) -> str:
    """
    Returns the region (Americas, EMEA, Pacific, China, or Other) for a given team name.
    """
    if not team_name:
        return "Other"
    
    # Normalize team name for comparison (handle encoding discrepancies)
    norm_name = team_name.lower().strip()
    
    for region, teams in REGION_MAP.items():
        for t in teams:
            t_norm = t.lower().strip()
            if t_norm == norm_name or t_norm in norm_name or norm_name in t_norm:
                return region
    return "Other"


def filter_slate_by_teams(slate: List[Dict[str, Any]], allowed_teams: set) -> List[Dict[str, Any]]:
    """
    Filters a player slate list to only include players whose team is in allowed_teams.
    Comparison is case-insensitive, stripped, and supports shortName/substring matching
    without falsely matching short abbreviations (e.g., 'TS') inside 'Esports'.
    """
    if not allowed_teams:
        return slate
    allowed_norm = {str(t).lower().strip() for t in allowed_teams if t}
    
    def matches_team(team_str: str) -> bool:
        if not team_str:
            return False
        t_norm = str(team_str).lower().strip()
        if t_norm in allowed_norm:
            return True
        for a in allowed_norm:
            if len(a) <= 4:
                if re.search(r'\b' + re.escape(a) + r'\b', t_norm):
                    return True
            else:
                if a in t_norm:
                    return True
        return False

    return [p for p in slate if matches_team(p.get("team", ""))]


