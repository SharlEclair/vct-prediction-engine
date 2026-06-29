"""
Data Ingestion Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 1).

Handles Task 1.1 (Target Variable Generation) and Task 1.2 (Baseline Clipping / Winsorization).
Includes mock telemetry generation for /v2/match/details JSON endpoint to support testing.
"""

import logging
import random
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_mock_match_telemetry(num_matches: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate mock /v2/match/details JSON telemetry data for testing.
    
    Args:
        num_matches (int): Number of mock matches to generate.
        seed (int): Random seed for reproducibility.
        
    Returns:
        List[Dict[str, Any]]: List of match telemetry dictionaries.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    teams = ["Sentinels", "Fnatic", "Paper Rex", "EDward Gaming", "Team Heretics", 
             "LOUD", "DRX", "Gen.G", "Natus Vincere", "NRG"]
    
    mock_matches = []
    player_id_counter = 100
    
    # Pre-define a pool of players for consistent tracking with latent skill
    player_pool = []
    for team in teams:
        for i in range(5):
            player_pool.append({
                "player_id": f"P_{player_id_counter}",
                "player_name": f"Player_{player_id_counter}",
                "team_name": team,
                "true_kpr": float(np.clip(np.random.normal(0.75, 0.15), 0.3, 1.2))
            })
            player_id_counter += 1

    team_players = {t: [p for p in player_pool if p["team_name"] == t] for t in teams}
    
    start_timestamp = 1700000000
    
    for match_idx in range(num_matches):
        team_a, team_b = random.sample(teams, 2)
        rounds_played = random.randint(13, 30) # Typical match length
        match_id = f"MATCH_{1000 + match_idx}"
        match_date = start_timestamp + match_idx * 86400 # 1 match per day in sequence
        
        segments = []
        for team, opp_team in [(team_a, team_b), (team_b, team_a)]:
            for player in team_players[team]:
                # Generate realistic correlated kill counts based on player true skill and rounds played
                base_kpr = player["true_kpr"] + np.random.normal(0.0, 0.05)
                base_kpr = max(0.1, base_kpr)
                kills = int(np.round(base_kpr * rounds_played))
                deaths = int(np.round(np.random.normal(0.7, 0.15) * rounds_played))
                assists = int(np.round(np.random.normal(0.3, 0.1) * rounds_played))
                first_bloods = int(np.round(np.random.normal(0.15, 0.08) * rounds_played))
                
                segments.append({
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "team_name": team,
                    "opponent_team_name": opp_team,
                    "kills": max(0, kills),
                    "deaths": max(0, deaths),
                    "assists": max(0, assists),
                    "first_bloods": max(0, first_bloods),
                    "rounds_played": rounds_played
                })
                
        mock_matches.append({
            "match_id": match_id,
            "match_timestamp": match_date,
            "segments": segments
        })
        
    logger.info("Generated %d mock match telemetry records.", num_matches)
    return mock_matches


def process_match_telemetry(telemetry_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Task 1.1: Target Variable Generation.
    Extract raw metrics (Kills, Deaths, Assists, First Bloods) and compute KPR (Kills Per Round).
    
    Args:
        telemetry_data (List[Dict[str, Any]]): Raw /v2/match/details JSON records.
        
    Returns:
        pd.DataFrame: Processed DataFrame containing match, player, team, raw metrics, and calculated KPR.
    """
    rows = []
    for match in telemetry_data:
        match_id = match.get("match_id")
        timestamp = match.get("match_timestamp")
        for seg in match.get("segments", []):
            rounds = seg.get("rounds_played", 1)
            kills = seg.get("kills", 0)
            deaths = seg.get("deaths", 0)
            assists = seg.get("assists", 0)
            first_bloods = seg.get("first_bloods", 0)
            
            kpr = kills / rounds if rounds > 0 else 0.0
            
            rows.append({
                "match_id": match_id,
                "match_timestamp": timestamp,
                "player_id": seg.get("player_id"),
                "player_name": seg.get("player_name"),
                "team_name": seg.get("team_name"),
                "opponent_team_name": seg.get("opponent_team_name"),
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "first_bloods": first_bloods,
                "rounds_played": rounds,
                "kpr": kpr
            })
            
    df = pd.DataFrame(rows)
    df.sort_values(by="match_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Task 1.1 complete: Extracted %d player-match rows and calculated KPR.", len(df))
    return df


def apply_winsorization(
    df: pd.DataFrame, 
    col: str = "kpr", 
    lower_quantile: float = 0.05, 
    upper_quantile: float = 0.95
) -> pd.DataFrame:
    """
    Task 1.2: Baseline Clipping (Winsorization).
    Mitigate extreme structural outliers by bounding target rate metrics strictly between 
    specified percentiles (default: 5th and 95th).
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the target column.
        col (str): Column name to winsorize (default 'kpr').
        lower_quantile (float): Lower percentile boundary (0.05).
        upper_quantile (float): Upper percentile boundary (0.95).
        
    Returns:
        pd.DataFrame: Modified DataFrame with a new clipped column (e.g., 'clipped_kpr').
    """
    df = df.copy()
    lower_bound = df[col].quantile(lower_quantile)
    upper_bound = df[col].quantile(upper_quantile)
    
    clipped_col_name = f"clipped_{col}"
    df[clipped_col_name] = np.clip(df[col], lower_bound, upper_bound)
    
    logger.info(
        "Task 1.2 complete: Winsorized '%s' (Bounds: [%.4f, %.4f]). Created '%s'.", 
        col, lower_bound, upper_bound, clipped_col_name
    )
    return df
