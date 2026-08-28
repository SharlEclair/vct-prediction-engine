import os
import sys
sys.path.insert(0, ".")

import json
import logging
import yaml
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Any, Tuple

from ml.feature_builder import load_and_normalize_all_matches, save_dataframe_parquet, RAW_DATA_DIR, FEATURES_DIR

logger = logging.getLogger("ml.dataset_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CONFIG_PATH = "config/ml.yaml"

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "split": {
            "train_cutoff": "2025-12-31",
            "val_cutoff": "2026-04-30"
        }
    }


def build_match_prediction_dataset(features_dir: str = FEATURES_DIR, raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Assembles match winner prediction dataset.
    Features: differential features between team1 and team2 pre-match statistics.
    Target: winner (1 if team1 wins, 0 if team2 wins).
    """
    team_df = pd.read_parquet(os.path.join(features_dir, "team_features.parquet"))
    matches = load_and_normalize_all_matches(raw_dir)
    
    rows = []
    
    for match in matches:
        match_id = match["match_id"]
        date = match["date"]
        t1 = match["teams"]["team1"]
        t2 = match["teams"]["team2"]
        winner = match.get("winner", "")
        
        if not winner or (winner != t1 and winner != t2):
            continue
            
        t1_sub = team_df[(team_df["match_id"] == match_id) & (team_df["team"] == t1)]
        t2_sub = team_df[(team_df["match_id"] == match_id) & (team_df["team"] == t2)]
        
        if t1_sub.empty or t2_sub.empty:
            continue
            
        r1 = t1_sub.iloc[0]
        r2 = t2_sub.iloc[0]
        
        feat_cols = [
            "matches_played", "win_rate", "full_buy_win_rate", "semi_buy_win_rate",
            "eco_conversion", "pistol_win_rate", "attack_round_win_rate",
            "defense_round_win_rate", "first_half_win_rate", "comeback_rate"
        ]
        
        row = {
            "match_id": match_id,
            "date": pd.to_datetime(date),
            "team1": t1,
            "team2": t2,
            "target": 1 if winner == t1 else 0
        }
        
        for col in feat_cols:
            row[f"t1_{col}"] = r1.get(col, 0.0)
            row[f"t2_{col}"] = r2.get(col, 0.0)
            row[f"diff_{col}"] = float(r1.get(col, 0.0)) - float(r2.get(col, 0.0))
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    logger.info(f"Built match prediction dataset: {df.shape}")
    return df


def build_map_prediction_dataset(features_dir: str = FEATURES_DIR, raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Assembles map winner prediction dataset.
    Features: team features, map features, map_name.
    Target: winner (1 if team1 wins map, 0 if team2 wins map).
    """
    team_df = pd.read_parquet(os.path.join(features_dir, "team_features.parquet"))
    map_df = pd.read_parquet(os.path.join(features_dir, "map_features.parquet"))
    matches = load_and_normalize_all_matches(raw_dir)
    
    rows = []
    
    for match in matches:
        match_id = match["match_id"]
        date = match["date"]
        t1 = match["teams"]["team1"]
        t2 = match["teams"]["team2"]
        
        t1_sub = team_df[(team_df["match_id"] == match_id) & (team_df["team"] == t1)]
        t2_sub = team_df[(team_df["match_id"] == match_id) & (team_df["team"] == t2)]
        
        if t1_sub.empty or t2_sub.empty:
            continue
            
        r1 = t1_sub.iloc[0]
        r2 = t2_sub.iloc[0]
        
        for m in match.get("maps", []):
            m_name = m.get("map_name", "Unknown")
            m_winner = m.get("winner", "")
            
            if not m_winner or (m_winner != t1 and m_winner != t2):
                continue
                
            m_sub = map_df[map_df["map_name"] == m_name]
            avg_rounds = m_sub.iloc[0]["avg_rounds"] if not m_sub.empty else 21.0
            
            row = {
                "match_id": match_id,
                "map_name": m_name,
                "date": pd.to_datetime(date),
                "team1": t1,
                "team2": t2,
                "diff_win_rate": float(r1.get("win_rate", 0.5)) - float(r2.get("win_rate", 0.5)),
                "diff_attack_win_rate": float(r1.get("attack_round_win_rate", 0.5)) - float(r2.get("attack_round_win_rate", 0.5)),
                "diff_defense_win_rate": float(r1.get("defense_round_win_rate", 0.5)) - float(r2.get("defense_round_win_rate", 0.5)),
                "avg_map_rounds": avg_rounds,
                "target": 1 if m_winner == t1 else 0
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    logger.info(f"Built map prediction dataset: {df.shape}")
    return df


def build_score_prediction_dataset(features_dir: str = FEATURES_DIR, raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Assembles map score prediction dataset.
    Targets: team1_rounds, team2_rounds.
    """
    team_df = pd.read_parquet(os.path.join(features_dir, "team_features.parquet"))
    matches = load_and_normalize_all_matches(raw_dir)
    
    rows = []
    
    for match in matches:
        match_id = match["match_id"]
        date = match["date"]
        t1 = match["teams"]["team1"]
        t2 = match["teams"]["team2"]
        
        t1_sub = team_df[(team_df["match_id"] == match_id) & (team_df["team"] == t1)]
        t2_sub = team_df[(team_df["match_id"] == match_id) & (team_df["team"] == t2)]
        
        if t1_sub.empty or t2_sub.empty:
            continue
            
        r1 = t1_sub.iloc[0]
        r2 = t2_sub.iloc[0]
        
        for m in match.get("maps", []):
            m_name = m.get("map_name", "Unknown")
            s = m.get("score", {})
            s1 = s.get("team1_score", 0)
            s2 = s.get("team2_score", 0)
            
            if s1 == 0 and s2 == 0:
                continue
                
            row = {
                "match_id": match_id,
                "map_name": m_name,
                "date": pd.to_datetime(date),
                "team1": t1,
                "team2": t2,
                "diff_win_rate": float(r1.get("win_rate", 0.5)) - float(r2.get("win_rate", 0.5)),
                "team1_rounds": s1,
                "team2_rounds": s2
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    logger.info(f"Built score prediction dataset: {df.shape}")
    return df


def split_dataset(df: pd.DataFrame, train_cutoff: str, val_cutoff: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically splits dataset into train, val, and test subsets."""
    df["date"] = pd.to_datetime(df["date"])
    train_mask = df["date"] <= pd.to_datetime(train_cutoff)
    val_mask = (df["date"] > pd.to_datetime(train_cutoff)) & (df["date"] <= pd.to_datetime(val_cutoff))
    test_mask = df["date"] > pd.to_datetime(val_cutoff)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def generate_all_datasets(features_dir: str = FEATURES_DIR, raw_dir: str = RAW_DATA_DIR):
    """Main execution to produce supervised datasets and split Parquet files."""
    cfg = load_config()
    train_cutoff = cfg["split"]["train_cutoff"]
    val_cutoff = cfg["split"]["val_cutoff"]
    
    os.makedirs(features_dir, exist_ok=True)
    
    # 1. Match prediction
    match_df = build_match_prediction_dataset(features_dir, raw_dir)
    match_df.to_csv(os.path.join(features_dir, "match_prediction.csv"), index=False)
    save_dataframe_parquet(match_df, os.path.join(features_dir, "match_prediction.parquet"))
    
    tr_match, val_match, te_match = split_dataset(match_df, train_cutoff, val_cutoff)
    save_dataframe_parquet(tr_match, os.path.join(features_dir, "match_train.parquet"))
    save_dataframe_parquet(val_match, os.path.join(features_dir, "match_val.parquet"))
    save_dataframe_parquet(te_match, os.path.join(features_dir, "match_test.parquet"))
    
    # 2. Map prediction
    map_df = build_map_prediction_dataset(features_dir, raw_dir)
    map_df.to_csv(os.path.join(features_dir, "map_prediction.csv"), index=False)
    save_dataframe_parquet(map_df, os.path.join(features_dir, "map_prediction.parquet"))
    
    # 3. Score prediction
    score_df = build_score_prediction_dataset(features_dir, raw_dir)
    score_df.to_csv(os.path.join(features_dir, "map_score.csv"), index=False)
    save_dataframe_parquet(score_df, os.path.join(features_dir, "map_score.parquet"))
    
    logger.info("Dataset Builder execution completed successfully!")


if __name__ == "__main__":
    generate_all_datasets()
