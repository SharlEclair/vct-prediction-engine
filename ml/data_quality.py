import os
import sys
sys.path.insert(0, ".")

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from scipy.stats import ks_2samp

from ml.feature_builder import FEATURES_DIR
from ml.train import get_feature_cols

logger = logging.getLogger("ml.data_quality")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REPORTS_DIR = "data/reports"

def generate_data_quality_report(features_dir: str = FEATURES_DIR, reports_dir: str = REPORTS_DIR) -> Dict[str, Any]:
    """
    Generates a pre-training Data Quality Report:
    - Missing values count & %
    - Duplicated matches & player observations
    - Class balance
    - Feature drift (KS-test between train and test distributions)
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    team_df = pd.read_parquet(os.path.join(features_dir, "team_features.parquet"))
    player_df = pd.read_parquet(os.path.join(features_dir, "player_features.parquet"))
    match_df = pd.read_parquet(os.path.join(features_dir, "match_prediction.parquet"))
    train_df = pd.read_parquet(os.path.join(features_dir, "match_train.parquet"))
    test_df = pd.read_parquet(os.path.join(features_dir, "match_test.parquet"))
    
    # 1. Duplicates
    dup_matches = int(match_df.duplicated(subset=["match_id"]).sum())
    dup_player_obs = int(player_df.duplicated(subset=["match_id", "player"]).sum())
    
    # 2. Missing values
    match_missing = match_df.isnull().sum().to_dict()
    player_missing = player_df.isnull().sum().to_dict()
    
    # 3. Class balance
    target_counts = match_df["target"].value_counts().to_dict()
    total_samples = len(match_df)
    class_balance = {
        "team1_wins": int(target_counts.get(1, 0)),
        "team2_wins": int(target_counts.get(0, 0)),
        "team1_win_ratio": float(target_counts.get(1, 0) / max(1, total_samples))
    }
    
    # 4. Feature Drift Check (Train vs Test KS-test)
    feature_cols = get_feature_cols(train_df)
    drift_results = {}
    
    for col in feature_cols:
        if col in train_df.columns and col in test_df.columns:
            tr_vals = train_df[col].dropna().values
            te_vals = test_df[col].dropna().values
            
            if len(tr_vals) > 0 and len(te_vals) > 0:
                stat, p_val = ks_2samp(tr_vals, te_vals)
                drift_results[col] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_val),
                    "drift_detected": bool(p_val < 0.05)
                }
                
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_matches_processed": total_samples,
        "duplicates": {
            "duplicate_matches": dup_matches,
            "duplicate_player_observations": dup_player_obs
        },
        "class_balance": class_balance,
        "missing_values": {
            "match_dataset": match_missing,
            "player_features": player_missing
        },
        "feature_drift_analysis": drift_results
    }
    
    report_path = os.path.join(reports_dir, "data_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Generated Data Quality Report at {report_path}")
    return report

if __name__ == "__main__":
    generate_data_quality_report()
