import os
import sys
sys.path.insert(0, ".")

import json
import joblib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("verify_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def verify_all_artifacts():
    logger.info("=== Starting Pipeline Verification & Artifact Audit ===")
    
    # 1. Verify Parquet and CSV Datasets
    files_to_check = [
        ("data/features/team_features.parquet", 100, 10),
        ("data/features/player_features.parquet", 1000, 10),
        ("data/features/map_features.parquet", 5, 3),
        ("data/features/match_prediction.csv", 1000, 10),
        ("data/features/match_prediction.parquet", 1000, 10),
        ("data/features/match_train.parquet", 500, 10),
        ("data/features/match_val.parquet", 10, 10),
        ("data/features/match_test.parquet", 10, 10),
    ]
    
    for path, min_rows, min_cols in files_to_check:
        assert os.path.exists(path), f"Missing expected dataset file: {path}"
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
            
        assert len(df) >= min_rows, f"{path} row count {len(df)} is below expected minimum {min_rows}"
        assert len(df.columns) >= min_cols, f"{path} column count {len(df.columns)} is below minimum {min_cols}"
        logger.info(f"VERIFIED dataset {path}: shape {df.shape}")
        
    # 2. Verify Saved Model Files and Inference Capability
    models_to_check = [
        "models/production/champion/match_winner_v1.pkl",
        "models/production/match_winner_v1.pkl",
        "models/production/map_winner_v1.pkl",
        "models/production/score_predictor_v1.pkl"
    ]
    
    for m_path in models_to_check:
        assert os.path.exists(m_path), f"Missing model checkpoint: {m_path}"
        pkg = joblib.load(m_path)
        assert "model" in pkg, f"Model payload missing 'model' key in {m_path}"
        model = pkg["model"]
        feature_cols = pkg.get("metadata", {}).get("feature_cols", [])
        num_feats = len(feature_cols) if feature_cols else 30
        dummy_input = np.zeros((1, num_feats))
        
        # Test inference
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(dummy_input)
            assert probs.shape[1] == 2, f"Invalid probability output shape {probs.shape}"
            assert 0.0 <= probs[0, 1] <= 1.0, f"Probability out of range: {probs[0, 1]}"
        elif hasattr(model, "predict"):
            preds = model.predict(dummy_input)
            assert len(preds) == 1, f"Invalid prediction shape {preds.shape}"
            
        logger.info(f"VERIFIED model inference for {m_path}")
        
    # 3. Verify Validation & Backtesting Reports
    reports_to_check = [
        "data/reports/data_quality_report.json",
        "data/reports/validation_report.json",
        "data/reports/backtest_results.json",
        "data/features/feature_manifest.json",
        "data/features/feature_registry.json",
        "docs/model_cards/match_winner_v1.md"
    ]
    
    for r_path in reports_to_check:
        assert os.path.exists(r_path), f"Missing report file: {r_path}"
        if r_path.endswith(".json"):
            with open(r_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict), f"Report {r_path} is not a valid JSON dictionary"
        logger.info(f"VERIFIED report existence: {r_path}")
        
    logger.info("=== ALL PIPELINE VERIFICATION CHECKS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    verify_all_artifacts()
