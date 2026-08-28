"""
Model Training Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 1).

Handles Task 1.5 (Regressor Training and Validation).
Trains an XGBoost regressor predicting continuous Expected Value (mu_TD) for player performance,
and validates MAE performance against the 4.37 naive baseline.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from data_ingestion import generate_mock_match_telemetry, process_match_telemetry, apply_winsorization
from feature_engineering import compute_bayesian_features, generate_odr_matrix, attach_odr_features
from utils.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_phase_1_dataset(num_matches: int = 200) -> Tuple[pd.DataFrame, list, str]:
    """
    Execute end-to-end telemetry ingestion, Winsorization, Bayesian tracker, and ODR matrix generation.
    
    Args:
        num_matches (int): Number of mock matches to ingest and process.
        
    Returns:
        Tuple[pd.DataFrame, list, str]: Processed dataset, feature column names, target column name.
    """
    logger.info("Starting Phase 1 pipeline data preparation...")
    
    # Task 1.1: Telemetry ingestion and KPR extraction
    telemetry = generate_mock_match_telemetry(num_matches=num_matches, seed=42)
    df = process_match_telemetry(telemetry)
    
    # Task 1.2: Winsorization on KPR (Removed for Phase 13)
    
    # Task 1.3: Bayesian feature calculation
    df = compute_bayesian_features(df)
    
    # Task 1.4: ODR matrix generation via Ridge regression
    odr_matrix = generate_odr_matrix(df, target_col="kpr", alpha_ridge=1.0)
    df = attach_odr_features(df, odr_matrix)
    
    # Target definition: We scale KPR to total kills or expected performance metric
    # Target variable for mu_TD regressor: Raw Kills per match segment (or continuous expected kills)
    target_col = "kills"
    
    feature_cols = [
        "kpr",
        "player_kpr_mu",
        "player_kpr_sigma",
        "opponent_odr"
    ]
    
    # Create lag/historical feature alignment to prevent data leakage in real time
    # Shift player features by 1 match to predict next match performance
    df.sort_values(by=["player_id", "match_timestamp"], inplace=True)
    for col in feature_cols:
        df[f"prev_{col}"] = df.groupby("player_id")[col].shift(1)
        
    lagged_feature_cols = [f"prev_{col}" for col in feature_cols]
    
    # Drop initial rows where lagged features are NaN
    df_clean = df.dropna(subset=lagged_feature_cols).copy()
    df_clean.reset_index(drop=True, inplace=True)
    
    logger.info("Dataset prepared successfully. Total samples for modeling: %d", len(df_clean))
    return df_clean, lagged_feature_cols, target_col


def train_and_evaluate_xgboost(
    df: pd.DataFrame, 
    feature_cols: list, 
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Any], xgb.XGBRegressor]:
    """
    Task 1.5: Train XGBoost regressor and evaluate MAE validation metrics against naive baseline (4.37).
    
    Args:
        df (pd.DataFrame): Modeling dataset.
        feature_cols (list): List of feature names.
        target_col (str): Target column name.
        test_size (float): Holdout test set ratio.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Tuple[Dict[str, Any], xgb.XGBRegressor]: Validation results and trained model instance.
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    logger.info("Training set shape: %s | Holdout test set shape: %s", X_train.shape, X_test.shape)
    
    # Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    
    # Naive baseline: predicting mean of training target
    naive_preds = np.full_like(y_test, fill_value=y_train.mean())
    sample_naive_mae = mean_absolute_error(y_test, naive_preds)
    
    # Load benchmark reference dynamically from config.yaml
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent
    config = load_config(str(root_dir / "config.yaml"))
    SPEC_NAIVE_BASELINE_MAE = float(config.get("BENCHMARKS", {}).get("spec_naive_baseline_mae", 4.37))
    
    outperforms_spec = model_mae < SPEC_NAIVE_BASELINE_MAE
    
    results = {
        "model_mae": float(model_mae),
        "sample_naive_mae": float(sample_naive_mae),
        "spec_baseline_mae": SPEC_NAIVE_BASELINE_MAE,
        "outperforms_spec_baseline": outperforms_spec,
        "feature_importances": dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
    }
    
    logger.info("=== TASK 1.5 VALIDATION RESULTS ===")
    logger.info("XGBoost Regressor MAE: %.4f", model_mae)
    logger.info("Sample Naive Mean MAE: %.4f", sample_naive_mae)
    logger.info("Spec Benchmark MAE   : %.4f", SPEC_NAIVE_BASELINE_MAE)
    logger.info("Outperforms Benchmark? %s", "YES" if outperforms_spec else "NO")
    logger.info("Feature Importances  : %s", results["feature_importances"])
    
    return results, model


def generate_slate_predictions(model: xgb.XGBRegressor, feature_cols: list) -> None:
    """
    Task 8.1: Generate expected value predictions (mu_TD) for the active players
    listed in current_slate.json and serialize to xgb_predictions.json.
    """
    from pathlib import Path
    import json
    
    # Establish root directory absolutely
    root_dir = Path(__file__).resolve().parent
    slate_path = root_dir / "data" / "processed" / "current_slate.json"
    ledger_path = root_dir / "data" / "processed" / "bayesian_player_ledger.json"
    out_path = root_dir / "data" / "processed" / "xgb_predictions.json"
    
    if not slate_path.exists():
        raise FileNotFoundError(f"Active slate file not found at {slate_path}")
        
    with open(slate_path, "r", encoding="utf-8") as f:
        slate = json.load(f)
        
    ledger = {}
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load Bayesian player ledger from {ledger_path}: {e}")
            
    predictions = {}
    for item in slate:
        player_id = item["player_id"]
        name = item["name"]
        salary = item["salary"]
        role = item["role"]
        
        # Look up in ledger
        p_feat = None
        for k, v in ledger.items():
            if k.lower().strip() == name.lower().strip():
                p_feat = v
                break
                
        if p_feat is not None:
            prev_player_kpr_mu = p_feat["kpr_mu"]
            prev_player_kpr_sigma = p_feat["kpr_sigma"]
            prev_kpr = prev_player_kpr_mu
        else:
            # Fallback default ACS based on role and salary if not in ledger
            role_defaults = {"Duelist": 220, "Initiator": 200, "Controller": 190, "Sentinel": 180, "Flex": 200}
            base_acs = role_defaults.get(role, 200)
            global_acs = base_acs * (salary / 8.0)
            prev_kpr = global_acs / 300.0
            prev_player_kpr_mu = prev_kpr
            prev_player_kpr_sigma = 0.20
            
        prev_opponent_odr = 0.0 # baseline neutral opponent ODR
        
        X_pred = pd.DataFrame([{
            "prev_kpr": prev_kpr,
            "prev_player_kpr_mu": prev_player_kpr_mu,
            "prev_player_kpr_sigma": prev_player_kpr_sigma,
            "prev_opponent_odr": prev_opponent_odr
        }])
        
        # Align columns to match model feature columns
        X_pred = X_pred[feature_cols]
        
        pred_kills = float(model.predict(X_pred)[0])
        
        # Translate kills prediction to Official VFL expected points
        # Official VFL scoring: Kills per map piecewise + Map Wins + Series Bonus + VLR Rating Bonus + MVP Placement Bonus
        map_k = pred_kills / 2.0 if pred_kills > 0 else (prev_kpr * 24.0) / 2.0
        if map_k <= 0:
            k_pts_map = -3
        elif map_k <= 4:
            k_pts_map = -1
        elif map_k <= 9:
            k_pts_map = 0
        else:
            k_pts_map = 1 + int((map_k - 10) // 5)

        # Rating estimation: 1.05 baseline, scaled by KPR relative to 0.72
        est_rating = 1.05 * (prev_kpr / 0.72)
        rating_scaling_bonus = 3 if est_rating >= 2.0 else (2 if est_rating >= 1.75 else (1 if est_rating >= 1.5 else 0))
        mvp_placement_bonus = 3 if prev_kpr >= 0.85 else (2 if prev_kpr >= 0.75 else (1 if prev_kpr >= 0.68 else 0))

        # Expected VFL score per match = (Kill pts per map * 2 maps) + Map Wins (+2.0) + Series Bonus (+1.5) + Rating Bonus + MVP Bonus
        projected_vfl_points = (k_pts_map * 2.0) + 2.0 + 1.5 + rating_scaling_bonus + mvp_placement_bonus
        
        # Write both player name and player id as keys to guarantee lookup compatibility
        predictions[name] = round(projected_vfl_points, 2)
        predictions[player_id] = round(projected_vfl_points, 2)
        
    # Write output predictions to data/processed/xgb_predictions.json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
        
    logger.info("Task 8.1 Complete: Generated and saved Official VFL expected value predictions to %s", out_path)


if __name__ == "__main__":
    df_model, features, target = prepare_phase_1_dataset(num_matches=300)
    results, trained_model = train_and_evaluate_xgboost(df_model, features, target)
    generate_slate_predictions(trained_model, features)
