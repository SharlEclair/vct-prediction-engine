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
from feature_engineering import compute_player_ema, generate_odr_matrix, attach_odr_features

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_phase_1_dataset(num_matches: int = 200) -> Tuple[pd.DataFrame, list, str]:
    """
    Execute end-to-end telemetry ingestion, Winsorization, EMA construction, and ODR matrix generation.
    
    Args:
        num_matches (int): Number of mock matches to ingest and process.
        
    Returns:
        Tuple[pd.DataFrame, list, str]: Processed dataset, feature column names, target column name.
    """
    logger.info("Starting Phase 1 pipeline data preparation...")
    
    # Task 1.1: Telemetry ingestion and KPR extraction
    telemetry = generate_mock_match_telemetry(num_matches=num_matches, seed=42)
    df = process_match_telemetry(telemetry)
    
    # Task 1.2: Winsorization on KPR
    df = apply_winsorization(df, col="kpr", lower_quantile=0.05, upper_quantile=0.95)
    
    # Task 1.3: EMA construction (slow alpha=0.1, rapid alpha=0.4)
    df = compute_player_ema(df, target_col="clipped_kpr", alphas=(0.1, 0.4))
    
    # Task 1.4: ODR matrix generation via Ridge regression
    odr_matrix = generate_odr_matrix(df, target_col="kpr", alpha_ridge=1.0)
    df = attach_odr_features(df, odr_matrix)
    
    # Target definition: We scale KPR to total kills or expected performance metric
    # Target variable for mu_TD regressor: Raw Kills per match segment (or continuous expected kills)
    target_col = "kills"
    
    feature_cols = [
        "clipped_kpr",
        "ema_kpr_alpha_0.1",
        "ema_kpr_alpha_0.4",
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
) -> Dict[str, Any]:
    """
    Task 1.5: Train XGBoost regressor and evaluate MAE validation metrics against naive baseline (4.37).
    
    Args:
        df (pd.DataFrame): Modeling dataset.
        feature_cols (list): List of feature names.
        target_col (str): Target column name.
        test_size (float): Holdout test set ratio.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Dict[str, Any]: Validation results including model MAE, baseline MAE, and improvement status.
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
    
    # Standard Benchmark reference from spec
    SPEC_NAIVE_BASELINE_MAE = 4.37
    
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
    
    return results


if __name__ == "__main__":
    df_model, features, target = prepare_phase_1_dataset(num_matches=300)
    results = train_and_evaluate_xgboost(df_model, features, target)
