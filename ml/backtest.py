import os
import sys
sys.path.insert(0, ".")

import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from ml.train import get_feature_cols, train_lightgbm_with_optuna
from ml.evaluate import compute_metrics

logger = logging.getLogger("ml.backtest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REPORTS_DIR = "data/reports"

def walk_forward_backtest(dataset_path: str = "data/features/match_prediction.parquet", n_splits: int = 5) -> Dict[str, Any]:
    """
    Performs expanding-window temporal walk-forward backtesting.
    Ensure training only uses past historical data before each test fold.
    """
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    total_len = len(df)
    min_train_size = int(total_len * 0.4)
    step_size = int((total_len - min_train_size) / n_splits)
    
    feature_cols = get_feature_cols(df)
    
    fold_results = []
    all_y_true = []
    all_y_prob = []
    
    logger.info(f"Starting Walk-Forward Backtest ({n_splits} folds) over {total_len} samples...")
    
    for i in range(n_splits):
        train_end_idx = min_train_size + i * step_size
        test_end_idx = min_train_size + (i + 1) * step_size if i < n_splits - 1 else total_len
        
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx:test_end_idx]
        
        if test_df.empty:
            continue
            
        X_train = train_df[feature_cols].fillna(0.0).values
        y_train = train_df["target"].values
        
        X_test = test_df[feature_cols].fillna(0.0).values
        y_test = test_df["target"].values
        
        # Split train into sub-val for optuna tuning inside fold
        sub_tr_size = int(len(X_train) * 0.8)
        X_sub_tr, y_sub_tr = X_train[:sub_tr_size], y_train[:sub_tr_size]
        X_sub_val, y_sub_val = X_train[sub_tr_size:], y_train[sub_tr_size:]
        
        model = train_lightgbm_with_optuna(X_sub_tr, y_sub_tr, X_sub_val, y_sub_val, n_trials=5)
        probs = model.predict_proba(X_test)[:, 1]
        
        fold_metric = compute_metrics(y_test, probs)
        fold_info = {
            "fold": i + 1,
            "train_range": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "test_range": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "metrics": fold_metric
        }
        fold_results.append(fold_info)
        
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())
        
        logger.info(f"Fold {i+1}/{n_splits} complete: Test Acc={fold_metric['accuracy']:.4f}, LogLoss={fold_metric['log_loss']:.4f}")
        
    overall_metrics = compute_metrics(np.array(all_y_true), np.array(all_y_prob))
    
    summary = {
        "model": "lightgbm_walk_forward",
        "n_splits": n_splits,
        "total_evaluated_samples": len(all_y_true),
        "overall_metrics": overall_metrics,
        "folds": fold_results
    }
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "backtest_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Saved backtest report to {report_path}")
    return summary


if __name__ == "__main__":
    res = walk_forward_backtest()
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {res['overall_metrics']['accuracy']:.4f}")
    print(f"Overall ROC-AUC: {res['overall_metrics']['roc_auc']:.4f}")
    print(f"Overall Log Loss: {res['overall_metrics']['log_loss']:.4f}")
