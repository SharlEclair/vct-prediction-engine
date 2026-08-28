import os
import sys
sys.path.insert(0, ".")

import json
import logging
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import shap

from ml.train import get_feature_cols, PRODUCTION_DIR
from ml.experiment_tracker import log_experiment

logger = logging.getLogger("ml.evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REPORTS_DIR = "data/reports"

def compute_calibration_errors(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_mask = (bin_assignments == i)
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:
            avg_acc = np.mean(y_true[bin_mask])
            avg_conf = np.mean(y_prob[bin_mask])
            abs_diff = np.abs(avg_acc - avg_conf)
            
            ece += (bin_size / total_samples) * abs_diff
            mce = max(mce, abs_diff)
            
    return float(ece), float(mce)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    ece, mce = compute_calibration_errors(y_true, y_prob)
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clipped)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece),
        "mce": float(mce)
    }


def plot_and_save_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_path: str):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, "s-", label="Model Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Proportion")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved calibration curve to {out_path}")


def compute_and_save_feature_importance(model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_cols: list, reports_dir: str = REPORTS_DIR):
    """
    Computes LightGBM gain importance, Permutation Importance, and SHAP values.
    Saves results to data/reports/feature_importance.json and data/reports/shap_summary.png.
    """
    importance_data = {}
    
    # 1. Native Model Feature Importance (Gain)
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        importance_data["native_gain_importance"] = {
            col: float(val) for col, val in zip(feature_cols, raw_imp)
        }
        
    # 2. Permutation Importance
    try:
        perm_res = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        importance_data["permutation_importance"] = {
            col: float(val) for col, val in zip(feature_cols, perm_res.importances_mean)
        }
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        
    # 3. SHAP Feature Importance
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification list vs array SHAP outputs
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1])
        else:
            sv = np.array(shap_values)
            
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        importance_data["shap_importance"] = {
            col: float(val) for col, val in zip(feature_cols, mean_abs_shap)
        }
        
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False)
        plt.tight_layout()
        shap_path = os.path.join(reports_dir, "shap_summary.png")
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot to {shap_path}")
    except Exception as e:
        logger.warning(f"SHAP evaluation warning: {e}")

    imp_path = os.path.join(reports_dir, "feature_importance.json")
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(importance_data, f, indent=2)
        
    logger.info(f"Saved feature importance JSON to {imp_path}")


def evaluate_models(features_dir: str = "data/features", reports_dir: str = REPORTS_DIR) -> Dict[str, Any]:
    os.makedirs(reports_dir, exist_ok=True)
    test_df = pd.read_parquet(os.path.join(features_dir, "match_test.parquet"))
    
    feature_cols = get_feature_cols(test_df)
    X_test = test_df[feature_cols].fillna(0.0).values
    y_test = test_df["target"].values
    
    results = {}
    
    # 1. Baseline Model
    baseline_pkg = joblib.load(os.path.join(PRODUCTION_DIR, "match_winner_baseline_v1.pkl"))
    b_model = baseline_pkg["model"]
    b_scaler = baseline_pkg["scaler"]
    X_scaled = b_scaler.transform(X_test)
    b_probs = b_model.predict_proba(X_scaled)[:, 1]
    results["baseline_logistic_regression"] = compute_metrics(y_test, b_probs)
    
    # Log baseline experiment
    log_experiment("baseline_logistic_regression", {"solver": "lbfgs"}, results["baseline_logistic_regression"])
    
    # 2. Main Model (LightGBM)
    main_pkg = joblib.load(os.path.join(PRODUCTION_DIR, "match_winner_v1.pkl"))
    m_model = main_pkg["model"]
    m_probs = m_model.predict_proba(X_test)[:, 1]
    results["lightgbm_main_model"] = compute_metrics(y_test, m_probs)
    
    # Log main model experiment
    hparams = getattr(m_model, "get_params", lambda: {})()
    log_experiment("lightgbm_main_model", hparams, results["lightgbm_main_model"])
    
    # Plot calibration curve & feature importance
    plot_and_save_calibration(y_test, m_probs, os.path.join(reports_dir, "calibration_curve.png"))
    compute_and_save_feature_importance(m_model, X_test, y_test, feature_cols, reports_dir)
    
    # Write evaluation report
    report_path = os.path.join(reports_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved evaluation report to {report_path}")
    return results


if __name__ == "__main__":
    res = evaluate_models()
    print("\n" + "="*60)
    print("VALIDATION METRICS SUMMARY")
    print("="*60)
    print(json.dumps(res, indent=2))
