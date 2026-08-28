import os
import sys
sys.path.insert(0, ".")

import json
import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna

# Disable verbose optuna logs by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("ml.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CHECKPOINT_DIR = "models/checkpoints"
PRODUCTION_DIR = "models/production"
CONFIG_PATH = "config/ml.yaml"

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"training": {"random_seed": 42, "optuna_trials": 15}}


def get_feature_cols(df: pd.DataFrame) -> list:
    ignore_cols = ["match_id", "date", "team1", "team2", "map_name", "target", "team1_rounds", "team2_rounds"]
    return [c for c in df.columns if c not in ignore_cols]


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_lightgbm_with_optuna(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 15) -> Any:
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        eps = 1e-15
        preds = np.clip(preds, eps, 1 - eps)
        logloss = -np.mean(y_val * np.log(preds) + (1 - y_val) * np.log(1 - preds))
        return logloss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({"objective": "binary", "random_state": 42, "verbosity": -1})
    
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def train_score_regressor(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_model_checkpoint(model: Any, name: str, scaler: Any = None, metadata: Dict[str, Any] = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    
    payload = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    chk_path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    prod_path = os.path.join(PRODUCTION_DIR, f"{name}.pkl")
    
    joblib.dump(payload, chk_path)
    joblib.dump(payload, prod_path)
    logger.info(f"Saved model checkpoint and production bundle for '{name}'")


def train_all_models(features_dir: str = "data/features"):
    cfg = load_config()
    n_trials = cfg.get("training", {}).get("optuna_trials", 15)
    
    train_df = pd.read_parquet(os.path.join(features_dir, "match_train.parquet"))
    val_df = pd.read_parquet(os.path.join(features_dir, "match_val.parquet"))
    
    feature_cols = get_feature_cols(train_df)
    X_tr = train_df[feature_cols].fillna(0.0).values
    y_tr = train_df["target"].values
    
    X_val = val_df[feature_cols].fillna(0.0).values
    y_val = val_df["target"].values
    
    logger.info(f"Training on {len(X_tr)} samples with {len(feature_cols)} features...")
    
    # 1. Baseline Logistic Regression
    lr_model, scaler = train_logistic_regression(X_tr, y_tr)
    save_model_checkpoint(lr_model, "match_winner_baseline_v1", scaler=scaler, metadata={"feature_cols": feature_cols})
    
    # 2. LightGBM Classifier (with Optuna)
    lgb_model = train_lightgbm_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=n_trials)
    save_model_checkpoint(lgb_model, "match_winner_v1", metadata={"feature_cols": feature_cols})
    
    # 3. XGBoost Map Winner Classifier
    map_df = pd.read_parquet(os.path.join(features_dir, "map_prediction.parquet"))
    map_cols = get_feature_cols(map_df)
    X_map = map_df[map_cols].fillna(0.0).values
    y_map = map_df["target"].values
    map_xgb = train_xgboost(X_map, y_map)
    save_model_checkpoint(map_xgb, "map_winner_v1", metadata={"feature_cols": map_cols})
    
    # 4. Score Predictor Regressor
    score_df = pd.read_parquet(os.path.join(features_dir, "map_score.parquet"))
    score_cols = get_feature_cols(score_df)
    X_score = score_df[score_cols].fillna(0.0).values
    y_score = score_df["team1_rounds"].values
    score_reg = train_score_regressor(X_score, y_score)
    save_model_checkpoint(score_reg, "score_predictor_v1", metadata={"feature_cols": score_cols})
    
    logger.info("All model training finished successfully!")


if __name__ == "__main__":
    train_all_models()
