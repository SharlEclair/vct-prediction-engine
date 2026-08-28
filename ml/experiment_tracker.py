import os
import sys
sys.path.insert(0, ".")

import glob
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("ml.experiment_tracker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

EXPERIMENTS_DIR = "experiments"

def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def log_experiment(model_type: str, hyperparameters: Dict[str, Any], metrics: Dict[str, Any],
                   dataset_version: str = "10.0.0", feature_version: str = "10.1") -> str:
    """
    Logs an experiment run with git commit hash, hyperparameters, and evaluation metrics.
    Saves to experiments/experiment_XXX.json.
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    existing_exp = glob.glob(os.path.join(EXPERIMENTS_DIR, "experiment_*.json"))
    exp_num = len(existing_exp) + 1
    exp_id = f"experiment_{exp_num:03d}"
    
    commit_hash = get_git_commit_hash()
    
    record = {
        "experiment_id": exp_id,
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": commit_hash,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "metrics": metrics
    }
    
    exp_path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        
    logger.info(f"Logged experiment record to {exp_path}")
    return exp_path


if __name__ == "__main__":
    log_experiment("lightgbm_baseline", {"learning_rate": 0.05, "n_estimators": 100}, {"accuracy": 0.65, "log_loss": 0.61})
