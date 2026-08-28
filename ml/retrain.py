import os
import sys
sys.path.insert(0, ".")

import logging
from datetime import datetime

from ml.data_quality import generate_data_quality_report
from ml.feature_builder import generate_feature_store
from ml.dataset_builder import generate_all_datasets
from ml.train import train_all_models
from ml.evaluate import evaluate_models
from ml.model_registry import register_model, promote_challenger_to_champion
from ml.model_card import generate_model_card

logger = logging.getLogger("ml.retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def run_retraining_pipeline(force_promote: bool = False):
    """
    Automated Retraining Workflow:
    1. Data Quality Checks
    2. Feature Store & Dataset Generation
    3. Retrain Challenger Models
    4. Run Validation & Calibration Metrics
    5. Compare against Champion and Promote if thresholds are met.
    """
    logger.info("=== Starting Automated Retraining Workflow ===")
    
    # 1. Data Quality
    dq_report = generate_data_quality_report()
    
    # 2. Features & Datasets
    generate_feature_store()
    generate_all_datasets()
    
    # 3. Train models
    train_all_models()
    
    # 4. Evaluate
    val_results = evaluate_models()
    lgb_metrics = val_results.get("lightgbm_main_model", {})
    
    # 5. Register Challenger
    model_name = "match_winner_v1"
    register_model(
        model_name,
        "models/checkpoints/match_winner_v1.pkl",
        metrics=lgb_metrics,
        status="challenger"
    )
    
    # 6. Promotion decision
    promoted = promote_challenger_to_champion(model_name, min_acc_gain=0.0) if force_promote else promote_challenger_to_champion(model_name, min_acc_gain=0.005)
    
    # 7. Generate Model Card
    generate_model_card(model_name, metrics=lgb_metrics)
    
    logger.info(f"=== Retraining Workflow Complete! Promoted to Champion: {promoted} ===")
    return promoted


if __name__ == "__main__":
    run_retraining_pipeline(force_promote=True)
