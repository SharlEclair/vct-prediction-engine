import os
import sys
sys.path.insert(0, ".")

import logging
import argparse
from datetime import datetime

from ml.data_quality import generate_data_quality_report
from ml.feature_builder import generate_feature_store
from ml.dataset_builder import generate_all_datasets
from ml.train import train_all_models
from ml.evaluate import evaluate_models
from ml.model_registry import register_model, promote_challenger_to_champion
from ml.model_card import generate_model_card
from ml.backtest import walk_forward_backtest

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def run_pipeline(skip_backtest: bool = False):
    """Executes complete end-to-end V10 MLOps & Machine Learning Pipeline."""
    start_time = datetime.now()
    logger.info("=== Starting VCT Prediction Engine MLOps Pipeline ===")

    # Step 1: Data Quality Report
    logger.info("--- Step 1: Generating Data Quality Report ---")
    generate_data_quality_report()

    # Step 2: Feature Store Foundation
    logger.info("--- Step 2: Generating Feature Store ---")
    generate_feature_store()

    # Step 3: Supervised Dataset Assembly
    logger.info("--- Step 3: Building Supervised Datasets ---")
    generate_all_datasets()

    # Step 4: Model Training & Experiment Tracking
    logger.info("--- Step 4: Training ML Models & Logging Experiments ---")
    train_all_models()

    # Step 5: Model Evaluation, ECE/MCE & SHAP Dashboard
    logger.info("--- Step 5: Evaluating Models, ECE/MCE & Feature Importance ---")
    val_results = evaluate_models()

    # Step 6: Model Registry & Staging
    logger.info("--- Step 6: Registering & Staging Production Models ---")
    lgb_metrics = val_results.get("lightgbm_main_model", {})
    register_model("match_winner_v1", "models/checkpoints/match_winner_v1.pkl", metrics=lgb_metrics, status="challenger")
    promote_challenger_to_champion("match_winner_v1", min_acc_gain=0.0)

    # Step 7: Model Cards
    logger.info("--- Step 7: Generating Model Cards ---")
    generate_model_card("match_winner_v1", metrics=lgb_metrics)

    # Step 8: Historical Backtesting
    if not skip_backtest:
        logger.info("--- Step 8: Running Walk-Forward Backtesting ---")
        walk_forward_backtest()
    else:
        logger.info("--- Step 8: Skipped Backtesting ---")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"=== ML Pipeline Completed Successfully in {elapsed:.2f} seconds ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCT Prediction Engine Master MLOps Pipeline")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip historical backtesting step")
    args = parser.parse_args()

    run_pipeline(skip_backtest=args.skip_backtest)
