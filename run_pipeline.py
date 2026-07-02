import os
import sys
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("run_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ROOT_DIR = Path(__file__).resolve().parent

def run_step(command, description):
    logger.info(f"=== STARTING STEP: {description} ===")
    logger.info(f"Command: {command}")
    
    # We call the python executable from the active virtualenv if possible
    python_exe = sys.executable
    full_command = [python_exe] + command
    
    try:
        result = subprocess.run(full_command, cwd=str(ROOT_DIR), check=True, capture_output=True, text=True)
        # Log standard output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  [STDOUT] {line}")
        logger.info(f"=== COMPLETED STEP: {description} successfully ===\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"=== FAILED STEP: {description} ===")
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.error(f"  [STDOUT] {line}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"  [STDERR] {line}")
        raise e

def main():
    logger.info("=== STARTING FULL SYSTEM UPDATE PIPELINE ===")
    
    try:
        # Step 1: Sync Map Pool
        run_step(["sync_map_pool.py"], "Sync Dynamic Map Pool Rotation")
        
        # Step 2: Scrape Patch Notes List
        run_step(["scrapers/wiki_scraper.py"], "Scrape Patch Notes List from VCT Wiki")
        
        # Step 3: Ingest Latest Patch Wikitext
        run_step(["scrapers/patch_ingestor.py"], "Fetch & Parse Wikitext for New Patches")
        
        # Step 4: Analyze Patches and Compute Concept Drift
        run_step(["patch_analyzer.py"], "Calculate Concept Drift & Update Nerf Registry")
        
        # Step 5: Retrain XGBoost Model & Save Predictions
        run_step(["model_training.py"], "Retrain XGBoost Regressor on Decayed Feature Matrices")
        
        # Step 6: Pipeline Reset Trigger
        predictions_path = ROOT_DIR / "data" / "processed" / "xgb_predictions.json"
        # We can also delete it just to make sure the state is clean (although model_training.py writes it)
        # But actually, keeping it is fine as model_training.py wrote the fresh ones.
        
        logger.info("=== FULL SYSTEM UPDATE PIPELINE COMPLETED SUCCESSFULLY ===")
        print("SUCCESS")
        
    except Exception as e:
        logger.error(f"Pipeline execution aborted due to error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
