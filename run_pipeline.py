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

def main(whitelist: str = None):
    logger.info("=== STARTING FULL SYSTEM UPDATE PIPELINE ===")
    
    try:
        # Step 1: Incremental VLR Scrape
        cmd_vlr = ["scrapers/incremental_vlr_scraper.py"]
        if whitelist:
            cmd_vlr += ["--whitelist", whitelist]
        run_step(cmd_vlr, "Incremental VLR Matches Scrape")
        
        # Step 2: Ingest VFL pricing/roster state
        run_step(["scrapers/vfl_scraper.py"], "Scrape VFL Slate & Player Projections")
        
        # Step 3: Sync Map Pool
        run_step(["sync_map_pool.py"], "Sync Dynamic Map Pool Rotation")
        
        # Step 4: Scrape Patch Notes List
        run_step(["scrapers/wiki_scraper.py"], "Scrape Patch Notes List from VCT Wiki")
        
        # Step 5: Ingest Latest Patch Wikitext
        run_step(["scrapers/patch_ingestor.py"], "Fetch & Parse Wikitext for New Patches")
        
        # Step 6: Analyze Patches and Compute Concept Drift
        run_step(["v8_patch_analyzer.py"], "Calculate Concept Drift & Update Nerf Registry")
        
        # Step 7: Build Features
        run_step(["feature_engineering.py"], "Build Feature Matrix Store")
        
        # Step 8: Retrain XGBoost Model & Save Predictions
        run_step(["model_training.py"], "Retrain XGBoost Regressor on Decayed Feature Matrices")
        
        logger.info("=== FULL SYSTEM UPDATE PIPELINE COMPLETED SUCCESSFULLY ===")
        print("SUCCESS")
        
    except Exception as e:
        logger.error(f"Pipeline execution aborted due to error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VCT Prediction Model Pipeline Orchestrator")
    parser.add_argument("--whitelist", type=str, default=None, help="Comma-separated whitelisted VLR events")
    args = parser.parse_args()
    
    main(whitelist=args.whitelist)
