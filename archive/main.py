import asyncio
import os
import json
import logging
import httpx
import pandas as pd
from api_client import get_match_details, get_player_stats
from wiki_scraper import scrape_patch_notes, scrape_agent_roles
from feature_engineering import build_feature_store
from historical_scraper import harvest_and_save_vct_match_ids
from model_pipeline import run_modeling_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_orchestrator")

RAW_DATA_DIR = os.path.join(".", "data", "raw")

async def main():
    logger.info("Initializing Scaled VCT ML Data Pipeline...")
    
    # Ensure raw output directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    logger.info(f"Ensured raw data directory exists at: {RAW_DATA_DIR}")
    
    # 1. Harvest Match IDs (scaled to 500 completed VCT matches)
    try:
        match_ids = await harvest_and_save_vct_match_ids(limit=500)
        logger.info(f"Harvested {len(match_ids)} Match IDs.")
    except Exception as e:
        logger.error(f"Failed to harvest Match IDs: {e}")
        # Try to load cached match IDs as fallback
        cache_path = os.path.join(".", "vct_match_ids.json")
        if os.path.exists(cache_path):
            logger.info("Loading cached match IDs from vct_match_ids.json as fallback...")
            with open(cache_path, "r", encoding="utf-8") as f:
                match_ids = json.load(f)
        else:
            raise e
            
    if not match_ids:
        logger.error("No Match IDs harvested or loaded. Pipeline cannot proceed.")
        return

    # Initialize shared AsyncClient
    async with httpx.AsyncClient() as client:
        # Fetch metadata
        logger.info("Fetching global metadata (stats, patch notes, agent roles)...")
        player_stats_task = get_player_stats(client)
        patch_notes_task = scrape_patch_notes(client)
        agent_roles_task = scrape_agent_roles(client)
        
        metadata_results = await asyncio.gather(
            player_stats_task, patch_notes_task, agent_roles_task, 
            return_exceptions=True
        )
        
        player_stats_res, patch_notes_res, agent_roles_res = metadata_results
        
        # --- Save Player Stats ---
        if isinstance(player_stats_res, Exception):
            logger.error(f"Failed to fetch player stats: {player_stats_res}")
        else:
            out_path = os.path.join(RAW_DATA_DIR, "player_stats.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(player_stats_res, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved player stats to {out_path}")
            
        # --- Save Patch Notes ---
        if isinstance(patch_notes_res, Exception):
            logger.error(f"Failed to scrape patch notes: {patch_notes_res}")
        else:
            out_path = os.path.join(RAW_DATA_DIR, "patch_notes.csv")
            patch_notes_res.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"Saved {len(patch_notes_res)} patch notes records to {out_path}")
            
        # --- Save Agent Roles ---
        if isinstance(agent_roles_res, Exception):
            logger.error(f"Failed to scrape agent roles: {agent_roles_res}")
        else:
            out_path = os.path.join(RAW_DATA_DIR, "agent_roles.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(agent_roles_res, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved {len(agent_roles_res)} agent role mappings to {out_path}")

        # --- Filter out already downloaded Match IDs ---
        todo_match_ids = []
        for mid in match_ids:
            file_path = os.path.join(RAW_DATA_DIR, f"match_{mid}.json")
            if not os.path.exists(file_path):
                todo_match_ids.append(mid)
                
        logger.info(f"Out of {len(match_ids)} Match IDs, {len(match_ids) - len(todo_match_ids)} are already cached. Downloading remaining {len(todo_match_ids)} matches...")

        # --- Save Match Details in Batches of 5 ---
        batch_size = 5
        todo_match_results = []
        
        if todo_match_ids:
            logger.info(f"Beginning batched ingestion for {len(todo_match_ids)} remaining match details...")
            for i in range(0, len(todo_match_ids), batch_size):
                batch_ids = todo_match_ids[i:i+batch_size]
                current_batch_num = i // batch_size + 1
                total_batches = (len(todo_match_ids) - 1) // batch_size + 1
                
                logger.info(f"Processing batch {current_batch_num}/{total_batches}: {batch_ids}")
                
                batch_tasks = [get_match_details(mid, client) for mid in batch_ids]
                batch_res = await asyncio.gather(*batch_tasks, return_exceptions=True)
                todo_match_results.extend(batch_res)
                
                # Sleep 0.5 seconds between batches to throttle rate
                await asyncio.sleep(0.5)
                
            # --- Save Match Details to Files ---
            success_count = 0
            for match_id, match_data in zip(todo_match_ids, todo_match_results):
                if isinstance(match_data, Exception):
                    logger.error(f"Failed to fetch match details for {match_id}: {match_data}")
                elif not match_data or match_data.get("status") == "error":
                    logger.error(f"Invalid or error response for match {match_id}: {match_data}")
                else:
                    out_path = os.path.join(RAW_DATA_DIR, f"match_{match_id}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(match_data, f, indent=4, ensure_ascii=False)
                    success_count += 1
                    
            logger.info(f"Ingested remaining matches: {success_count} successfully saved, {len(todo_match_ids) - success_count} failed.")
        else:
            logger.info("All matches are already cached. Skipping download.")

    logger.info("Pipeline data ingestion complete.")
    
    # 2. Build Feature Store
    import sys
    if "--no-fe" not in sys.argv:
        logger.info("Running Phase 2: Point-In-Time Feature Engineering...")
        try:
            build_feature_store()
            logger.info("Feature engineering complete.")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise e
    else:
        logger.info("Skipping feature engineering (--no-fe specified).")

    # 3. Model Pipeline & SHAP Explainability
    logger.info("Running Phase 3: Modeling & Explainability...")
    try:
        results = run_modeling_pipeline()
        logger.info("Modeling pipeline complete.")
        print("\n" + "="*50)
        print("VCT PREDICTION PIPELINE RESULTS")
        print("="*50)
        print(f"Test Match: {results['team_a']} vs {results['team_b']} (ID: {results['test_match_id']})")
        print(f"Predicted Probability of {results['team_a']} Winning: {results['win_prob_team_a']:.2%}")
        print(f"Predicted Probability of {results['team_b']} Winning: {results['win_prob_team_b']:.2%}")
        print(f"SHAP explanation image saved to: {results['shap_plot_path']}")
        print("="*50 + "\n")
    except Exception as e:
        logger.error(f"Modeling pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    # Ensure proper event loop policy on Windows to avoid warnings
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
