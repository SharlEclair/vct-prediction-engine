import os
import json
import asyncio
import logging
import httpx
from datetime import datetime
import re
from api_client import get_match_details
from v5_simulation_engine import parse_simulation_match_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("harvest_2025_t1")

BASE_URL = "http://localhost:3000"
RAW_DATA_DIR = "./data/raw"

# Standard exclusions plus Challengers, Ascension, off-season, Game Changers
exclude_keywords = [
    'challengers', 'game changers', 'gc', 'premier', 'grassroots', 
    'fortress', 'collegiate', 'university', 'showmatch', 'community', 
    'trial', 'open qualifier', 'cup', 'weekly', 'monthly', 'amateur',
    'ascension', 'lcq', 'last chance', 'contenders', 'rivals', 'gamechangers',
    'off-season', 'off season'
]

async def harvest_2025_t1_match_ids():
    logger.info("Harvesting VCT 2025 Tier 1 match IDs...")
    match_ids = []
    
    # We probe pages 1 to 200 (covering late 2025 to early 2025)
    async with httpx.AsyncClient() as client:
        for page in range(1, 201):
            url = f"{BASE_URL}/v2/match?q=results&from_page={page}&to_page={page}"
            try:
                r = await client.get(url, timeout=30.0)
                data = r.json()
                segments = data.get("data", {}).get("segments", [])
                if not segments:
                    logger.info(f"Page {page} empty. Stopping page traversal.")
                    break
                
                # Check year of the first match in segment to verify if we are in 2025
                # Note: results segments do not have full dates, but we can query one match page if needed,
                # or estimate based on page range.
                # Let's inspect tournament name and retrieve match detail date for the first match to verify the year.
                sample_match_page = segments[0].get('match_page', '')
                if sample_match_page:
                    sample_mid = sample_match_page.strip('/').split('/')[0]
                    # Fetch details for the first match to verify year
                    details = await get_match_details(sample_mid, client)
                    if details and details.get("status") != "error":
                        date_str = details["data"]["segments"][0].get("date", "")
                        dt = parse_simulation_match_date(date_str)
                        logger.info(f"Page {page} sample match date: {date_str} (parsed year: {dt.year})")
                        if dt.year > 2025:
                            # Skip pages in 2026
                            continue
                        elif dt.year < 2025:
                            # Stop traversal if we've gone back to 2024
                            logger.info(f"Reached year {dt.year} (prior to 2025) on page {page}. Stopping.")
                            break
                
                added_in_page = 0
                for s in segments:
                    tournament = s.get('tournament_name', '')
                    name_lower = tournament.lower()
                    
                    # Ensure VCT Tier 1 criteria:
                    # 1. Must contain year '2025'
                    # 2. Must contain VCT brand keyword
                    # 3. Must contain Tier 1 league/tournament indicator
                    # 4. Must NOT contain any exclusions (challengers, ascension, etc.)
                    is_t1_vct = False
                    if '2025' in name_lower:
                        if ('vct' in name_lower or 'champions tour' in name_lower or 'valorant champions' in name_lower):
                            t1_indicators = ['kickoff', 'masters', 'champions', 'stage', 'americas', 'emea', 'pacific', 'cn']
                            if any(ind in name_lower for ind in t1_indicators):
                                if not any(ex in name_lower for ex in exclude_keywords):
                                    is_t1_vct = True
                            
                    if is_t1_vct:
                        match_page = s.get('match_page', '')
                        if match_page:
                            match_id = match_page.strip('/').split('/')[0]
                            if match_id and match_id not in match_ids:
                                match_ids.append(match_id)
                                added_in_page += 1
                                
                logger.info(f"Page {page}: processed. Added {added_in_page} VCT Tier 1 matches. Total 2025 Tier 1 matches collected: {len(match_ids)}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                
    # Save the match IDs list to a file
    temp_path = "./data/vct_2025_t1_match_ids.json"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(match_ids, f, indent=4)
        
    logger.info(f"Saved {len(match_ids)} 2025 VCT Tier 1 match IDs to {temp_path}")
    return match_ids

async def download_match_details(match_ids):
    logger.info(f"Starting ingestion of match details for {len(match_ids)} matches...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Load already cached matches to avoid duplicate work
    cached_count = 0
    todo_ids = []
    for mid in match_ids:
        path = os.path.join(RAW_DATA_DIR, f"match_{mid}.json")
        if os.path.exists(path):
            cached_count += 1
        else:
            todo_ids.append(mid)
            
    logger.info(f"{cached_count} matches already cached. {len(todo_ids)} to download.")
    
    if not todo_ids:
        logger.info("All matches are already downloaded!")
        return
        
    # Download in batches of 5
    batch_size = 5
    async with httpx.AsyncClient() as client:
        for i in range(0, len(todo_ids), batch_size):
            batch = todo_ids[i : i + batch_size]
            logger.info(f"Downloading batch {i // batch_size + 1}: {batch}")
            
            tasks = [get_match_details(mid, client) for mid in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for mid, data in zip(batch, results):
                if isinstance(data, Exception):
                    logger.error(f"Error downloading match {mid}: {data}")
                elif not data or data.get("status") == "error":
                    logger.error(f"Invalid match details for match {mid}: {data}")
                else:
                    out_path = os.path.join(RAW_DATA_DIR, f"match_{mid}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
            
            await asyncio.sleep(0.5)
            
    logger.info("Match details download complete.")

async def main():
    ids = await harvest_2025_t1_match_ids()
    if ids:
        await download_match_details(ids)

if __name__ == "__main__":
    asyncio.run(main())
