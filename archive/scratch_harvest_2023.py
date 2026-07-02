import os
import json
import asyncio
import logging
import httpx
from api_client import get_match_details

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("harvest_2023")

BASE_URL = "http://localhost:3000"
RAW_DATA_DIR = "./data/raw"

exclude_keywords = [
    'game changers', 'gc', 'premier', 'grassroots', 'fortress', 
    'collegiate', 'university', 'showmatch', 'community', 'trial',
    'open qualifier', 'cup', 'weekly', 'monthly', 'amateur'
]
vct_keywords = ['challengers', 'masters', 'champions', 'vct', 'champions tour']

async def harvest_2023_match_ids():
    temp_path = "./data/vct_2023_match_ids.json"
    if os.path.exists(temp_path):
        logger.info("Found cached 2023 VCT match IDs, loading to skip harvesting pages...")
        with open(temp_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Harvesting 2023 VCT match IDs...")
    match_ids = []
    
    # Probe pages 220 to 330
    async with httpx.AsyncClient() as client:
        for page in range(220, 331):
            url = f"{BASE_URL}/v2/match?q=results&from_page={page}&to_page={page}"
            try:
                r = await client.get(url, timeout=30.0)
                data = r.json()
                segments = data.get("data", {}).get("segments", [])
                if not segments:
                    logger.info(f"Page {page} empty. Stopping page traversal.")
                    break
                
                added_in_page = 0
                for s in segments:
                    tournament = s.get('tournament_name', '')
                    name_lower = tournament.lower()
                    
                    is_vct = False
                    if any(kw in name_lower for kw in vct_keywords):
                        if not any(ex in name_lower for ex in exclude_keywords):
                            is_vct = True
                            
                    if is_vct:
                        match_page = s.get('match_page', '')
                        if match_page:
                            match_id = match_page.strip('/').split('/')[0]
                            if match_id and match_id not in match_ids:
                                match_ids.append(match_id)
                                added_in_page += 1
                                
                logger.info(f"Page {page}: processed. Found {added_in_page} VCT matches. Total 2023 matches collected: {len(match_ids)}")
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                
    # Save the match IDs list to a temporary file
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(match_ids, f, indent=4)
        
    logger.info(f"Saved {len(match_ids)} 2023 VCT match IDs to {temp_path}")
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
    ids = await harvest_2023_match_ids()
    if ids:
        await download_match_details(ids)

if __name__ == "__main__":
    asyncio.run(main())
