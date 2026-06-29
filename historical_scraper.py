import os
import json
import asyncio
import random
import logging
from curl_cffi import requests
from vlr_scraper import is_tier1_event

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("historical_scraper")

BASE_URL = "http://localhost:3000"

async def harvest_and_save_vct_match_ids(limit: int = 500) -> list[str]:
    """Harvests the Match IDs for the last `limit` completed VCT matches.
    Filters out amateur/tier-3 tournaments. Saves to vct_match_ids.json.
    """
    logger.info(f"Starting Match ID harvesting to find last {limit} completed VCT matches...")
    match_ids = []
    page = 1
    
    consecutive_failures = 0
    async with requests.AsyncSession(impersonate="chrome") as client:
        while len(match_ids) < limit:
            if consecutive_failures >= 3:
                logger.error("Too many consecutive failures fetching match IDs. Aborting harvest.")
                break
            url = f"{BASE_URL}/v2/match?q=results&from_page={page}&to_page={page}"
            try:
                logger.info(f"Fetching result page {page}...")
                response = await client.get(url, timeout=30.0)
                if response.status_code != 200:
                    raise Exception(f"HTTP status {response.status_code}")
                data = response.json()
                
                segments = data.get("data", {}).get("segments", [])
                if not segments:
                    logger.warning("No more match segments returned by the API. Stopping harvest.")
                    break
                
                added_in_page = 0
                for s in segments:
                    tournament = s.get('tournament_name', '')
                    
                    if is_tier1_event(tournament):
                        match_page = s.get('match_page', '')
                        if match_page:
                            # Extract ID from e.g. "/248272/kiwoom-drx-vs-bilibili-gaming"
                            match_id = match_page.strip('/').split('/')[0]
                            if match_id and match_id not in match_ids:
                                match_ids.append(match_id)
                                added_in_page += 1
                                if len(match_ids) >= limit:
                                    break
                                    
                logger.info(f"Page {page} processed. Added {added_in_page} VCT matches. Current count: {len(match_ids)}")
                
                # If page added nothing and we are deep, check if we're hitting a wall
                if len(segments) < 50:
                    logger.info("Reached end of available historical pages in local API.")
                    break
                    
                consecutive_failures = 0
                page += 1
                # Small delay between requests to keep the server happy
                await asyncio.sleep(0.3)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error fetching page {page}: {e}. Retrying in 2 seconds (attempt {consecutive_failures}/3)...")
                await asyncio.sleep(2.0)
                
    # Save to JSON file if matches were harvested
    if match_ids:
        out_path = os.path.join(".", "vct_match_ids.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(match_ids, f, indent=4)
        
    logger.info(f"Successfully harvested {len(match_ids)} VCT Match IDs and saved to {out_path}")
    return match_ids

if __name__ == "__main__":
    asyncio.run(harvest_and_save_vct_match_ids(500))
