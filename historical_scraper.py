import os
import json
import asyncio
import logging
import httpx

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
    
    # We will exclude these strings from tournament names to filter out amateur/tier-3
    exclude_keywords = [
        'game changers', 'gc', 'premier', 'grassroots', 'fortress', 
        'collegiate', 'university', 'showmatch', 'community', 'trial',
        'open qualifier', 'cup', 'weekly', 'monthly', 'amateur'
    ]
    
    # We require tournament names to contain at least one of these main VCT event keywords
    vct_keywords = ['challengers', 'masters', 'champions', 'vct', 'champions tour']
    
    async with httpx.AsyncClient() as client:
        while len(match_ids) < limit:
            url = f"{BASE_URL}/v2/match?q=results&from_page={page}&to_page={page}"
            try:
                logger.info(f"Fetching result page {page}...")
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                segments = data.get("data", {}).get("segments", [])
                if not segments:
                    logger.warning("No more match segments returned by the API. Stopping harvest.")
                    break
                
                added_in_page = 0
                for s in segments:
                    tournament = s.get('tournament_name', '')
                    name_lower = tournament.lower()
                    
                    # Filtering criteria
                    is_vct = False
                    if any(kw in name_lower for kw in vct_keywords):
                        if not any(ex in name_lower for ex in exclude_keywords):
                            is_vct = True
                            
                    if is_vct:
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
                    
                page += 1
                # Small delay between requests to keep the server happy
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}. Retrying in 2 seconds...")
                await asyncio.sleep(2.0)
                
    # Save to JSON file
    out_path = os.path.join(".", "vct_match_ids.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(match_ids, f, indent=4)
        
    logger.info(f"Successfully harvested {len(match_ids)} VCT Match IDs and saved to {out_path}")
    return match_ids

if __name__ == "__main__":
    asyncio.run(harvest_and_save_vct_match_ids(500))
