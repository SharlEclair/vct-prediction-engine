import os
import sys
import re
import json
import logging
import glob
from datetime import datetime
from selectolax.parser import HTMLParser

# Ensure parent and current directories are in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scrapers.vlr_scraper import (
    is_tier1_event,
    parse_vlr_match,
    clean_text,
    fetch_url_with_curl,
    VLR_BASE_URL
)
from feature_engineering import parse_match_date

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("incremental_vlr_scraper")

RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")

def get_latest_local_match_date() -> datetime:
    """Scan all raw match files to find the latest match timestamp."""
    latest_date = datetime(2023, 1, 1)  # Default fallback date
    match_files = glob.glob(os.path.join(RAW_DIR, "match_*.json"))
    
    logger.info(f"Scanning {len(match_files)} local raw matches for the latest timestamp...")
    for filepath in match_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = data.get("data", {}).get("segments", [])
            if segments:
                date_str = segments[0].get("date")
                if date_str:
                    m_date = parse_match_date(date_str)
                    if m_date > latest_date:
                        latest_date = m_date
        except Exception as e:
            continue
            
    logger.info(f"Most recent cached match date found: {latest_date}")
    return latest_date

def run_incremental_scrape():
    """Paginate and scrape new VCT Tier 1 matches until hitting a match older than latest cached date."""
    latest_local_date = get_latest_local_match_date()
    os.makedirs(RAW_DIR, exist_ok=True)
    
    page = 1
    new_matches_scraped = 0
    stop_scraping = False
    
    while not stop_scraping:
        results_url = f"{VLR_BASE_URL}/matches/results?page={page}"
        logger.info(f"Fetching matches results page {page}: {results_url}")
        html_text = fetch_url_with_curl(results_url)
        if not html_text:
            logger.warning("Empty response or request failed. Stopping.")
            break
            
        parser = HTMLParser(html_text)
        match_items = parser.css("a.wf-module-item")
        if not match_items:
            logger.info("No more match items found. Ending crawl.")
            break
            
        logger.info(f"Found {len(match_items)} match items on page {page}.")
        for item in match_items:
            href = item.attributes.get("href", "")
            if not href:
                continue
                
            # Extract match ID
            match_id = href.strip("/").split("/")[0]
            if not match_id.isdigit():
                continue
                
            # Extract tournament/event
            tourney_elem = item.css_first("div.match-item-event")
            tourney_name = clean_text(tourney_elem.text()) if tourney_elem else ""
            
            # We only care about Tier 1 events
            if not is_tier1_event(tourney_name):
                continue
                
            # Check if we already have it
            dest_filepath = os.path.join(RAW_DIR, f"match_{match_id}.json")
            if os.path.exists(dest_filepath):
                logger.info(f"Match {match_id} already exists locally. Skipping details page fetch.")
                # We still need to parse the date to see if we should stop. But wait!
                # Since we already have this match, let's load it from disk to check the date!
                try:
                    with open(dest_filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    segments = data.get("data", {}).get("segments", [])
                    if segments and segments[0].get("date"):
                        m_date = parse_match_date(segments[0]["date"])
                        if m_date < latest_local_date:
                            logger.info(f"Encountered existing match {match_id} from {m_date} which is older than {latest_local_date}. Stop signal reached.")
                            stop_scraping = True
                            break
                except Exception:
                    pass
                continue
                
            # Scrape match details
            logger.info(f"New Tier 1 match detected: {match_id} ({tourney_name}). Scraping details...")
            segments = parse_vlr_match(match_id)
            if not segments:
                logger.warning(f"Could not parse match details for {match_id}. Skipping.")
                continue
                
            # Check date from segments
            date_str = segments[0].get("date")
            if date_str:
                m_date = parse_match_date(date_str)
                if m_date < latest_local_date:
                    logger.info(f"Scraped match {match_id} date {m_date} is older than latest local date {latest_local_date}. Stopping scraper.")
                    stop_scraping = True
                    break
            
            # Save the new match details
            payload = {
                "status": "success",
                "data": {
                    "status": 200,
                    "segments": segments
                }
            }
            with open(dest_filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved new match {match_id} successfully.")
            new_matches_scraped += 1
            
        if stop_scraping:
            break
            
        page += 1
        
    logger.info(f"Incremental scrape complete. Added {new_matches_scraped} new matches.")
    # Return count for parent processes to read
    print(f"NEW_MATCHES_SCRAPED:{new_matches_scraped}")
    return new_matches_scraped

if __name__ == "__main__":
    run_incremental_scrape()
