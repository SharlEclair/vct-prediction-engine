"""
Bulk Scraper for VCT 2025 and 2026 matches:
- Kickoff
- Stage 1
- Stage 2
- Masters
- Champions

Extracts full Schema v1.0 JSONs (Overview, Performance, Economy) for all target matches and saves them to data/raw/match_{id}.json.
"""
import os
import sys
import re
import json
import time
import logging
from datetime import datetime
from selectolax.parser import HTMLParser

# Ensure root workspace is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scrapers.vlr_scraper import (
    parse_vlr_match,
    fetch_url_with_curl,
    clean_text,
    is_tier1_event,
    VLR_BASE_URL
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("scrape_vct_2025_2026")

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
TARGET_IDS_FILE = os.path.join(ROOT_DIR, "data", "vct_2025_2026_target_ids.json")


def is_target_vct_event(event_name: str) -> bool:
    """
    Checks if an event is a VCT 2025 or 2026 Tier-1 match:
    - Kickoff
    - Stage 1
    - Stage 2
    - Masters
    - Champions
    """
    if not event_name:
        return False
    
    name_lower = event_name.lower()
    
    # 1. Year check
    if not ("2025" in name_lower or "2026" in name_lower):
        return False
        
    # 2. Strict Tier 1 check (excludes Game Changers, Challengers, Ascension, etc.)
    if not is_tier1_event(event_name):
        return False
        
    # 3. Stage / Tournament check
    stages = ["kickoff", "stage 1", "stage 2", "masters", "champions"]
    return any(stg in name_lower for stg in stages)


def harvest_target_match_ids(max_pages: int = 55) -> list[dict]:
    """
    Paginates through VLR match results pages and collects all matching
    VCT 2025 & 2026 Kickoff, Stage 1, Stage 2, Masters, and Champions match IDs.
    """
    logger.info(f"Starting harvest of VCT 2025 & 2026 match IDs (max_pages={max_pages})...")
    os.makedirs(os.path.dirname(TARGET_IDS_FILE), exist_ok=True)
    
    harvested = []
    seen_ids = set()
    consecutive_non_target_pages = 0
    
    for page in range(1, max_pages + 1):
        url = f"{VLR_BASE_URL}/matches/results?page={page}"
        logger.info(f"Fetching results page {page}/{max_pages}: {url}")
        html_text = fetch_url_with_curl(url)
        if not html_text:
            logger.warning(f"Failed to fetch results page {page}. Skipping.")
            continue
            
        parser = HTMLParser(html_text)
        items = parser.css("a.wf-module-item")
        if not items:
            logger.info("No more match items found. Ending harvest.")
            break
            
        page_target_count = 0
        has_2025_or_2026_matches = False
        
        for item in items:
            href = item.attributes.get("href", "")
            if not href:
                continue
                
            m_id = href.strip("/").split("/")[0]
            if not m_id.isdigit():
                continue
                
            event_el = item.css_first("div.match-item-event")
            event_name = clean_text(event_el.text()) if event_el else ""
            
            eta_el = item.css_first("div.match-item-date")
            date_txt = clean_text(eta_el.text()) if eta_el else ""
            
            if "2025" in event_name or "2026" in event_name or "2025" in date_txt or "2026" in date_txt:
                has_2025_or_2026_matches = True
                
            if is_target_vct_event(event_name):
                if m_id not in seen_ids:
                    seen_ids.add(m_id)
                    harvested.append({
                        "match_id": m_id,
                        "event": event_name,
                        "date_context": date_txt
                    })
                    page_target_count += 1
                    
        logger.info(f"Page {page}: found {page_target_count} target VCT matches (Total cumulative: {len(harvested)}).")
        
        if not has_2025_or_2026_matches:
            consecutive_non_target_pages += 1
            if consecutive_non_target_pages >= 3:
                logger.info("Encountered 3 consecutive pages with no 2025/2026 matches. Stopping harvest.")
                break
        else:
            consecutive_non_target_pages = 0
            
    # Save target list to disk
    with open(TARGET_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(harvested, f, indent=2)
        
    logger.info(f"Harvest complete! Collected {len(harvested)} VCT 2025/2026 match IDs.")
    return harvested


def scrape_all_vct_matches(force_rescrape: bool = False, max_pages: int = 55):
    """Scrapes all harvested VCT 2025 & 2026 matches in Schema v1.0 format."""
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Load or harvest target match IDs
    if os.path.exists(TARGET_IDS_FILE) and not force_rescrape:
        try:
            with open(TARGET_IDS_FILE, "r", encoding="utf-8") as f:
                target_matches = json.load(f)
            logger.info(f"Loaded {len(target_matches)} target matches from {TARGET_IDS_FILE}.")
        except Exception:
            target_matches = harvest_target_match_ids(max_pages=max_pages)
    else:
        target_matches = harvest_target_match_ids(max_pages=max_pages)
        
    total = len(target_matches)
    success_count = 0
    skipped_count = 0
    fail_count = 0
    
    logger.info(f"Starting scraping of {total} VCT 2025/2026 matches into {RAW_DIR}...")
    
    for idx, item in enumerate(target_matches, start=1):
        m_id = item["match_id"]
        event_name = item.get("event", "VCT Match")
        fpath = os.path.join(RAW_DIR, f"match_{m_id}.json")
        
        # Check if already scraped in Schema v1.0
        if os.path.exists(fpath) and not force_rescrape:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if existing_data.get("schema_version") == "1.0":
                    logger.info(f"[{idx}/{total}] Match {m_id} already exists in Schema v1.0. Skipping.")
                    skipped_count += 1
                    continue
            except Exception:
                pass
                
        logger.info(f"[{idx}/{total}] Scraping match {m_id} ({event_name})...")
        try:
            match_data = parse_vlr_match(m_id)
            if match_data and match_data.get("overview"):
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(match_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                logger.info(f"  -> Successfully saved match_{m_id}.json (Schema v1.0)")
            else:
                fail_count += 1
                logger.warning(f"  -> Failed to parse details for match {m_id}")
        except Exception as e:
            fail_count += 1
            logger.error(f"  -> Error scraping match {m_id}: {e}")
            
    logger.info("=" * 60)
    logger.info(f"VCT 2025/2026 Scrape Finished!")
    logger.info(f"  - Total Targets: {total}")
    logger.info(f"  - Successfully Scraped: {success_count}")
    logger.info(f"  - Already Cached (v1.0): {skipped_count}")
    logger.info(f"  - Failed: {fail_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bulk scrape VCT 2025 & 2026 matches in Schema v1.0")
    parser.add_argument("--force", action="store_true", help="Force re-scraping even if cached v1.0 file exists")
    parser.add_argument("--max-pages", type=int, default=55, help="Maximum results pages to harvest (default: 55)")
    args = parser.parse_args()
    
    scrape_all_vct_matches(force_rescrape=args.force, max_pages=args.max_pages)
