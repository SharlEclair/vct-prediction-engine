"""
Rescrape all 2026 matches using updated vlr_scraper.py and clean_match_data.py.
"""
import sys
sys.path.insert(0, ".")

import glob
import json
import logging
import os
import re
import time
from scrapers.vlr_scraper import parse_vlr_match
from scrapers.clean_match_data import clean_single_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_target_match_ids(target_years: list[str] = ["2024", "2025", "2026"]) -> list[str]:
    raw_files = glob.glob("data/raw/match_*.json")
    target_matches = []
    
    for f in raw_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                d = json.load(file)
                segments = d.get("data", {}).get("segments", [])
                if not segments and isinstance(d.get("segments"), list):
                    segments = d.get("segments")
                    
                is_target = False
                for seg in segments:
                    date_str = str(seg.get("date", ""))
                    event_str = str(seg.get("event", ""))
                    
                    if any(y in date_str or y in event_str for y in target_years):
                        is_target = True
                        break
                        
                if is_target:
                    m_id = re.search(r"match_(\d+)\.json", f)
                    if m_id:
                        target_matches.append(m_id.group(1))
        except Exception:
            pass
            
    return sorted(list(set(target_matches)))

def rescrape_all_target_matches():
    match_ids = get_target_match_ids(["2024", "2025", "2026"])
    total = len(match_ids)
    logging.info(f"Found {total} matches from 2024, 2025, and 2026 to re-scrape.")
    
    success_count = 0
    fail_count = 0
    
    for idx, m_id in enumerate(match_ids, start=1):
        fpath = f"data/raw/match_{m_id}.json"
        logging.info(f"[{idx}/{total}] Re-scraping match {m_id}...")
        try:
            segments = parse_vlr_match(m_id)
            if segments:
                out_payload = {"status": "success", "data": {"status": 200, "segments": segments}}
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(out_payload, f, indent=4)
                
                clean_single_file(fpath)
                success_count += 1
                logging.info(f"Successfully re-scraped & cleaned match {m_id} ({len(segments)} maps)")
            else:
                logging.warning(f"No segment data returned for match {m_id}")
                fail_count += 1
        except Exception as e:
            logging.error(f"Error re-scraping match {m_id}: {e}")
            fail_count += 1
            
        time.sleep(1.5)
        
    logging.info(f"Re-scrape finished! Successfully updated: {success_count}/{total} matches (Failed: {fail_count}).")

if __name__ == "__main__":
    rescrape_all_target_matches()
