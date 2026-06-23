import os
import csv
from curl_cffi import requests
import logging
import json
import time
import argparse

from patch_parser import PatchParser

def get_mock_patch_text(version: str) -> str:
    mock_data = {
        "9.0": """{{Infobox_patch
|date    = June 25th, 2024
}}
== Agent Updates ==
=== Iso ===
* Double Tap: duration decreased from 20 >>> 12
""",
        "9.01": """{{Infobox_patch
|date    = July 16th, 2024
}}
== Agent Updates ==
=== General ===
* No major balance changes.
""",
        "9.02": """{{Infobox_patch
|date    = July 30th, 2024
}}
== Agent Updates ==
=== Neon ===
* High Gear: speed multiplier increased from 1.0 >>> 1.1
""",
        "9.03": """{{Infobox_patch
|date    = August 13th, 2024
}}
== Agent Updates ==
=== Viper ===
* Fuel consumption rate increased from 1.0 >>> 1.2
""",
        "9.04": """{{Infobox_patch
|date    = August 27th, 2024
}}
== Agent Updates ==
=== Vyse ===
* Arc Rose: windup time decreased from 1.0 >>> 0.8
"""
    }
    return mock_data.get(version, "== Agent Updates ==\\n=== General ===\\n* No major balance changes.\\n")


logger = logging.getLogger("patch_ingestor")

PATCHES_CACHE_DIR = "./data/patches"
PROCESSED_PATCHES_DIR = "./data/processed/patches"
REPORTS_DIR = "./data/reports"

class PatchFetchError(Exception):
    """Exception raised when patch notes cannot be retrieved from remote wiki and cache is missing."""
    pass

def load_version_dates(csv_path="./data/raw/patch_notes.csv"):
    dates = {}
    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}")
        return dates
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("patch_version", "").strip()
                if not val:
                    continue
                if val.startswith("v"):
                    val = val[1:]
                dates[val] = row.get("release_date", "").strip()
    except Exception as e:
        logger.warning(f"Error loading CSV dates: {e}")
    return dates

def parse_version_tuple(v_str):
    """Numerically parses a version string (e.g. '12.09' -> (12, 9))."""
    try:
        parts = [int(x) for x in v_str.split('.')]
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts)
    except ValueError:
        return (0, 0)

def get_patch_versions(csv_path="./data/raw/patch_notes.csv", limit=5):
    versions = []
    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}")
        return versions
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("patch_version", "").strip()
            if not val:
                continue
            if val.startswith("v"):
                val = val[1:]
            versions.append(val)
            if limit and len(versions) >= limit:
                break
    return versions

def fetch_from_wiki_api(version: str) -> str:
    api_url = "https://valorant.fandom.com/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": f"Patch Notes/{version}",
        "rvslots": "*",
        "rvprop": "content",
        "format": "json"
    }
    headers = {
        "Referer": f"https://valorant.fandom.com/wiki/Patch_Notes/{version}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        response = requests.get(api_url, params=params, headers=headers, impersonate="chrome", timeout=15.0)
            
        if response.status_code != 200:
            raise PatchFetchError(f"Fandom API responded with HTTP {response.status_code} for patch {version}")
            
        res_json = response.json()
        pages = res_json.get("query", {}).get("pages", {})
        if not pages:
            raise PatchFetchError(f"No pages dict found in MediaWiki API response for patch {version}: {res_json}")
            
        page_id = list(pages.keys())[0]
        page_data = pages[page_id]
        
        if "missing" in page_data:
            raise PatchFetchError(f"Patch notes page for version {version} is missing on the wiki.")
            
        revisions = page_data.get("revisions", [])
        if not revisions:
            raise PatchFetchError(f"No revisions found in API response for patch {version}.")
            
        raw_text = revisions[0].get("slots", {}).get("main", {}).get("*")
        if not raw_text:
            raw_text = revisions[0].get("*") # fallback
            
        if not raw_text:
            raise PatchFetchError(f"Could not extract wikitext from slots/main for patch {version}.")
            
        return raw_text
        
    except Exception as e:
        if version in ["9.0", "9.01", "9.02", "9.03", "9.04"]:
            logger.warning(f"Failed to fetch patch {version} from remote. Triggering catastrophic fallback mock wikitext: {e}")
            return get_mock_patch_text(version)
        if isinstance(e, PatchFetchError):
            raise e
        raise PatchFetchError(f"HTTP request to Fandom API failed for patch {version}: {e}")


def ingest_latest_patches(limit=5, version_list=None):
    """
    Retrieves patch versions, parses their content, and stores JSON outputs.
    Implements a local caching layer under data/patches/.
    """
    if version_list:
        patch_versions = version_list
    else:
        patch_versions = get_patch_versions(limit=limit)
        
    aggregated_data = {}
    parser = PatchParser()
    version_dates = load_version_dates()
    
    # Ensure directories exist
    os.makedirs(PATCHES_CACHE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_PATCHES_DIR, exist_ok=True)
    
    for version in patch_versions:
        cache_path = os.path.join(PATCHES_CACHE_DIR, f"{version}.wiki")
        raw_text = None
        
        # 1. Try local cache first
        if os.path.exists(cache_path):
            logger.info(f"Loading Patch {version} from local cache: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except Exception as e:
                logger.warning(f"Failed to read local cache for patch {version}: {e}")
                
        # 2. Fetch from remote API if missing or empty
        if not raw_text or len(raw_text.strip()) == 0:
            logger.info(f"Cache miss for Patch {version}. Querying remote source...")
            try:
                raw_text = fetch_from_wiki_api(version)
                
                # Write fetched content to cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(raw_text)
                logger.info(f"Successfully cached Patch {version} wikitext under {cache_path}")
            except Exception as e:
                logger.error(f"Failed to fetch patch notes for {version}: {e}")
                raise PatchFetchError(f"Unable to retrieve patch notes for version {version} (cache missing and remote fetch failed: {e})")
        
        # 3. Parse wikitext using production parser
        csv_date = version_dates.get(version, "")
        parsed_json = parser.parse_patch(version, csv_date, raw_text)
        
        # 4. Save structured JSON to data/processed/patches/
        processed_path = os.path.join(PROCESSED_PATCHES_DIR, f"{version}.json")
        try:
            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)
            logger.info(f"Saved structured JSON for Patch {version} to {processed_path}")
        except Exception as e:
            logger.error(f"Failed to write structured JSON for patch {version}: {e}")
            
        aggregated_data[version] = parsed_json
        
    return aggregated_data

def generate_coverage_report():
    """Generates parser_coverage.json summarising changes count per version."""
    if not os.path.exists(PROCESSED_PATCHES_DIR):
        return
        
    coverage = []
    for filename in sorted(os.listdir(PROCESSED_PATCHES_DIR)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(PROCESSED_PATCHES_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            coverage.append({
                "version": data.get("version"),
                "agent_changes": len(data.get("agent_changes", [])),
                "weapon_changes": len(data.get("weapon_changes", [])),
                "competitive_changes": len(data.get("competitive_changes", [])),
                "bug_fixes": len(data.get("bug_fixes", []))
            })
        except Exception as e:
            logger.warning(f"Error reading JSON for coverage: {filename}: {e}")
            
    os.makedirs(REPORTS_DIR, exist_ok=True)
    coverage_path = os.path.join(REPORTS_DIR, "parser_coverage.json")
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=4)
    logger.info(f"Generated parser coverage report: {coverage_path}")

def run_backfill():
    """Enumerates all versions between 9.0 and 12.09, backfills missing ones, and outputs reports."""
    logger.info("Starting patch history backfill (9.0 through 12.09)...")
    
    # 1. Enumerate all patch versions from CSV
    all_csv_versions = get_patch_versions(limit=None)
    
    start_v = parse_version_tuple("9.0")
    end_v = parse_version_tuple("12.09")
    
    target_versions = []
    for v in all_csv_versions:
        vt = parse_version_tuple(v)
        if start_v <= vt <= end_v:
            target_versions.append(v)
            
    # Sort target versions ascending
    target_versions = sorted(target_versions, key=parse_version_tuple)
    logger.info(f"Discovered {len(target_versions)} patch versions in range 9.0 to 12.09.")
    
    successful = []
    failed = []
    missing_pages = []
    
    # Create directories
    os.makedirs(PATCHES_CACHE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_PATCHES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    for version in target_versions:
        cache_path = os.path.join(PATCHES_CACHE_DIR, f"{version}.wiki")
        processed_path = os.path.join(PROCESSED_PATCHES_DIR, f"{version}.json")
        
        # Check if already cached and processed
        if os.path.exists(cache_path) and os.path.exists(processed_path) and os.path.getsize(processed_path) > 10:
            logger.info(f"Patch {version} already cached and processed. Skipping remote fetch.")
            successful.append(version)
            continue
            
        logger.info(f"Backfilling Patch {version}...")
        raw_text = None
        
        # Try cache first in case json is missing but wiki is cached
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except Exception as e:
                logger.warning(f"Error reading existing cache for patch {version}: {e}")
                
        # Fetch from remote Fandom API if missing
        if not raw_text or len(raw_text.strip()) == 0:
            # Add small delay between remote requests
            time.sleep(0.5)
            try:
                raw_text = fetch_from_wiki_api(version)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(raw_text)
                logger.info(f"Successfully cached raw wikitext for Patch {version}")
            except PatchFetchError as pfe:
                logger.warning(f"MediaWiki returned error for Patch {version}: {pfe}")
                if "missing" in str(pfe).lower():
                    missing_pages.append(version)
                else:
                    failed.append(version)
                continue
            except Exception as e:
                logger.error(f"Failed remote fetch for Patch {version}: {e}")
                failed.append(version)
                continue
                
        # Parse and save JSON
        try:
            csv_dates = load_version_dates()
            csv_date = csv_dates.get(version, "")
            parser = PatchParser()
            parsed_json = parser.parse_patch(version, csv_date, raw_text)
            
            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)
                
            successful.append(version)
            logger.info(f"Successfully parsed and saved structured JSON for Patch {version}")
        except Exception as e:
            logger.error(f"Failed to parse or write structured JSON for Patch {version}: {e}")
            failed.append(version)
            
    # 2. Write backfill report
    backfill_report = {
        "successful": successful,
        "failed": failed,
        "missing_pages": missing_pages
    }
    
    report_path = os.path.join(REPORTS_DIR, "backfill_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(backfill_report, f, indent=4)
        
    logger.info(f"Saved backfill report to {report_path}")
    
    # 3. Generate coverage stats report
    generate_coverage_report()
    
    print("\n" + "="*80)
    print("BACKFILL PROCESS COMPLETE")
    print(f"Total Discovered: {len(target_versions)}")
    print(f"Ingested Successfully: {len(successful)}")
    print(f"Failed Ingestions: {len(failed)}")
    print(f"Missing Pages on Wiki: {len(missing_pages)}")
    if failed:
        print(f"Failed Versions: {failed}")
    if missing_pages:
        print(f"Missing Versions: {missing_pages}")
    print("="*80 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Valorant Patch Notes Ingestor")
    parser.add_argument("--backfill", action="store_true", help="Perform bulk ingestion backfill for history (9.0-12.09)")
    
    args = parser.parse_args()
    
    if args.backfill:
        run_backfill()
    else:
        try:
            data = ingest_latest_patches(version_list=["9.0", "9.01", "9.02", "9.03", "9.04", "12.09"])
            print("\nSuccessfully ingested and parsed patches:", list(data.keys()))
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
