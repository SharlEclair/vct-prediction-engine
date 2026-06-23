import os
import csv
import httpx
import logging
import json

from patch_parser import PatchParser

logger = logging.getLogger("patch_ingestor")

PATCHES_CACHE_DIR = "./data/patches"
PROCESSED_PATCHES_DIR = "./data/processed/patches"

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
            # Remove 'v' prefix for Fandom Wiki URLs (e.g. v9.02 -> 9.02)
            if val.startswith("v"):
                val = val[1:]
            
            versions.append(val)
            if limit and len(versions) >= limit:
                break
    return versions

def fetch_from_wiki_api(version: str) -> str:
    """
    Queries the Fandom MediaWiki API to fetch raw wikitext revision content for a patch.
    """
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://valorant.fandom.com/wiki/Patch_Notes/{version}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    logger.info(f"Querying Fandom API for Patch Notes/{version}...")
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(api_url, params=params, headers=headers, timeout=15.0)
            
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
            raw_text = revisions[0].get("*") # fallback for older API schemas
            
        if not raw_text:
            raise PatchFetchError(f"Could not extract wikitext from slots/main for patch {version}.")
            
        return raw_text
        
    except httpx.HTTPError as he:
        raise PatchFetchError(f"HTTP request to Fandom API failed for patch {version}: {he}")
    except Exception as e:
        if not isinstance(e, PatchFetchError):
            raise PatchFetchError(f"Unexpected error when querying Fandom API for patch {version}: {e}")
        raise e

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        data = ingest_latest_patches(version_list=["9.0", "9.01", "9.02", "9.03", "9.04", "12.09"])
        print("\nSuccessfully ingested and parsed patches:", list(data.keys()))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
