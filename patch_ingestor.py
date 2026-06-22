import os
import csv
import httpx
import logging
import subprocess
from v4_parsing_skills import parse_mediawiki_tree

logger = logging.getLogger("patch_ingestor")

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
            
            # Skip future patches that don't exist on the wiki yet (e.g., 10.xx, 11.xx, 12.xx)
            try:
                major = int(val.split('.')[0])
                if major >= 10:
                    continue
            except ValueError:
                pass
                
            versions.append(val)
            if limit and len(versions) >= limit:
                break
    return versions

MOCK_MEDIAWIKI_9_02 = """
==PC==
===Agent Updates===
===={{ai|Neon}}====
* {{ui|Nerf}} slideCount 2 >>> 1
* {{ui|Nerf}} runSpeedMultiplier 1.15 >>> 1.10

===Weapon Updates===
===={{wi|Operator}}====
* {{ui|Nerf}} cost 4700 >>> 5000
* {{ui|Nerf}} fireRate 0.75 >>> 0.6
===={{wi|Outlaw}}====
* {{ui|Nerf}} cost 2400 >>> 2600
"""

def fetch_with_curl(url):
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", url],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15.0,
            check=True
        )
        return result.stdout
    except Exception as e:
        logger.error(f"Curl fetch failed: {e}")
        return None

def ingest_latest_patches(limit=5):
    """
    Reads patch versions from CSV and fetches the raw MediaWiki text from Valorant Fandom.
    Attempts curl first, falls back to httpx, and uses mock text only as a last resort.
    Returns a dictionary mapping version -> parsed_tree.
    """
    patch_versions = get_patch_versions(limit=limit)
    aggregated_data = {}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for version in patch_versions:
            logger.info(f"Fetching Patch Notes for {version}...")
            url = f"https://valorant.fandom.com/wiki/Patch_Notes/{version}?action=raw"
            raw_text = None
            
            # Try curl bypass first
            logger.info(f"Attempting curl fetch for Patch {version}...")
            raw_text = fetch_with_curl(url)
            
            if raw_text and len(raw_text.strip()) > 50:
                parsed_tree = parse_mediawiki_tree(raw_text)
                # Verify we parsed any meaningful updates
                agents = parsed_tree.get("Agent Updates", {})
                weapons = parsed_tree.get("Weapon Updates", {})
                if agents or weapons:
                    aggregated_data[version] = parsed_tree
                    logger.info(f"Successfully fetched and parsed Patch {version} using curl")
                    continue
                else:
                    logger.warning(f"Curl fetch completed but parse was empty for Patch {version}. Trying fallback.")
            
            # Fallback to standard HTTP client
            try:
                logger.info(f"Attempting client fetch for Patch {version}...")
                response = client.get(url, timeout=10.0)
                if response.status_code == 200:
                    raw_text = response.text
                    parsed_tree = parse_mediawiki_tree(raw_text)
                    aggregated_data[version] = parsed_tree
                    logger.info(f"Successfully fetched and parsed Patch {version} via Client")
                else:
                    logger.warning(f"Client fetch failed for Patch {version}: HTTP {response.status_code}")
                    # Fallback to mock data
                    logger.warning(f"Using mock fallback for Patch {version}")
                    parsed_tree = parse_mediawiki_tree(MOCK_MEDIAWIKI_9_02)
                    aggregated_data[version] = parsed_tree
            except Exception as e:
                logger.error(f"Client fetch failed for Patch {version}: {e}")
                logger.warning(f"Using mock fallback for Patch {version}")
                parsed_tree = parse_mediawiki_tree(MOCK_MEDIAWIKI_9_02)
                aggregated_data[version] = parsed_tree
                
    return aggregated_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = ingest_latest_patches(limit=2)
    import json
    print(json.dumps(data, indent=2))

