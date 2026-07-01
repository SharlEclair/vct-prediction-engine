import os
import json
import glob
import logging
from pathlib import Path

logger = logging.getLogger("sync_map_pool")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
CONFIG_PATH = ROOT_DIR / "config.yaml"

def sync_active_map_pool():
    logger.info("Scanning match telemetry to extract active map rotation...")
    
    files = glob.glob(os.path.join(RAW_DIR, "match_*.json"))
    if not files:
        logger.warning("No raw match files found. Keeping existing map pool.")
        return
        
    # Sort files by filename or time to check recent matches
    # We can inspect the match date or segment timestamp
    matches = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
            segment = content["data"]["segments"][0]
            date_str = segment.get("date", "")
            matches.append((f, date_str))
        except Exception:
            pass
            
    # If no matches could be loaded, skip
    if not matches:
        logger.warning("Failed to parse raw match telemetry dates. Skipping map pool sync.")
        return
        
    # Extract unique maps from the last 30 matches (representing the current VCT competitive map rotation)
    # We reverse sort by date string (or just use filename order since VLR matches are typically numbered chronologically)
    matches.sort(key=lambda x: os.path.basename(x[0]), reverse=True)
    recent_matches = matches[:30]
    
    active_maps = set()
    for filepath, _ in recent_matches:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = json.load(file)
            segment = content["data"]["segments"][0]
            for map_data in segment.get("maps", []):
                map_name = map_data.get("map_name", "").strip()
                if map_name and map_name.lower() not in ["tbd", "unknown"]:
                    # Capitalize properly (e.g., 'ascent' -> 'Ascent')
                    active_maps.add(map_name.capitalize())
        except Exception:
            pass
            
    if not active_maps:
        logger.warning("No active maps detected from recent matches. Keeping current config.")
        return
        
    # Filter to known competitive maps (optional, but good to clean up typo/mock maps if any)
    known_maps = {"Ascent", "Bind", "Haven", "Lotus", "Sunset", "Pearl", "Fracture", "Abyss", "Icebox", "Breeze", "Split"}
    filtered_maps = sorted(list(active_maps & known_maps))
    
    if not filtered_maps:
        # If overlap is empty, default to active_maps sorted
        filtered_maps = sorted(list(active_maps))
        
    logger.info(f"Active maps detected: {filtered_maps}")
    
    # Overwrite config.yaml COMPETITIVE_MAP_POOL
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            
        lines = content.splitlines()
        new_lines = []
        in_map_pool = False
        
        for line in lines:
            if line.strip().startswith("COMPETITIVE_MAP_POOL:"):
                new_lines.append("COMPETITIVE_MAP_POOL:")
                for m in filtered_maps:
                    new_lines.append(f"  - \"{m}\"")
                in_map_pool = True
                continue
            if in_map_pool:
                if line.strip().startswith("-") or line.strip() == "":
                    continue
                else:
                    in_map_pool = False
            new_lines.append(line)
            
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        logger.info(f"Successfully updated COMPETITIVE_MAP_POOL in config.yaml with {len(filtered_maps)} maps.")
    else:
        logger.error(f"config.yaml not found at {CONFIG_PATH}")

if __name__ == "__main__":
    sync_active_map_pool()
