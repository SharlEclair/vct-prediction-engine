import os
import json
import logging
import numpy as np
from scipy.spatial.distance import jensenshannon
from feature_engineering import load_raw_matches

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("meta_engine")

PROCESSED_DIR = os.path.join(".", "data", "processed")

def build_patch_distance_matrix():
    logger.info("Loading matches...")
    matches = load_raw_matches()
    
    # 1. Collect all unique patches and agents
    unique_patches = sorted(list(set(m["patch"] for m in matches if m.get("patch"))))
    
    unique_agents = set()
    for m in matches:
        for map_data in m.get("maps", []):
            for team_key in ["team1", "team2"]:
                for p in map_data.get("players", {}).get(team_key, []):
                    if p.get("agent"):
                        unique_agents.add(p["agent"])
    unique_agents = sorted(list(unique_agents))
    
    logger.info(f"Found {len(unique_patches)} unique patches: {unique_patches}")
    logger.info(f"Found {len(unique_agents)} unique agents: {unique_agents}")
    
    # 2. Count agent picks per patch
    counts = {p: {a: 0 for a in unique_agents} for p in unique_patches}
    for m in matches:
        patch = m["patch"]
        for map_data in m.get("maps", []):
            for team_key in ["team1", "team2"]:
                for p in map_data.get("players", {}).get(team_key, []):
                    agent = p.get("agent")
                    if agent in counts[patch]:
                        counts[patch][agent] += 1
                        
    # 3. Compute probability distributions
    probs = {}
    for patch in unique_patches:
        total = sum(counts[patch].values())
        if total > 0:
            probs[patch] = [counts[patch][agent] / total for agent in unique_agents]
        else:
            probs[patch] = [1.0 / len(unique_agents) for agent in unique_agents]
            
    # 4. Compute pairwise JSD matrix
    matrix = {}
    for p1 in unique_patches:
        matrix[p1] = {}
        for p2 in unique_patches:
            dist = jensenshannon(probs[p1], probs[p2])
            if np.isnan(dist):
                dist = 0.0
            matrix[p1][p2] = float(dist)
            
    # 5. Save the matrix
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    matrix_path = os.path.join(PROCESSED_DIR, "patch_distance_matrix.json")
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=4)
        
    logger.info(f"JSD Patch Distance Matrix successfully saved to {matrix_path}")
    logger.info(f"Matrix shape: {len(unique_patches)} x {len(unique_patches)}")
    
    # Print the matrix nicely
    print("\n--- JSD Patch Distance Matrix ---")
    print(f"Patches: {unique_patches}")
    for p1 in unique_patches:
        row_str = "  ".join(f"{matrix[p1][p2]:.4f}" for p2 in unique_patches)
        print(f"{p1:6s}: {row_str}")
    print("---------------------------------\n")
    
    return len(unique_patches)

if __name__ == "__main__":
    build_patch_distance_matrix()
