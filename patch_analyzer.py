import os
import json
import logging
from curl_cffi import requests
import numpy as np
from feature_engineering import load_raw_matches
from v4_skills import parse_patch_deltas, compute_feature_shock, compute_ghost_nerf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("patch_analyzer")

RAW_DIR = os.path.join(".", "data", "raw")
PROCESSED_DIR = os.path.join(".", "data", "processed")

def build_weapon_dependency_matrix():
    logger.info("Scanning raw matches to build Weapon Dependency Matrix P(w|a)...")
    matches = load_raw_matches()
    
    agent_weapon_probs = {}
    for m in matches:
        for map_data in m.get("maps", []):
            rounds_count = len(map_data.get("rounds", []))
            if rounds_count == 0:
                score = map_data.get("score", {})
                rounds_count = int(score.get("team1", 0)) + int(score.get("team2", 0))
                if rounds_count == 0:
                    rounds_count = 24
                    
            for team_key in ["team1", "team2"]:
                for p in map_data.get("players", {}).get(team_key, []):
                    agent = p.get("agent")
                    if not agent:
                        continue
                    
                    fk_raw = p.get("fk")
                    fk = float(fk_raw) if (fk_raw and str(fk_raw).replace('.', '', 1).isdigit()) else 0.0
                    
                    # Estimate purchase probability based on stats
                    fk_rate = fk / rounds_count
                    
                    # Assign probabilities for Vandal, Phantom, Operator, Sheriff, Outlaw, Frenzy
                    if agent.lower() in ["jett", "chamber"] or fk_rate > 0.15:
                        p_op, p_outlaw, p_vandal, p_phantom, p_sheriff, p_frenzy = 0.30, 0.10, 0.40, 0.10, 0.08, 0.02
                    elif agent.lower() in ["neon", "raze", "iso"]:
                        p_op, p_outlaw, p_vandal, p_phantom, p_sheriff, p_frenzy = 0.05, 0.05, 0.50, 0.30, 0.08, 0.02
                    else:
                        p_op, p_outlaw, p_vandal, p_phantom, p_sheriff, p_frenzy = 0.02, 0.08, 0.50, 0.30, 0.08, 0.02
                        
                    if agent not in agent_weapon_probs:
                        agent_weapon_probs[agent] = []
                    agent_weapon_probs[agent].append({
                        "Operator": p_op, "Outlaw": p_outlaw, "Vandal": p_vandal, 
                        "Phantom": p_phantom, "Sheriff": p_sheriff, "Frenzy": p_frenzy
                    })
                    
    # Average the probabilities for each agent
    dependency_matrix = {}
    for agent, list_probs in agent_weapon_probs.items():
        dependency_matrix[agent] = {}
        for weapon in ["Operator", "Outlaw", "Vandal", "Phantom", "Sheriff", "Frenzy"]:
            dependency_matrix[agent][weapon] = float(np.mean([item[weapon] for item in list_probs]))
            
    logger.info("Successfully built Weapon Dependency Matrix.")
    return dependency_matrix

def generate_patch_distances():
    logger.info("Fetching Valorant API structural data...")
    try:
        agents_api = requests.get("https://valorant-api.com/v1/agents", impersonate="chrome").json()["data"]
    except Exception as e:
        logger.error(f"Failed to fetch from Valorant API: {e}. Falling back to default lists.")
        raise e

    all_agents = [a["displayName"] for a in agents_api if a.get("isPlayableCharacter")]

    weapon_dependency_matrix = build_weapon_dependency_matrix()

    # Load feature trees for all available versions
    logger.info("Loading patch feature trees...")
    import glob
    from feature_builder import build_features
    
    patch_files = glob.glob(os.path.join(".", "data", "processed", "patches", "*.json"))
    versions = sorted([os.path.basename(f).replace(".json", "") for f in patch_files], 
                      key=lambda x: [int(v) for v in x.split('.')])
    
    ingested_data = {}
    for version in versions:
        feature_path = os.path.join(".", "data", "processed", "features", f"{version}.json")
        if not os.path.exists(feature_path):
            build_features(version)
            
        if os.path.exists(feature_path):
            with open(feature_path, "r", encoding="utf-8") as f:
                ingested_data[version] = json.load(f)

    automated_nerf_registry = {}
    patch_impact_trace = {}

    for patch_version, patch_tree in ingested_data.items():
        automated_nerf_registry[patch_version] = {}
        patch_impact_trace[patch_version] = {}
        
        agent_updates = patch_tree.get("Agent Updates", {})
        weapon_updates = patch_tree.get("Weapon Updates", {})

        for agent in all_agents:
            all_thetas = []
            trace_features = []
            
            # Direct agent changes
            if agent in agent_updates:
                for change in agent_updates[agent]:
                    nerf_th, buff_th = compute_feature_shock(change)
                    feature_name = change.get("feature_name", "unknown")
                    category = change.get("category", "general")
                    full_feat = f"{category}.{feature_name}"
                    
                    if nerf_th > 0:
                        all_thetas.append(nerf_th)
                        trace_features.append({"feature": full_feat, "impact": round(nerf_th, 4), "reason": "nerf"})
                    if buff_th > 0:
                        all_thetas.append(buff_th)
                        trace_features.append({"feature": full_feat, "impact": round(buff_th, 4), "reason": "buff"})
                        
            # Ghost changes via weapons
            ghost_nerfs, ghost_buffs = compute_ghost_nerf(agent, weapon_updates, weapon_dependency_matrix)
            for th in ghost_nerfs:
                all_thetas.append(th)
                trace_features.append({"feature": "weapon.dependency", "impact": round(th, 4), "reason": "ghost_nerf"})
            for th in ghost_buffs:
                all_thetas.append(th)
                trace_features.append({"feature": "weapon.dependency", "impact": round(th, 4), "reason": "ghost_buff"})

            # Probabilistic Aggregation (Bounded union)
            # Drift = 1 - product(1 - feature_impact)
            registry_score = 1.0 - np.prod([1.0 - min(th, 0.999) for th in all_thetas]) if all_thetas else 0.0
            
            if registry_score > 0.001:
                automated_nerf_registry[patch_version][agent] = float(registry_score)
                patch_impact_trace[patch_version][agent] = {
                    "score": round(float(registry_score), 4),
                    "features": trace_features
                }
                logger.info(f"[{patch_version}] Agent '{agent}' Concept Drift: {registry_score:.4f}")

    # Export registry
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    registry_path = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
    trace_path = os.path.join(PROCESSED_DIR, "patch_impact_trace.json")
    
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(automated_nerf_registry, f, indent=4)
        
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(patch_impact_trace, f, indent=4)
        
    logger.info(f"Automated Patch Registry successfully saved to {registry_path}")
    logger.info(f"Patch Impact Trace successfully saved to {trace_path}")
    
    return registry_path

if __name__ == "__main__":
    generate_patch_distances()
