import os
import re
import json
import logging
import httpx
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engineering import load_raw_matches
from v4_skills import parse_patch_deltas, compute_delta_p_agent, compute_ghost_nerf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("patch_analyzer")

RAW_DIR = os.path.join(".", "data", "raw")
PROCESSED_DIR = os.path.join(".", "data", "processed")

# Mock Riot Patch Notes for 9.02 to prove the pipeline works
MOCK_PATCH_NOTES_9_02 = """
>>> VALORANT PATCH NOTES 9.02 <<<
Weapons updates:
- Operator cost 4700 >>> 5000
- Operator fireRate 0.75 >>> 0.6
- Outlaw cost 2400 >>> 2600
Agents updates:
- Neon slideCount 2 >>> 1
- Neon runSpeedMultiplier 1.15 >>> 1.10
"""

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
                    acs_raw = p.get("acs")
                    acs = float(acs_raw) if (acs_raw and str(acs_raw).replace('.', '', 1).isdigit()) else 200.0
                    
                    # Estimate purchase probability based on stats
                    fk_rate = fk / rounds_count
                    
                    # Assign probabilities for Vandal, Phantom, Operator, Sheriff, Outlaw, Frenzy
                    if agent.lower() in ["jett", "chamber"] or fk_rate > 0.15:
                        p_op = 0.30
                        p_outlaw = 0.10
                        p_vandal = 0.40
                        p_phantom = 0.10
                        p_sheriff = 0.08
                        p_frenzy = 0.02
                    elif agent.lower() in ["neon", "raze", "iso"]:
                        p_op = 0.05
                        p_outlaw = 0.05
                        p_vandal = 0.50
                        p_phantom = 0.30
                        p_sheriff = 0.08
                        p_frenzy = 0.02
                    else:
                        p_op = 0.02
                        p_outlaw = 0.08
                        p_vandal = 0.50
                        p_phantom = 0.30
                        p_sheriff = 0.08
                        p_frenzy = 0.02
                        
                    if agent not in agent_weapon_probs:
                        agent_weapon_probs[agent] = []
                    agent_weapon_probs[agent].append({
                        "Operator": p_op,
                        "Outlaw": p_outlaw,
                        "Vandal": p_vandal,
                        "Phantom": p_phantom,
                        "Sheriff": p_sheriff,
                        "Frenzy": p_frenzy
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
    # Hit Valorant API for structural data
    try:
        weapons_api = httpx.get("https://valorant-api.com/v1/weapons").json()["data"]
        agents_api = httpx.get("https://valorant-api.com/v1/agents").json()["data"]
    except Exception as e:
        logger.error(f"Failed to fetch from Valorant API: {e}. Falling back to default lists.")
        raise e

    # Build weapon feature vectors
    # Features: [cost, fireRate, magazineSize, reloadTimeSeconds, runSpeedMultiplier]
    weapon_vectors = {}
    for w in weapons_api:
        name = w["displayName"]
        cost = float(w.get("shopData", {}).get("cost", 1000.0) if w.get("shopData") else 1000.0)
        fire_rate = float(w.get("weaponStats", {}).get("fireRate", 10.0) if w.get("weaponStats") else 10.0)
        mag_size = float(w.get("weaponStats", {}).get("magazineSize", 30.0) if w.get("weaponStats") else 30.0)
        reload_time = float(w.get("weaponStats", {}).get("reloadTimeSeconds", 2.0) if w.get("weaponStats") else 2.0)
        speed_mult = float(w.get("weaponStats", {}).get("runSpeedMultiplier", 1.0) if w.get("weaponStats") else 1.0)
        weapon_vectors[name] = np.array([cost, fire_rate, mag_size, reload_time, speed_mult])

    # Build agent feature vectors
    # Features: [slideCount, runSpeedMultiplier]
    agent_vectors = {}
    for a in agents_api:
        if not a.get("isPlayableCharacter"):
            continue
        name = a["displayName"]
        if name == "Neon":
            slide_count = 2.0
            speed_mult = 1.15
        else:
            slide_count = 2.0
            speed_mult = 1.0
        agent_vectors[name] = np.array([slide_count, speed_mult])

    # Fit StandardScaler for weapons and agents
    X_weapons = np.array(list(weapon_vectors.values()))
    weapon_scaler = StandardScaler()
    weapon_scaler.fit(X_weapons)

    X_agents = np.array(list(agent_vectors.values()))
    agent_scaler = StandardScaler()
    agent_scaler.fit(X_agents)

    # Weights for calculation (equal weighting)
    weapon_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    agent_weights = np.array([1.0, 1.0])

    # Parse NLP deltas
    logger.info("Parsing mock patch notes text block using NLP and RegEx...")
    deltas = parse_patch_deltas(MOCK_PATCH_NOTES_9_02)
    cleaned_deltas = {}
    for metric, (old_val, new_val) in deltas.items():
        clean_metric = re.sub(r'^[-\s*]+', '', metric).strip()
        cleaned_deltas[clean_metric] = (old_val, new_val)

    # Feature indices
    weapon_feature_indices = {
        "cost": 0,
        "fireRate": 1,
        "magazineSize": 2,
        "reloadTimeSeconds": 3,
        "runSpeedMultiplier": 4
    }
    agent_feature_indices = {
        "slideCount": 0,
        "runSpeedMultiplier": 1
    }

    # Construct past and current vectors
    past_weapon_vectors = {k: v.copy() for k, v in weapon_vectors.items()}
    current_weapon_vectors = {k: v.copy() for k, v in weapon_vectors.items()}
    
    past_agent_vectors = {k: v.copy() for k, v in agent_vectors.items()}
    current_agent_vectors = {k: v.copy() for k, v in agent_vectors.items()}

    # Update current vectors from parsed deltas
    for metric, (old_val, new_val) in cleaned_deltas.items():
        parts = metric.split()
        if len(parts) >= 2:
            target_name = parts[0]
            feature_name = parts[1]
            if target_name in current_weapon_vectors and feature_name in weapon_feature_indices:
                idx = weapon_feature_indices[feature_name]
                current_weapon_vectors[target_name][idx] = new_val
                logger.info(f"Updated Weapon '{target_name}' {feature_name}: {old_val} -> {new_val}")
            elif target_name in current_agent_vectors and feature_name in agent_feature_indices:
                idx = agent_feature_indices[feature_name]
                current_agent_vectors[target_name][idx] = new_val
                logger.info(f"Updated Agent '{target_name}' {feature_name}: {old_val} -> {new_val}")

    # Build weapon dependency matrix
    weapon_dependency_matrix = build_weapon_dependency_matrix()

    # Calculate penalties for each agent
    automated_nerf_registry = {"9.02": {}}
    
    for agent in sorted(list(agent_vectors.keys())):
        # Calculate Delta P_agent
        v_past_agent = past_agent_vectors[agent]
        v_curr_agent = current_agent_vectors[agent]
        delta_p_agent = compute_delta_p_agent(v_past_agent, v_curr_agent, agent_scaler, agent_weights)

        # Calculate Delta P_ghost (weapon nerf penalty)
        delta_p_ghost = compute_ghost_nerf(
            agent, current_weapon_vectors, past_weapon_vectors,
            weapon_dependency_matrix, weapon_scaler, weapon_weights
        )

        # Final penalty = max(delta_p_agent, delta_p_ghost)
        delta_p_final = max(delta_p_agent, delta_p_ghost)

        if delta_p_final > 0.01: # Store non-trivial values
            automated_nerf_registry["9.02"][agent] = float(delta_p_final)
            logger.info(f"Agent '{agent}' Nerf Penalty: {delta_p_final:.4f} (Agent Nerf: {delta_p_agent:.4f}, Ghost Nerf: {delta_p_ghost:.4f})")

    # Export registry
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    registry_path = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
    
    # Preserve legacy manual registry patches if needed, or overwrite
    # Let's save it directly as requested
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(automated_nerf_registry, f, indent=4)
        
    logger.info(f"Automated Patch Nerf Registry successfully saved to {registry_path}")
    print("\n--- Ghost Nerf Analysis JSON Output ---")
    print(json.dumps(automated_nerf_registry, indent=4))
    print("----------------------------------------\n")
    
    return registry_path

if __name__ == "__main__":
    generate_patch_distances()
