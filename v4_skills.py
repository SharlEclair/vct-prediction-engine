import re
import numpy as np

# Baseline Elasticities
CATEGORY_ELASTICITIES = {
    "combat": 1.2,
    "ability": 1.0,
    "movement": 1.0,
    "economy": 0.8,
    "projectile": 0.4,
    "general": 0.5
}

def get_ability_weight(ability_name):
    """Assign power budget weight based on ability slot / type."""
    if not ability_name:
        return 0.15
    name = ability_name.lower()
    if any(x in name for x in ["ultimate", "blade storm", "tour de force", "resurrection", "annihilation"]):
        return 0.30
    if any(x in name for x in ["signature", "dash", "tailwind", "toxic screen", "high gear"]):
        return 0.40
    if "general" in name:
        return 0.15
    return 0.15 # Basic utility

def parse_patch_deltas(patch_text_block):
    """Parses Riot's strict '>>>' syntax to extract mechanical deltas."""
    pattern = re.compile(r'(?P<metric>.+?)\s+(?P<old_val>\d+(?:\.\d+)?)[a-zA-Z%]*\s*>>>\s*(?P<new_val>\d+(?:\.\d+)?)[a-zA-Z%]*')
    deltas = {}
    for line in patch_text_block.split('\n'):
        match = pattern.search(line)
        if match:
            metric = match.group('metric').strip()
            old_v, new_v = float(match.group('old_val')), float(match.group('new_val'))
            deltas[metric] = (old_v, new_v)
    return deltas

def compute_feature_shock(change):
    """
    Computes BLOPS feature shock (theta_c). 
    Returns (nerf_theta, buff_theta).
    """
    category = change.get("category", "general")
    ability = change.get("ability", "General")
    direction = change.get("type", "adjustment").lower()
    
    beta = CATEGORY_ELASTICITIES.get(category, 0.5)
    w_ab = get_ability_weight(ability)
    k = 0.5 # Half-saturation constant
    
    values = change.get("values")
    if not values or values.get("old") is None or values.get("new") is None:
        # Text-only qualitative shock
        if direction == "rework":
            r_c = 0.50
        elif direction == "removal":
            r_c = 0.80
        else:
            r_c = 0.25 # mechanic_change or adjustment
    else:
        old_val = float(values["old"])
        new_val = float(values["new"])
        epsilon = 0.0001
        
        if old_val == 0:
            r_c = abs(new_val)
        else:
            r_c = abs(new_val - old_val) / max(abs(old_val), epsilon)
            
    # Non-linear saturation
    shock_c = r_c / (r_c + k)
    
    # Compute Theta
    theta_c = beta * w_ab * shock_c
    
    if direction == "nerf":
        return theta_c, 0.0
    elif direction == "buff":
        return 0.0, theta_c
    else:
        # Adjustments contribute to both as general drift
        return theta_c * 0.5, theta_c * 0.5

def compute_ghost_nerf(agent_id, current_patch_weapon_changes, weapon_dependency_matrix):
    """Evaluates weapon dependency shifts. Returns lists of thetas: (ghost_nerfs, ghost_buffs)."""
    ghost_nerf_thetas = []
    ghost_buff_thetas = []
    
    if agent_id not in weapon_dependency_matrix:
        return [], []
        
    for weapon_id in weapon_dependency_matrix[agent_id]:
        prob_purchase = weapon_dependency_matrix[agent_id][weapon_id]
        if weapon_id in current_patch_weapon_changes:
            for change in current_patch_weapon_changes[weapon_id]:
                nerf_th, buff_th = compute_feature_shock(change)
                # Scale the theta directly by purchase probability for bounded aggregation later
                if nerf_th > 0:
                    ghost_nerf_thetas.append(nerf_th * prob_purchase)
                if buff_th > 0:
                    ghost_buff_thetas.append(buff_th * prob_purchase)
                
    return ghost_nerf_thetas, ghost_buff_thetas