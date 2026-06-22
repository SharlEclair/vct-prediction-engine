import spacy
import re
import numpy as np

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

def compute_delta_p_agent(v_past, v_current, scaler, weights, gamma=1.0):
    """Computes bounded patch distance using Weighted RBF over standardized Euclidean space."""
    v_past_scaled = scaler.transform(v_past.reshape(1, -1))
    v_curr_scaled = scaler.transform(v_current.reshape(1, -1))
    sq_dist = np.sum(weights * (v_curr_scaled - v_past_scaled) ** 2)
    distance_score = 1 - np.exp(-gamma * sq_dist)
    return distance_score

def compute_ghost_nerf(agent_id, current_patch_weapon_vectors, past_patch_weapon_vectors, weapon_dependency_matrix, scaler, weapon_weights):
    """Evaluates weapon dependency shifts (Ghost Nerfs)."""
    ghost_penalty = 0.0
    # Ensure agent_id exists in our telemetry matrix
    if agent_id not in weapon_dependency_matrix:
        return 0.0
        
    for weapon_id in weapon_dependency_matrix[agent_id]:
        prob_purchase = weapon_dependency_matrix[agent_id][weapon_id]
        if weapon_id in past_patch_weapon_vectors and weapon_id in current_patch_weapon_vectors:
            w_past = past_patch_weapon_vectors[weapon_id]
            w_curr = current_patch_weapon_vectors[weapon_id]
            weapon_shift = compute_delta_p_agent(w_past, w_curr, scaler, weapon_weights)
            ghost_penalty += prob_purchase * weapon_shift
    return ghost_penalty