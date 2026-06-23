import numpy as np
from sklearn.preprocessing import StandardScaler
import httpx
import json
from patch_analyzer import build_weapon_dependency_matrix
from v4_skills import compute_delta_p_agent

weapons_api = httpx.get('https://valorant-api.com/v1/weapons').json()['data']
agents_api = httpx.get('https://valorant-api.com/v1/agents').json()['data']

weapon_vectors = {}
for w in weapons_api:
    name_w = w['displayName']
    shop_data = w.get('shopData') or {}
    weapon_stats = w.get('weaponStats') or {}
    cost = float(shop_data.get('cost', 1000.0))
    fire = float(weapon_stats.get('fireRate', 10.0))
    mag = float(weapon_stats.get('magazineSize', 30.0))
    reload = float(weapon_stats.get('reloadTimeSeconds', 2.0))
    speed = float(weapon_stats.get('runSpeedMultiplier', 1.0))
    weapon_vectors[name_w] = np.array([cost, fire, mag, reload, speed])

agent_vectors = {}
for a in agents_api:
    if not a.get('isPlayableCharacter'): continue
    name_a = a['displayName']
    slide = 2.0
    mult = 1.15 if name_a == 'Neon' else 1.0
    agent_vectors[name_a] = np.array([slide, mult])

weapon_scaler = StandardScaler().fit(np.array(list(weapon_vectors.values())))
agent_scaler = StandardScaler().fit(np.array(list(agent_vectors.values())))
weapon_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
agent_weights = np.array([1.0, 1.0])

def test_math(agent_name, feat_idx, old_val, new_val):
    print(f"\n--- {agent_name} ---")
    past_vec = agent_vectors[agent_name].copy()
    past_vec[feat_idx] = old_val
    cur_vec = agent_vectors[agent_name].copy()
    cur_vec[feat_idx] = new_val
    
    # Just to mimic how patch_analyzer uses the baseline scale
    # Note patch_analyzer does NOT refit scaler. It uses the default agent_vectors.
    v_past_scaled = agent_scaler.transform(past_vec.reshape(1, -1))
    v_cur_scaled = agent_scaler.transform(cur_vec.reshape(1, -1))
    print(f"Past scaled: {v_past_scaled}")
    print(f"Curr scaled: {v_cur_scaled}")
    
    sq_dist = np.sum(agent_weights * (v_cur_scaled - v_past_scaled) ** 2)
    print(f"Sq dist: {sq_dist}")
    delta = 1 - np.exp(-1.0 * sq_dist)
    print(f"Delta: {delta}")

test_math("Neon", 0, 2.0, 1.0)
test_math("Clove", 1, 1.0, 3.0)
test_math("Breach", 1, 1.0, 2400.0)
