# audit_trace.py
"""Trace the full execution path for specific registry entries.
Generates a detailed markdown report with:
1. Raw wiki lines used
2. Parser output objects
3. Feature builder output objects
4. Analyzer intermediate variables
5. Final registry score insertion point
"""
import os, json, re, logging
from patch_parser import PatchParser
from feature_builder import build_features
from patch_analyzer import generate_patch_distances
from patch_analyzer import compute_delta_p_agent, compute_ghost_nerf

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('trace')

# Helpers to capture raw lines
def find_raw_lines(version, name):
    path = os.path.join('data', 'patches', f'{version}.wiki')
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if name.lower() in line.lower() and line.strip().startswith('*'):
                lines.append((idx, line.rstrip()))
    return lines

def trace_version_agent(version, name):
    logger.info(f"\n## Trace for {name} in patch {version}\n")
    # 1. Raw wiki lines
    raw = find_raw_lines(version, name)
    logger.info('**Raw wiki lines**')
    for ln, txt in raw:
        logger.info(f"Line {ln}: `{txt}`")

    # 2. Parser output
    wiki_path = os.path.join('data','patches', f'{version}.wiki')
    with open(wiki_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    parser = PatchParser()
    parsed = parser.parse_patch(version, '', raw_text)
    # locate object
    obj = None
    for ch in parsed.get('agent_changes', []):
        if ch.get('agent','').lower() == name.lower():
            obj = ch
            break
    if not obj:
        for ch in parsed.get('weapon_changes', []):
            if ch.get('weapon','').lower() == name.lower():
                obj = ch
                break
    logger.info('\n**Parser output object**')
    logger.info(json.dumps(obj, indent=4))

    # 3. Feature builder output
    stats = build_features(version)
    feat_path = os.path.join('data','processed','features', f'{version}.json')
    with open(feat_path, 'r', encoding='utf-8') as f:
        feats = json.load(f)
    fobj = None
    if name in feats.get('Agent Updates', {}):
        fobj = feats['Agent Updates'][name][0]
    elif name in feats.get('Weapon Updates', {}):
        fobj = feats['Weapon Updates'][name][0]
    logger.info('\n**Feature builder object**')
    logger.info(json.dumps(fobj, indent=4))

    # 4. Analyzer calculations – we re‑run the core loop for this version only
    # Load vectors (same as in patch_analyzer)
    import numpy as np
    from feature_engineering import load_raw_matches

    # weapon vectors
    import httpx
    weapons_api = httpx.get('https://valorant-api.com/v1/weapons').json()['data']
    weapon_vectors = {}
    for w in weapons_api:
        name_w = w['displayName']
        
        # Safe extraction for shopData
        shop_data = w.get('shopData')
        if shop_data is None:
            shop_data = {}
            
        # Safe extraction for weaponStats
        weapon_stats = w.get('weaponStats')
        if weapon_stats is None:
            weapon_stats = {}
            
        cost = float(shop_data.get('cost', 1000.0))
        fire = float(weapon_stats.get('fireRate', 10.0))
        mag = float(weapon_stats.get('magazineSize', 30.0))
        reload = float(weapon_stats.get('reloadTimeSeconds', 2.0))
        speed = float(weapon_stats.get('runSpeedMultiplier', 1.0))
        
        weapon_vectors[name_w] = np.array([cost, fire, mag, reload, speed])
    # agent vectors
    agents_api = httpx.get('https://valorant-api.com/v1/agents').json()['data']
    agent_vectors = {}
    for a in agents_api:
        if not a.get('isPlayableCharacter'):
            continue
        name_a = a['displayName']
        slide = 2.0
        mult = 1.15 if name_a == 'Neon' else 1.0
        agent_vectors[name_a] = np.array([slide, mult])
    # scalers
    from sklearn.preprocessing import StandardScaler
    weapon_scaler = StandardScaler().fit(np.array(list(weapon_vectors.values())))
    agent_scaler = StandardScaler().fit(np.array(list(agent_vectors.values())))
    # dependency matrix
    from patch_analyzer import build_weapon_dependency_matrix
    weapon_dep = build_weapon_dependency_matrix()
    # Past/current copies
    past_weapon = {k:v.copy() for k,v in weapon_vectors.items()}
    curr_weapon = {k:v.copy() for k,v in weapon_vectors.items()}
    past_agent = {k:v.copy() for k,v in agent_vectors.items()}
    curr_agent = {k:v.copy() for k,v in agent_vectors.items()}
    # Apply feature updates for this version
    patch_tree = {}  # load from processed features file
    with open(feat_path, 'r', encoding='utf-8') as f:
        patch_tree = json.load(f)
    # Agent updates
    for ag, changes in patch_tree.get('Agent Updates', {}).items():
        for ch in changes:
            feat = ch.get('feature_name')
            if ag in curr_agent and feat in {0:'slideCount',1:'runSpeedMultiplier'}:
                idx = 0 if feat == 'slideCount' else 1
                if ch.get('values'):
                    new = ch['values'].get('new')
                    if new is not None:
                        curr_agent[ag][idx] = float(new)
                        logger.info(f"Updated agent {ag} {feat} -> {new}")
    # Weapon updates
    for wp, changes in patch_tree.get('Weapon Updates', {}).items():
        for ch in changes:
            feat = ch.get('feature_name')
            if wp in curr_weapon and feat in {'cost':0,'fireRate':1,'magazineSize':2,'reloadTimeSeconds':3,'runSpeedMultiplier':4}:
                idx = {'cost':0,'fireRate':1,'magazineSize':2,'reloadTimeSeconds':3,'runSpeedMultiplier':4}[feat]
                if ch.get('values'):
                    new = ch['values'].get('new')
                    if new is not None:
                        curr_weapon[wp][idx] = float(new)
                        logger.info(f"Updated weapon {wp} {feat} -> {new}")
    # Compute delta for target
    delta_agent = compute_delta_p_agent(past_agent[name], curr_agent[name], agent_scaler, np.array([1.0,1.0]))
    delta_ghost = compute_ghost_nerf(name, curr_weapon, past_weapon, weapon_dep, weapon_scaler, np.array([1.0]*5))
    delta_final = max(delta_agent, delta_ghost)
    logger.info('\n**Analyzer calculations**')
    logger.info(f"delta_p_agent = {delta_agent}")
    logger.info(f"delta_p_ghost = {delta_ghost}")
    logger.info(f"delta_p_final = {delta_final}")
    # Where it gets written
    logger.info('\n**Registry insertion point**')
    logger.info('In patch_analyzer.generate_patch_distances, line ~225:')
    logger.info(f"automated_nerf_registry['{version}']['{name}'] = float(delta_p_final)")
    logger.info(f"Final registry value = {delta_final}\n")

if __name__ == '__main__':
    for ver, agent in [('9.11','Neon'), ('10.04','Clove'), ('12.00','Breach')]:
        trace_version_agent(ver, agent)
