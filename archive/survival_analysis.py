# survival_analysis.py
"""Run a detailed survival analysis of the feature pipeline inside
patch_analyzer.generate_patch_distances().
It reproduces the core loops of patch_analyzer but records per‑feature
statistics required by the user.
"""
import os, json, logging
from collections import defaultdict, Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import httpx
from feature_builder import build_features
from v4_skills import compute_delta_p_agent, compute_ghost_nerf

# ---------------------------------------------------------------------------
# Helper to load raw wiki and find lines for a given agent/patch (used for three cases)
def find_raw_wiki_line(version, agent_name):
    path = os.path.join('data', 'patches', f'{version}.wiki')
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if agent_name.lower() in line.lower() and line.strip().startswith('*'):
                lines.append((i, line.rstrip()))
    return lines

# ---------------------------------------------------------------------------
# Setup logging (mirrors patch_analyzer)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('survival_analysis')

# ---------------------------------------------------------------------------
# Load Valorant API data (same as patch_analyzer)
weapons_api = httpx.get('https://valorant-api.com/v1/weapons').json()['data']
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

agents_api = httpx.get('https://valorant-api.com/v1/agents').json()['data']
agent_vectors = {}
for a in agents_api:
    if not a.get('isPlayableCharacter'):
        continue
    name_a = a['displayName']
    # default values (match patch_analyzer defaults)
    slide = 2.0
    mult = 1.15 if name_a == 'Neon' else 1.0
    agent_vectors[name_a] = np.array([slide, mult])

# Scalers & weights (same as patch_analyzer)
weapon_scaler = StandardScaler().fit(np.array(list(weapon_vectors.values())))
agent_scaler = StandardScaler().fit(np.array(list(agent_vectors.values())))
weapon_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
agent_weights = np.array([1.0, 1.0])

# Weapon dependency matrix (reuse function from patch_analyzer) – copy here
from patch_analyzer import build_weapon_dependency_matrix
weapon_dependency_matrix = build_weapon_dependency_matrix()

# ---------------------------------------------------------------------------
# Feature indices (mirrors patch_analyzer)
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

# ---------------------------------------------------------------------------
# Prepare per‑patch collection structures
patch_report = {}
exclusion_counter = Counter()
registry_entries = {}

# ---------------------------------------------------------------------------
# Load all feature trees (same logic as patch_analyzer)
import glob
patch_files = glob.glob(os.path.join('.', 'data', 'processed', 'patches', '*.json'))
versions = sorted([os.path.basename(f).replace('.json', '') for f in patch_files],
                  key=lambda x: [int(v) for v in x.split('.')])

for version in versions:
    # Ensure feature file exists
    feature_path = os.path.join('.', 'data', 'processed', 'features', f"{version}.json")
    if not os.path.exists(feature_path):
        build_features(version)
    with open(feature_path, 'r', encoding='utf-8') as f:
        patch_tree = json.load(f)

    # Initialise report entry for this version
    rpt = {
        "parsed_changes": 0,
        "generated_features": 0,
        "numeric_features": 0,
        "mapped_features": 0,
        "ignored": [],  # list of dicts with reason
        "vector_updates": [],
        "agents_nonzero_delta": [],
        "agents_in_registry": []
    }
    patch_report[version] = rpt

    # Copy base vectors
    past_weapon = {k: v.copy() for k, v in weapon_vectors.items()}
    cur_weapon = {k: v.copy() for k, v in weapon_vectors.items()}
    past_agent = {k: v.copy() for k, v in agent_vectors.items()}
    cur_agent = {k: v.copy() for k, v in agent_vectors.items()}

    # ------------------- Agent updates -------------------
    for agent_name, changes in patch_tree.get('Agent Updates', {}).items():
        for change in changes:
            rpt["parsed_changes"] += 1
            rpt["generated_features"] += 1
            feature_name = change.get('feature_name')
            # Determine numeric status
            new_val = None
            if change.get('values') is not None:
                new_val = change['values'].get('new')
                if isinstance(new_val, (int, float)):
                    rpt["numeric_features"] += 1
            # Validation chain
            if agent_name not in cur_agent:
                reason = 'agent lookup failed'
                rpt["ignored"].append({"agent": agent_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            if feature_name not in agent_feature_indices:
                reason = 'feature_name does not exist in vector'
                rpt["ignored"].append({"agent": agent_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            if change.get('values') is None:
                reason = 'values is None'
                rpt["ignored"].append({"agent": agent_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            if new_val is None:
                reason = 'new value is None'
                rpt["ignored"].append({"agent": agent_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            # weight check – both agent weights are 1.0 in current code, but keep logic for future
            weight = agent_weights[agent_feature_indices[feature_name]]
            if weight == 0:
                reason = 'weight == 0'
                rpt["ignored"].append({"agent": agent_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            # Apply update
            idx = agent_feature_indices[feature_name]
            old_val = cur_agent[agent_name][idx]
            cur_agent[agent_name][idx] = float(new_val)
            rpt["mapped_features"] += 1
            rpt["vector_updates"].append({
                "agent": agent_name,
                "feature_name": feature_name,
                "old_value": old_val,
                "new_value": float(new_val)
            })

    # ------------------- Weapon updates (similar tracking) -------------------
    for weapon_name, changes in patch_tree.get('Weapon Updates', {}).items():
        for change in changes:
            rpt["parsed_changes"] += 1
            rpt["generated_features"] += 1
            feature_name = change.get('feature_name')
            new_val = None
            if change.get('values') is not None:
                new_val = change['values'].get('new')
                if isinstance(new_val, (int, float)):
                    rpt["numeric_features"] += 1
            if weapon_name not in cur_weapon:
                reason = 'weapon lookup failed'
                rpt["ignored"].append({"weapon": weapon_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            if feature_name not in weapon_feature_indices:
                reason = 'feature_name does not exist in vector'
                rpt["ignored"].append({"weapon": weapon_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            if change.get('values') is None:
                reason = 'values is None'
                rpt["ignored"].append({"weapon": weapon_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            if new_val is None:
                reason = 'new value is None'
                rpt["ignored"].append({"weapon": weapon_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            weight = weapon_weights[weapon_feature_indices[feature_name]]
            if weight == 0:
                reason = 'weight == 0'
                rpt["ignored"].append({"weapon": weapon_name, "feature_name": feature_name, "reason": reason})
                exclusion_counter[reason] += 1
                continue
            idx = weapon_feature_indices[feature_name]
            old_val = cur_weapon[weapon_name][idx]
            cur_weapon[weapon_name][idx] = float(new_val)
            rpt["mapped_features"] += 1
            rpt["vector_updates"].append({
                "weapon": weapon_name,
                "feature_name": feature_name,
                "old_value": old_val,
                "new_value": float(new_val)
            })

    # ------------------- Compute penalties -------------------
    for agent in sorted(cur_agent.keys()):
        delta_agent = compute_delta_p_agent(past_agent[agent], cur_agent[agent], agent_scaler, agent_weights)
        delta_ghost = compute_ghost_nerf(
            agent, cur_weapon, past_weapon,
            weapon_dependency_matrix, weapon_scaler, weapon_weights
        )
        delta_final = max(delta_agent, delta_ghost)
        if delta_final > 0.01:
            rpt["agents_nonzero_delta"].append({"agent": agent, "delta": delta_final})
        if delta_final > 0.01:
            # This mimics registry threshold logic
            rpt["agents_in_registry"].append({"agent": agent, "value": float(delta_final)})
            # store for later summary table
            if version not in registry_entries:
                registry_entries[version] = {}
            registry_entries[version][agent] = float(delta_final)

# ---------------------------------------------------------------------------
# Produce markdown tables

def md_table(headers, rows):
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join(["---" for _ in headers]) + "|"
    body = "\n".join(["| " + " | ".join(str(v) for v in row) + " |" for row in rows])
    return "\n".join([header_line, sep_line, body])

# 1️⃣  Overall per‑patch table
overall_rows = []
for v in versions:
    r = patch_report[v]
    overall_rows.append([
        v,
        r["parsed_changes"],
        r["generated_features"],
        r["numeric_features"],
        r["mapped_features"],
        len([i for i in r["ignored"] if i.get('reason') == 'feature_name does not exist in vector']),
        len([i for i in r["ignored"] if i.get('reason') == 'weight == 0']),
        len([i for i in r["ignored"] if i.get('reason') == 'values is None']),
        len([i for i in r["ignored"] if i.get('reason') == 'agent lookup failed']),
        len(r["vector_updates"]),
        len(r["agents_nonzero_delta"]),
        len(r["agents_in_registry"])
    ])

md_overall = md_table([
    "Patch", "Parsed", "Generated", "Numeric", "Mapped", "Ignored‑no‑vector", "Ignored‑zero‑weight", "Ignored‑no‑values", "Ignored‑agent‑lookup", "Vec‑changed", "Agents‑Δ>0.01", "Agents‑in‑registry"
], overall_rows)

# 2️⃣  Exclusion reasons ranking
rank_rows = sorted(exclusion_counter.items(), key=lambda x: x[1], reverse=True)
md_exclusions = md_table(["Reason", "Count"], rank_rows)

# 3️⃣  Detailed path for Neon, Clove, Breach
special_cases = [
    ("9.11", "Neon"),
    ("10.04", "Clove"),
    ("12.00", "Breach")
]
case_sections = []
for ver, agent in special_cases:
    # raw wiki lines
    raw = find_raw_wiki_line(ver, agent)
    raw_md = "\n".join([f"* Line {ln}: `{txt}`" for ln, txt in raw]) if raw else "(none found)"
    # parsed object – same as first entry in patch_tree
    feature_path = os.path.join('data', 'processed', 'features', f"{ver}.json")
    with open(feature_path, 'r', encoding='utf-8') as f:
        ft = json.load(f)
    parsed_obj = None
    if agent in ft.get('Agent Updates', {}):
        parsed_obj = ft['Agent Updates'][agent][0]
    elif agent in ft.get('Weapon Updates', {}):
        parsed_obj = ft['Weapon Updates'][agent][0]
    # feature object is the same dict
    feature_obj = parsed_obj
    # Find the vector update that corresponds (old/new values)
    upd = None
    for u in patch_report[ver]["vector_updates"]:
        if u.get('agent') == agent:
            upd = u
            break
    # Compute scaled diff and squared distance for the agent vector
    if upd:
        idx = agent_feature_indices[upd['feature_name']]
        past_vec = past_agent[agent].copy()
        cur_vec = cur_agent[agent].copy()
        # apply the update to a copy to compute
        cur_vec[idx] = upd['new_value']
        v_past_scaled = agent_scaler.transform(past_vec.reshape(1, -1))
        v_cur_scaled = agent_scaler.transform(cur_vec.reshape(1, -1))
        sq_dist = np.sum(agent_weights * (v_cur_scaled - v_past_scaled) ** 2)
        delta_agent = compute_delta_p_agent(past_vec, cur_vec, agent_scaler, agent_weights)
        delta_final = delta_agent  # ghost penalty is zero for agents
        registry_val = registry_entries.get(ver, {}).get(agent)
    else:
        upd = {}
        sq_dist = delta_agent = delta_final = registry_val = None
    sec = f"## {agent} (patch {ver})\n\n**Raw wiki lines**\n{raw_md}\n\n**Parsed object**\n```json\n{json.dumps(parsed_obj, indent=4)}\n```\n\n**Feature object**\n```json\n{json.dumps(feature_obj, indent=4)}\n```\n\n**Vector update**\n```json\n{json.dumps(upd, indent=4)}\n```\n\n**Scaled vector diff** – old scaled: `{v_past_scaled}` / new scaled: `{v_cur_scaled}`\n\n**Squared distance**: `{sq_dist}`\n\n**Delta p (agent)**: `{delta_agent}`\n\n**Final registry value**: `{registry_val}`\n"
    case_sections.append(sec)

# ---------------------------------------------------------------------------
# Write results to markdown artifact
output_md = f"# Survival Analysis Report\n\n## Per‑patch Summary\n\n{md_overall}\n\n## Exclusion Reasons Ranking\n\n{md_exclusions}\n\n## Detailed Paths for Neon, Clove, Breach\n\n" + "\n---\n".join(case_sections)

with open('survival_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(output_md)

print('Survival analysis completed. Report written to survival_analysis_report.md')
