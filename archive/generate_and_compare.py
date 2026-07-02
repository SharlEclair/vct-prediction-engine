# generate_and_compare.py
"""Run patch_analyzer.generate_patch_distances() and compare to the existing
automated_patch_nerf_registry.json. Output a simple diff report.
"""
import os, json, sys
import pathlib

# Ensure workspace is on sys.path
workspace = pathlib.Path(__file__).parent
sys.path.append(str(workspace))

import patch_analyzer

# Paths
existing_path = os.path.join('data', 'processed', 'automated_patch_nerf_registry.json')
# Generate fresh registry
new_registry_path = patch_analyzer.generate_patch_distances()

# Load both
with open(existing_path, 'r', encoding='utf-8') as f:
    old = json.load(f)
with open(new_registry_path, 'r', encoding='utf-8') as f:
    new = json.load(f)

# Compute diff
diffs = []
for version in set(old.keys()).union(new.keys()):
    old_agents = old.get(version, {})
    new_agents = new.get(version, {})
    for agent in set(old_agents.keys()).union(new_agents.keys()):
        old_val = old_agents.get(agent)
        new_val = new_agents.get(agent)
        if old_val != new_val:
            diffs.append((version, agent, old_val, new_val))

print('--- Diff Report ---')
print(f'Total differing entries: {len(diffs)}')
for v, a, o, n in diffs:
    print(f'Version {v}, Agent {a}: old={o}, new={n}')

# Also output paths for evidence
print('\nGenerated registry path:', new_registry_path)
print('Existing registry path:', existing_path)
