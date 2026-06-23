import json
import numpy as np
from feature_engineering import load_raw_matches
import matplotlib.pyplot as plt
import os

REGISTRY_PATH = "data/processed/automated_patch_nerf_registry.json"

def validate_drift_correlation():
    print("--- PREDICTIVE ERROR VALIDATION ---")
    matches = load_raw_matches()
    
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        
    # Group player performances by patch
    player_patch_acs = {}
    
    for m in matches:
        patch = m.get("patch")
        if not patch: continue
        
        for map_data in m.get("maps", []):
            for team_key in ["team1", "team2"]:
                for p in map_data.get("players", {}).get(team_key, []):
                    name = p.get("name")
                    agent = p.get("agent")
                    acs = p.get("acs")
                    if name and agent and acs is not None:
                        acs_val = float(str(acs).replace('.', '', 1)) if str(acs).replace('.', '', 1).isdigit() else 0.0
                        key = (name, agent, patch)
                        if key not in player_patch_acs:
                            player_patch_acs[key] = []
                        player_patch_acs[key].append(acs_val)
                        
    # Average ACS per player-agent-patch
    for k in player_patch_acs:
        player_patch_acs[k] = np.mean(player_patch_acs[k])
        
    # Find transitions
    # Sort patches by version
    patches = sorted(list(set([k[2] for k in player_patch_acs.keys()])), key=lambda x: [int(v) for v in x.split('.')])
    
    patch_transitions = []
    for i in range(len(patches)-1):
        patch_transitions.append((patches[i], patches[i+1]))
        
    errors = []
    blops_scores = []
    
    for old_p, new_p in patch_transitions:
        for agent, score in registry.get(new_p, {}).items():
            if score < 0.05: continue # Only care about meaningful changes
            
            # Find players who played this agent in both patches
            agent_players = [k[0] for k in player_patch_acs.keys() if k[1] == agent and k[2] == old_p]
            
            patch_maes = []
            for player in agent_players:
                old_key = (player, agent, old_p)
                new_key = (player, agent, new_p)
                
                if new_key in player_patch_acs:
                    old_acs = player_patch_acs[old_key]
                    new_acs = player_patch_acs[new_key]
                    mae = abs(new_acs - old_acs)
                    patch_maes.append(mae)
                    
            if patch_maes:
                avg_mae = np.mean(patch_maes)
                errors.append(avg_mae)
                blops_scores.append(score)
                print(f"[{old_p} -> {new_p}] {agent}: BLOPS={score:.3f} | ACS MAE={avg_mae:.1f} (n={len(patch_maes)})")
                
    if len(errors) > 0:
        correlation = np.corrcoef(blops_scores, errors)[0, 1]
        print(f"\nPearson Correlation (BLOPS vs ACS Prediction Error): {correlation:.3f}")
    else:
        print("Not enough matching player data across patches to compute correlation.")

if __name__ == "__main__":
    validate_drift_correlation()
