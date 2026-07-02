import json
import numpy as np
import random

REGISTRY_PATH = "data/processed/automated_patch_nerf_registry.json"
TRACE_PATH = "data/processed/patch_impact_trace.json"

def audit_distribution():
    print("--- BLOPS DISTRIBUTION AUDIT ---")
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        
    all_scores = []
    cases = {}
    
    for patch, agents in registry.items():
        for agent, score in agents.items():
            all_scores.append(score)
            cases[f"{agent} ({patch})"] = score
            
    if not all_scores:
        print("No scores found.")
        return
        
    all_scores = np.array(all_scores)
    
    print(f"Total entries: {len(all_scores)}")
    print(f"Mean: {np.mean(all_scores):.4f}")
    print(f"Median: {np.median(all_scores):.4f}")
    print(f"Std Dev: {np.std(all_scores):.4f}")
    print(f"Max: {np.max(all_scores):.4f}")
    
    b1 = np.sum(all_scores < 0.05) / len(all_scores)
    b2 = np.sum((all_scores >= 0.05) & (all_scores < 0.25)) / len(all_scores)
    b3 = np.sum((all_scores >= 0.25) & (all_scores < 0.50)) / len(all_scores)
    b4 = np.sum(all_scores >= 0.50) / len(all_scores)
    
    print("\nBuckets:")
    print(f"< 0.05 (Minor Drift): {b1:.1%}")
    print(f"0.05 - 0.25 (Moderate Drift): {b2:.1%}")
    print(f"0.25 - 0.50 (Significant Drift): {b3:.1%}")
    print(f"> 0.50 (Major Rework): {b4:.1%}")
    
    # Specific Cases
    print("\nSanity Checks:")
    checks = ["Neon (9.11)", "Breach (12.00)", "Clove (10.04)", "Vyse (11.08)", "Jett (9.10)"]
    for c in checks:
        print(f"{c}: {cases.get(c, 'N/A')}")
        
    # Downstream Decay
    print("\n--- DECAY CURVE AUDIT ---")
    test_deltas = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    for d in test_deltas:
        weight = np.exp(-2.0 * d)
        print(f"Delta: {d:.2f} -> Remaining History Weight: {weight:.1%}")

def audit_features():
    print("\n--- FEATURE TRACE AUDIT ---")
    with open(TRACE_PATH, "r") as f:
        traces = json.load(f)
        
    all_features = []
    unknown_count = 0
    total_count = 0
    
    for patch, agents in traces.items():
        for agent, data in agents.items():
            for feat in data.get("features", []):
                all_features.append({"agent": agent, "patch": patch, **feat})
                total_count += 1
                if feat.get("feature", "").startswith("general.raw_text") or "mechanic_change" in feat.get("feature", ""):
                    unknown_count += 1
                    
    print(f"Total Trace Items: {total_count}")
    print(f"Unknown/Raw Fallbacks: {unknown_count} ({unknown_count/total_count:.1%})")
    
    print("\nRandom Sample of 10 Traces:")
    random.seed(42)
    sample = random.sample(all_features, min(10, total_count))
    for s in sample:
        print(f"{s['patch']} {s['agent']}: {s['feature']} ({s['impact']})")

if __name__ == "__main__":
    audit_distribution()
    audit_features()
