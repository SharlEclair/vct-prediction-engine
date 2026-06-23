import os
import json
import pandas as pd

def main():
    # Load comparison_results.csv
    df = pd.read_csv("comparison_results.csv")
    
    # Sort by blops_delta to get top 10 highest BLOPS scores
    df_sorted_blops = df.sort_values(by="blops_delta", ascending=False)
    print("--- TOP 10 HIGHEST BLOPS SCORES ---")
    top_10_blops = df_sorted_blops.head(10)[["patch", "agent", "blops_delta"]]
    for idx, row in top_10_blops.iterrows():
        print(f"Patch {row['patch']} | Agent: {row['agent']} | BLOPS Delta: {row['blops_delta']:.4f}")
        
    # We need to calculate "actual error increase" for each patch-agent transition.
    # Let's load the data from audit_blops_calibration.py's logic or compute it.
    # To be consistent, let's run the transition calculations again or compute error_increase.
    # Wait, we can run a script that computes error_increase for each transition, sorts them, and lists the top 10.
    
    import audit_blops_calibration
    matches = audit_blops_calibration.load_raw_matches()
    patch_dates = audit_blops_calibration.load_patch_dates()
    player_baselines = audit_blops_calibration.load_player_baselines()
    df_perf = audit_blops_calibration.get_player_performances(matches)
    unique_patches = sorted(list(df_perf["patch"].dropna().unique()), key=lambda x: [int(v) for v in x.split('.')])
    
    transition_data = []
    
    for idx in range(len(unique_patches) - 1):
        patch_x = unique_patches[idx]
        patch_y = unique_patches[idx + 1]
        
        df_target_y = df_perf[df_perf["patch"] == patch_y]
        df_target_x = df_perf[df_perf["patch"] == patch_x]
        
        agents_y = df_target_y["agent"].dropna().unique()
        
        for agent in agents_y:
            if not agent:
                continue
                
            df_agent_y = df_target_y[df_target_y["agent"] == agent]
            df_agent_x = df_target_x[df_target_x["agent"] == agent]
            
            if df_agent_y.empty or df_agent_x.empty:
                continue
                
            # Compute Model A prediction error on patch X
            errors_x = []
            for _, row in df_agent_x.iterrows():
                player = row["player"]
                ts_target = row["timestamp"]
                acs_target = row["acs"]
                
                df_p_hist = df_perf[(df_perf["player"] == player) & (df_perf["timestamp"] < ts_target)]
                if df_p_hist.empty:
                    baseline = player_baselines.get(player, {"acs": 200.0, "kast": 0.70})
                    pred_acs = baseline["acs"]
                else:
                    delta_days = (ts_target - df_p_hist["timestamp"]).dt.total_seconds() / 86400.0
                    time_decay = np_exp = 2.71828 # simple math fallback
                    import numpy as np
                    time_decay = np.exp(-0.02 * delta_days)
                    w_sum = sum(time_decay)
                    pred_acs = sum(df_p_hist["acs"] * time_decay) / w_sum if w_sum > 0 else player_baselines.get(player, {"acs": 200.0})["acs"]
                errors_x.append(abs(acs_target - pred_acs))
                
            # Compute Model A prediction error on patch Y
            errors_y = []
            for _, row in df_agent_y.iterrows():
                player = row["player"]
                ts_target = row["timestamp"]
                acs_target = row["acs"]
                
                df_p_hist = df_perf[(df_perf["player"] == player) & (df_perf["timestamp"] < ts_target)]
                if df_p_hist.empty:
                    baseline = player_baselines.get(player, {"acs": 200.0, "kast": 0.70})
                    pred_acs = baseline["acs"]
                else:
                    delta_days = (ts_target - df_p_hist["timestamp"]).dt.total_seconds() / 86400.0
                    import numpy as np
                    time_decay = np.exp(-0.02 * delta_days)
                    w_sum = sum(time_decay)
                    pred_acs = sum(df_p_hist["acs"] * time_decay) / w_sum if w_sum > 0 else player_baselines.get(player, {"acs": 200.0})["acs"]
                errors_y.append(abs(acs_target - pred_acs))
                
            mae_x = sum(errors_x) / len(errors_x) if errors_x else 0.0
            mae_y = sum(errors_y) / len(errors_y) if errors_y else 0.0
            error_increase = mae_y - mae_x
            
            blops_delta = df[(df["patch"] == patch_y) & (df["agent"] == agent)]["blops_delta"].values
            blops_val = blops_delta[0] if len(blops_delta) > 0 else 0.0
            
            transition_data.append({
                "patch": patch_y,
                "agent": agent,
                "blops_delta": blops_val,
                "error_increase": error_increase
            })
            
    df_trans = pd.DataFrame(transition_data)
    
    df_sorted_err = df_trans.sort_values(by="error_increase", ascending=False)
    print("\n--- TOP 10 HIGHEST ACTUAL ERROR INCREASES ---")
    top_10_err = df_sorted_err.head(10)[["patch", "agent", "error_increase"]]
    for idx, row in top_10_err.iterrows():
        print(f"Patch {row['patch']} | Agent: {row['agent']} | Error Increase: {row['error_increase']:.4f}")
        
    # Compare overlap
    blops_set = set(zip(top_10_blops["patch"], top_10_blops["agent"]))
    err_set = set(zip(top_10_err["patch"], top_10_err["agent"]))
    overlap = blops_set.intersection(err_set)
    
    print(f"\nOverlap Count: {len(overlap)}")
    if overlap:
        print("Overlapping Transitions:")
        for patch, agent in overlap:
            print(f"  Patch {patch} | Agent: {agent}")
    else:
        print("No overlapping transitions found in the top 10.")

if __name__ == "__main__":
    main()
