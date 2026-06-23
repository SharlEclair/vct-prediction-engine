import os
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
from feature_engineering import load_raw_matches

RAW_DIR = os.path.join(".", "data", "raw")
PROCESSED_DIR = os.path.join(".", "data", "processed")

# Baseline/Old Registry definition
baseline_registry = {
    "9.11": {
        "Neon": 0.6321205588285577
    },
    "10.04": {
        "Clove": 1.0
    },
    "12.00": {
        "Breach": 1.0
    }
}

def load_patch_dates():
    patch_dates = {}
    csv_path = os.path.join(RAW_DIR, "patch_notes.csv")
    if os.path.exists(csv_path):
        try:
            df_patches = pd.read_csv(csv_path)
            for _, row in df_patches.iterrows():
                version = str(row['patch_version']).strip().lower()
                if version.startswith('v'):
                    version = version[1:]
                date_str_val = str(row['release_date'])
                clean_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str_val)
                parsed_dt = datetime.strptime(clean_date, '%B %d, %Y')
                patch_dates[version] = parsed_dt
        except Exception as e:
            pass
    return patch_dates

def load_player_baselines():
    player_stats_path = os.path.join(RAW_DIR, "player_stats.json")
    baseline_lookup = {}
    if os.path.exists(player_stats_path):
        with open(player_stats_path, "r", encoding="utf-8") as f:
            player_stats_baseline = json.load(f)["data"]["segments"]
        for ps in player_stats_baseline:
            p_name = ps["player"]
            acs_b = float(ps.get("average_combat_score", 200.0))
            kast_str = ps.get("kill_assists_survived_traded", "70%")
            kast_b = float(kast_str.replace("%", "")) / 100.0 if "%" in kast_str else 0.70
            baseline_lookup[p_name] = {"acs": acs_b, "kast": kast_b}
    return baseline_lookup

def get_player_performances(matches):
    player_performances = []
    for m in matches:
        match_id = m['match_id']
        ts = m['timestamp']
        patch = m.get('patch')
        
        # Player map performance tracking
        player_map_stats = {}
        for map_data in m.get('maps', []):
            rounds_count = len(map_data.get('rounds', []))
            if rounds_count == 0:
                score = map_data.get('score', {})
                rounds_count = int(score.get('team1', 0)) + int(score.get('team2', 0))
                if rounds_count == 0:
                    rounds_count = 24
                    
            for team_key in ['team1', 'team2']:
                for p in map_data.get('players', {}).get(team_key, []):
                    p_name = p['name']
                    acs_val = float(p['acs']) if (p.get('acs') and str(p['acs']).isdigit()) else 0.0
                    kast_str = p.get('kast', '')
                    kast_val = float(kast_str.replace('%', '')) / 100.0 if (kast_str and '%' in kast_str) else 0.70
                    
                    if p_name not in player_map_stats:
                        player_map_stats[p_name] = []
                    player_map_stats[p_name].append({
                        'acs': acs_val,
                        'kast': kast_val,
                        'agent': p.get('agent', '')
                    })
                    
        for p_name, stats_list in player_map_stats.items():
            avg_acs = sum(s['acs'] for s in stats_list) / len(stats_list)
            avg_kast = sum(s['kast'] for s in stats_list) / len(stats_list)
            
            from collections import Counter
            agent_counts = Counter(s['agent'] for s in stats_list if s.get('agent'))
            most_common_agent = agent_counts.most_common(1)[0][0] if agent_counts else ""
            
            player_performances.append({
                'player': p_name,
                'match_id': match_id,
                'timestamp': ts,
                'patch': patch,
                'agent': most_common_agent,
                'acs': avg_acs,
                'kast': avg_kast
            })
    return pd.DataFrame(player_performances)

def compute_delta_penalty(agent, patch_hist, patch_target, registry, patch_dates):
    if not patch_hist or not patch_target or patch_hist == patch_target:
        return 0.0
    dt_hist = patch_dates.get(patch_hist.lower())
    dt_target = patch_dates.get(patch_target.lower())
    if dt_hist is None or dt_target is None or dt_hist >= dt_target:
        return 0.0
    
    penalty = 0.0
    for patch, nerf_agents in registry.items():
        dt_patch = patch_dates.get(patch.lower())
        if dt_patch is not None:
            if dt_hist < dt_patch <= dt_target:
                penalty += nerf_agents.get(agent, 0.0)
    return penalty

def run_calibration():
    print("Loading VCT match data...")
    matches = load_raw_matches()
    
    print("Loading registries...")
    blops_registry_path = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
    with open(blops_registry_path, "r", encoding="utf-8") as f:
        blops_registry = json.load(f)
        
    patch_distance_matrix_path = os.path.join(PROCESSED_DIR, "patch_distance_matrix.json")
    with open(patch_distance_matrix_path, "r", encoding="utf-8") as f:
        patch_distance_matrix = json.load(f)
        
    patch_dates = load_patch_dates()
    player_baselines = load_player_baselines()
    df_perf = get_player_performances(matches)
    
    unique_patches = sorted(list(df_perf["patch"].dropna().unique()), key=lambda x: [int(v) for v in x.split('.')])
    
    # 1. Test different lambdas
    lambdas = [-1.0, -1.5, -2.0, -2.5, -3.0]
    calibration_results = {}
    
    for lmb in lambdas:
        print(f"Evaluating lambda = {lmb} ...")
        all_errors = []
        all_sq_errors = []
        all_weights = []
        
        for idx in range(len(unique_patches) - 1):
            patch_x = unique_patches[idx]
            patch_y = unique_patches[idx + 1]
            
            df_target = df_perf[df_perf["patch"] == patch_y]
            agents_y = df_target["agent"].dropna().unique()
            
            for agent in agents_y:
                if not agent:
                    continue
                df_agent_target = df_target[df_target["agent"] == agent]
                
                for _, row in df_agent_target.iterrows():
                    player = row["player"]
                    ts_target = row["timestamp"]
                    acs_target = row["acs"]
                    
                    df_p_hist = df_perf[(df_perf["player"] == player) & (df_perf["timestamp"] < ts_target)]
                    
                    if df_p_hist.empty:
                        baseline = player_baselines.get(player, {"acs": 200.0, "kast": 0.70})
                        pred_acs = baseline["acs"]
                    else:
                        delta_days = (ts_target - df_p_hist["timestamp"]).dt.total_seconds() / 86400.0
                        time_decay = np.exp(-0.02 * delta_days)
                        
                        is_same_agent = (df_p_hist["agent"] == agent).astype(float)
                        delta_p_global = np.array([patch_distance_matrix.get(p_hist, {}).get(patch_y, 0.0) for p_hist in df_p_hist["patch"]])
                        delta_p_agent = np.array([compute_delta_penalty(agent, p_hist, patch_y, blops_registry, patch_dates) for p_hist in df_p_hist["patch"]])
                        
                        # Apply candidate lambda (note: lmb is negative, so exp(lmb * delta) matches exp(-lambda * delta))
                        state_penalty = is_same_agent * np.exp(lmb * delta_p_agent) + (1.0 - is_same_agent) * np.exp(-0.5 * delta_p_global)
                        weight = time_decay * state_penalty
                        
                        w_sum = sum(weight)
                        if w_sum > 0:
                            pred_acs = sum(df_p_hist["acs"] * weight) / w_sum
                        else:
                            baseline = player_baselines.get(player, {"acs": 200.0, "kast": 0.70})
                            pred_acs = baseline["acs"]
                            
                    all_errors.append(abs(acs_target - pred_acs))
                    all_sq_errors.append((acs_target - pred_acs) ** 2)
                    all_weights.append(1.0)
                    
        mae = np.mean(all_errors)
        rmse = np.sqrt(np.mean(all_sq_errors))
        calibration_results[lmb] = {"mae": mae, "rmse": rmse}
        print(f"  MAE: {mae:.5f} | RMSE: {rmse:.5f}")
        
    # Find best lambda
    best_lmb = min(calibration_results.keys(), key=lambda x: calibration_results[x]["mae"])
    print(f"\nBest lambda based on MAE: {best_lmb}")
    
    # 2. Correlation Analysis
    # Let's compute:
    # - Observed performance error increase: MAE(Model A, patch Y, agent) - MAE(Model A, patch X, agent)
    # - Observed disruption: |mean_acs(agent on patch Y) - mean_acs(agent on patch X)|
    # - BLOPS score: blops_registry[patch Y][agent]
    
    transition_data = []
    
    for idx in range(len(unique_patches) - 1):
        patch_x = unique_patches[idx]
        patch_y = unique_patches[idx + 1]
        
        # Get target performances
        df_target_y = df_perf[df_perf["patch"] == patch_y]
        df_target_x = df_perf[df_perf["patch"] == patch_x]
        
        agents_y = df_target_y["agent"].dropna().unique()
        
        for agent in agents_y:
            if not agent:
                continue
                
            # Filter targets
            df_agent_y = df_target_y[df_target_y["agent"] == agent]
            df_agent_x = df_target_x[df_target_x["agent"] == agent]
            
            if df_agent_y.empty or df_agent_x.empty:
                continue
                
            # Compute mean ACS
            mean_acs_x = df_agent_x["acs"].mean()
            mean_acs_y = df_agent_y["acs"].mean()
            disruption = abs(mean_acs_y - mean_acs_x)
            
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
                    time_decay = np.exp(-0.02 * delta_days)
                    w_sum = sum(time_decay)
                    pred_acs = sum(df_p_hist["acs"] * time_decay) / w_sum if w_sum > 0 else player_baselines.get(player, {"acs": 200.0})["acs"]
                errors_y.append(abs(acs_target - pred_acs))
                
            mae_x = np.mean(errors_x)
            mae_y = np.mean(errors_y)
            error_increase = mae_y - mae_x
            
            blops_delta = blops_registry.get(patch_y, {}).get(agent, 0.0)
            
            transition_data.append({
                "patch": patch_y,
                "agent": agent,
                "blops_delta": blops_delta,
                "error_increase": error_increase,
                "disruption": disruption
            })
            
    df_trans = pd.DataFrame(transition_data)
    
    # Filter to cases where blops_delta > 0 or error_increase is defined
    df_filtered = df_trans.dropna()
    
    if len(df_filtered) > 1:
        pearson_corr, p_val_p = pearsonr(df_filtered["blops_delta"], df_filtered["error_increase"])
        spearman_corr, p_val_s = spearmanr(df_filtered["blops_delta"], df_filtered["disruption"])
        print(f"\nPearson correlation (BLOPS score vs observed performance error increase): {pearson_corr:.5f} (p-value: {p_val_p:.5e})")
        print(f"Spearman correlation (BLOPS ranking vs observed disruption ranking): {spearman_corr:.5f} (p-value: {p_val_s:.5e})")
    else:
        pearson_corr, spearman_corr = 0.0, 0.0
        print("\nNot enough data points to calculate correlation.")
        
    # Save DECAY_CALIBRATION_REPORT.md
    with open("DECAY_CALIBRATION_REPORT.md", "w") as f:
        f.write("# Decay Calibration Analysis\n\n")
        f.write("This report documents the performance of the historical performance concept drift model under alternative lambda values.\n\n")
        f.write("## Mathematical Scaling of exp(lambda * delta)\n\n")
        f.write(f"The decay is computed as `exp(lambda * delta_p_agent)`. We evaluated five candidate lambda values on the complete VCT match dataset:\n\n")
        f.write("| Lambda Coefficient | Overall ACS MAE | Overall ACS RMSE |\n")
        f.write("| --- | --- | --- |\n")
        for lmb in lambdas:
            mae_val = calibration_results[lmb]["mae"]
            rmse_val = calibration_results[lmb]["rmse"]
            f.write(f"| {lmb} | {mae_val:.5f} | {rmse_val:.5f} |\n")
        f.write("\n")
        f.write(f"**Best Lambda:** `{best_lmb}`\n\n")
        f.write("## Correlation Analysis\n\n")
        f.write(f"- **Pearson Correlation** (BLOPS score vs observed performance error increase): `{pearson_corr:.5f}`\n")
        f.write(f"- **Spearman Correlation** (BLOPS ranking vs observed disruption ranking): `{spearman_corr:.5f}`\n")
        
    print("\nCalibration report saved to DECAY_CALIBRATION_REPORT.md")

if __name__ == "__main__":
    run_calibration()
