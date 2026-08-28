import os
import json
import re
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from feature_engineering import load_raw_matches

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("blops_backtest")

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
            logger.error(f"Failed to load patch notes: {e}")
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

def main():
    logger.info("Loading VCT match data...")
    matches = load_raw_matches()
    
    logger.info("Loading registries...")
    blops_registry_path = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
    with open(blops_registry_path, "r", encoding="utf-8") as f:
        blops_registry = json.load(f)
        
    patch_distance_matrix_path = os.path.join(PROCESSED_DIR, "patch_distance_matrix.json")
    with open(patch_distance_matrix_path, "r", encoding="utf-8") as f:
        patch_distance_matrix = json.load(f)
        
    patch_dates = load_patch_dates()
    player_baselines = load_player_baselines()
    
    # Process player performances
    df_perf = get_player_performances(matches)
    
    # Get chronological patch list
    unique_patches = sorted(list(df_perf["patch"].dropna().unique()), key=lambda x: [int(v) for v in x.split('.')])
    logger.info(f"Loaded {len(unique_patches)} unique patches: {unique_patches}")
    
    # Track results
    results = []
    
    # Run backtest for each transition
    for idx in range(len(unique_patches) - 1):
        patch_x = unique_patches[idx]
        patch_y = unique_patches[idx + 1]
        
        logger.info(f"Processing transition: {patch_x} -> {patch_y}")
        
        # Get target performances in patch_y
        df_target = df_perf[df_perf["patch"] == patch_y]
        
        # Find agents played in patch_y
        agents_y = df_target["agent"].dropna().unique()
        
        for agent in agents_y:
            if not agent:
                continue
                
            # Filter targets to this agent
            df_agent_target = df_target[df_target["agent"] == agent]
            post_matches = len(df_agent_target)
            
            # Count pre-patch matches of this agent in patch_x
            df_pre_agent = df_perf[(df_perf["patch"] == patch_x) & (df_perf["agent"] == agent)]
            pre_matches = len(df_pre_agent)
            
            if post_matches == 0:
                continue
                
            # Delta values
            old_delta = baseline_registry.get(patch_y, {}).get(agent, 0.0)
            blops_delta = blops_registry.get(patch_y, {}).get(agent, 0.0)
            
            # Compute errors for the player-matches
            errors_a = []
            errors_b = []
            errors_c = []
            
            sq_errors_a = []
            sq_errors_b = []
            sq_errors_c = []
            
            kast_errors_a = []
            kast_errors_b = []
            kast_errors_c = []
            
            for _, row in df_agent_target.iterrows():
                player = row["player"]
                ts_target = row["timestamp"]
                acs_target = row["acs"]
                kast_target = row["kast"]
                
                # Get historical matches of player
                df_p_hist = df_perf[(df_perf["player"] == player) & (df_perf["timestamp"] < ts_target)]
                
                if df_p_hist.empty:
                    # Fallback to baseline
                    baseline = player_baselines.get(player, {"acs": 200.0, "kast": 0.70})
                    pred_acs_a = baseline["acs"]
                    pred_acs_b = baseline["acs"]
                    pred_acs_c = baseline["acs"]
                    
                    pred_kast_a = baseline["kast"]
                    pred_kast_b = baseline["kast"]
                    pred_kast_c = baseline["kast"]
                else:
                    # Calculate weights and predict
                    # Time decay
                    delta_days = (ts_target - df_p_hist["timestamp"]).dt.total_seconds() / 86400.0
                    time_decay = np.exp(-0.02 * delta_days)
                    
                    is_same_agent = (df_p_hist["agent"] == agent).astype(float)
                    
                    # Global drift delta
                    delta_p_global = np.array([patch_distance_matrix.get(p_hist, {}).get(patch_y, 0.0) for p_hist in df_p_hist["patch"]])
                    
                    # Compute agent deltas for Model B and C
                    delta_p_agent_b = np.array([compute_delta_penalty(agent, p_hist, patch_y, baseline_registry, patch_dates) for p_hist in df_p_hist["patch"]])
                    delta_p_agent_c = np.array([compute_delta_penalty(agent, p_hist, patch_y, blops_registry, patch_dates) for p_hist in df_p_hist["patch"]])
                    
                    # Model A: No patch decay (history weights unchanged)
                    weight_a = time_decay
                    
                    # Model B: Old registry logic
                    state_penalty_b = is_same_agent * np.exp(-2.0 * delta_p_agent_b) + (1.0 - is_same_agent) * np.exp(-0.5 * delta_p_global)
                    weight_b = time_decay * state_penalty_b
                    
                    # Model C: BLOPS registry
                    state_penalty_c = is_same_agent * np.exp(-2.0 * delta_p_agent_c) + (1.0 - is_same_agent) * np.exp(-0.5 * delta_p_global)
                    weight_c = time_decay * state_penalty_c
                    
                    # Compute weighted predictions
                    def get_weighted_pred(vals, weights, default):
                        w_sum = sum(weights)
                        return sum(vals * weights) / w_sum if w_sum > 0 else default
                        
                    baseline = player_baselines.get(player, {"acs": 200.0, "kast": 0.70})
                    
                    pred_acs_a = get_weighted_pred(df_p_hist["acs"], weight_a, baseline["acs"])
                    pred_acs_b = get_weighted_pred(df_p_hist["acs"], weight_b, baseline["acs"])
                    pred_acs_c = get_weighted_pred(df_p_hist["acs"], weight_c, baseline["acs"])
                    
                    pred_kast_a = get_weighted_pred(df_p_hist["kast"], weight_a, baseline["kast"])
                    pred_kast_b = get_weighted_pred(df_p_hist["kast"], weight_b, baseline["kast"])
                    pred_kast_c = get_weighted_pred(df_p_hist["kast"], weight_c, baseline["kast"])
                    
                # Accumulate errors
                errors_a.append(abs(acs_target - pred_acs_a))
                errors_b.append(abs(acs_target - pred_acs_b))
                errors_c.append(abs(acs_target - pred_acs_c))
                
                sq_errors_a.append((acs_target - pred_acs_a)**2)
                sq_errors_b.append((acs_target - pred_acs_b)**2)
                sq_errors_c.append((acs_target - pred_acs_c)**2)
                
                kast_errors_a.append(abs(kast_target - pred_kast_a))
                kast_errors_b.append(abs(kast_target - pred_kast_b))
                kast_errors_c.append(abs(kast_target - pred_kast_c))
                
            if errors_a:
                mae_a = np.mean(errors_a)
                mae_b = np.mean(errors_b)
                mae_c = np.mean(errors_c)
                
                rmse_a = np.sqrt(np.mean(sq_errors_a))
                rmse_b = np.sqrt(np.mean(sq_errors_b))
                rmse_c = np.sqrt(np.mean(sq_errors_c))
                
                kast_mae_a = np.mean(kast_errors_a)
                kast_mae_b = np.mean(kast_errors_b)
                kast_mae_c = np.mean(kast_errors_c)
                
                results.append({
                    "patch": patch_y,
                    "agent": agent,
                    "pre_patch_matches": pre_matches,
                    "post_patch_matches": post_matches,
                    "old_delta": old_delta,
                    "blops_delta": blops_delta,
                    "model_a_error": mae_a,
                    "model_b_error": mae_b,
                    "model_c_error": mae_c,
                    "model_a_rmse": rmse_a,
                    "model_b_rmse": rmse_b,
                    "model_c_rmse": rmse_c,
                    "model_a_kast_mae": kast_mae_a,
                    "model_b_kast_mae": kast_mae_b,
                    "model_c_kast_mae": kast_mae_c
                })
                
    df_results = pd.DataFrame(results)
    
    # Save the requested comparison_results.csv with only the specified columns first
    csv_cols = [
        "patch", "agent", "pre_patch_matches", "post_patch_matches", 
        "old_delta", "blops_delta", "model_a_error", "model_b_error", "model_c_error"
    ]
    df_csv = df_results[csv_cols]
    csv_output_path = os.path.join(PROCESSED_DIR, "comparison_results.csv")
    # Also save to current directory for downstream compatibility if needed
    df_csv.to_csv("comparison_results.csv", index=False)
    df_csv.to_csv(csv_output_path, index=False)
    logger.info(f"Saved patch backtest comparison results to {csv_output_path} and ./comparison_results.csv")
    
    # Output metrics to verify
    total_matches_evaluated = df_results["post_patch_matches"].sum()
    logger.info(f"Total player-match observations evaluated: {total_matches_evaluated}")
    
    # Compute overall metrics weighted by post_patch_matches
    def weighted_avg(vals, weights):
        return np.sum(vals * weights) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        
    weights = df_results["post_patch_matches"].values
    
    overall_mae_a = weighted_avg(df_results["model_a_error"].values, weights)
    overall_mae_b = weighted_avg(df_results["model_b_error"].values, weights)
    overall_mae_c = weighted_avg(df_results["model_c_error"].values, weights)
    
    overall_rmse_a = np.sqrt(weighted_avg(df_results["model_a_rmse"].values ** 2, weights))
    overall_rmse_b = np.sqrt(weighted_avg(df_results["model_b_rmse"].values ** 2, weights))
    overall_rmse_c = np.sqrt(weighted_avg(df_results["model_c_rmse"].values ** 2, weights))
    
    logger.info("Overall ACS MAE:")
    logger.info(f"  Model A (No Decay): {overall_mae_a:.4f}")
    logger.info(f"  Model B (Old Registry): {overall_mae_b:.4f}")
    logger.info(f"  Model C (BLOPS Registry): {overall_mae_c:.4f}")
    
    logger.info("Overall ACS RMSE:")
    logger.info(f"  Model A (No Decay): {overall_rmse_a:.4f}")
    logger.info(f"  Model B (Old Registry): {overall_rmse_b:.4f}")
    logger.info(f"  Model C (BLOPS Registry): {overall_rmse_c:.4f}")

if __name__ == "__main__":
    main()
