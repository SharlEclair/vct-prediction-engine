import os
import logging
import re
import json
from datetime import datetime
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
import shap
import matplotlib.pyplot as plt

# Set up logging
logger = logging.getLogger("model_pipeline")

def run_modeling_pipeline():
    logger.info("Starting Phase 3: Modeling & Explainable AI...")
    
    # Define paths
    processed_dir = os.path.join(".", "data", "processed")
    x_path = os.path.join(processed_dir, "X_features.csv")
    y_path = os.path.join(processed_dir, "y_target.csv")
    shap_output_path = os.path.join(processed_dir, "shap_explanation.png")
    
    # 1. Load the processed CSVs
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Processed data files not found. Ensure features are built. "
            f"Expected {x_path} and {y_path}"
        )
        
    logger.info(f"Loading processed features from {x_path}")
    X_df = pd.read_csv(x_path)
    logger.info(f"Loading targets from {y_path}")
    y_df = pd.read_csv(y_path)
    
    # Merge datasets on match_id to ensure proper alignment
    data = pd.merge(X_df, y_df, on="match_id")
    
    # Sort chronologically by timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Total dataset size after merge: {len(data)} matches.")
    
    # 2. Implement 2D composite decay weights relative to the target match (the last match in the dataset)
    # Load raw matches to get player agent details
    from feature_engineering import load_raw_matches
    matches = load_raw_matches()
    
    # Target match is the last match
    target_match = matches[-1]
    target_timestamp = target_match["timestamp"]
    target_patch = target_match["patch"]
    
    # Extract target agents
    from collections import Counter
    target_agents = {}
    for map_data in target_match.get("maps", []):
        for team_key in ["team1", "team2"]:
            for p in map_data.get("players", {}).get(team_key, []):
                p_name = p["name"]
                agent = p["agent"]
                if p_name not in target_agents:
                    target_agents[p_name] = []
                target_agents[p_name].append(agent)
    for p_name, agents in target_agents.items():
        target_agents[p_name] = Counter(agents).most_common(1)[0][0]
        
    # Load patch release dates
    patch_dates = {}
    csv_path = os.path.join(".", "data", "raw", "patch_notes.csv")
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
            
    # Load patch nerf registry and patch distance matrix
    nerf_registry_path = os.path.join(processed_dir, "automated_patch_nerf_registry.json")
    with open(nerf_registry_path, "r", encoding="utf-8") as f:
        nerf_registry = json.load(f)
        
    distance_matrix_path = os.path.join(processed_dir, "patch_distance_matrix.json")
    with open(distance_matrix_path, "r", encoding="utf-8") as f:
        patch_distance_matrix = json.load(f)
        
    def get_agent_nerf_penalty(agent: str, p_hist: str, p_target: str) -> float:
        if p_hist == p_target:
            return 0.0
        dt_hist = patch_dates.get(p_hist.lower() if p_hist else '')
        dt_target = patch_dates.get(p_target.lower() if p_target else '')
        if dt_hist is None or dt_target is None:
            return 0.0
        if dt_hist >= dt_target:
            return 0.0
        penalty = 0.0
        for patch, nerf_agents in nerf_registry.items():
            dt_patch = patch_dates.get(patch.lower())
            if dt_patch is not None:
                if dt_hist < dt_patch <= dt_target:
                    penalty += nerf_agents.get(agent, 0.0)
        return penalty
        
    match_weights = {}
    for m in matches:
        m_id = m["match_id"]
        m_patch = m["patch"]
        m_ts = m["timestamp"]
        
        delta_days = (target_timestamp - m_ts).total_seconds() / 86400.0
        time_decay = np.exp(-0.02 * delta_days)
        
        # JSD distance
        delta_p_global = patch_distance_matrix.get(m_patch, {}).get(target_patch, 0.0)
        
        # Player-level weights
        player_weights = []
        for map_data in m.get("maps", []):
            for team_key in ["team1", "team2"]:
                for p in map_data.get("players", {}).get(team_key, []):
                    p_name = p["name"]
                    agent = p["agent"]
                    
                    target_agent = target_agents.get(p_name)
                    is_same_agent = 1 if (target_agent and agent == target_agent) else 0
                    
                    delta_p_agent = get_agent_nerf_penalty(target_agent, m_patch, target_patch) if is_same_agent else 0.0
                    
                    state_penalty = is_same_agent * np.exp(-2.0 * delta_p_agent) + (1 - is_same_agent) * np.exp(-0.5 * delta_p_global)
                    player_weights.append(time_decay * state_penalty)
                    
        if player_weights:
            match_weights[m_id] = float(np.mean(player_weights))
        else:
            match_weights[m_id] = float(time_decay * np.exp(-0.5 * delta_p_global))
            
    # Apply weights to data
    data['sample_weight'] = data['match_id'].map(match_weights).fillna(1.0)
    data['delta_days'] = (target_timestamp - data['timestamp']).dt.total_seconds() / 86400.0
    
    logger.info("Calculated time deltas and 2D composite weights:")
    for idx, row in data.iterrows():
        logger.info(
            f"  Match {row['match_id']} | Date: {row['timestamp']} | "
            f"Delta: {row['delta_days']:.4f} days | Weight: {row['sample_weight']:.6f}"
        )
        
    # Prepare features and target vectors
    exclude_cols = ['match_id', 'timestamp', 'y_target', 'delta_days', 'sample_weight']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].copy()
    y = data['y_target'].copy()
    weights = data['sample_weight'].copy()
    
    # Identify and clean categorical features
    cat_features = list(X.select_dtypes(include=['object', 'category']).columns)
    logger.info(f"Identified categorical features: {cat_features}")
    for col in cat_features:
        X[col] = X[col].astype(str).fillna('None')
        
    # 3. Chronological Train/Test Split
    X_train = X.iloc[:-1].reset_index(drop=True)
    y_train = y.iloc[:-1].reset_index(drop=True)
    w_train = weights.iloc[:-1].reset_index(drop=True)
    
    X_test = X.iloc[-1:].reset_index(drop=True)
    y_test = y.iloc[-1:].reset_index(drop=True)
    test_match_id = data['match_id'].iloc[-1]
    
    logger.info(f"Split data: Train shape={X_train.shape}, Test shape={X_test.shape}")
    logger.info(f"Test Match ID: {test_match_id}")
    
    # 4. Instantiate CatBoost Pools and run cross-validation
    logger.info("Instantiating CatBoost Pools...")
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features,
        weight=w_train
    )
    
    cv_params = {
        'iterations': 100,
        'learning_rate': 0.05,
        'depth': 4,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'verbose': 0
    }
    
    logger.info("Running TimeSeries cross-validation to prevent temporal leakage...")
    cv_results = cv(
        pool=train_pool,
        params=cv_params,
        fold_count=5,
        type='TimeSeries',
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Log CV results
    best_loss_idx = np.argmin(cv_results['test-Logloss-mean'])
    best_loss = cv_results['test-Logloss-mean'][best_loss_idx]
    logger.info(f"Cross-validation completed. Best CV Logloss: {best_loss:.4f} at iteration {best_loss_idx}")
    
    # 5. CatBoost Classifier Training
    logger.info("Initializing and training CatBoostClassifier on training pool...")
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=4,
        random_seed=42,
        verbose=0
    )
    
    model.fit(train_pool)
    logger.info("Model training completed successfully.")
    
    # Save the trained model to vct_model.cbm
    model_save_path = os.path.join(processed_dir, "vct_model.cbm")
    model.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Run predict_proba to output the confidence percentage for the target match
    # class 1 matches Team A win probability
    probs = model.predict_proba(X_test)[0]
    win_prob_team_a = probs[1]
    win_prob_team_b = probs[0]
    
    logger.info(f"Predictions for test match {test_match_id}:")
    logger.info(f"  Team A ({X_test['team_a_name'].iloc[0]}) win probability: {win_prob_team_a:.2%}")
    logger.info(f"  Team B ({X_test['team_b_name'].iloc[0]}) win probability: {win_prob_team_b:.2%}")
    
    # 5. Explainable AI (SHAP) Integration
    logger.info("Calculating SHAP values for the test match prediction...")
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)
    
    # Generate and save SHAP waterfall plot
    logger.info(f"Generating SHAP waterfall plot and saving to {shap_output_path}...")
    plt.figure(figsize=(10, 6))
    
    # Generate waterfall plot for the test instance (index 0 in test set)
    shap.plots.waterfall(explanation[0], show=False)
    plt.title(f"SHAP Waterfall Plot for Match {test_match_id}", fontsize=14, pad=20)
    plt.tight_layout()
    
    # Ensure processed dir exists
    os.makedirs(processed_dir, exist_ok=True)
    plt.savefig(shap_output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info("SHAP waterfall explanation saved.")
    
    return {
        "test_match_id": test_match_id,
        "team_a": X_test['team_a_name'].iloc[0],
        "team_b": X_test['team_b_name'].iloc[0],
        "win_prob_team_a": win_prob_team_a,
        "win_prob_team_b": win_prob_team_b,
        "shap_plot_path": shap_output_path
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_modeling_pipeline()
