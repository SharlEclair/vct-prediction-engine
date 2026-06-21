import os
import logging
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
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
    
    # 2. Implement exponential time-decay weights
    # Calculate time delta in days relative to the most recent match in the dataset
    max_timestamp = data['timestamp'].max()
    data['delta_days'] = (max_timestamp - data['timestamp']).dt.total_seconds() / 86400.0
    
    decay_constant = 0.02
    data['sample_weight'] = np.exp(-decay_constant * data['delta_days'])
    
    logger.info("Calculated time deltas and exponential weights:")
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
    # The last match in the dataset is the test set, all preceding matches are the training set
    X_train = X.iloc[:-1].reset_index(drop=True)
    y_train = y.iloc[:-1].reset_index(drop=True)
    w_train = weights.iloc[:-1].reset_index(drop=True)
    
    X_test = X.iloc[-1:].reset_index(drop=True)
    y_test = y.iloc[-1:].reset_index(drop=True)
    test_match_id = data['match_id'].iloc[-1]
    
    logger.info(f"Split data: Train shape={X_train.shape}, Test shape={X_test.shape}")
    logger.info(f"Test Match ID: {test_match_id}")
    
    # 4. CatBoost Classifier Training
    logger.info("Initializing and training CatBoostClassifier...")
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=4,
        cat_features=cat_features,
        random_seed=42,
        verbose=0
    )
    
    model.fit(X_train, y_train, sample_weight=w_train)
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
