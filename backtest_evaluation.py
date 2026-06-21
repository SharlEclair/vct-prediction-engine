import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

def run_backtest():
    # 1. Load data
    processed_dir = os.path.join(".", "data", "processed")
    x_path = os.path.join(processed_dir, "X_features.csv")
    y_path = os.path.join(processed_dir, "y_target.csv")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Processed features or target files not found at {processed_dir}")
        
    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    
    data = pd.merge(X_df, y_df, on="match_id")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Chronological Split (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"Total Matches: {len(data)}")
    print(f"Train Matches: {len(train_data)} (First 80%)")
    print(f"Test Matches: {len(test_data)} (Final 20%)")
    
    # Calculate exponential time-decay weights on train set relative to max train date
    max_train_ts = train_data['timestamp'].max()
    train_data['delta_days'] = (max_train_ts - train_data['timestamp']).dt.total_seconds() / 86400.0
    train_data['sample_weight'] = np.exp(-0.02 * train_data['delta_days'])
    
    # Define features
    v1_features = [
        'team_a_name', 'team_b_name',
        'team_a_historical_acs_ema', 'team_a_historical_kast_ema', 'team_a_historical_duel_diff', 'team_a_historical_avg_loadout',
        'team_b_historical_acs_ema', 'team_b_historical_kast_ema', 'team_b_historical_duel_diff', 'team_b_historical_avg_loadout',
        'map_1_name', 'map_1_veto_weight',
        'map_2_name', 'map_2_veto_weight',
        'map_3_name', 'map_3_veto_weight'
    ]
    
    v2_features = [col for col in X_df.columns if col not in ['match_id', 'timestamp']]
    
    # Check that V1 features are indeed a subset of what's in X_features
    for col in v1_features:
        if col not in X_df.columns:
            raise KeyError(f"Feature {col} not found in X_features.csv")
            
    # Categorical features list
    all_cats = ['team_a_name', 'team_b_name', 'map_1_name', 'map_2_name', 'map_3_name', 'map_4_name', 'map_5_name']
    cat_v1 = [col for col in v1_features if col in all_cats]
    cat_v2 = [col for col in v2_features if col in all_cats]
    
    # Handle missing categoricals or clean string formatting
    for col in all_cats:
        train_data[col] = train_data[col].astype(str).fillna('None')
        test_data[col] = test_data[col].astype(str).fillna('None')
        
    # Prepare datasets
    X_train_v1 = train_data[v1_features].copy()
    X_test_v1 = test_data[v1_features].copy()
    
    X_train_v2 = train_data[v2_features].copy()
    X_test_v2 = test_data[v2_features].copy()
    
    y_train = train_data['y_target'].copy()
    y_test = test_data['y_target'].copy()
    w_train = train_data['sample_weight'].copy()
    
    # 3. Model Training
    print("\nTraining V1 Baseline Model...")
    model_v1 = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=4,
        cat_features=cat_v1,
        random_seed=42,
        verbose=0
    )
    model_v1.fit(X_train_v1, y_train, sample_weight=w_train)
    
    print("Training V2 Tactical Model...")
    model_v2 = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=4,
        cat_features=cat_v2,
        random_seed=42,
        verbose=0
    )
    model_v2.fit(X_train_v2, y_train, sample_weight=w_train)
    
    # 4. Predictions & Probability Estimation
    preds_v1 = model_v1.predict(X_test_v1)
    probs_v1 = model_v1.predict_proba(X_test_v1)[:, 1]
    
    preds_v2 = model_v2.predict(X_test_v2)
    probs_v2 = model_v2.predict_proba(X_test_v2)[:, 1]
    
    # 5. Metrics Calculations
    metrics = {}
    for name, y_pred, y_prob in [("V1 Baseline", preds_v1, probs_v1), ("V2 Tactical", preds_v2, probs_v2)]:
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob)
        metrics[name] = {"Accuracy": acc, "Log-Loss": loss, "F1-Score": f1, "ROC-AUC": auc}
        
    # Formatted side-by-side console report
    print("\n" + "="*59)
    print(f"{'METRIC':<20} | {'V1 BASELINE':<12} | {'V2 TACTICAL':<12} | {'DELTA':<10}")
    print("="*59)
    for m_key in ["Accuracy", "Log-Loss", "F1-Score", "ROC-AUC"]:
        val1 = metrics["V1 Baseline"][m_key]
        val2 = metrics["V2 Tactical"][m_key]
        delta = val2 - val1
        print(f"{m_key:<20} | {val1:<12.4f} | {val2:<12.4f} | {delta:<+10.4f}")
    print("="*59 + "\n")
    
    # 6. ROC curves plotting
    plt.figure(figsize=(8, 6))
    
    # Plot V1 ROC
    disp_v1 = RocCurveDisplay.from_predictions(
        y_test, probs_v1, 
        name="V1 Baseline", 
        ax=plt.gca()
    )
    if hasattr(disp_v1, 'line_') and disp_v1.line_ is not None:
        disp_v1.line_.set_color("#1f77b4")
        disp_v1.line_.set_linewidth(2)
        
    # Plot V2 ROC
    disp_v2 = RocCurveDisplay.from_predictions(
        y_test, probs_v2, 
        name="V2 Tactical", 
        ax=plt.gca()
    )
    if hasattr(disp_v2, 'line_') and disp_v2.line_ is not None:
        disp_v2.line_.set_color("#ff7f0e")
        disp_v2.line_.set_linewidth(2)
    
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title("ROC Curve Comparison: V1 Baseline vs V2 Tactical Model", fontsize=12, pad=15)
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    
    os.makedirs(processed_dir, exist_ok=True)
    plot_path = os.path.join(processed_dir, "roc_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"ROC Curve Comparison Plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    run_backtest()
