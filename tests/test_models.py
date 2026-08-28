import os
import sys
sys.path.insert(0, ".")

import joblib
import pytest
from ml.train import train_all_models, PRODUCTION_DIR

def test_model_training_and_persistence():
    train_all_models()
    
    baseline_path = os.path.join(PRODUCTION_DIR, "match_winner_baseline_v1.pkl")
    main_path = os.path.join(PRODUCTION_DIR, "match_winner_v1.pkl")
    map_path = os.path.join(PRODUCTION_DIR, "map_winner_v1.pkl")
    score_path = os.path.join(PRODUCTION_DIR, "score_predictor_v1.pkl")
    
    assert os.path.exists(baseline_path)
    assert os.path.exists(main_path)
    assert os.path.exists(map_path)
    assert os.path.exists(score_path)
    
    main_pkg = joblib.load(main_path)
    assert "model" in main_pkg
    assert "metadata" in main_pkg
