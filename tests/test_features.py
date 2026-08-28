import os
import sys
sys.path.insert(0, ".")

import pandas as pd
import pytest
from ml.feature_builder import generate_feature_store, FEATURES_DIR

def test_feature_store_generation():
    generate_feature_store()
    
    team_p = os.path.join(FEATURES_DIR, "team_features.parquet")
    player_p = os.path.join(FEATURES_DIR, "player_features.parquet")
    map_p = os.path.join(FEATURES_DIR, "map_features.parquet")
    manifest_p = os.path.join(FEATURES_DIR, "feature_manifest.json")
    
    assert os.path.exists(team_p)
    assert os.path.exists(player_p)
    assert os.path.exists(map_p)
    assert os.path.exists(manifest_p)
    
    team_df = pd.read_parquet(team_p)
    player_df = pd.read_parquet(player_p)
    map_df = pd.read_parquet(map_p)
    
    assert not team_df.empty
    assert "win_rate" in team_df.columns
    assert "full_buy_win_rate" in team_df.columns
    
    assert not player_df.empty
    assert "ACS_EMA" in player_df.columns
    assert "rating_EMA" in player_df.columns
    
    assert not map_df.empty
    assert "map_name" in map_df.columns
