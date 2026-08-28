import os
import sys
sys.path.insert(0, ".")

import pandas as pd
import pytest
from ml.dataset_builder import generate_all_datasets, FEATURES_DIR, split_dataset

def test_dataset_builder_pipeline():
    generate_all_datasets()
    
    match_csv = os.path.join(FEATURES_DIR, "match_prediction.csv")
    match_pq = os.path.join(FEATURES_DIR, "match_prediction.parquet")
    train_pq = os.path.join(FEATURES_DIR, "match_train.parquet")
    val_pq = os.path.join(FEATURES_DIR, "match_val.parquet")
    test_pq = os.path.join(FEATURES_DIR, "match_test.parquet")
    
    assert os.path.exists(match_csv)
    assert os.path.exists(match_pq)
    assert os.path.exists(train_pq)
    assert os.path.exists(val_pq)
    assert os.path.exists(test_pq)
    
    df = pd.read_parquet(match_pq)
    assert not df.empty
    assert "target" in df.columns
    assert "diff_win_rate" in df.columns

def test_chronological_splits():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2025-05-01", "2026-02-01", "2026-05-15"],
        "val": [1, 2, 3, 4]
    })
    tr, val, te = split_dataset(df, train_cutoff="2025-12-31", val_cutoff="2026-04-30")
    
    assert len(tr) == 2
    assert len(val) == 1
    assert len(te) == 1
