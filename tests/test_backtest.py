import os
import sys
sys.path.insert(0, ".")

import json
import pytest
from ml.backtest import walk_forward_backtest, REPORTS_DIR

def test_walk_forward_backtest():
    res = walk_forward_backtest(n_splits=3)
    
    report_path = os.path.join(REPORTS_DIR, "backtest_results.json")
    assert os.path.exists(report_path)
    
    assert "overall_metrics" in res
    assert "accuracy" in res["overall_metrics"]
    assert len(res["folds"]) == 3
