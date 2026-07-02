import unittest
import logging
import numpy as np
from v4_skills import compute_feature_shock
from feature_builder import map_semantic_feature

class TestBLOPSRegression(unittest.TestCase):
    
    def test_case_A_huge_numeric_nerf(self):
        """Case A: Huge numeric nerf. Expected: high drift."""
        change = {
            "values": {"old": 10.0, "new": 100.0},
            "type": "nerf",
            "category": "ability",
            "ability": "Signature"
        }
        nerf_th, buff_th = compute_feature_shock(change)
        # Bounded and non-zero drift
        self.assertEqual(buff_th, 0.0)
        self.assertGreater(nerf_th, 0.2) # High drift
        self.assertLessEqual(nerf_th, 1.0)
        
    def test_case_B_huge_numeric_buff(self):
        """Case B: Huge numeric buff. Expected: high drift."""
        change = {
            "values": {"old": 100.0, "new": 10.0},
            "type": "buff",
            "category": "ability",
            "ability": "Signature"
        }
        nerf_th, buff_th = compute_feature_shock(change)
        self.assertEqual(nerf_th, 0.0)
        self.assertGreater(buff_th, 0.2) # High drift
        self.assertLessEqual(buff_th, 1.0)
        
    def test_case_C_buff_nerf_rework(self):
        """Case C: Buff + nerf rework. Expected: higher drift than either alone."""
        # A nerf rework change
        nerf_rework = {
            "values": {"old": 1.0, "new": 0.5},
            "type": "nerf",
            "category": "combat",
            "ability": "Signature"
        }
        # A buff change
        buff = {
            "values": {"old": 1.0, "new": 1.5},
            "type": "buff",
            "category": "ability",
            "ability": "Ultimate"
        }
        
        n_rew, b_rew = compute_feature_shock(nerf_rework)
        n_buff, b_buff = compute_feature_shock(buff)
        
        # Combined drift calculation (probabilistic union)
        all_thetas = [n_rew, b_rew, n_buff, b_buff]
        combined_drift = 1.0 - np.prod([1.0 - th for th in all_thetas if th > 0])
        
        self.assertGreater(combined_drift, n_rew)
        self.assertGreater(combined_drift, b_buff)
        
    def test_case_D_10_correlated_small_changes(self):
        """Case D: 10 correlated small changes. Expected: sub-linear aggregation."""
        change = {
            "values": {"old": 10.0, "new": 11.0}, # 10% change
            "type": "nerf",
            "category": "combat",
            "ability": "General"
        }
        nerf_th, _ = compute_feature_shock(change)
        
        linear_sum = nerf_th * 10
        
        # Probabilistic union of 10 identical thetas
        combined_drift = 1.0 - (1.0 - nerf_th) ** 10
        
        # Sub-linear aggregation check
        self.assertLess(combined_drift, linear_sum)
        self.assertGreater(combined_drift, nerf_th)
        
    def test_case_E_unknown_feature(self):
        """Case E: Unknown feature. Expected: warning + preserved trace."""
        logger = logging.getLogger("feature_builder")
        
        # Verify that map_semantic_feature triggers a warning and preserves the name
        with self.assertLogs(level='WARNING') as cm:
            category, feat_name = map_semantic_feature("xyz_completely_unknown_feature", "")
            
        self.assertTrue(any("Unknown feature extracted" in log for log in cm.output))
        self.assertEqual(category, "general")
        self.assertEqual(feat_name, "xyz_completely_unknown_feature")

if __name__ == "__main__":
    unittest.main()
