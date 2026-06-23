import unittest
from feature_builder import map_semantic_feature
from v4_skills import compute_feature_shock
import numpy as np

class TestAdversarialPatchAnalyzer(unittest.TestCase):

    def test_1_ultimate_cost_vs_ability_cost(self):
        cat1, feat1 = map_semantic_feature("ultimate cost", "increased from 7 to 8")
        cat2, feat2 = map_semantic_feature("ability cost", "increased cost to 200")
        self.assertEqual((cat1, feat1), ("economy", "ultimate_cost"))
        self.assertEqual((cat2, feat2), ("economy", "cost"))
        self.assertNotEqual(feat1, feat2)

    def test_2_cast_speed_vs_movement_speed(self):
        cat1, feat1 = map_semantic_feature("cast speed", "reduced windup time")
        cat2, feat2 = map_semantic_feature("movement speed", "reduced run velocity")
        self.assertEqual((cat1, feat1), ("ability", "cast_time"))
        self.assertEqual((cat2, feat2), ("movement", "movement_speed"))
        self.assertNotEqual((cat1, feat1), (cat2, feat2))

    def test_3_weapon_reload_vs_movement(self):
        cat1, feat1 = map_semantic_feature("weapon reload speed reduced", "")
        self.assertEqual((cat1, feat1), ("combat", "reload"))
        self.assertNotEqual(feat1, "movement_speed")

    def test_4_text_only_rework_non_zero(self):
        change = {
            "values": None,
            "type": "rework",
            "category": "ability",
            "ability": "Signature"
        }
        nerf, buff = compute_feature_shock(change)
        # It's a text-only nerf default
        self.assertGreater(nerf, 0.0)
        self.assertGreater(buff, 0.0)

    def test_5_huge_buff_does_not_cancel_nerf(self):
        buff_change = {"values": {"old": 10, "new": 100}, "type": "buff", "category": "ability"}
        nerf_change = {"values": {"old": 10, "new": 1}, "type": "nerf", "category": "combat"}
        
        nerf1, buff1 = compute_feature_shock(buff_change)
        nerf2, buff2 = compute_feature_shock(nerf_change)
        
        # Buffs and Nerfs should now both be collected and NOT subtracted in the analyzer loop
        # We test that the independent thetas are both positive.
        self.assertEqual(nerf1, 0.0)
        self.assertGreater(buff1, 0.0)
        
        self.assertGreater(nerf2, 0.0)
        self.assertEqual(buff2, 0.0)
        
        # In patch_analyzer.py, all thetas are combined: 1 - prod(1 - th)
        # So total drift > nerf2 and total drift > buff1. They do NOT cancel.
        drift = 1 - ((1 - min(buff1, 0.999)) * (1 - min(nerf2, 0.999)))
        self.assertGreater(drift, nerf2)

    def test_6_probabilistic_union_not_3x(self):
        change1 = {"values": {"old": 10, "new": 5}, "type": "nerf", "category": "combat"}
        change2 = {"values": {"old": 10, "new": 5}, "type": "nerf", "category": "combat"}
        change3 = {"values": {"old": 10, "new": 5}, "type": "nerf", "category": "combat"}
        
        th1, _ = compute_feature_shock(change1)
        th2, _ = compute_feature_shock(change2)
        th3, _ = compute_feature_shock(change3)
        
        linear_sum = th1 + th2 + th3
        prob_union = 1 - ((1 - th1) * (1 - th2) * (1 - th3))
        
        self.assertLess(prob_union, linear_sum)
        self.assertGreater(prob_union, th1)

    def test_7_unknown_feature_preserved(self):
        cat, feat = map_semantic_feature("some invisible ghost mechanic")
        self.assertEqual(cat, "general")
        self.assertEqual(feat, "some invisible ghost mechanic")

    def test_8_projectile_buff_no_false_nerf(self):
        # Breach projectile speed buff
        change = {
            "values": {"old": 2000.0, "new": 2400.0},
            "type": "buff",
            "category": "projectile",
            "ability": "Flashpoint"
        }
        nerf, buff = compute_feature_shock(change)
        self.assertEqual(nerf, 0.0)
        self.assertGreater(buff, 0.0)

    def test_9_neon_slide_count_detected(self):
        cat, feat = map_semantic_feature("slideCount")
        self.assertEqual(cat, "movement")
        self.assertEqual(feat, "slide_count")

    def test_10_clove_duration_is_ability(self):
        cat, feat = map_semantic_feature("Move speed duration")
        self.assertEqual(cat, "ability")
        self.assertEqual(feat, "duration")

if __name__ == "__main__":
    unittest.main()
