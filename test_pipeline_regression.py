import unittest
import numpy as np
from feature_builder import map_semantic_feature
from v4_skills import compute_feature_impact
from patch_analyzer import generate_patch_distances

class TestPatchPipelineRegression(unittest.TestCase):

    def test_breach_projectile_speed_not_movement(self):
        category, feature = map_semantic_feature("Projectile speed")
        self.assertEqual(category, "projectile")
        self.assertEqual(feature, "velocity")
        self.assertNotEqual(feature, "movement_speed")

    def test_clove_duration_not_movement(self):
        category, feature = map_semantic_feature("Move speed duration")
        self.assertEqual(category, "ability")
        self.assertEqual(feature, "duration")
        self.assertNotEqual(feature, "movement_speed")

    def test_neon_slide_count_extraction(self):
        category, feature = map_semantic_feature("slideCount")
        self.assertEqual(category, "movement")
        self.assertEqual(feature, "slide_count")

    def test_buff_not_nerf(self):
        change = {
            "values": {"old": 2000.0, "new": 2400.0},
            "weight": 0.5,
            "type": "buff"
        }
        nerf, buff = compute_feature_impact(change, gamma=1.0)
        self.assertEqual(nerf, 0.0)
        self.assertGreater(buff, 0.0)

    def test_nerf_impact(self):
        change = {
            "values": {"old": 2.0, "new": 1.0},
            "weight": 0.8,
            "type": "nerf"
        }
        nerf, buff = compute_feature_impact(change, gamma=1.0)
        self.assertGreater(nerf, 0.0)
        self.assertEqual(buff, 0.0)
        
    def test_registry_generation_completes(self):
        try:
            registry_path = generate_patch_distances()
            self.assertIsNotNone(registry_path)
        except Exception as e:
            self.fail(f"generate_patch_distances raised exception: {e}")

if __name__ == "__main__":
    unittest.main()
