import unittest

import numpy as np

from legged_lab.perception.surface_audit import audit_surfaces, observed_next_tread_coverage
from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor


class SurfaceAuditTest(unittest.TestCase):
    def surface(self, height=0., y_range=(-0.34, 0.34)):
        x, y = np.meshgrid(np.arange(0.2, 1.5, 0.008), np.arange(*y_range, 0.008))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height)))
        return TreadSurfaceExtractor(SurfaceValidationCfg()).extract(points)

    def test_height_mismatch_is_a_failure_even_if_xy_is_inside(self):
        result = self.surface()
        records, _ = audit_surfaces(result, np.zeros(3), np.eye(3), np.array([[0., 2., 0.15]]))
        self.assertEqual(records[0]["safe_center_height_violations"], records[0]["safe_centers"])

    def test_unmatched_truth_does_not_silently_pass(self):
        result = self.surface()
        records, _ = audit_surfaces(result, np.zeros(3), np.eye(3), np.array([[3., 4., 0.]]))
        self.assertEqual(records[0]["safe_center_truth_violations"], records[0]["safe_centers"])

    def test_upper_platform_is_not_a_next_downward_target(self):
        result = self.surface()
        result.direction = -1
        feet = np.array([[0., 0.136, 0.], [0., -0.136, 0.]])
        coverage = observed_next_tread_coverage(result, np.zeros(3), np.eye(3), feet)
        self.assertEqual(coverage["centers"], 0)

    def test_left_only_candidates_are_not_double_support(self):
        result = self.surface(y_range=(0., 0.34))
        result.direction = -1
        feet = np.array([[0., 0.136, 0.15], [0., -0.136, 0.15]])
        coverage = observed_next_tread_coverage(result, np.zeros(3), np.eye(3), feet)
        self.assertGreater(coverage["left_centers"], 0)
        self.assertEqual(coverage["right_centers"], 0)
        self.assertFalse(coverage["both_feet_same_surface"])

    def test_nearest_upward_plane_is_used_even_if_it_has_no_candidates(self):
        cfg = SurfaceValidationCfg()
        x, y = np.meshgrid(np.arange(0.2, 1.5, 0.008), np.arange(-0.34, 0.34, 0.008))
        heights = np.where(x < 0.4, 0.11, 0.22)
        points = np.column_stack((x.ravel(), y.ravel(), heights.ravel()))
        result = TreadSurfaceExtractor(cfg).extract(points)
        result.direction = 1
        feet = np.array([[0., 0.136, 0.033], [0., -0.136, 0.033]])
        coverage = observed_next_tread_coverage(result, np.zeros(3), np.eye(3), feet)
        self.assertEqual(coverage["centers"], 0)
        self.assertTrue(any(s.safe_center_mask.any() for s in result.surfaces))


if __name__ == "__main__":
    unittest.main()
