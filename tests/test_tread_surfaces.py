import math
import unittest

import numpy as np

from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor, depth_horizontal_mask


class TreadSurfacesTest(unittest.TestCase):
    def points(self, height, x_range=(0.16, 1.98), y_range=(-0.34, 0.34)):
        x, y = np.meshgrid(np.arange(*x_range, 0.008), np.arange(*y_range, 0.008), indexing="ij")
        z = np.broadcast_to(height(x, y), x.shape)
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    def extractor(self):
        return TreadSurfaceExtractor(SurfaceValidationCfg())

    def test_left_only_surface_cannot_support_right_foot(self):
        full = self.extractor().extract(self.points(lambda x, y: 0.0))
        left = self.extractor().extract(self.points(lambda x, y: 0.0, y_range=(-0.34, -0.01)))
        self.assertTrue(full.footprint_supported(0, [0.8, 0.15]))
        self.assertFalse(left.footprint_supported(0, [0.8, 0.15]))
        self.assertTrue(left.footprint_supported(0, [0.8, -0.18]))

    def test_eleven_degree_surface_is_rejected(self):
        result = self.extractor().extract(self.points(lambda x, y: 0.2 * x))
        self.assertGreater(len(result.surfaces), 0)
        self.assertFalse(any(surface.valid for surface in result.surfaces))
        self.assertTrue(all(surface.rejection == "tilt" for surface in result.surfaces))
        self.assertFalse((result.grid_labels >= 0).any())

    def test_vertical_riser_does_not_become_support_plane(self):
        y, z = np.meshgrid(np.arange(-0.34, 0.34, 0.005), np.arange(-0.5, 0.5, 0.005))
        points = np.column_stack((np.full(y.size, 0.5), y.ravel(), z.ravel()))
        result = self.extractor().extract(points)
        self.assertFalse(any(surface.valid for surface in result.surfaces))

    def test_terminal_platform_is_detected_for_both_directions(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                def heights(x, y):
                    levels = np.clip(np.floor((x - 0.4) / 0.32) + 1, 0, 3)
                    return direction * 0.15 * levels
                result = self.extractor().extract(self.points(heights))
                terminal = [s for s in result.surfaces if s.valid and
                            abs(s.centroid[2] - direction * 0.45) < 0.005]
                self.assertTrue(terminal)
                self.assertEqual(terminal[0].kind, "platform_candidate")
                self.assertEqual(result.direction, direction)

    def test_unknown_hole_inside_bounds_is_not_filled(self):
        points = self.points(lambda x, y: 0.0)
        hole = (np.abs(points[:, 0] - 0.8) < 0.05) & (np.abs(points[:, 1]) < 0.05)
        result = self.extractor().extract(points[~hole])
        self.assertEqual(len(result.surfaces), 1)
        self.assertFalse(result.footprint_supported(0, [0.8, 0.0]))
        self.assertTrue(result.footprint_supported(0, [1.5, 0.0]))

    def test_32cm_tread_has_no_24cm_sole_with_two_5cm_margins(self):
        extractor = TreadSurfaceExtractor(SurfaceValidationCfg(edge_margin=0.05))
        result = extractor.extract(self.points(lambda x, y: 0.0, x_range=(0.2, 0.52)))
        self.assertTrue(result.surfaces[0].valid)
        self.assertFalse(result.surfaces[0].safe_center_mask.any())
        self.assertTrue(result.footprint_supported(0, [0.36, 0], include_margin=False))
        self.assertFalse(result.footprint_supported(0, [0.36, 0], include_margin=True))

    def test_critical_margin_changes_side_with_stair_direction(self):
        result = self.extractor().extract(self.points(lambda x, y: 0.0))
        result.direction = 1
        np.testing.assert_allclose(result.support_margins(), [0.03, 0.02, 0.02])
        result.direction = -1
        np.testing.assert_allclose(result.support_margins(), [0.02, 0.03, 0.02])

    def test_32cm_tread_can_offer_2cm_clearance_without_overhang(self):
        result = self.extractor().extract(self.points(lambda x, y: 0.0, x_range=(0.2, 0.52)))
        self.assertTrue(result.surfaces[0].safe_center_mask.any())
        self.assertTrue(result.footprint_supported(0, [0.36, 0], include_margin=True))

    def test_heading_follows_stair_edges_not_robot_x(self):
        angle = math.radians(-8)
        def heights(x, y):
            forward = math.cos(angle) * x + math.sin(angle) * y
            return 0.15 * np.clip(np.floor((forward - 0.4) / 0.32) + 1, 0, 5)
        result = self.extractor().extract(self.points(heights))
        self.assertIsNotNone(result.heading_rad)
        self.assertAlmostEqual(math.degrees(result.heading_rad), -8, delta=2.5)

    def test_all_invalid_depth_returns_empty_geometry(self):
        points = np.full((20, 3), np.nan)
        result = self.extractor().extract(points)
        self.assertEqual(result.surfaces, [])
        self.assertFalse(result.grid_observed.any())

    def test_invalid_local_normal_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            depth_horizontal_mask(np.ones((20, 20)), np.eye(3), np.eye(3), window_size=2)
        with self.assertRaises(ValueError):
            depth_horizontal_mask(np.ones((20, 20)), np.eye(3), np.eye(3), radius=0)

    def test_local_normals_separate_riser_from_floor(self):
        intrinsic = np.array([[100., 0., 80.], [0., 100., 40.], [0., 0., 1.]])
        rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        v = np.arange(160)[:, None]
        depth = np.broadcast_to(60./np.maximum(v-40, 1), (160, 160)).astype(np.float32).copy()
        depth[:100] = 1.
        mask = depth_horizontal_mask(depth, intrinsic, rotation, max_tilt_deg=45.)
        self.assertFalse(mask[30:80, 30:130].any())
        self.assertTrue(mask[120:140, 30:130].all())
        depth[125, 80] = 0.
        mask = depth_horizontal_mask(depth, intrinsic, rotation, max_tilt_deg=45.)
        self.assertFalse(mask[125, 80])

    def test_candidate_offsets_agree_with_full_foot_checks(self):
        def heights(x, y):
            return 0.15*np.floor((x-0.4)/0.32)
        for angle in (0., 0.2, 8., -8.):
            cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            points = self.points(lambda x, y: 0.0)
            points[:, 2] = heights(cosine*points[:, 0]+sine*points[:, 1], 0)
            result = self.extractor().extract(points)
            for surface in result.surfaces:
                centers = result.candidate_centers(surface.surface_id)
                for center in centers[::max(1, len(centers)//12)]:
                    self.assertTrue(result.footprint_supported(surface.surface_id, center, result.heading_rad or 0))

    def test_subcell_search_can_fit_narrow_observed_strip(self):
        extractor = TreadSurfaceExtractor(SurfaceValidationCfg(target_tracking_margin=0.))
        result = extractor.extract(self.points(lambda x, y: 0., x_range=(0.39, 0.69)))
        result.direction = 1
        surface = result.surfaces[0]
        surface.safe_center_mask, surface.safe_center_offsets = extractor._safe_centers(
            surface, math.radians(0.2), result.support_margins())
        centers = result.candidate_centers(0)
        self.assertGreater(len(centers), 0)
        self.assertTrue(any(np.linalg.norm(offset) > 0 for offset in surface.safe_center_offsets[surface.safe_center_mask]))
        self.assertTrue(all(result.footprint_supported(0, center, math.radians(0.2)) for center in centers))

    def test_planned_centers_allow_small_tracking_error_without_relaxing_acceptance(self):
        cfg = SurfaceValidationCfg(min_forward=-.35, max_forward=1.1, lateral_half_width=.6)
        extractor = TreadSurfaceExtractor(cfg)
        x, y = np.meshgrid(np.arange(-.32, .94, .006), np.arange(-.38, .38, .006))
        z = np.where(x < .22, 0., np.where(x < .54, .11, .22))
        result = extractor.extract(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
        tread = next(s for s in result.surfaces if abs(s.centroid[2]-.11) < .01)
        centers = result.candidate_centers(tread.surface_id)
        self.assertGreater(len(centers), 0)
        center = centers[np.argmin(np.linalg.norm(centers-[.38, -.13], axis=1))]
        for error in (-.002, .002):
            for yaw in (-.005, .005):
                self.assertTrue(result.footprint_supported(tread.surface_id, center+[error, 0.], yaw))
        self.assertFalse(result.footprint_supported(tread.surface_id, [.525, -.13]))


if __name__ == "__main__":
    unittest.main()
