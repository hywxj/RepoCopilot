import math
import unittest

import numpy as np

from legged_lab.perception.surface_memory import SurfaceMemory, SurfaceMemoryCfg
from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor


class SurfaceMemoryTest(unittest.TestCase):
    def setUp(self):
        self.cfg = SurfaceValidationCfg()
        self.extractor = TreadSurfaceExtractor(self.cfg)
        self.memory = SurfaceMemory(self.cfg)
        self.position = np.array([-12., 4., 1.])
        self.rotation = np.eye(3)

    def points(self, height=0., ys=(-0.34, 0.34)):
        x, y = np.meshgrid(np.arange(0.2, 1.5, 0.008), np.arange(*ys, 0.008), indexing="ij")
        return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height)))

    def update(self, points, time, position=None, rotation=None):
        snapshot = self.extractor.extract(points)
        return self.memory.update(points, snapshot, self.position if position is None else position,
                                  self.rotation if rotation is None else rotation, time)

    def test_id_survives_translation_and_yaw(self):
        points = self.points()
        first = self.update(points, 0.)
        identity = first.surfaces[0].track_id
        angle = math.radians(8.)
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        position = self.position + [0.06, 0., 0.]
        transformed = (points+self.position-position) @ rotation
        second = self.update(transformed, 0.04, position, rotation)
        self.assertEqual(second.surfaces[0].track_id, identity)
        world = second.surfaces[0].centroid @ rotation.T + position
        self.assertAlmostEqual(world[2], 1., places=6)

    def test_actual_complementary_observations_can_support_both_feet(self):
        left = self.update(self.points(ys=(-0.34, -0.005)), 0.)
        self.assertFalse(left.footprint_supported(0, [0.8, 0.15]))
        both = self.update(self.points(ys=(0.005, 0.34)), 0.04)
        self.assertTrue(both.footprint_supported(0, [0.8, -0.15]))
        self.assertTrue(both.footprint_supported(0, [0.8, 0.15]))

    def test_unknown_hole_is_not_filled_by_history(self):
        points = self.points()
        hole = (np.abs(points[:, 0]-0.8) < 0.06) & (np.abs(points[:, 1]) < 0.06)
        for time in (0., 0.04, 0.08):
            result = self.update(points[~hole], time)
        self.assertFalse(result.footprint_supported(0, [0.8, 0.]))

    def test_observations_expire_even_when_nothing_new_is_seen(self):
        self.update(self.points(), 0.)
        expired = self.update(np.empty((0, 3)), 1.1)
        self.assertEqual(expired.surfaces, [])
        self.assertEqual(self.memory.tracks, [])

    def test_teleport_clears_previous_support(self):
        self.update(self.points(), 0.)
        result = self.update(np.empty((0, 3)), 0.04, self.position+[1., 0., 0.])
        self.assertEqual(result.surfaces, [])

    def test_repeated_frame_is_not_added_again(self):
        first = self.update(self.points(), 0.)
        repeated = self.update(self.points(), 0.)
        self.assertIs(first, repeated)
        self.assertEqual(len(self.memory.frames), 1)

    def test_near_foot_observation_is_validated_before_forward_roi_clips_it(self):
        x, y = np.meshgrid(np.arange(-0.3, 0.1, 0.008), np.arange(-0.34, 0.34, 0.008))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        snapshot = self.extractor.extract(points)
        self.assertEqual(snapshot.surfaces, [])
        result = self.memory.update(points, snapshot, self.position, self.rotation, 0.)
        self.assertTrue(result.footprint_supported(0, [-0.1, 0.]))

    def test_invalid_sensor_points_do_not_enter_memory(self):
        points = self.points()
        valid = points[:, 1] < 0.
        snapshot = self.extractor.extract(points, valid)
        result = self.memory.update(points, snapshot, self.position, self.rotation, 0., valid)
        self.assertFalse(result.footprint_supported(0, [0.8, 0.15]))

    def test_clock_restart_clears_old_observations_and_changes_generation(self):
        self.update(self.points(), 1.)
        generation = self.memory.generation
        result = self.update(np.empty((0, 3)), 0.)
        self.assertEqual(result.surfaces, [])
        self.assertGreater(self.memory.generation, generation)

    def test_stalled_camera_history_expires_on_current_clock(self):
        self.update(self.points(), 0.)
        expired = self.memory.refresh(self.position, self.rotation, 1.1)
        self.assertEqual(expired.surfaces, [])
        self.assertEqual(self.memory.tracks, [])

    def test_repeated_acquisition_reprojects_world_history_without_adding_points(self):
        first = self.update(self.points(), 0.)
        world = first.surfaces[0].centroid+self.position
        moved = self.memory.update(np.empty((0, 3)), self.extractor.extract(np.empty((0, 3))),
                                   self.position+[0.06, 0., 0.], self.rotation, 0.)
        np.testing.assert_allclose(moved.surfaces[0].centroid+self.position+[0.06, 0., 0.], world, atol=0.004)
        self.assertEqual(len(self.memory.frames), 1)

    def test_changed_height_invalidates_old_support(self):
        self.update(self.points(), 0.)
        changed = self.update(self.points(height=0.15), 0.04)
        self.assertTrue(changed.surfaces)
        self.assertTrue(all(abs(s.centroid[2]-0.15) < 0.005 for s in changed.surfaces))

    def test_batched_height_invalidation_matches_cell_lookup(self):
        rng = np.random.default_rng(42)
        size = self.cfg.grid_size
        cells = rng.integers(-150, 150, size=(1000, 2))
        fresh = np.column_stack(((cells+0.25)*size, rng.choice([0., 0.11], 1000)))
        old = np.concatenate((fresh.copy(), fresh.copy(),
                              np.column_stack((rng.uniform(-6., 6., (500, 2)), np.zeros(500)))))
        old[1000:2000, 2] += 0.10
        self.memory.frames.append((0.04, old))
        self.memory.frames.append((0.08, np.empty((0, 3))))
        unique, inverse = np.unique(np.floor(fresh[:, :2]/size).astype(np.int64),
                                    axis=0, return_inverse=True)
        means = np.bincount(inverse, weights=fresh[:, 2])/np.bincount(inverse)
        heights = {tuple(cell): height for cell, height in zip(unique, means)}
        expected = np.array([heights.get(tuple(key), np.nan)
                             for key in np.floor(old[:, :2]/size).astype(np.int64)])
        keep = ~np.isfinite(expected) | (np.abs(old[:, 2]-expected) <= self.memory.cfg.height_tolerance_m)
        self.memory._invalidate_changed_heights(fresh)
        self.assertEqual(self.memory.frames[0][0], 0.04)
        np.testing.assert_array_equal(self.memory.frames[0][1], old[keep])
        self.assertEqual(self.memory.frames[1][1].shape, (0, 3))

    def test_empty_height_update_leaves_observed_history_unchanged(self):
        old = self.points()
        self.memory.frames.append((0., old.copy()))
        self.memory._invalidate_changed_heights(np.empty((0, 3)))
        np.testing.assert_array_equal(self.memory.frames[0][1], old)

    def test_reset_and_invalid_time(self):
        self.update(self.points(), 0.)
        self.memory.reset()
        self.assertFalse(self.memory.frames)
        with self.assertRaises(ValueError):
            self.update(self.points(), math.nan)


if __name__ == "__main__":
    unittest.main()
