"""Real rendered-depth calibration, visibility, memory and clone isolation."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

try:
    import mujoco
    import qpsolvers
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo and qpsolvers are required")
class MujocoDepthGeometryTest(unittest.TestCase):
    def episode(self, direction=1):
        from legged_lab.scripts.mujoco_stair_teacher import TeachingEpisode

        episode = TeachingEpisode(direction=direction, geometry_source="depth")
        self.addCleanup(episode.close)
        episode.data.qpos[0] += .05
        mujoco.mj_forward(episode.model, episode.data)
        return episode

    def test_rendered_up_and_down_find_full_foot_regions(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                episode = self.episode(direction)
                geometry = episode.refresh_geometry()
                source = episode.depth_source
                self.assertIsNone(episode.world_points)
                self.assertEqual(source.last_depth.shape, (720, 1280))
                self.assertEqual(source.last_rgb.shape, (720, 1280, 3))
                self.assertEqual(geometry.direction, direction)
                self.assertGreater(source.last_valid_points, 1000)
                plan = episode.controller.plan(geometry, episode.measurement())
                self.assertIsNotNone(plan)
                np.testing.assert_allclose(plan.targets[:, 2], .11, atol=.002)
                for foot in range(2):
                    local = (plan.targets[foot]-plan.root_position) @ plan.rotation
                    self.assertTrue(plan.geometry.footprint_supported(plan.surface_id, local[:2],
                                                                     plan.geometry.heading_rad))
                self.assertLess(len(source.memory.frames[-1][1]), 35000)

    def test_missing_pixels_do_not_create_support_or_refresh_old_tracks(self):
        episode = self.episode()
        source = episode.depth_source
        geometry = episode.refresh_geometry()
        plan = episode.controller.plan(geometry, episode.measurement())
        original_seen = plan.last_seen.copy()
        blank = np.full_like(source.last_depth, np.nan)
        episode.data.time = .04
        source.ingest(episode.data, blank)
        self.assertEqual(source.last_frame_time, .04)
        tracked = [s for s in source.memory.result.surfaces if s.track_id == plan.track_id]
        self.assertTrue(tracked)
        self.assertTrue(all(s.last_observed_time == original_seen[0] for s in tracked))
        source.memory.reset()
        episode.data.time = .08
        missing = source.ingest(episode.data, blank)
        self.assertEqual(missing.surfaces, [])
        self.assertIsNone(episode.controller.plan(missing, episode.measurement()))

    def test_masked_tread_hole_is_not_filled_by_plane_fitting(self):
        episode = self.episode()
        source = episode.depth_source
        episode.refresh_geometry()
        depth = source.last_depth.copy()
        intrinsic = source.intrinsic
        v, u = np.indices(depth.shape)
        xyz = np.stack(((u-intrinsic[0, 2])*depth/intrinsic[0, 0],
                        (v-intrinsic[1, 2])*depth/intrinsic[1, 1], depth), axis=-1)
        world = xyz @ source.last_camera_rotation.T+source.last_camera_position
        hole = ((np.abs(world[..., 0]-.38) < .10) & (np.abs(world[..., 1]-.136) < .16)
                & (np.abs(world[..., 2]-.11) < .01))
        self.assertTrue(hole.any())
        depth[hole] = np.nan
        source.memory.reset()
        episode.data.time = .04
        geometry = source.ingest(episode.data, depth)
        self.assertIsNone(episode.controller.plan(geometry, episode.measurement()))

    def test_prediction_clone_cannot_render_or_refresh_world_lock(self):
        episode = self.episode()
        source = episode.depth_source
        geometry = episode.refresh_geometry()
        plan = episode.controller.plan(geometry, episode.measurement())
        original_targets = plan.targets.copy()
        clone = copy.deepcopy(source, {id(episode.model): episode.model})
        self.assertTrue(clone.frozen)
        self.assertIsNone(clone.renderer)
        with self.assertRaisesRegex(RuntimeError, "cannot acquire"):
            clone.render(episode.data)
        moved = copy.copy(episode.data)
        moved.qpos[0] += .03
        moved.time = .08
        clone.update(moved)
        self.assertEqual(clone.last_frame_time, 0.)
        self.assertEqual(source.last_frame_time, 0.)
        self.assertEqual(source.frame_count, 1)
        np.testing.assert_array_equal(plan.targets, original_targets)
        moved.time = 1.1
        self.assertEqual(clone.update(moved).surfaces, [])
        self.assertTrue(source.memory.result.surfaces)

    def test_dynamic_initial_lock_and_features_use_real_depth_frames(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode

        for direction in (1, -1):
            with self.subTest(direction=direction):
                episode = DynamicTeachingEpisode(direction=direction, geometry_source="depth")
                self.addCleanup(episode.close)
                self.assertIsNone(episode._context.world_points)
                self.assertGreaterEqual(episode._context.depth_source.frame_count, 3)
                self.assertEqual(episode.target_region["region_source"], "locked_observed_mask_from_rendered_depth")
                original = episode.controller.lock.targets.copy()
                features = episode.actor_features()
                self.assertTrue(np.all((features[36:38] > 0.) & (features[36:38] < 1.)))
                sample = episode.tick()
                self.assertEqual(sample["phase"], "SHIFT_LEAD")
                np.testing.assert_array_equal(episode.controller.lock.targets, original)

    def test_tracking_cache_preserves_observation_age_and_expires_without_extraction(self):
        episode = self.episode()
        source = episode.depth_source
        original = episode.refresh_geometry()
        seen = {s.track_id: s.last_observed_time for s in original.surfaces}
        with patch.object(source.memory.extractor, "extract", side_effect=AssertionError("unexpected extraction")):
            episode.data.time = .02
            cached = episode.refresh_geometry(tracking_only=True)
            self.assertEqual(cached.timestamp_s, original.timestamp_s)
            self.assertEqual({s.track_id: s.last_observed_time for s in cached.surfaces}, seen)
            self.assertEqual(source.frame_count, 1)
            episode.data.time = 1.1
            expired = source.update(episode.data, acquire=False, tracking_only=True)
            self.assertEqual(expired.surfaces, [])
            self.assertEqual(source.last_frame_time, 0.)
        generation = source.memory.generation
        episode.data.qpos[0] += .6
        episode.data.time = 1.11
        jumped = source.update(episode.data, acquire=False, tracking_only=True)
        self.assertEqual(jumped.surfaces, [])
        self.assertEqual(source.memory.generation, generation+1)
        episode.data.time = 1.10
        source.update(episode.data, acquire=False, tracking_only=True)
        self.assertEqual(source.memory.generation, generation+2)


if __name__ == "__main__":
    unittest.main()
