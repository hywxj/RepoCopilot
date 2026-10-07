"""Physical regressions for the explicitly scoped single-step candidates."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    import qpsolvers
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo and qpsolvers are required")
class DynamicStairMotionTest(unittest.TestCase):
    def test_up_and_down_finish_with_motor_limits_and_regional_support(self):
        from scipy.spatial.transform import Rotation
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        from legged_lab.perception.stair_step_controller import StepPhase

        for direction in (1, -1):
            with self.subTest(direction=direction):
                episode = DynamicTeachingEpisode(direction=direction)
                model, data = episode.model, episode.data
                initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
                initial_time = data.time
                teacher = episode.teacher
                hips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_z_link")
                        for s in ("l", "r")]
                ankles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_x_link")
                          for s in ("l", "r")]
                loaded_angles, pelvis_roll = [], []
                minimum_margin = np.inf
                # Preserve the measured-state time relation used by exported traces.
                for _ in range(550):
                    mujoco.mj_forward(model, data)
                    measured = episode.measurement()
                    load = measured.contact_forces[:, 2]/measured.body_weight
                    vectors = data.xpos[hips]-data.xpos[ankles]
                    angles = np.abs(np.arctan2(vectors[:, 1], vectors[:, 2]))
                    loaded_angles.extend(angles[load >= .60])
                    pelvis_roll.append(abs(Rotation.from_matrix(
                        data.xmat[teacher.pelvis_id].reshape(3, 3)).as_euler("xyz")[0]))
                    q = data.qpos[teacher.qpos]
                    limits = model.jnt_range[teacher.joints]
                    minimum_margin = min(minimum_margin, np.minimum(q-limits[:, 0], limits[:, 1]-q).min())
                    sample = episode.tick()
                    np.testing.assert_array_equal(data.qfrc_applied, 0.)
                    np.testing.assert_array_equal(data.xfrc_applied, 0.)
                    if sample["success"]:
                        break

                self.assertTrue(sample["success"], sample)
                self.assertEqual(sample["phase"], "COMPLETE")
                self.assertTrue(episode.confirmed.all())
                self.assertGreaterEqual(sample["stable_for_s"], .25)
                self.assertLess(np.linalg.norm(sample["root_velocity"]), .01)
                self.assertLess(np.linalg.norm(sample["root_angular_velocity"]), .01)
                self.assertTrue((np.asarray(sample["load_fraction"]) > .2).all())
                np.testing.assert_allclose(np.asarray(sample["feet"])[:, 2], .11, atol=.002)
                # Empirical regression ceilings, not universal gait-quality limits.
                self.assertLess(max(loaded_angles), np.deg2rad(12.5))
                self.assertLess(max(pelvis_roll), np.deg2rad(2.))
                self.assertGreater(minimum_margin, 0.)
                torques = np.stack([row[4] for row in episode.trace])
                self.assertTrue((torques >= model.actuator_ctrlrange[:, 0]-1.e-7).all())
                self.assertTrue((torques <= model.actuator_ctrlrange[:, 1]+1.e-7).all())
                for foot in range(2):
                    self.assertTrue(episode.controller._physical_support(
                        episode.measurement(), foot, min_load=.2, record_plant=False))

                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"candidate.npz"
                    episode.save_trace(path)
                    with np.load(path, allow_pickle=False) as trace:
                        self.assertEqual(trace["motor_torques"].shape, (len(episode.samples), 4, model.nu))
                        self.assertEqual(trace["step_features"].shape, (len(episode.samples), len(StepPhase)))
                        np.testing.assert_allclose(np.diff(trace["state_time"]), .01, atol=1.e-8)
                        np.testing.assert_allclose(trace["state_time"], [s["time"] for s in episode.samples])
                        np.testing.assert_allclose(trace["qpos"][0], initial_qpos)
                        self.assertEqual(str(trace["step_features_semantics"]),
                                         "phase_onehot_only_not_policy_observation")
                        self.assertFalse(bool(trace["motion_quality_validated"]))
                        self.assertFalse(bool(trace["learned_policy"]))
                        self.assertTrue(bool(trace["force_oracle"]))

                # A GUI loop must retain its attached MuJoCo objects and reset
                # every controller reader/reference, not only the displayed pose.
                reset = episode.reset()
                self.assertIs(reset.model, model)
                self.assertIs(reset.data, data)
                self.assertIs(reset._context.model, model)
                self.assertIs(reset._context.data, data)
                self.assertIs(reset.teacher.data, data)
                np.testing.assert_allclose(data.qpos, initial_qpos, atol=1.e-12)
                np.testing.assert_allclose(data.qvel, initial_qvel, atol=1.e-12)
                self.assertAlmostEqual(data.time, initial_time)
                self.assertFalse(reset.confirmed.any())
                self.assertEqual(reset.tick()["phase"], "SHIFT_LEAD")

    def test_incomplete_motion_cannot_be_exported(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        episode = DynamicTeachingEpisode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"candidate.npz"
            with self.assertRaisesRegex(ValueError, "physically successful"):
                episode.save_trace(path)
            self.assertFalse(path.exists())

    def test_initial_conditions_are_physical_reproducible_and_lock_is_bounded(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        configuration = dict(initial_forward_offset=.054, initial_lateral_offset=.002,
                             initial_yaw=.002, initial_joint_velocity_noise=.005, seed=7)
        first = DynamicTeachingEpisode(**configuration)
        second = DynamicTeachingEpisode(**configuration)
        np.testing.assert_allclose(first.initial_qpos, second.initial_qpos)
        np.testing.assert_allclose(first.initial_qvel, second.initial_qvel)
        self.assertAlmostEqual(first.initial_qpos[0], .054)
        self.assertAlmostEqual(first.initial_qpos[1], .002)
        self.assertAlmostEqual(first.initial_qpos[6], np.sin(.001))
        self.assertGreater(np.linalg.norm(first.initial_qvel[6:]), 0.)
        # The obstacle stays at its real location despite initial robot offsets.
        self.assertAlmostEqual(first.model.geom_pos[
            mujoco.mj_name2id(first.model, mujoco.mjtObj.mjOBJ_GEOM, "tread_0"), 0], .38)
        np.testing.assert_allclose(first.data.qpos, second.data.qpos)
        with self.assertRaisesRegex(RuntimeError, "No valid tread lock"):
            DynamicTeachingEpisode(lock_timeout_s=.01)
        with self.assertRaisesRegex(ValueError, "Initial conditions"):
            DynamicTeachingEpisode(initial_yaw=np.nan)

    def test_descent_forward_offsets_keep_original_contact_acceptance(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        for offset in (.045, .055):
            with self.subTest(offset=offset):
                episode = DynamicTeachingEpisode(direction=-1, initial_forward_offset=offset)
                for _ in range(550):
                    sample = episode.tick()
                    if sample["success"]:
                        break
                self.assertTrue(sample["success"])
                self.assertGreaterEqual(sample["stable_for_s"], .25)
                for foot in range(2):
                    self.assertTrue(episode.controller._physical_support(
                        episode.measurement(), foot, min_load=.2, record_plant=False))

    def test_failed_preview_releases_context_before_constructor_returns(self):
        from unittest.mock import patch
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        with patch("legged_lab.perception.mujoco_stair_motion.plan_com_preview",
                   side_effect=RuntimeError("preview failure")), patch(
                       "legged_lab.scripts.mujoco_stair_teacher.TeachingEpisode.close") as close:
            with self.assertRaisesRegex(RuntimeError, "preview failure"):
                DynamicTeachingEpisode()
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
