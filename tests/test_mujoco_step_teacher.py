from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import mujoco
    import qpsolvers
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo and qpsolvers are required")
class MujocoStepTeacherTest(unittest.TestCase):
    def test_geometry_has_32cm_treads_and_down_floor_is_not_a_1cm_obstacle(self):
        from legged_lab.scripts.mujoco_stair_teacher import build_model
        for direction in (1, -1):
            model, data = build_model(direction=direction)
            for level in range(2):
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tread_{level}")
                self.assertAlmostEqual(2*model.geom_size[geom, 0], .32)
                top = (level+1)*.11 if direction > 0 else (1-level)*.11
                self.assertAlmostEqual(model.geom_pos[geom, 2]+model.geom_size[geom, 2], top)
            self.assertFalse((model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH).any())
            self.assertEqual(model.nv, model.nu+6)

    def test_standing_uses_joint_motors_with_real_external_support(self):
        from legged_lab.scripts.mujoco_stair_teacher import build_model
        from legged_lab.perception.mujoco_step_teacher import MujocoStepTeacher
        model, data = build_model()
        teacher = MujocoStepTeacher(model, data)
        measured = teacher.reader.measurement(data)
        root, soles = measured.root_position.copy(), measured.sole_positions.copy()
        for _ in range(80):
            data.ctrl[:] = teacher.control(root, soles, [True, True], model.opt.timestep, [.5, .5])
            self.assertTrue((data.ctrl >= model.actuator_ctrlrange[:, 0]-1.e-8).all())
            self.assertTrue((data.ctrl <= model.actuator_ctrlrange[:, 1]+1.e-8).all())
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        load = teacher.reader.contact_forces_world(data)[:, 2]/teacher.reader.body_weight
        np.testing.assert_allclose(load, [.5, .5], atol=.02)
        np.testing.assert_allclose(data.qpos[:3], root, atol=.002)
        np.testing.assert_array_equal(data.qfrc_applied, 0.)
        np.testing.assert_array_equal(data.xfrc_applied, 0.)
        self.assertLess(teacher.max_dynamics_residual, 1.e-7)
        # Independent x/y friction boxes wrongly accept this diagonal demand.
        matrix = teacher.last_problem[2][2*model.nu:2*model.nu+4]
        demand = np.zeros(teacher.last_solution.shape)
        demand[model.nv:model.nv+3] = [.6, .6, 1.]
        self.assertGreater((matrix @ demand).max(), .1)
        for wrench in teacher.last_solution[model.nv:].reshape(-1, 6):
            self.assertLessEqual(abs(wrench[0])+abs(wrench[1]), .7*wrench[2]+1.e-8)
            self.assertLessEqual(abs(wrench[3]), .018*wrench[2]+1.e-8)

    def test_failed_episodes_cannot_be_saved_as_expert_data(self):
        from legged_lab.scripts.mujoco_stair_teacher import TeachingEpisode
        episode = TeachingEpisode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"invalid.npz"
            with self.assertRaisesRegex(ValueError, "physically successful"):
                episode.save_trace(path)
            self.assertFalse(path.exists())

    def test_physical_up_and_down_complete_with_aligned_expert_data(self):
        from legged_lab.scripts.mujoco_stair_teacher import TeachingEpisode
        from legged_lab.perception.stair_step_controller import StepPhase
        for direction in (1, -1):
            with self.subTest(direction=direction):
                episode = TeachingEpisode(direction=direction)
                while episode.data.time < 15.:
                    sample = episode.tick()
                    if sample["success"]:
                        break
                self.assertEqual(episode.controller.phase, StepPhase.COMPLETE)
                self.assertTrue(episode.controller.support_valid.all())
                self.assertTrue((np.array(sample["load_fraction"]) >= .2).all())
                self.assertGreater(sample["root"][0], .30)
                np.testing.assert_allclose(np.array(sample["feet"])[:, 2], .11, atol=.002)
                np.testing.assert_array_equal(episode.data.qfrc_applied, 0.)
                np.testing.assert_array_equal(episode.data.xfrc_applied, 0.)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"expert.npz"
                    episode.save_trace(path)
                    with np.load(path, allow_pickle=False) as trace:
                        self.assertEqual(trace["motor_torques"].shape, (len(episode.samples), 4, 29))
                        np.testing.assert_allclose(np.diff(trace["state_time"]), .01, atol=1.e-8)
                        np.testing.assert_allclose(trace["step_features"][:, 24:26], 0.)
                        np.testing.assert_allclose(trace["step_features"][:, 34:36], 0.)
                        self.assertTrue(bool(trace["force_oracle"]))
                        self.assertFalse(bool(trace["learned_policy"]))
                        self.assertTrue(np.isfinite(trace["motor_torques"]).all())


if __name__ == "__main__":
    unittest.main()
