"""Named action mapping, isolated prediction, and physical held-action checks."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

try:
    import mujoco
    import qpsolvers
except ImportError:
    mujoco = None


CONFIG = Path(__file__).resolve().parents[1]/"legged_lab/configs/elf3_stair_position.yaml"


@unittest.skipIf(mujoco is None, "MuJoCo and qpsolvers are required")
class PositionBridgeTest(unittest.TestCase):
    def test_named_inverse_pd_and_motor_saturation(self):
        from legged_lab.scripts.mujoco_stair_teacher import build_model
        from legged_lab.perception.mujoco_position_bridge import MujocoPositionInterface

        model, data = build_model(.11, .32, 1)
        interface = MujocoPositionInterface(model, CONFIG)
        self.assertEqual(interface.physics_steps, 8)
        self.assertAlmostEqual(interface.dt, .02)
        self.assertEqual(interface.joint_names[0], "l_shoulder_y_joint")
        self.assertEqual(interface.joint_names[-1], "r_ankle_x_joint")
        self.assertAlmostEqual(interface.kp[0], 48.)
        self.assertAlmostEqual(interface.kp[-1], 30.)
        self.assertAlmostEqual(interface.default[0], .2)
        self.assertAlmostEqual(interface.default[23], .44)
        data.qvel[interface.dofs] = np.linspace(-.2, .2, model.nu)
        torque = np.linspace(-2., 2., model.nu)
        before = data.qpos.copy(), data.qvel.copy(), float(data.time)
        action = interface.action_from_torque(data, torque)
        np.testing.assert_allclose(interface.torque_from_action(data, action), torque, atol=1.e-12)
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        self.assertEqual(data.time, before[2])
        saturated = interface.torque_from_action(data, np.full(model.nu, 100.))
        np.testing.assert_allclose(saturated, model.actuator_ctrlrange[:, 1])
        with self.assertRaises(ValueError):
            interface.target_from_action(np.full(model.nu, np.nan))

    def test_prediction_isolated_from_real_state_and_supervisor(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        from legged_lab.perception.mujoco_position_bridge import MujocoPositionInterface, DynamicPositionBridge

        episode = DynamicTeachingEpisode()
        interface = MujocoPositionInterface(episode.model, CONFIG)
        bridge = DynamicPositionBridge(episode, interface)
        original = {
            "qpos": episode.data.qpos.copy(), "qvel": episode.data.qvel.copy(),
            "ctrl": episode.data.ctrl.copy(), "time": float(episode.data.time),
            "plants": episode.plants.copy(), "touching": episode.touching.copy(),
            "confirmed": episode.confirmed.copy(), "dwell": episode.support_dwell.copy(),
            "torso": episode.teacher.torso_rpy_reference.copy(),
        }
        action = bridge.action()
        for field in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(episode.data, field), original[field])
        self.assertEqual(episode.data.time, original["time"])
        for field in ("plants", "touching", "confirmed"):
            np.testing.assert_array_equal(getattr(episode, field), original[field])
        np.testing.assert_array_equal(episode.support_dwell, original["dwell"])
        np.testing.assert_array_equal(episode.teacher.torso_rpy_reference, original["torso"])
        self.assertEqual(len(episode.samples), 0)
        self.assertEqual(len(episode.trace), 0)
        held = action.copy()
        with patch.object(episode.teacher, "control", side_effect=AssertionError("Live teacher called")):
            for _ in range(interface.physics_steps):
                episode._advance(lambda ep, sample: interface.torque_from_action(ep.data, held))
        self.assertAlmostEqual(episode.data.time-original["time"], .02)
        np.testing.assert_array_equal(action, held)

    def test_up_and_down_have_regional_support_with_fixed_20ms_actions(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        from legged_lab.perception.mujoco_position_bridge import MujocoPositionInterface, DynamicPositionBridge

        for direction in (1, -1):
            with self.subTest(direction=direction):
                episode = DynamicTeachingEpisode(direction=direction)
                interface = MujocoPositionInterface(episode.model, CONFIG)
                bridge = DynamicPositionBridge(episode, interface)
                for _ in range(260):
                    action = bridge.action()
                    target = interface.target_from_action(action).copy()
                    started = float(episode.data.time)
                    with patch.object(episode.teacher, "control", side_effect=AssertionError("Live teacher called")):
                        for _ in range(interface.physics_steps):
                            sample, torque = episode._advance(
                                lambda ep, sample: interface.torque_from_action(ep.data, action))
                            np.testing.assert_array_equal(interface.target_from_action(action), target)
                            self.assertTrue((torque >= episode.model.actuator_ctrlrange[:, 0]).all())
                            self.assertTrue((torque <= episode.model.actuator_ctrlrange[:, 1]).all())
                            np.testing.assert_array_equal(episode.data.qfrc_applied, 0.)
                            np.testing.assert_array_equal(episode.data.xfrc_applied, 0.)
                    self.assertAlmostEqual(episode.data.time-started, .02)
                    if sample["success"]:
                        break
                self.assertTrue(sample["success"], sample)
                self.assertTrue(episode.confirmed.all())
                self.assertGreaterEqual(sample["stable_for_s"], .25)
                mujoco.mj_forward(episode.model, episode.data)
                measurement = episode.measurement()
                for foot in range(2):
                    self.assertTrue(episode.controller._physical_support(
                        measurement, foot, min_load=.2, record_plant=False))
                np.testing.assert_allclose(measurement.sole_positions[:, 2], .11, atol=.002)

    def test_event_prediction_reuses_cached_preparation_and_leaves_live_events_untouched(self):
        from legged_lab.perception.stair_event_supervisor import EventDrivenStairEpisode
        from legged_lab.perception.mujoco_position_bridge import MujocoPositionInterface, DynamicPositionBridge
        from legged_lab.perception.stair_position_observation import StairPositionObservation

        episode = EventDrivenStairEpisode()
        self.addCleanup(episode.close)
        interface = MujocoPositionInterface(episode.model, CONFIG)
        StairPositionObservation(interface).observe(episode)
        prepared = episode._prepared
        original = (episode.event_phase, episode.event_count, episode.event_last_time,
                    episode.event_gate_dwell, float(episode.data.time), episode.data.qpos.copy())
        calls = []
        original_events = EventDrivenStairEpisode._physical_events

        def observed_events(clone, measured, *args, **kwargs):
            calls.append((clone, measured.timestamp))
            return original_events(clone, measured, *args, **kwargs)

        bridge = DynamicPositionBridge(episode, interface, refine=False)
        with patch.object(EventDrivenStairEpisode, "_physical_events", observed_events):
            action = bridge.action()
        self.assertEqual(len(calls), interface.physics_steps-1)  # First state was already prepared by observe().
        self.assertEqual(len({timestamp for _, timestamp in calls}), len(calls))
        self.assertTrue(all(clone is not episode for clone, _ in calls))
        self.assertIs(episode._prepared, prepared)
        self.assertEqual((episode.event_phase, episode.event_count, episode.event_last_time,
                          episode.event_gate_dwell, float(episode.data.time)), original[:5])
        np.testing.assert_array_equal(episode.data.qpos, original[5])
        self.assertTrue(np.isfinite(action).all())

    def test_prediction_cannot_restore_a_disabled_teacher_through_shared_context(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        from legged_lab.perception.stair_event_supervisor import EventDrivenStairEpisode
        from legged_lab.perception.mujoco_position_bridge import MujocoPositionInterface, DynamicPositionBridge

        def _teacher_disabled(*args, **kwargs):
            raise RuntimeError("Teacher disabled for isolation test")

        for episode_type in (DynamicTeachingEpisode, EventDrivenStairEpisode):
            with self.subTest(episode_type=episode_type.__name__):
                episode = episode_type()
                self.addCleanup(episode.close)
                self.assertIs(episode.teacher, episode._context.teacher)
                interface = MujocoPositionInterface(episode.model, CONFIG)
                bridge = DynamicPositionBridge(episode, interface, refine=False)
                original_time = float(episode.data.time)
                original_position = episode.data.qpos.copy()
                with patch.object(episode._context.teacher, "control", _teacher_disabled):
                    with self.assertRaisesRegex(RuntimeError, "disabled for isolation test"):
                        bridge.action()
                    self.assertEqual(episode.data.time, original_time)
                    np.testing.assert_array_equal(episode.data.qpos, original_position)
                    # Explicit held-target PD remains executable, without recovering
                    # or calling the original inverse-dynamics instance method.
                    action = interface.action_from_target(episode.data.qpos[interface.qpos])
                    episode._advance(lambda ep, _: interface.torque_from_action(ep.data, action))
                self.assertGreater(episode.data.time, original_time)


if __name__ == "__main__":
    unittest.main()
