"""The student observes the current dynamic references and never direct loads."""

import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

try:
    import mujoco
    import qpsolvers
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo and qpsolvers are required")
class PositionObservationTest(unittest.TestCase):
    def test_dynamic_snapshot_is_idempotent_and_history_contains_applied_action(self):
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        from legged_lab.perception.mujoco_position_bridge import MujocoPositionInterface
        from legged_lab.perception.stair_position_observation import StairPositionObservation

        episode = DynamicTeachingEpisode(direction=-1)
        config = Path(__file__).resolve().parents[1]/"legged_lab/configs/elf3_stair_position.yaml"
        interface = MujocoPositionInterface(episode.model, config)
        observer = StairPositionObservation(interface)
        before = episode.data.qpos.copy()
        observation = observer.observe(episode)
        support_time = episode.support_dwell.copy()
        first_sample = episode.prepare_step()[0]
        self.assertIs(episode.prepare_step()[0], first_sample)
        np.testing.assert_array_equal(episode.support_dwell, support_time)
        np.testing.assert_array_equal(episode.data.qpos, before)
        self.assertEqual(observation.shape, (1000,))
        self.assertEqual(observation[999], -1.)
        np.testing.assert_array_equal(observation[984:986], 0.)
        np.testing.assert_array_equal(observation[994:996], 0.)
        np.testing.assert_array_equal(observation[992:994], [0., 1.])
        np.testing.assert_array_equal(observation[:96], observation[864:960])
        measured = episode.measurement()
        expected = (np.array(first_sample["foot_reference"])-measured.sole_positions) @ measured.yaw_rotation
        np.testing.assert_allclose(observation[978:984], expected.ravel(), atol=1.e-7)

        # The executable student path must not call inverse dynamics, including
        # when preparing the phase/reference snapshot for its observation.
        episode.teacher.control = Mock(side_effect=AssertionError("teacher must stay off"))
        action = interface.action_from_target(episode.data.qpos[interface.qpos])
        for _ in range(interface.physics_steps):
            episode._advance(lambda ep, _: interface.torque_from_action(ep.data, action))
        observer.last_action = action.copy()
        second = observer.observe(episode)
        episode.teacher.control.assert_not_called()
        np.testing.assert_array_equal(second[:864], observation[96:960])
        np.testing.assert_allclose(second[931:960], action, atol=1.e-6)
        self.assertAlmostEqual(episode.data.time-first_sample["time"], .02)


if __name__ == "__main__":
    unittest.main()
