import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "stair_step_rewards", Path(__file__).resolve().parents[1]/"legged_lab/mdp/stair_step_rewards.py")
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


class StairTrackingRewardTest(unittest.TestCase):
    def test_error_penalty_increases_with_drift_without_an_idle_bonus(self):
        error = torch.tensor([0., .5, 1., 2., 4.])
        env = SimpleNamespace(step_reward_metrics={"body_reference": 1/(1+error.square())})
        actual = rewards.stair_step_metric(env, "body_reference", quadratic_error=True)
        torch.testing.assert_close(actual, torch.tensor([0., -.25, -1., -4., -9.]))

    def test_default_reference_metric_is_unchanged(self):
        value = torch.tensor([.3, .8])
        env = SimpleNamespace(step_reward_metrics={"heading": value})
        self.assertIs(rewards.stair_step_metric(env, "heading"), value)
        with self.assertRaises(ValueError):
            rewards.stair_step_metric(env, "heading", quadratic_error=True)

    def test_zero_metric_has_a_finite_capped_penalty(self):
        env = SimpleNamespace(step_reward_metrics={"feet_reference": torch.zeros(2)})
        torch.testing.assert_close(rewards.stair_step_metric(env, "feet_reference", quadratic_error=True),
                                   torch.full((2,), -9.))


if __name__ == "__main__":
    unittest.main()
