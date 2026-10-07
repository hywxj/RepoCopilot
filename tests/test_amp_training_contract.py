"""CPU checks for the AMP optimizer and continuous-task reward contract."""

import contextlib
import copy
import io
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils.utils import Normalizer


class _ExpertFrames:
    def __init__(self, state, next_state):
        self.state = state
        self.next_state = next_state

    def feed_forward_generator(self, num_batches, batch_size):
        assert batch_size == len(self.state)
        for _ in range(num_batches):
            yield self.state, self.next_state


class AMPTrainingContractTest(unittest.TestCase):
    def test_real_ppo_update_uses_raw_statistics_and_one_feature_space(self):
        torch.manual_seed(31)
        np.random.seed(31)
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(4, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=0.2)
        discriminator = Discriminator(140, 1.0, [8], device="cpu")
        normalizer = Normalizer(70)
        normalizer.update(np.stack([np.full(70, 16.0), np.full(70, 24.0)]))
        before_normalizer = copy.deepcopy(normalizer)
        expert = _ExpertFrames(24.0 + torch.randn(8, 70), 27.0 + torch.randn(8, 70))
        original_expert = (expert.state.clone(), expert.next_state.clone())
        algorithm = AMPPPO(
            policy, discriminator, expert, normalizer, amp_replay_buffer_size=16,
            num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-3, device="cpu",
        )
        algorithm.init_storage("rl", 2, 4, [4], [4], [2])
        with torch.no_grad():
            for step in range(4):
                obs = torch.randn(2, 4)
                algorithm.act(obs, obs, 20.0 + step + torch.randn(2, 70))
                algorithm.process_env_step(
                    torch.tensor([0.2 + step, -0.1 * step]), torch.zeros(2, dtype=torch.bool), {},
                    21.0 + step + torch.randn(2, 70),
                )
            algorithm.compute_returns(torch.randn(2, 4))

        # Record actual replay draws; the test still uses the real replay sampler,
        # rollout storage, PPO loss, discriminator loss and Adam optimizer.
        replay_generator = algorithm.amp_storage.feed_forward_generator
        sampled_pairs = []

        def record_replay(*args):
            for pair in replay_generator(*args):
                sampled_pairs.append(tuple(frame.clone() for frame in pair))
                yield pair

        classification_inputs = []
        hook = discriminator.register_forward_pre_hook(
            lambda module, args: classification_inputs.append(args[0].detach().clone())
        )
        policy_before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
        discriminator_before = [parameter.detach().clone() for parameter in discriminator.parameters()]
        replay_before = (algorithm.amp_storage.states.clone(), algorithm.amp_storage.next_states.clone())
        with (
            patch.object(algorithm.amp_storage, "feed_forward_generator", side_effect=record_replay),
            patch.object(discriminator, "compute_grad_pen", wraps=discriminator.compute_grad_pen) as penalty,
            patch.object(normalizer, "update", wraps=normalizer.update) as update_statistics,
        ):
            losses = algorithm.update()
        hook.remove()

        self.assertTrue(all(math.isfinite(value) for value in losses.values()), losses)
        self.assertGreater(losses["amp_grad_pen"], 0.0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(policy_before, policy.actor.parameters())))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(discriminator_before, discriminator.parameters())))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters()))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in discriminator.parameters()))

        policy_state, policy_next_state = sampled_pairs[0]
        expected_policy = torch.cat([
            before_normalizer.normalize_torch(policy_state, "cpu"),
            before_normalizer.normalize_torch(policy_next_state, "cpu"),
        ], dim=-1)
        expected_expert = torch.cat([
            before_normalizer.normalize_torch(expert.state, "cpu"),
            before_normalizer.normalize_torch(expert.next_state, "cpu"),
        ], dim=-1)
        self.assertEqual(len(classification_inputs), 2)
        torch.testing.assert_close(classification_inputs[0], expected_policy)
        torch.testing.assert_close(classification_inputs[1], expected_expert)
        penalty.assert_called_once()
        torch.testing.assert_close(torch.cat(penalty.call_args.args, dim=-1), expected_expert)

        raw_frames = torch.cat([policy_state, policy_next_state, expert.state, expert.next_state]).numpy()
        update_statistics.assert_called_once()
        np.testing.assert_array_equal(update_statistics.call_args.args[0], raw_frames)
        expected_statistics = copy.deepcopy(before_normalizer)
        expected_statistics.update(raw_frames)
        np.testing.assert_allclose(normalizer.mean, expected_statistics.mean)
        np.testing.assert_allclose(normalizer.var, expected_statistics.var)
        self.assertEqual(normalizer.count, before_normalizer.count + 32)
        self.assertGreater(float(normalizer.mean.mean()), 20.0)
        torch.testing.assert_close(expert.state, original_expert[0])
        torch.testing.assert_close(expert.next_state, original_expert[1])
        torch.testing.assert_close(algorithm.amp_storage.states, replay_before[0])
        torch.testing.assert_close(algorithm.amp_storage.next_states, replay_before[1])

    @staticmethod
    def _constant_discriminator():
        discriminator = Discriminator(6, 2.0, [8], device="cpu", task_reward_lerp=0.4)
        with torch.no_grad():
            for parameter in discriminator.parameters():
                parameter.zero_()
        return discriminator

    def test_reward_time_integration_and_motion_mask_leave_task_term_unscaled(self):
        discriminator = self._constant_discriminator()
        state = torch.zeros(2, 3)
        task_reward = torch.tensor([0.08, 0.12])
        mask = torch.tensor([True, False])
        reward, prediction = discriminator.predict_amp_reward(
            state, state, task_reward, reward_dt=0.02, motion_mask=mask,
        )
        # d=0 -> style rate = 2 * 0.75 = 1.5; task_reward already contains dt.
        torch.testing.assert_close(reward, torch.tensor([0.6 * 1.5 * 0.02 + 0.4 * 0.08, 0.4 * 0.12]))
        torch.testing.assert_close(prediction, torch.zeros(2, 1))
        twice_duration, _ = discriminator.predict_amp_reward(
            state, state, 2 * task_reward, reward_dt=0.04, motion_mask=mask,
        )
        torch.testing.assert_close(twice_duration, 2 * reward)

    def test_omitted_time_scaling_preserves_legacy_reward(self):
        discriminator = self._constant_discriminator()
        task = torch.tensor([0.1])
        reward, _ = discriminator.predict_amp_reward(torch.zeros(1, 3), torch.zeros(1, 3), task)
        self.assertEqual(tuple(reward.shape), (1,))
        torch.testing.assert_close(reward, 0.6 * torch.tensor([1.5]) + 0.4 * task)

    def test_invalid_duration_and_ambiguous_mask_fail_early(self):
        discriminator = self._constant_discriminator()
        state = torch.zeros(2, 3)
        for duration in (0.0, -0.02, float("inf"), float("nan")):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "reward_dt"):
                discriminator.predict_amp_reward(state, state, torch.zeros(2), reward_dt=duration)
        for mask in (torch.ones(2), torch.ones(2, 1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "motion_mask"):
                discriminator.predict_amp_reward(state, state, torch.zeros(2), motion_mask=mask)


if __name__ == "__main__":
    unittest.main()
