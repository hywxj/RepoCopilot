import unittest

import torch
from torch import nn

from rsl_rl.modules.actor_critic import ActorCritic, GatedResidualActorCritic, _GatedResidualActor


class ResidualActionScaleTest(unittest.TestCase):
    def test_conditioning_updates_cannot_change_stop_feedback(self):
        policy = GatedResidualActorCritic(
            5, 6, 2, actor_hidden_dims=[4], critic_hidden_dims=[4], residual_hidden_dims=[4],
            base_actor_obs_dim=2, residual_scale=1., active_base_action_scale=0., squash_residual=False,
            initialize_stair_actor_from_blind=True, train_stair_conditioning_only=True,
        )
        original = ActorCritic(2, 3, 2, actor_hidden_dims=[4], critic_hidden_dims=[4])
        policy.load_state_dict(original.state_dict())
        stopping = torch.tensor([[.2, -.1, 1., 1., 1.]])
        active = torch.tensor([[.2, -.1, 1., 0., 1.]])
        before = policy.act_inference(stopping).detach().clone()
        feedback = {key: value.detach().clone() for key, value in policy.actor.residual_actor.state_dict().items()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=.02)
        for _ in range(10):
            optimizer.zero_grad()
            (policy.act_inference(active)-2.).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(policy.act_inference(stopping), before, atol=0., rtol=0.)
        current = policy.actor.residual_actor.state_dict()
        torch.testing.assert_close(current['0.weight'][:, :2], feedback['0.weight'][:, :2], atol=0., rtol=0.)
        for key in current:
            if key != '0.weight':
                torch.testing.assert_close(current[key], feedback[key], atol=0., rtol=0.)
        self.assertGreater(float((current['0.weight'][:, 2:]-feedback['0.weight'][:, 2:]).abs().max()), 0.)
        inference = GatedResidualActorCritic(5, 6, 2, actor_hidden_dims=[4], critic_hidden_dims=[4],
                                           residual_hidden_dims=[4], base_actor_obs_dim=2,
                                           residual_scale=1., active_base_action_scale=0., squash_residual=False)
        inference.load_state_dict(policy.state_dict())
        torch.testing.assert_close(inference.act_inference(stopping), before, atol=0., rtol=0.)

    def test_phase_exploration_only_changes_sampling_not_inference(self):
        policy = GatedResidualActorCritic(
            5, 6, 2, actor_hidden_dims=[4], residual_hidden_dims=[4], base_actor_obs_dim=2,
            init_noise_std=.1, min_action_std=.04, max_action_std=.15,
            low_noise_obs_indices=[2], low_noise_scale=.1,
        )
        obs = torch.tensor([[0., 0., 1., 0., 1.], [0., 0., 0., 1., 1.]])
        before = policy.act_inference(obs).detach().clone()
        policy.update_distribution(obs)
        torch.testing.assert_close(policy.action_std, torch.tensor([[.01, .01], [.1, .1]]))
        torch.testing.assert_close(policy.action_mean, before)

    def test_phase_exploration_rejects_invalid_configuration(self):
        for indices, scale in (([5], .1), ([-1], .1), ([2], 0.), ([2], 1.1)):
            with self.subTest(indices=indices, scale=scale), self.assertRaises(ValueError):
                GatedResidualActorCritic(5, 6, 2, actor_hidden_dims=[4], base_actor_obs_dim=2,
                                        low_noise_obs_indices=indices, low_noise_scale=scale)

    def test_observation_scaling_does_not_touch_blind_branch(self):
        base = nn.Linear(2, 2)
        residual = nn.Sequential(nn.Linear(4, 2))
        actor = _GatedResidualActor(base, residual, 2, -1, 1., squash_residual=False,
                                    residual_observation_scales=[1., 1., 12.5, 1.])
        obs = torch.tensor([[.2, -.1, .02, 0.], [.2, -.1, .02, 1.]])
        torch.testing.assert_close(actor(obs)[0], base(obs[:, :2])[0])
        expected = base(obs[:, :2])+obs[:, -1:]*residual(obs*torch.tensor([1., 1., 12.5, 1.]))
        torch.testing.assert_close(actor(obs), expected)

    def test_old_checkpoint_scale_migration_preserves_actions(self):
        kwargs = dict(actor_hidden_dims=[4], residual_hidden_dims=[4], base_actor_obs_dim=2,
                      residual_scale=1., active_base_action_scale=0., squash_residual=False)
        old = GatedResidualActorCritic(5, 6, 2, **kwargs)
        with torch.no_grad():
            old.actor.residual_actor[-1].weight.normal_(0., .1)
        state = old.state_dict()
        state.pop('actor.residual_observation_scales')
        new = GatedResidualActorCritic(5, 6, 2, residual_observation_scales=[1., 1., 12.5, 8.3, 1.], **kwargs)
        self.assertTrue(new.load_state_dict(state))
        self.assertTrue(new.reset_optimizer_on_load)
        obs = torch.randn(8, 5)
        obs[:, -1] = torch.linspace(-1, 1, 8)
        torch.testing.assert_close(new.act_inference(obs), old.act_inference(obs), atol=1.e-6, rtol=1.e-5)
        restored = GatedResidualActorCritic(5, 6, 2, residual_observation_scales=[1., 1., 12.5, 8.3, 1.], **kwargs)
        restored.load_state_dict(new.state_dict())
        self.assertFalse(restored.reset_optimizer_on_load)
        torch.testing.assert_close(restored.act_inference(obs), new.act_inference(obs))

    def test_scale_rejects_nonfinite_or_mismatched_configuration(self):
        for scales in ([1., 0., 1.], [1., float('nan'), 1.], [1., 1.]):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                _GatedResidualActor(nn.Linear(2, 2), nn.Sequential(nn.Linear(3, 2)), 2, -1, 1.,
                                    residual_observation_scales=scales)

    def test_full_stair_branch_keeps_blind_action_when_gate_is_closed(self):
        base = nn.Linear(2, 2)
        residual = nn.Sequential(nn.Linear(3, 2))
        with torch.no_grad():
            base.weight.zero_()
            base.bias.fill_(0.3)
            residual[0].weight.zero_()
            residual[0].bias.fill_(1.0)
        actor = _GatedResidualActor(
            base, residual, base_actor_obs_dim=2, gate_obs_index=-1,
            residual_scale=0.80,
        )
        output = actor(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        torch.testing.assert_close(output[0], torch.full((2,), 0.3))
        torch.testing.assert_close(output[1], torch.full((2,), 0.3 + 0.8 * torch.tanh(torch.tensor(1.0))))

    def test_leg_scale_changes_only_gated_residual_and_not_checkpoint_state(self):
        base = nn.Linear(2, 2)
        residual = nn.Sequential(nn.Linear(3, 2))
        with torch.no_grad():
            base.weight.zero_()
            base.bias.zero_()
            residual[0].weight.zero_()
            residual[0].bias.fill_(1.0)
        actor = _GatedResidualActor(
            base, residual, base_actor_obs_dim=2, gate_obs_index=-1,
            residual_scale=0.2, residual_action_scales=[1.0, 3.0],
        )
        output = actor(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        expected = 0.2 * torch.tanh(torch.tensor(1.0))
        self.assertTrue(torch.allclose(output[0], torch.tensor([expected, 3.0 * expected])))
        self.assertTrue(torch.allclose(output[1], torch.zeros(2)))
        self.assertNotIn("residual_action_scales", actor.state_dict())

    def test_stair_actor_can_replace_the_blind_gait_and_blend_continuously(self):
        base = nn.Linear(2, 2)
        residual = nn.Sequential(nn.Linear(3, 2))
        with torch.no_grad():
            base.weight.zero_()
            base.bias.fill_(2.)
            residual[0].weight.zero_()
            residual[0].bias.fill_(0.)
        actor = _GatedResidualActor(
            base, residual, base_actor_obs_dim=2, gate_obs_index=-1,
            residual_scale=3., active_base_action_scale=0.,
        )
        output = actor(torch.tensor([[0., 0., 0.], [0., 0., 0.5], [0., 0., 1.], [0., 0., -1.]]))
        torch.testing.assert_close(output, torch.tensor([[2., 2.], [1., 1.], [0., 0.], [0., 0.]]))
        self.assertNotIn('active_base_action_scale', actor.state_dict())

    def test_replacement_mode_keeps_residual_gradients_and_frozen_base(self):
        base = nn.Linear(2, 2)
        residual = nn.Sequential(nn.Linear(3, 2))
        for p in base.parameters():
            p.requires_grad_(False)
        actor = _GatedResidualActor(
            base, residual, base_actor_obs_dim=2, gate_obs_index=-1,
            residual_scale=3., active_base_action_scale=0.,
        )
        actor(torch.tensor([[0., 0., 1.]])).sum().backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertGreater(residual[0].bias.grad.abs().sum().item(), 0.)

    def test_independent_stair_actor_initialization_preserves_gait_for_every_gate(self):
        original = ActorCritic(2, 3, 2, actor_hidden_dims=[4], critic_hidden_dims=[4])
        policy = GatedResidualActorCritic(
            5, 6, 2, actor_hidden_dims=[4], critic_hidden_dims=[4], residual_hidden_dims=[4],
            base_actor_obs_dim=2, residual_scale=1., active_base_action_scale=0.,
            squash_residual=False, initialize_stair_actor_from_blind=True,
        )
        initial_critic = {key: value.clone() for key, value in policy.critic.state_dict().items()}
        self.assertFalse(policy.load_state_dict(original.state_dict()))
        for key, value in policy.critic.state_dict().items():
            torch.testing.assert_close(value, initial_critic[key])
        obs = torch.randn(4, 5)
        obs[:, -1] = torch.tensor([0., 0.5, 1., -1.])
        torch.testing.assert_close(policy.act_inference(obs), original.act_inference(obs[:, :2]))
        self.assertTrue(all(not p.requires_grad for p in policy.actor.base_actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in policy.actor.residual_actor.parameters()))
        torch.testing.assert_close(policy.actor.residual_actor[0].weight[:, 2:], torch.zeros(4, 3))
        with torch.no_grad():
            policy.actor.residual_actor[-1].bias.add_(0.4)
        updated = policy.act_inference(obs)-original.act_inference(obs[:, :2])
        torch.testing.assert_close(updated, obs[:, -1:].abs()*torch.full((4, 2), 0.4), atol=1e-6, rtol=1e-5)

    def test_blind_initialized_stair_actor_rejects_mismatched_architecture(self):
        with self.assertRaisesRegex(ValueError, 'matching hidden layers'):
            GatedResidualActorCritic(
                5, 6, 2, actor_hidden_dims=[4], residual_hidden_dims=[3], base_actor_obs_dim=2,
                residual_scale=1., active_base_action_scale=0., squash_residual=False,
                initialize_stair_actor_from_blind=True,
            )

    def test_low_noise_stair_skill_uses_its_configured_bounds(self):
        policy = GatedResidualActorCritic(
            5, 6, 2, actor_hidden_dims=[4], residual_hidden_dims=[4], base_actor_obs_dim=2,
            residual_scale=1., active_base_action_scale=0., squash_residual=False,
            initialize_stair_actor_from_blind=True, init_noise_std=0.02,
            min_action_std=0.01, max_action_std=0.10,
        )
        obs = torch.zeros(1, 5)
        obs[:, -1] = 1.
        original = ActorCritic(2, 3, 2, actor_hidden_dims=[4], critic_hidden_dims=[4], init_noise_std=0.9)
        policy.load_state_dict(original.state_dict())
        policy.update_distribution(obs)
        torch.testing.assert_close(policy.action_std, torch.full((1, 2), 0.02))
        with torch.no_grad():
            policy.std.fill_(0.001)
        policy.update_distribution(obs)
        torch.testing.assert_close(policy.action_std, torch.full((1, 2), 0.01))
        with torch.no_grad():
            policy.std.fill_(0.5)
        policy.update_distribution(obs)
        torch.testing.assert_close(policy.action_std, torch.full((1, 2), 0.10))


if __name__ == "__main__":
    unittest.main()
