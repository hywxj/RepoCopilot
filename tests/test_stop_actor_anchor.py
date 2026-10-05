import unittest

import torch
from torch import nn

from rsl_rl.algorithms.amp_ppo import AMPPPO


class StopActorAnchorTest(unittest.TestCase):
    def setUp(self):
        self.algorithm = AMPPPO.__new__(AMPPPO)
        self.algorithm.device = "cpu"
        self.algorithm.stop_actor_anchor = None
        self.algorithm.stop_anchor_indices = []
        self.algorithm.stop_anchor_coef = 0.
        self.actor = nn.Linear(4, 2)
        self.obs = torch.tensor([[.2, -.1, 1., 0.], [.3, .1, 0., 1.]])

    def test_only_observe_actions_are_protected_and_anchor_stays_frozen(self):
        self.algorithm.set_stop_actor_anchor(self.actor, [2], 50.)
        with torch.no_grad():
            self.actor.bias.add_(.1)
        actions = self.actor(self.obs)
        loss = self.algorithm.stop_anchor_loss(self.obs, actions)
        torch.testing.assert_close(loss, torch.tensor(.01))
        actions.retain_grad()
        loss.backward()
        self.assertGreater(actions.grad[0].abs().sum(), 0.)
        torch.testing.assert_close(actions.grad[1], torch.zeros(2))
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in self.algorithm.stop_actor_anchor.parameters()))

    def test_empty_phase_batch_keeps_a_zero_differentiable_loss(self):
        self.algorithm.set_stop_actor_anchor(self.actor, [2], 50.)
        obs = self.obs[1:]
        actions = self.actor(obs)
        loss = self.algorithm.stop_anchor_loss(obs, actions)
        loss.backward()
        self.assertEqual(float(loss), 0.)

    def test_default_does_not_change_existing_ppo(self):
        loss = self.algorithm.stop_anchor_loss(self.obs, self.actor(self.obs))
        self.assertEqual(float(loss), 0.)

    def test_invalid_anchor_is_rejected(self):
        for indices, coefficient in (([], 1.), ([2], 0.)):
            with self.assertRaises(ValueError):
                self.algorithm.set_stop_actor_anchor(self.actor, indices, coefficient)


if __name__ == "__main__":
    unittest.main()
