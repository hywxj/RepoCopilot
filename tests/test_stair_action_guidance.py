import unittest

import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.storage import RolloutStorage


class StairActionGuidanceTest(unittest.TestCase):
    def test_only_active_labels_train_actions_and_teacher_is_detached(self):
        actions = torch.zeros(2, 2, requires_grad=True)
        teacher = torch.tensor([[.2, -.2, 1.], [10., 10., 0.]], requires_grad=True)
        loss = AMPPPO.stair_action_teacher_loss(actions, teacher)
        torch.testing.assert_close(loss, torch.tensor(.04))
        loss.backward()
        torch.testing.assert_close(actions.grad[1], torch.zeros(2))
        self.assertIsNone(teacher.grad)

    def test_no_labels_keeps_default_loss_zero(self):
        actions = torch.zeros(2, 2, requires_grad=True)
        for teacher in (None, torch.zeros(2, 3)):
            self.assertEqual(float(AMPPPO.stair_action_teacher_loss(actions, teacher)), 0.)

    @staticmethod
    def transition(step, guided):
        transition = RolloutStorage.Transition()
        ids = torch.arange(3).float()+step*3
        transition.observations = ids[:, None].repeat(1, 2)
        transition.privileged_observations = transition.observations
        transition.actions = transition.action_mean = torch.zeros(3, 2)
        transition.action_sigma = torch.ones(3, 2)
        transition.values = torch.zeros(3, 1)
        transition.actions_log_prob = transition.rewards = transition.dones = torch.zeros(3)
        if guided:
            transition.stair_teacher_actions = torch.stack((ids, ids+1, torch.ones(3)), dim=1)
        return transition

    def test_guidance_stays_aligned_with_observations_after_minibatch_shuffle(self):
        storage = RolloutStorage("rl", 3, 2, [2], [2], [2])
        for step in range(2):
            storage.add_transitions(self.transition(step, True))
        for batch in storage.mini_batch_generator(2, 2):
            self.assertEqual(len(batch), 13)
            torch.testing.assert_close(batch[0][:, 0], batch[12][:, 0])
        storage.clear()
        for step in range(2):
            storage.add_transitions(self.transition(step, False))
        self.assertEqual(float(storage.stair_teacher_actions.sum()), 0.)

    def test_unguided_ppo_keeps_existing_minibatch_contract(self):
        storage = RolloutStorage("rl", 3, 2, [2], [2], [2])
        for step in range(2):
            storage.add_transitions(self.transition(step, False))
        self.assertTrue(all(len(batch) == 12 for batch in storage.mini_batch_generator(2, 1)))


if __name__ == "__main__":
    unittest.main()
