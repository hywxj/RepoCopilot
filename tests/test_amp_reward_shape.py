import unittest

import torch

from rsl_rl.modules.discriminator import Discriminator


class AMPRewardShapeTest(unittest.TestCase):
    def test_single_and_multiple_environments_keep_batch_axis(self):
        for batch in (1, 4):
            model = Discriminator(6, 0., [8], device="cpu", task_reward_lerp=1.)
            task = torch.arange(batch, dtype=torch.float32)
            reward, prediction = model.predict_amp_reward(torch.zeros(batch, 3), torch.zeros(batch, 3), task)
            self.assertEqual(tuple(reward.shape), (batch,))
            self.assertEqual(tuple(prediction.shape), (batch, 1))
            torch.testing.assert_close(reward, task)


if __name__ == '__main__':
    unittest.main()
