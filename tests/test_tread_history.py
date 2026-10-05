import unittest

import torch

from legged_lab.perception.stair_geometry import (
    centered_tread_support,
    compensate_tread_history,
    match_sole_to_treads,
    unsafe_tread_touchdown,
)


class TreadHistoryTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.zeros(1, 1, 10)
        self.features[0, 0, 0:2] = torch.tensor((0.15, 0.275))
        self.features[0, 0, 5] = 1.0
        self.positions = torch.zeros(1, 1, 2)
        self.headings = torch.tensor([[[1.0, 0.0]]])
        self.heading = torch.tensor([[1.0, 0.0]])

    def compensate(self, robot_x):
        return compensate_tread_history(
            self.features,
            self.positions,
            self.headings,
            torch.tensor([[robot_x, 0.0]]),
            self.heading,
            max_treads=1,
            max_forward=2.0,
            rear_limit=0.20,
        )[0, 0]

    def test_tread_under_foot_is_preserved_with_negative_near_edge(self):
        tread = self.compensate(0.45)
        self.assertEqual(tread[5].item(), 1.0)
        self.assertAlmostEqual(tread[0].item() * 2.0, -0.15, places=5)
        self.assertAlmostEqual(tread[1].item() * 2.0, 0.10, places=5)

    def test_tread_entirely_behind_foot_is_invalid(self):
        tread = self.compensate(0.80)
        self.assertEqual(tread[5].item(), 0.0)

    def test_trailing_foot_can_retain_a_tread_behind_the_root(self):
        tread = compensate_tread_history(
            self.features, self.positions, self.headings,
            torch.tensor([[0.80, 0.0]]), self.heading,
            max_treads=1, max_forward=2.0, rear_limit=0.60,
        )[0, 0]
        self.assertEqual(tread[5].item(), 1.0)
        self.assertAlmostEqual(tread[1].item() * 2.0, -0.25, places=5)

    def test_sole_must_be_at_tread_height(self):
        treads = torch.tensor([[[0.10, 0.40, 0.0, 0.30, 0.9, 1.0]]])
        near = torch.tensor([[[0.15]]])
        far = torch.tensor([[[0.35]]])
        heights = torch.tensor([[1.20]])
        supported, _, _, good_score = match_sole_to_treads(
            near, far, torch.tensor([[[1.20]]]), treads, heights, 0.03, 0.04, 0.10
        )
        too_high, _, _, high_score = match_sole_to_treads(
            near, far, torch.tensor([[[1.35]]]), treads, heights, 0.03, 0.04, 0.10
        )
        self.assertTrue(supported.item())
        self.assertFalse(too_high.item())
        self.assertGreater(good_score.item(), high_score.item())

    def test_narrow_tread_rejects_edge_landing_even_with_enough_overlap(self):
        treads = torch.tensor([[[0.40, 0.62, 0.0, 0.22, 0.9, 1.0]]])
        centers = torch.tensor([[[0.51], [0.53], [0.55]]])
        near = centers - 0.103
        far = centers + 0.103
        supported, overlap, _, _ = match_sole_to_treads(
            near, far, torch.ones_like(near), treads, torch.ones(1, 1), 0.05, 0.08, 0.06
        )
        centered, error, tolerance = centered_tread_support(
            near, far, treads, supported, center_fraction=0.40
        )
        self.assertTrue(supported.all())
        self.assertGreater(overlap[0, 2, 0].item(), 0.08)
        self.assertEqual(centered.flatten().tolist(), [True, True, False])
        self.assertAlmostEqual(tolerance[0, 0, 0].item(), 0.022, places=3)
        self.assertAlmostEqual(error[0, 2, 0].item(), 0.04, places=3)

    def test_unsafe_touchdown_ignores_upper_platform_and_blind_mode(self):
        first = torch.tensor([[True, True, True]])
        safe = torch.tensor([[False, False, False]])
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        overlap = torch.tensor([[[0.05], [0.05], [0.05]]])
        height_error = torch.tensor([[[0.0], [0.15], [0.0]]])
        unsafe = unsafe_tread_touchdown(
            first, safe, valid, overlap, height_error,
            torch.tensor([-1]), height_tolerance=0.06,
        )
        self.assertEqual(unsafe.tolist(), [[True, False, True]])
        blind = unsafe_tread_touchdown(
            first, safe, valid, overlap, height_error,
            torch.tensor([0]), height_tolerance=0.06,
        )
        self.assertFalse(blind.any())


if __name__ == "__main__":
    unittest.main()
