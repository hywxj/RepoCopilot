from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from legged_lab.perception.stair_step_controller import body_height_limit
from legged_lab.perception.stair_skill_teacher import StairSkillTeacher


class StairBodyReferenceTest(unittest.TestCase):
    def test_leg_reach_matches_elf3_model(self):
        model = ET.parse(Path(__file__).resolve().parents[1]/"legged_lab/assets/elf3_lite/urdf/elf3.urdf")
        lengths = [np.linalg.norm(np.fromstring(model.find(f".//joint[@name='{name}']/origin").get("xyz"), sep=" "))
                   for name in ("l_knee_y_joint", "l_ankle_y_joint")]
        self.assertAlmostEqual(sum(lengths), .64)

    def test_height_cannot_rise_a_full_tread_while_rear_leg_stays_on_source(self):
        hips = np.array([[0., .136, -.3825], [0., -.136, -.3825]])
        feet = np.array([[.36, .15, .11], [0., -.15, 0.]])
        ankles = feet+[-.03, 0., .04]
        root_xy = np.array([.36, .15])
        limit = body_height_limit(root_xy, hips, ankles, .63)
        self.assertLess(limit, 1.)
        root = np.array([*root_xy, limit])
        self.assertTrue((np.linalg.norm(root+hips-ankles, axis=1) <= .63+1.e-9).all())
        upper_feet = feet.copy()
        upper_feet[1] = [.36, -.15, .11]
        upper_limit = body_height_limit(root_xy, hips, upper_feet+[-.03, 0., .04], .63)
        self.assertGreater(upper_limit, limit+.10)

    def test_unreachable_horizontal_reference_is_rejected(self):
        self.assertIsNone(body_height_limit(np.array([1., 0.]), np.zeros((2, 3)), np.zeros((2, 3)), .63))

    def test_motor_reset_projects_body_height_with_feet_instead_of_midair_height(self):
        teacher = StairSkillTeacher(torch.zeros(2, 3), .22, .32, .11, 1, torch.full((2,), 1000.))
        root = torch.tensor([[0., 0., 1.08]]).repeat(2, 1)
        feet = torch.tensor([[[0., .15, .04], [0., -.15, .04]]]).repeat(2, 1, 1)
        teacher.update(root, torch.eye(3).repeat(2, 1, 1), feet, torch.zeros_like(feet),
                       feet[:, :, None].repeat(1, 1, 4, 1), torch.tensor([0., 0., 1.]).expand(2, 2, 3),
                       torch.tensor([0., 0., 500.]).expand(2, 2, 3), torch.zeros_like(root),
                       torch.zeros_like(root), .02)
        torch.testing.assert_close(teacher.initial_root[:, 2], torch.full((2,), 1.04))


if __name__ == "__main__":
    unittest.main()
