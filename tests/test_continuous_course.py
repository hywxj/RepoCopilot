"""Geometry/endpoint safety checks for the independent continuous baseline."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from legged_lab.utils.continuous_course import (
    ContinuousCourse, continuous_course_mesh, course_outcomes,
    footprint_edge_margin, local_height_features, support_height,
)


def feet_at(x, floor=0.):
    corners = torch.tensor([[-.09, -.042, 0.], [-.09, .042, 0.],
                            [.15, -.042, 0.], [.15, .042, 0.]])
    corners = corners[None, None].repeat(1, 2, 1, 1)
    corners[..., 0] += x
    corners[:, 0, :, 1] += .13
    corners[:, 1, :, 1] -= .13
    corners[..., 2] += floor
    return corners


class ContinuousCourseTests(unittest.TestCase):
    def test_mesh_and_queries_agree_at_every_surface_including_first_down_edge(self):
        for direction in ("flat", "up", "down"):
            for height in (.02, .04, .06, .11, .16):
                with self.subTest(direction=direction, height=height):
                    c = ContinuousCourse(direction=direction, step_height=height)
                    meshes, origin = continuous_course_mesh(0., SimpleNamespace(course=c, size=(c.length, c.width)))
                    self.assertEqual(origin.tolist(), [c.spawn_x, c.width/2, c.spawn_height])
                    for mesh in meshes:
                        center_x = (mesh.bounds[0, 0]+mesh.bounds[1, 0])/2-origin[0]
                        expected = support_height(torch.tensor(center_x), c).item()+origin[2]
                        self.assertAlmostEqual(mesh.bounds[1, 2], expected)
                    x = torch.tensor([c.first_edge-1.e-4, c.first_edge+1.e-4])
                    delta = support_height(x, c).diff().item()
                    self.assertAlmostEqual(delta, {"flat": 0., "up": height, "down": -height}[direction], places=6)
                    for mesh in meshes:
                        self.assertAlmostEqual(mesh.bounds[0, 2], -.02)

    def test_height_grid_rotates_with_heading_and_invalid_samples_are_masked(self):
        c = ContinuousCourse(direction="up")
        root = torch.tensor([[0., 0., 1.], [0., 0., 1.], [4.9, 1.45, 1.]])
        features = local_height_features(root, torch.tensor([0., torch.pi/2, 0.]), c)
        self.assertEqual(features.shape, (3, 90))
        self.assertTrue(torch.isfinite(features).all())
        self.assertGreater(features[0, :45].max().item(), -1.)
        torch.testing.assert_close(features[1, :45], torch.full((45,), -1.))
        invalid = features[2, 45:] == 0
        self.assertTrue(invalid.any())
        self.assertTrue((features[2, :45][invalid] == 0).all())

    def test_goal_requires_whole_feet_on_platform_and_loaded_support_without_dwell(self):
        for direction in ("flat", "up", "down"):
            c = ContinuousCourse(direction=direction)
            floor = support_height(torch.tensor(c.goal_x), c).item()
            root = torch.tensor([[c.goal_x, 0., 1.+floor]])
            gravity = torch.tensor([[0., 0., -1.]])
            feet = feet_at(c.goal_x, floor)
            load = torch.tensor([[100., 100.]])
            no = torch.tensor([False])
            success, failed, *_ = course_outcomes(root, gravity, feet, load, no, no, c)
            self.assertTrue(success.item())
            self.assertFalse(failed.item())
            for reason in ("airborne", "one_unloaded", "collision", "timeout", "low_root", "tilt", "outside", "foot_on_edge", "foot_outside"):
                rp, g, fp, f, hit, timeout = root.clone(), gravity.clone(), feet.clone(), load.clone(), no.clone(), no.clone()
                if reason == "airborne": fp[..., 2] += .1
                elif reason == "one_unloaded": f[0, 1] = 0.
                elif reason == "collision": hit[:] = True
                elif reason == "timeout": timeout[:] = True
                elif reason == "low_root": rp[:, 2] = floor+.3
                elif reason == "tilt": g[:, 2] = -.5
                elif reason == "outside": rp[:, 1] = c.lateral_limit+.1
                elif reason == "foot_on_edge": fp[0, 0, 0, 0] = c.final_edge-.01 if direction != "flat" else c.goal_x-.6
                elif reason == "foot_outside": fp[0, 0, 0, 1] = c.width/2+.01
                with self.subTest(direction=direction, reason=reason):
                    self.assertFalse(course_outcomes(rp, g, fp, f, hit, timeout, c)[0].item())

    def test_signed_margin_reports_loaded_foot_overhang_instead_of_center_only(self):
        c = ContinuousCourse(direction="up")
        safe_x = c.first_edge+.12
        safe = footprint_edge_margin(feet_at(safe_x), torch.full((1, 2), safe_x), c)
        np.testing.assert_allclose(safe.numpy(), .03, atol=1.e-6)
        edge_x = c.first_edge+.03
        overhang = footprint_edge_margin(feet_at(edge_x), torch.full((1, 2), edge_x), c)
        np.testing.assert_allclose(overhang.numpy(), -.06, atol=1.e-6)

    def test_no_single_step_progress_is_a_completion_event(self):
        c = ContinuousCourse(direction="up")
        for i in range(c.num_steps):
            x = c.first_edge+i*c.step_depth+.12
            floor = support_height(torch.tensor(x), c).item()
            out = course_outcomes(torch.tensor([[x, 0., floor+1.]]), torch.tensor([[0., 0., -1.]]),
                                  feet_at(x, floor), torch.tensor([[100., 100.]]),
                                  torch.tensor([False]), torch.tensor([False]), c)
            self.assertFalse(out[0].item())


if __name__ == "__main__":
    unittest.main()
