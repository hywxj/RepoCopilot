"""Broad-phase cropping must reproduce the original full-grid SAT exactly."""
import math
from types import SimpleNamespace
import unittest

import numpy as np

from legged_lab.perception.tread_surfaces import (
    SurfaceGeometryResult, SurfaceValidationCfg, TreadSurfaceExtractor,
    _footprint_grid_window, footprint_cells, footprint_mask_supported,
)


def full_grid_supported(grid, observed, center, yaw, cfg, margins):
    rear, front, side = margins
    length = .5*(cfg.foot_rear_extent+cfg.foot_front_extent)
    width = cfg.foot_half_width+side
    corners = np.array([[-length-rear, -width], [length+front, -width],
                        [length+front, width], [-length-rear, width]])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    corners = corners @ rotation.T+np.asarray(center)
    if (corners[:, 0].min() < cfg.min_forward or corners[:, 0].max() > cfg.max_forward
            or np.abs(corners[:, 1]).max() > cfg.lateral_half_width):
        return False
    covered = footprint_cells(grid-np.asarray(center), yaw, cfg, margins)
    return bool(covered.any() and observed[covered].all())


class FootprintQueryTest(unittest.TestCase):
    def test_random_yaw_center_margins_and_observed_holes(self):
        rng = np.random.default_rng(1841)
        hits = 0
        for spacing in (.01, .02, .037):
            cfg = SurfaceValidationCfg(min_forward=-.4, max_forward=2.,
                                       lateral_half_width=.7, grid_size=spacing)
            grid = TreadSurfaceExtractor(cfg).grid_xy
            for _ in range(400):
                center = rng.uniform([-.5, -.8], [2.1, .8])
                yaw = rng.uniform(-4*math.pi, 4*math.pi)
                margins = rng.uniform(0., .12, 3)
                observed = np.ones(grid.shape[:2], dtype=bool)
                if rng.random() < .5:
                    observed[rng.random(observed.shape) < .015] = False
                expected = full_grid_supported(grid, observed, center, yaw, cfg, margins)
                actual = footprint_mask_supported(grid, observed, center, yaw, cfg, margins)
                self.assertEqual(actual, expected)
                hits += actual
                # Compare membership itself, so all-false support cannot hide
                # a skipped cell, including unobserved cells in the crop.
                rear, front, side = margins
                length = .5*(cfg.foot_rear_extent+cfg.foot_front_extent)
                width = cfg.foot_half_width+side
                polygon = np.array([[-length-rear, -width], [length+front, -width],
                                    [length+front, width], [-length-rear, width]])
                rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                                     [math.sin(yaw), math.cos(yaw)]])
                window = _footprint_grid_window(grid, polygon @ rotation.T+center, spacing)
                cropped = np.zeros(grid.shape[:2], dtype=bool)
                cropped[window] = footprint_cells(grid[window]-center, yaw, cfg, margins)
                np.testing.assert_array_equal(cropped, footprint_cells(grid-center, yaw, cfg, margins))
        self.assertGreater(hits, 100)

    def test_boundary_and_zero_area_touch_are_unchanged(self):
        cfg = SurfaceValidationCfg(min_forward=0., max_forward=1., lateral_half_width=.5,
                                   foot_rear_extent=.1, foot_front_extent=.1, foot_half_width=.04)
        grid = TreadSurfaceExtractor(cfg).grid_xy
        observed = np.ones(grid.shape[:2], dtype=bool)
        for yaw in (0., math.pi/2, math.pi, -.3, .3):
            for center in ([.1, 0.], [.9, 0.], [.5, .46], [.5, -.46], [.5, 0.]):
                for delta in (-1.e-8, -1.e-12, 0., 1.e-12, 1.e-8):
                    query = np.asarray(center)+[delta, delta]
                    for margins in ((0., 0., 0.), (.02, .01, .01), (.01, .02, .01)):
                        self.assertEqual(footprint_mask_supported(grid, observed, query, yaw, cfg, margins),
                                         full_grid_supported(grid, observed, query, yaw, cfg, margins))
        # This row only touches x=.6: an unknown zero-area neighbour must not
        # become a false rejection. Moving 1e-6 into it must reject support.
        observed[30, :] = False
        self.assertTrue(footprint_mask_supported(grid, observed, [.5, 0.], 0., cfg, (0., 0., 0.)))
        self.assertFalse(footprint_mask_supported(grid, observed, [.500001, 0.], 0., cfg, (0., 0., 0.)))

    def test_custom_nonregular_and_mutated_grids_fall_back(self):
        cfg = SurfaceValidationCfg(min_forward=0., max_forward=1., lateral_half_width=.5)
        original = TreadSurfaceExtractor(cfg).grid_xy
        polygon = np.array([[.3, -.1], [.6, -.1], [.6, .1], [.3, .1]])
        altered = original.copy()
        altered[1, 1, 0] += .15
        nonuniform = original.copy()
        nonuniform[1, :, 0] += .005
        for grid in (altered, nonuniform, original[::-1], original.transpose(1, 0, 2), original.reshape(-1, 2)):
            window = _footprint_grid_window(grid, polygon, cfg.grid_size)
            self.assertEqual(grid[window].shape, grid.shape)
            mask = np.ones(grid.shape[:-1], dtype=bool)
            self.assertEqual(footprint_mask_supported(grid, mask, [.45, 0.], .2, cfg, (.03, .02, .02)),
                             full_grid_supported(grid, mask, [.45, 0.], .2, cfg, (.03, .02, .02)))
        # A previous fast query must not leave a stale regular-grid cache.
        self.assertLess(original[_footprint_grid_window(original, polygon, cfg.grid_size)].size, original.size)
        original[1, 1] = [.45, 0.]
        observed = np.ones(original.shape[:2], dtype=bool)
        observed[1, 1] = False
        self.assertFalse(footprint_mask_supported(original, observed, [.45, 0.], 0., cfg, (0., 0., 0.)))

    def test_perception_uses_exact_full_grid_predicate(self):
        cfg = SurfaceValidationCfg(min_forward=0., max_forward=1., lateral_half_width=.5)
        grid = TreadSurfaceExtractor(cfg).grid_xy
        mask = np.ones(grid.shape[:2], dtype=bool)
        mask[25, 25] = False
        surface = SimpleNamespace(surface_id=0, valid=True, observed_mask=mask)
        result = SurfaceGeometryResult(cfg, grid, np.zeros(mask.shape), mask, np.zeros(mask.shape),
                                       np.zeros(0), surfaces=[surface])
        rng = np.random.default_rng(148)
        for direction in (-1, 1):
            result.direction = direction
            for _ in range(100):
                center, yaw = rng.uniform([0., -.5], [1., .5]), rng.uniform(-math.pi, math.pi)
                for include_margin in (False, True):
                    margins = result.support_margins() if include_margin else (0., 0., 0.)
                    self.assertEqual(result.footprint_supported(0, center, yaw, include_margin),
                                     full_grid_supported(grid, mask, center, yaw, cfg, margins))



if __name__ == "__main__":
    unittest.main()
