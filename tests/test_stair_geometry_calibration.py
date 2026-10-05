import unittest
from types import SimpleNamespace

import torch

from legged_lab.perception.stair_geometry import StairGeometryExtractor


class StairGeometryCalibrationTest(unittest.TestCase):
    def extractor(self, offset):
        cfg = SimpleNamespace(
            min_forward=0.15,
            max_forward=2.0,
            profile_bin_size=0.025,
            max_treads=4,
            min_riser_height=0.08,
            max_riser_height=0.21,
            min_tread_depth=0.16,
            max_tread_depth=0.36,
            max_tread_height_std=0.035,
            min_tread_confidence=0.25,
            roughness_reference=0.04,
            descending_edge_offset_m=offset,
        )
        return StairGeometryExtractor(cfg)

    def test_only_valid_descending_treads_are_shifted(self):
        raw = self.extractor(0.0)
        calibrated = self.extractor(-0.065)
        x = raw.profile_x
        descending_profile = -0.15 * torch.floor(((x - 0.40) / 0.25).clamp_min(0.0))
        ascending_profile = -descending_profile

        raw_down = raw.extract_from_profile(descending_profile)
        calibrated_down = calibrated.extract_from_profile(descending_profile)
        valid = raw_down.treads[..., 5] > 0.5
        self.assertTrue(valid.any())
        self.assertTrue((raw_down.direction < 0).all())
        torch.testing.assert_close(
            calibrated_down.treads[..., :2][valid],
            raw_down.treads[..., :2][valid] - 0.065,
        )
        self.assertTrue((calibrated_down.treads[..., :2][~valid] == 0.0).all())

        raw_up = raw.extract_from_profile(ascending_profile)
        calibrated_up = calibrated.extract_from_profile(ascending_profile)
        self.assertTrue((raw_up.direction > 0).all())
        torch.testing.assert_close(calibrated_up.treads, raw_up.treads)

    def test_32cm_treads_are_detected_in_both_directions(self):
        extractor = self.extractor(0.0)
        x = extractor.profile_x
        levels = torch.floor(((x - 0.40) / 0.32).clamp_min(0.0))
        for direction in (1, -1):
            with self.subTest(direction=direction):
                result = extractor.extract_from_profile(direction * 0.15 * levels)
                valid = result.treads[0, :, 5] > 0.5
                self.assertGreaterEqual(int(valid.sum()), 3)
                self.assertEqual(result.direction.item(), direction)
                self.assertTrue((result.treads[0, valid, 3] - 0.32).abs().max() <= 0.025)
                self.assertGreaterEqual(result.stair_confidence.item(), 0.65)

    def test_surface_diagnostic_does_not_change_actor_geometry(self):
        legacy = self.extractor(0.0)
        cfg = legacy.cfg
        cfg.processing_width, cfg.processing_height = 32, 18
        cfg.lateral_half_width = 0.35
        cfg.min_height_from_root, cfg.max_height_from_root = -1.5, 0.35
        cfg.surface_validation_enabled = True
        cfg.surface_processing_width, cfg.surface_processing_height = 64, 36
        diagnostic = StairGeometryExtractor(cfg)
        depth = torch.ones(1, 36, 64)
        intrinsics = torch.tensor([[[45., 0., 32.], [0., 45., 18.], [0., 0., 1.]]])
        position = torch.zeros(1, 3)
        rotation = torch.tensor([[1., 0., 0., 0.]])
        inputs = (depth, intrinsics, position, rotation, position, rotation)
        raw = legacy.extract(*inputs)
        validated = diagnostic.extract(*inputs)
        torch.testing.assert_close(validated.features, raw.features, rtol=0, atol=0)
        torch.testing.assert_close(validated.treads, raw.treads, rtol=0, atol=0)
        self.assertEqual(validated.processed_image_shape, (36, 64))
        self.assertEqual(validated.surface_geometry[0].point_labels.size, 36 * 64)
        cfg.surface_memory_enabled = True
        with_memory = StairGeometryExtractor(cfg)
        remembered = with_memory.extract(*inputs, frame_timestamp=torch.tensor([0.]))
        torch.testing.assert_close(remembered.features, raw.features, rtol=0, atol=0)
        torch.testing.assert_close(remembered.treads, raw.treads, rtol=0, atol=0)
        self.assertEqual(len(remembered.surface_memory), 1)
        self.assertEqual(remembered.surface_memory[0].memory_frame_count, 1)
        with_memory.reset_surface_memory([0])
        self.assertFalse(with_memory.surface_memories[0].frames)

    def test_memory_cannot_be_enabled_without_an_acquisition_timestamp(self):
        extractor = self.extractor(0.0)
        cfg = extractor.cfg
        cfg.processing_width, cfg.processing_height = 32, 18
        cfg.lateral_half_width = 0.35
        cfg.min_height_from_root, cfg.max_height_from_root = -1.5, 0.35
        cfg.surface_validation_enabled, cfg.surface_memory_enabled = True, True
        diagnostic = StairGeometryExtractor(cfg)
        depth = torch.ones(1, 18, 32)
        intrinsics = torch.tensor([[[20., 0., 16.], [0., 20., 9.], [0., 0., 1.]]])
        position = torch.zeros(1, 3)
        rotation = torch.tensor([[1., 0., 0., 0.]])
        with self.assertRaisesRegex(ValueError, "timestamps"):
            diagnostic.extract(depth, intrinsics, position, rotation, position, rotation)


if __name__ == "__main__":
    unittest.main()
