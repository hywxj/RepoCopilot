import unittest

import torch

from legged_lab.perception.foothold_control import (
    bounded_overshoot_joint_step,
    confirmed_support_transition,
    capture_point_offset,
    current_tread_match,
    damped_joint_step,
    foothold_lock_transition,
    nonfoot_collision_risk,
    select_next_treads,
    stair_yaw_feedback,
    stair_support_coverage_transition,
    stair_width_promotion,
    standing_support_quality,
    supported_stair_progress,
    swing_foot_trajectory,
    terminal_platform_support,
    terminal_platform_height,
    verified_stair_progress_transition,
)


class FootholdControlTest(unittest.TestCase):
    def test_standing_reward_prefers_balanced_stationary_support(self):
        force = torch.zeros(3, 2, 3)
        force[..., 2] = torch.tensor([[50., 50.], [100., 0.], [50., 50.]])
        speed = torch.zeros_like(force)
        speed[2, :, 0] = 0.10
        result = standing_support_quality(force, speed, torch.full((3,), 100.))
        self.assertEqual(result.shape, (3,))
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertGreater(result[0].item(), result[1].item())
        self.assertGreater(result[0].item(), result[2].item())

    def test_standing_reward_scales_by_each_robots_weight(self):
        force = torch.tensor([[[0., 0., 50.], [0., 0., 50.]],
                              [[0., 0., 200.], [0., 0., 200.]]])
        result = standing_support_quality(force, torch.zeros_like(force), torch.tensor([100., 400.]))
        self.assertTrue(torch.allclose(result, torch.ones(2)))

    def test_standing_reward_single_batch_remains_finite_without_support(self):
        force = torch.zeros(1, 2, 3)
        result = standing_support_quality(force, force, torch.zeros(1))
        self.assertEqual(result.shape, (1,))
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(result.item(), 0.2)

    def test_width_curriculum_requires_stable_terminal_and_six_of_eight_levels(self):
        stable = torch.tensor([True, True, False, True])
        covered = torch.tensor(
            [
                [True] * 6 + [False] * 2,
                [True] * 5 + [False] * 3,
                [True] * 8,
                [True] * 8,
            ]
        )
        self.assertEqual(
            stair_width_promotion(stable, covered).tolist(),
            [True, False, False, True],
        )

    def test_overshoot_guard_only_retracts_predicted_forward_overrun(self):
        jacobian = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        delta = torch.tensor([[0.08, 0.0, 0.0], [0.01, 0.0, 0.0]])
        correction = bounded_overshoot_joint_step(
            jacobian, delta, torch.tensor([0.45, 0.45]),
            torch.tensor([0.50, 0.50]), max_joint_step=0.05,
        )
        self.assertLess(correction[0, 0].item(), -0.029)
        self.assertTrue(torch.allclose(correction[1], torch.zeros(3)))
        self.assertTrue((correction.abs() <= 0.05).all())

    def test_guard_can_keep_a_target_just_behind_the_ankle(self):
        treads = torch.tensor([[[0.30, 0.50, 0.0, 0.20, 0.9, 1.0]]])
        args = (
            treads, torch.tensor([[0.65]]),
            torch.tensor([[0.43, 0.10]]),
            torch.tensor([[0.80, 0.80]]),
            torch.tensor([[False, True]]), torch.tensor([-1]),
        )
        _, _, ordinary = select_next_treads(*args)
        _, _, guarded = select_next_treads(*args, min_forward_gap=-0.06)
        self.assertFalse(ordinary[0, 0])
        self.assertTrue(guarded[0, 0])

    def test_swing_target_retains_signed_error_until_the_far_edge(self):
        treads = torch.tensor([[[0.30, 0.50, 0.0, 0.20, 0.9, 1.0]]])
        args = (
            treads, torch.tensor([[0.65]]),
            torch.tensor([[0.50, 0.10]]), torch.tensor([[0.80, 0.80]]),
            torch.tensor([[False, True]]), torch.tensor([-1]),
        )
        selected, _, valid = select_next_treads(*args, min_forward_gap=-0.12)
        self.assertTrue(valid[0, 0].item())
        error = 0.5 * (selected[0, 0, 0] + selected[0, 0, 1]) - args[2][0, 0]
        self.assertAlmostEqual(error.item(), -0.10)

    def test_physical_coverage_requires_centered_stable_support_for_three_frames(self):
        count = torch.zeros(1, 2, 3, dtype=torch.long)
        covered = torch.zeros(1, 3, dtype=torch.bool)
        foot_x = torch.tensor([[0.625, 0.0]])
        sole_z = torch.tensor([[0.30, 0.45]])
        force = torch.tensor([[[0.0, 0.0, 80.0], [0.0, 0.0, 0.0]]])
        speed = torch.zeros(1, 2)
        args = (torch.tensor([0.50]), torch.tensor([0.25]),
                torch.tensor([0.45]), torch.tensor([0.15]))
        for expected in (0, 0, 1, 0):
            count, covered, increment = stair_support_coverage_transition(
                count, covered, foot_x, sole_z, force, speed, *args, descending=True
            )
            self.assertEqual(increment.item(), expected)
        self.assertEqual(covered.tolist(), [[True, False, False]])

        count.zero_()
        covered.zero_()
        foot_x[0, 0] = 0.68
        for _ in range(3):
            count, covered, increment = stair_support_coverage_transition(
                count, covered, foot_x, sole_z, force, speed, *args, descending=True
            )
        self.assertFalse(covered.any())
        for _ in range(3):
            count, covered, increment = stair_support_coverage_transition(
                count, covered, foot_x, sole_z, force, speed, *args, descending=True,
                center_tolerance_fraction=0.45, max_center_tolerance=0.12,
            )
        self.assertTrue(covered[0, 0])

        count.zero_()
        covered.zero_()
        foot_x[0, 0] = 1.10
        sole_z[0, 0] = 0.0
        for _ in range(3):
            count, covered, increment = stair_support_coverage_transition(
                count, covered, foot_x, sole_z, force, speed, *args, descending=True
            )
        self.assertEqual(covered.tolist(), [[False, False, True]])

    def test_terminal_platform_height_accounts_for_downstairs_spawn(self):
        start = torch.tensor([1.20, 0.0])
        riser = torch.tensor([0.15, 0.15])
        self.assertTrue(torch.allclose(
            terminal_platform_height(start, riser, 8, descending=True),
            torch.tensor([0.0, -1.20]),
        ))
        self.assertTrue(torch.allclose(
            terminal_platform_height(start, riser, 8, descending=False),
            torch.tensor([2.40, 1.20]),
        ))

    def test_terminal_platform_requires_stable_upward_support_at_correct_height(self):
        sole_z = torch.tensor([[0.0, 0.2], [0.0, 0.0], [0.0, 0.0]])
        force = torch.tensor([
            [[0.0, 0.0, 80.0], [0.0, 0.0, 0.0]],
            [[80.0, 0.0, 10.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 80.0], [0.0, 0.0, 0.0]],
        ])
        speed = torch.tensor([[0.02, 0.0], [0.02, 0.0], [0.50, 0.0]])
        self.assertEqual(
            terminal_platform_support(sole_z, force, speed, torch.zeros(3)).tolist(),
            [True, False, False],
        )

    def test_nonfoot_collision_risk_uses_peak_history_force(self):
        force = torch.zeros(3, 2, 2, 3)
        force[1, 0, 1, 2] = 170.0
        force[2, 1, 0, 2] = 600.0
        self.assertTrue(torch.allclose(
            nonfoot_collision_risk(force), torch.tensor([0.0, 0.5, 1.0])
        ))

    def test_landing_requires_three_stable_frames_and_emits_once(self):
        count = torch.zeros(1, 2, dtype=torch.long)
        events = []
        for valid in (True, True, False, True, True, True, True, False, True, True, True):
            count, event = confirmed_support_transition(
                count, torch.tensor([[valid, False]]), 3
            )
            events.append(bool(event[0, 0]))
        self.assertEqual([index for index, event in enumerate(events) if event], [5, 10])

    def test_identity_jacobian_tracks_position_error(self):
        jacobian = torch.eye(3).unsqueeze(0)
        error = torch.tensor([[0.03, 0.00, 0.05]])
        correction = damped_joint_step(jacobian, error, damping=0.01)
        self.assertTrue(torch.allclose(correction, error, atol=1.0e-4))

    def test_singular_jacobian_remains_finite_and_bounded(self):
        jacobian = torch.zeros(2, 3, 6)
        error = torch.ones(2, 3)
        correction = damped_joint_step(jacobian, error, max_joint_step=0.12)
        self.assertTrue(torch.isfinite(correction).all())
        self.assertTrue((correction.abs() <= 0.12).all())

    def test_ascending_target_excludes_old_lower_tread(self):
        treads = torch.tensor([[[0.25, 0.45, 0.0, 0.20, 0.8, 1.0],
                                [0.47, 0.67, 0.0, 0.20, 0.8, 1.0]]])
        target, height, valid = select_next_treads(
            treads,
            torch.tensor([[0.0, 0.15]]),
            torch.tensor([[0.15, 0.05]]),
            torch.tensor([[0.15, 0.15]]),
            torch.tensor([[False, True]]),
            torch.tensor([1]),
        )
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[0, 1])
        self.assertAlmostEqual(target[0, 0, 0].item(), 0.47)
        self.assertAlmostEqual(height[0, 0].item(), 0.15, places=5)

    def test_descending_target_excludes_old_upper_tread(self):
        treads = torch.tensor([[[0.25, 0.45, 0.0, 0.20, 0.8, 1.0],
                                [0.47, 0.67, 0.0, 0.20, 0.8, 1.0]]])
        target, height, valid = select_next_treads(
            treads,
            torch.tensor([[0.81, 0.51]]),
            torch.tensor([[0.15, 0.05]]),
            torch.tensor([[0.66, 0.66]]),
            torch.tensor([[False, True]]),
            torch.tensor([-1]),
        )
        self.assertTrue(valid[0, 0])
        self.assertAlmostEqual(target[0, 0, 0].item(), 0.47)
        self.assertAlmostEqual(height[0, 0].item(), 0.51, places=5)

    def test_preview_exposes_next_lower_tread_before_swing(self):
        treads = torch.tensor([[[0.25, 0.45, 0.0, 0.20, 0.8, 1.0],
                                [0.47, 0.67, 0.0, 0.20, 0.8, 1.0]]])
        _, _, hidden = select_next_treads(
            treads, torch.tensor([[0.80, 0.65]]),
            torch.tensor([[0.10, 0.10]]), torch.tensor([[0.80, 0.80]]),
            torch.ones(1, 2, dtype=torch.bool), torch.tensor([-1]),
        )
        selected, _, preview = select_next_treads(
            treads, torch.tensor([[0.80, 0.65]]),
            torch.tensor([[0.10, 0.10]]), torch.tensor([[0.80, 0.80]]),
            torch.ones(1, 2, dtype=torch.bool), torch.tensor([-1]),
            preview_stance=True,
        )
        self.assertFalse(hidden.any())
        self.assertTrue(preview.all())
        self.assertAlmostEqual(selected[0, 0, 0].item(), 0.47, places=5)

    def test_target_requires_support_and_correct_mode(self):
        treads = torch.tensor([[[0.25, 0.45, 0.0, 0.20, 0.8, 1.0]]])
        _, _, unsupported = select_next_treads(
            treads, torch.tensor([[0.15]]), torch.zeros(1, 2),
            torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool), torch.tensor([1]),
        )
        _, _, blind = select_next_treads(
            treads, torch.tensor([[0.15]]), torch.zeros(1, 2),
            torch.zeros(1, 2), torch.tensor([[False, True]]), torch.tensor([0]),
        )
        self.assertFalse(unsupported.any())
        self.assertFalse(blind.any())

    def test_heading_feedback_turns_toward_center_and_is_bounded(self):
        feedback = stair_yaw_feedback(
            torch.tensor([0.2, -0.2, 2.0]),
            torch.tensor([0.3, -0.3, 1.0]),
            torch.zeros(3),
        )
        self.assertLess(feedback[0].item(), 0.0)
        self.assertGreater(feedback[1].item(), 0.0)
        self.assertAlmostEqual(feedback[2].item(), -0.45)

    def test_locked_target_survives_detector_dropout_and_releases_on_contact(self):
        locked = torch.tensor([[True, True]])
        mode = torch.tensor([[1, 1]], dtype=torch.int8)
        no_candidate = torch.zeros(1, 2, dtype=torch.bool)
        kept, acquire = foothold_lock_transition(
            locked, mode, torch.tensor([[False, True]]), torch.tensor([1]), no_candidate
        )
        self.assertEqual(kept.tolist(), [[True, False]])
        self.assertFalse(acquire.any())
        switched, _ = foothold_lock_transition(
            locked, mode, torch.zeros(1, 2, dtype=torch.bool), torch.tensor([-1]), no_candidate
        )
        self.assertFalse(switched.any())

    def test_target_lock_does_not_acquire_on_contact_or_blind_mode(self):
        valid, acquired = foothold_lock_transition(
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.int8),
            torch.tensor([[True, False]]),
            torch.tensor([1]),
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertEqual(valid.tolist(), [[False, True]])
        self.assertEqual(acquired.tolist(), [[False, True]])
        valid, acquired = foothold_lock_transition(
            valid, torch.tensor([[0, 1]], dtype=torch.int8),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.tensor([0]),
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertFalse(valid.any())
        self.assertFalse(acquired.any())

    def test_historical_target_requires_current_high_confidence_tread_for_control(self):
        selected = torch.tensor([[[0.40, 0.60, 0.0, 0.20, 0.9, 1.0]]])
        current = torch.tensor([[[0.42, 0.62, 0.0, 0.20, 0.8, 1.0]]])
        self.assertTrue(current_tread_match(selected, current)[0, 0])
        current[..., 4] = 0.5
        self.assertFalse(current_tread_match(selected, current)[0, 0])
        current[..., 4] = 0.9
        current[..., :2] += 0.12
        self.assertFalse(current_tread_match(selected, current)[0, 0])

    def test_capture_point_includes_body_velocity_and_foot_offset(self):
        offset = capture_point_offset(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[[0.1, 0.2, 0.0]]]),
            torch.tensor([[1.0, 0.0]]),
        )
        self.assertAlmostEqual(offset[0, 0, 0].item(), (1.0 / 9.81) ** 0.5 - 0.1, places=5)
        self.assertAlmostEqual(offset[0, 0, 1].item(), -0.2, places=5)

    def test_swing_trajectory_reaches_target_and_clears_step(self):
        start = torch.tensor([[[0.0, 0.1, 0.8]]])
        target = torch.tensor([[[0.3, 0.1, 0.65]]])
        elapsed = torch.tensor([[0.0, 0.14, 0.28, 0.40]])
        trajectory = swing_foot_trajectory(
            start.expand(1, 4, 3), target.expand(1, 4, 3), elapsed, 0.28, 0.10
        )
        self.assertTrue(torch.allclose(trajectory[0, 0], start[0, 0]))
        self.assertTrue(torch.allclose(trajectory[0, 2], target[0, 0]))
        self.assertTrue(torch.allclose(trajectory[0, 3], target[0, 0]))
        self.assertAlmostEqual(trajectory[0, 1, 2].item(), 0.825, places=5)
        self.assertTrue(torch.allclose(trajectory[..., 1], torch.full((1, 4), 0.1)))

    def test_stair_progress_requires_forward_position_and_foot_support_height(self):
        width = torch.tensor([0.20])
        height = torch.tensor([0.16])
        start = torch.tensor([0.0])
        level = torch.tensor([0])
        level, increment = supported_stair_progress(
            torch.tensor([0.55]), torch.tensor([0.15]), start, width, height, level, 0.50, 6
        )
        self.assertEqual((level.item(), increment.item()), (1, 1))
        level, increment = supported_stair_progress(
            torch.tensor([0.75]), torch.tensor([0.15]), start, width, height, level, 0.50, 6
        )
        self.assertEqual((level.item(), increment.item()), (1, 0))
        level, increment = supported_stair_progress(
            torch.tensor([0.75]), torch.tensor([0.32]), start, width, height, level, 0.50, 6
        )
        self.assertEqual((level.item(), increment.item()), (2, 1))

    def test_descending_progress_starts_after_upper_platform(self):
        width = torch.tensor([0.20])
        height = torch.tensor([0.16])
        start = torch.tensor([0.96])
        level = torch.tensor([0])
        level, increment = supported_stair_progress(
            torch.tensor([0.60]), torch.tensor([0.80]), start, width, height,
            level, torch.tensor([0.70]), 6, direction=-1,
        )
        self.assertEqual((level.item(), increment.item()), (0, 0))
        level, increment = supported_stair_progress(
            torch.tensor([0.75]), torch.tensor([0.80]), start, width, height,
            level, torch.tensor([0.70]), 6, direction=-1,
        )
        self.assertEqual((level.item(), increment.item()), (1, 1))

    def test_verified_progress_requires_confirmed_next_level_tread(self):
        observed = torch.tensor([2, 2, 1, 1])
        verified = torch.zeros(4, dtype=torch.long)
        confirmed = torch.tensor([[False, False], [True, False], [True, False], [True, False]])
        heights = torch.tensor([[0.75, 0.0], [0.90, 0.0], [0.75, 0.0], [0.75, 0.0]])
        next_level, credited = verified_stair_progress_transition(
            observed, verified, confirmed, heights,
            torch.full((4,), 0.90), torch.full((4,), 0.15), direction=-1,
        )
        self.assertEqual(credited.tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(next_level.tolist(), [0, 0, 1, 1])
        next_level, credited = verified_stair_progress_transition(
            observed[2:3], next_level[2:3],
            torch.tensor([[True, False]]), torch.tensor([[0.60, 0.0]]),
            torch.tensor([0.90]), torch.tensor([0.15]), direction=-1,
        )
        self.assertEqual(credited.tolist(), [0.0])

    def test_verified_progress_recovers_after_skipped_unsafe_step_without_back_credit(self):
        verified = torch.tensor([1])
        observed = torch.tensor([4])
        confirmed = torch.tensor([[True, False]])
        start = torch.tensor([0.90])
        riser = torch.tensor([0.15])
        verified, credited = verified_stair_progress_transition(
            observed, verified, confirmed, torch.tensor([[0.45, 0.0]]),
            start, riser, direction=-1,
        )
        self.assertEqual((verified.item(), credited.item()), (3, 1.0))
        verified, credited = verified_stair_progress_transition(
            observed, verified, confirmed, torch.tensor([[0.45, 0.0]]),
            start, riser, direction=-1,
        )
        self.assertEqual((verified.item(), credited.item()), (3, 0.0))


if __name__ == "__main__":
    unittest.main()
