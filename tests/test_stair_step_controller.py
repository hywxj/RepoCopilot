import unittest
import copy
import json
import math
from unittest.mock import patch

import numpy as np

from legged_lab.perception.stair_step_controller import (
    StairStepController, StepControlCfg, StepMeasurement, StepPhase,
)
from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor
from legged_lab.perception.surface_memory import SurfaceMemory


class StairStepTest(unittest.TestCase):
    def make_case(self, direction=1):
        cfg = SurfaceValidationCfg(min_forward=-0.35, max_forward=1.0)
        extractor = TreadSurfaceExtractor(cfg)
        x, y = np.meshgrid(np.arange(-0.34, 0.95, 0.006), np.arange(-0.34, 0.34, 0.006))
        height = np.where(x < 0.22, -1., np.where(x < 0.54, -1.+direction*0.11, -1.+direction*0.22))
        geometry = extractor.extract(np.column_stack((x.ravel(), y.ravel(), height.ravel())))
        geometry.timestamp_s = 0.
        for s in geometry.surfaces:
            s.track_id = s.surface_id+10
            s.last_observed_time = 0.
        m = StepMeasurement(0., 0, np.array([0., 0., 1.]), np.eye(3),
                            np.array([[0., 0.15, 0.], [0., -0.15, 0.]]),
                            np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 3)),
                            np.array([[0., 0., 500.], [0., 0., 500.]]),
                            np.zeros(3), np.zeros(3), 1000.)
        return StairStepController(StepControlCfg(confirmation_s=0.04, settle_s=0.08)), geometry, m

    def tick(self, c, g, m, count=1):
        for _ in range(count):
            m.timestamp += 0.02
            g.timestamp_s = m.timestamp
            for s in g.surfaces:
                s.last_observed_time = m.timestamp
            c.update(g, m, 0.02)

    def locked(self, direction=1):
        c, g, m = self.make_case(direction)
        self.tick(c, g, m, 5)
        self.assertEqual(c.phase, StepPhase.SHIFT_LEAD)
        return c, g, m

    def test_gate_is_off_without_stair_geometry(self):
        c, g, m = self.make_case()
        g.direction = 0
        self.tick(c, g, m, 20)
        self.assertFalse(c.active)
        self.assertIsNone(c.lock)

    def test_body_reference_places_model_com_on_fixed_support_without_accumulating_offset(self):
        c, g, m = self.locked()
        m.hip_offsets = np.array([[0., .136, -.3825], [0., -.136, -.3825]])
        m.com_offset = np.array([.02, .04, -.28])
        support = c.reference_feet[1-c.lead, :2].copy()
        c.features(m)
        reference = c.reference_root.copy()
        np.testing.assert_allclose(reference[:2]+m.com_offset[:2], support)
        c.features(m)
        c.reward_metrics(m, .02)
        np.testing.assert_array_equal(c.reference_root, reference)

    def test_observe_stops_in_place_and_lowers_a_nearby_airborne_foot(self):
        c, g, m = self.make_case()
        m.sole_positions[0, 2] = 0.035
        m.contact_forces[:, 2] = [0., 1000.]
        entry_root = m.root_position.copy()
        entry_xy = m.sole_positions[:, :2].copy()
        self.tick(c, g, m, 3)
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        np.testing.assert_array_equal(c.reference_root, entry_root)
        np.testing.assert_array_equal(c.reference_feet[:, :2], entry_xy)
        np.testing.assert_array_equal(c.reference_feet[:, 2], [0., 0.])
        m.root_position[0] += 0.05
        self.tick(c, g, m)
        np.testing.assert_array_equal(c.reference_root, entry_root)

    def test_lowering_rewards_touchdown_load_and_moves_com_toward_landing_foot(self):
        for phase in (StepPhase.LOWER_LEAD, StepPhase.LOWER_TRAIL):
            with self.subTest(phase=phase):
                c, g, m = self.locked()
                c._transition(phase, m)
                swing = c.lead if phase == StepPhase.LOWER_LEAD else 1-c.lead
                stance = 1-swing
                m.contact_forces[:, 2] = 0.
                m.contact_forces[swing, 2] = c.cfg.touchdown_load_fraction*m.body_weight
                m.contact_forces[stance, 2] = (1-c.cfg.touchdown_load_fraction)*m.body_weight
                m.sole_positions[swing] = c.lock.targets[swing]
                self.assertAlmostEqual(c.reward_metrics(m, .02)['load_reference'], 1.)
                m.hip_offsets = np.array([[0., .136, -.3825], [0., -.136, -.3825]])
                m.com_offset = np.array([.02, .04, -.28])
                source = c.reference_feet[stance, :2] if phase == StepPhase.LOWER_LEAD else c.lock.targets[stance, :2]
                anchor = source+c.cfg.touchdown_load_fraction*(c.lock.targets[swing, :2]-source)
                c.features(m)
                np.testing.assert_allclose(c.reference_root[:2]+m.com_offset[:2], anchor)

    def test_airborne_or_wrong_place_contact_does_not_authorize_touchdown_com_shift(self):
        for phase in (StepPhase.LOWER_LEAD, StepPhase.LOWER_TRAIL):
            c, g, m = self.locked()
            c._transition(phase, m)
            swing = c.lead if phase == StepPhase.LOWER_LEAD else 1-c.lead
            stance = 1-swing
            m.hip_offsets = np.array([[0., .136, -.3825], [0., -.136, -.3825]])
            m.com_offset = np.array([.02, .04, -.28])
            anchor = (c.reference_feet[stance, :2] if phase == StepPhase.LOWER_LEAD
                      else c.lock.targets[stance, :2]).copy()
            for load in (0., .15*m.body_weight):
                m.contact_forces[swing, 2] = load
                c.constrain_body_reference(m)
                np.testing.assert_allclose(c.reference_root[:2]+m.com_offset[:2], anchor)
            m.sole_positions[swing] = c.lock.targets[swing]
            m.contact_forces[swing, 2] = .02*m.body_weight
            c.constrain_body_reference(m)
            np.testing.assert_allclose(c.reference_root[:2]+m.com_offset[:2],
                                       anchor+c.cfg.touchdown_load_fraction*(c.lock.targets[swing, :2]-anchor))
            self.assertIsNone(c.planted_positions[swing])
            self.assertFalse(c._physical_support(m, swing))

    def test_observe_does_not_project_a_loaded_or_different_level_foot(self):
        c, g, m = self.make_case()
        m.sole_positions[0, 2] = 0.11
        self.tick(c, g, m, 3)
        self.assertAlmostEqual(c.reference_feet[0, 2], 0.11)
        c, g, m = self.make_case()
        m.sole_positions[0, 2] = 0.17
        m.contact_forces[:, 2] = [0., 1000.]
        self.tick(c, g, m, 3)
        self.assertAlmostEqual(c.reference_feet[0, 2], 0.17)

    def test_reference_rewards_keep_stance_credit_and_distant_tracking_signal(self):
        c, g, m = self.locked()
        c.reference_feet = m.sole_positions.copy()
        c.reference_feet[0, 0] += 0.40
        far = c.reward_metrics(m, 0.02)['feet_reference']
        self.assertGreater(far, 0.5)
        m.sole_positions[0, 0] += 0.10
        nearer = c.reward_metrics(m, 0.02)['feet_reference']
        self.assertGreater(nearer, far)
        c.reference_root[1] += 0.20
        self.assertGreater(c.reward_metrics(m, 0.02)['body_reference'], 0.1)
        m.contact_forces[:, 2] = [0., 1000.]
        self.assertGreater(c.reward_metrics(m, 0.02)['load_reference'], 0.)

    def test_shift_body_reference_does_not_follow_source_foot_drift(self):
        for phase in (StepPhase.SHIFT_LEAD, StepPhase.SHIFT_TRAIL):
            with self.subTest(phase=phase):
                c, g, m = self.locked()
                c._transition(phase, m)
                swing = c.lead if phase == StepPhase.SHIFT_LEAD else 1-c.lead
                stance = 1-swing
                expected = c.reference_feet[stance, :2].copy()
                self.tick(c, g, m)
                m.sole_positions[:, 0] -= 0.10
                m.root_position[0] -= 0.10
                self.tick(c, g, m)
                np.testing.assert_array_equal(c.reference_root[:2], expected)
                self.assertEqual(c.phase, phase)
                self.assertFalse(c.shift_requirements['swing_unloaded'])
                c.reset()
                self.assertEqual(c.shift_requirements, {})

    def test_new_support_height_before_lock_fails_without_authorizing_a_step(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                c, g, m = self.make_case(direction)
                m.root_velocity[0] = 0.3
                self.tick(c, g, m, 4)
                self.assertEqual(c.phase, StepPhase.OBSERVE)
                m.sole_positions[0, 2] = direction*0.11
                self.tick(c, g, m)
                self.assertEqual(c.phase, StepPhase.RECOVER)
                self.assertEqual(c.failure_reason, 'unexpected_support_height_before_lock')
                self.assertTrue(c.unsafe_contact_event)
                self.assertFalse(c.success_event)
                self.assertIsNone(c.lock)

    def test_source_toe_contact_or_airborne_foot_is_not_a_new_support_level(self):
        c, g, m = self.make_case()
        m.root_velocity[0] = 0.3
        self.tick(c, g, m, 4)
        angle = np.deg2rad(40.)
        m.foot_rotations[0] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        m.sole_positions[0, 2] = 0.12*np.sin(angle)
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        m.sole_positions[0, 2] = 0.17
        m.contact_forces[0, 2] = 0.
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.OBSERVE)

    def test_repeated_camera_frame_does_not_confirm_detection(self):
        c, g, m = self.make_case()
        m.camera_timestamp = 0.
        self.tick(c, g, m, 10)
        self.assertEqual(c.phase, StepPhase.BLIND)
        self.assertEqual(c.detection_count, 1)

    def test_plans_up_and_down_from_observation_not_terrain_truth(self):
        for direction in (1, -1):
            c, g, m = self.make_case(direction)
            lock = c.plan(g, m)
            self.assertIsNotNone(lock)
            np.testing.assert_allclose(lock.targets[:, 2], direction*0.11, atol=0.001)
            self.assertGreaterEqual(lock.targets[0, 1]-lock.targets[1, 1], 0.16)

    def test_plan_keeps_non_grid_aligned_lateral_stance_when_tread_allows_it(self):
        for direction in (1, -1):
            c, g, m = self.make_case(direction)
            m.sole_positions[:, 1] = [.137, -.141]
            lock = c.plan(g, m)
            self.assertIsNotNone(lock)
            np.testing.assert_allclose(lock.targets[:, 1], m.sole_positions[:, 1], atol=1.e-12)

    def test_nearest_unusable_level_cannot_be_skipped(self):
        c, g, m = self.make_case()
        nearest = min((s for s in g.surfaces if float(s.height_at(s.centroid[:2])) > -0.95),
                      key=lambda s: s.height_at(s.centroid[:2]))
        nearest.safe_center_mask[:] = False
        self.assertIsNone(c.plan(g, m))

    def test_heading_must_be_aligned_before_locking(self):
        c, g, m = self.make_case()
        g.heading_rad = np.deg2rad(8)
        self.assertIsNone(c.plan(g, m))

    def test_observe_turns_toward_observed_heading_without_forward_motion(self):
        for angle in (-20., 20.):
            with self.subTest(angle=angle):
                c, g, m = self.make_case()
                g.heading_rad = np.deg2rad(angle)
                self.tick(c, g, m, 4)
                self.assertEqual(c.phase, StepPhase.OBSERVE)
                self.assertIsNone(c.lock)
                command = c.walking_command(m)
                np.testing.assert_array_equal(command[:2], [0., 0.])
                self.assertAlmostEqual(command[2], np.sign(angle)*c.cfg.alignment_max_yaw_rate)
                self.assertFalse(c.observe_requirements['heading_aligned'])
                self.assertEqual(c.reward_metrics(m, 0.02)['stop_angular_motion'], 0.)

    def test_alignment_compensates_robot_yaw_without_forcing_a_target_lock(self):
        c, g, m = self.make_case()
        g.heading_rad = np.deg2rad(10.)
        self.tick(c, g, m, 4)
        angle = np.deg2rad(6.)
        m.yaw_rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        self.assertAlmostEqual(np.rad2deg(c.alignment_heading_error(m)), 4.)
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])
        g.heading_rad = np.deg2rad(4.)
        with patch.object(c, 'plan', return_value=None):
            self.tick(c, g, m, 2)
        self.assertTrue(c.observe_requirements['heading_aligned'])
        # Alignment alone cannot create a plan when coverage is unavailable.
        self.assertFalse(c.observe_requirements['target'])
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        self.assertIsNone(c.lock)
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])

    def test_alignment_requires_fresh_heading_and_actual_support_feedback(self):
        c, g, m = self.make_case()
        g.heading_rad = np.deg2rad(12.)
        self.tick(c, g, m, 4)
        m.contact_forces = None
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])
        self.assertEqual(c.alignment_block_reason, 'load_feedback_unavailable')
        m.contact_forces = np.array([[0., 0., 500.], [0., 0., 500.]])
        m.timestamp += c.cfg.target_max_age_s+0.01
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])
        self.assertEqual(c.alignment_block_reason, 'heading_observation_unavailable')
        self.assertEqual(c.reward_metrics(m, 0.02)['heading'], 0.)

    def test_alignment_is_blocked_on_split_levels_or_rapid_body_motion(self):
        c, g, m = self.make_case()
        g.heading_rad = np.deg2rad(12.)
        self.tick(c, g, m, 4)
        m.sole_positions[0, 2] = 0.11
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])
        self.assertEqual(c.alignment_block_reason, 'feet_not_on_same_level')
        m.sole_positions[0, 2] = 0.
        m.root_velocity[0] = 0.2
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])
        self.assertEqual(c.alignment_block_reason, 'body_motion_too_fast')

    def test_alignment_does_not_act_in_blind_or_locked_execution(self):
        c, g, m = self.make_case()
        g.heading_rad = np.deg2rad(12.)
        self.tick(c, g, m)
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])
        c, g, m = self.locked()
        g.heading_rad = np.deg2rad(12.)
        self.tick(c, g, m)
        np.testing.assert_array_equal(c.walking_command(m), [0., 0., 0.])

    def test_stair_actor_remains_active_while_aligning_before_lock(self):
        c, g, m = self.make_case()
        self.assertEqual(c.action_gate_target(m), 0.)
        g.heading_rad = np.deg2rad(12.)
        self.tick(c, g, m, 4)
        self.assertEqual(c.action_gate_target(m), 1.)
        g.heading_rad = 0.
        self.tick(c, g, m)
        self.assertEqual(c.action_gate_target(m), 1.)
        c, g, m = self.locked(-1)
        self.assertEqual(c.action_gate_target(m), -1.)

    def test_expired_target_cannot_be_locked(self):
        c, g, m = self.make_case()
        m.timestamp = 1.
        self.assertIsNone(c.plan(g, m))

    def test_changed_preview_requires_a_new_confirmation_window(self):
        c, g, m = self.make_case()
        self.tick(c, g, m, 4)
        self.assertAlmostEqual(c.confirmed, 0.02)
        changed = copy.deepcopy(c.preview)
        changed.targets[:, 1] += 0.04
        with patch.object(c, 'plan', return_value=changed):
            self.tick(c, g, m)
            self.assertEqual(c.phase, StepPhase.OBSERVE)
            self.assertFalse(c.observe_requirements['target_consistent'])
            self.assertEqual(c.confirmed, 0.)
            self.tick(c, g, m, 2)
        self.assertEqual(c.phase, StepPhase.SHIFT_LEAD)
        np.testing.assert_array_equal(c.lock.targets, changed.targets)

    def test_same_position_on_a_different_track_does_not_inherit_confirmation(self):
        c, g, m = self.make_case()
        self.tick(c, g, m, 4)
        changed = copy.deepcopy(c.preview)
        changed.track_id += 100
        with patch.object(c, 'plan', return_value=changed):
            self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        self.assertEqual(c.confirmed, 0.)
        self.assertIsNone(c.lock)

    def test_preview_drift_is_measured_from_confirmation_start_not_last_frame(self):
        c, g, m = self.make_case()
        c.cfg.confirmation_s = 0.08
        self.tick(c, g, m, 4)
        first = copy.deepcopy(c.preview)
        for shift in (0.015, 0.030):
            changed = copy.deepcopy(first)
            changed.targets[:, 1] += shift
            with patch.object(c, 'plan', return_value=changed):
                self.tick(c, g, m)
        self.assertEqual(c.confirmed, 0.)
        self.assertIsNone(c.lock)

    def test_small_preview_jitter_does_not_restart_stable_confirmation(self):
        c, g, m = self.make_case()
        self.tick(c, g, m, 4)
        changed = copy.deepcopy(c.preview)
        changed.targets[:, 1] += 0.005
        with patch.object(c, 'plan', return_value=changed):
            self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.SHIFT_LEAD)

    def test_replanning_uses_same_world_tread_after_translation_yaw_and_body_bob(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                c, g, m = self.make_case(direction)
                x, y = np.meshgrid(np.arange(-0.34, 0.95, 0.006), np.arange(-0.34, 0.34, 0.006))
                z = np.where(x < 0.22, -1., np.where(x < 0.54, -1.+direction*0.11, -1.+direction*0.22))
                points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
                world = points + m.root_position
                memory = SurfaceMemory(g.cfg)
                first = memory.update(points, g, m.root_position, m.yaw_rotation, m.timestamp)
                before = c.plan(first, m)
                self.assertIsNotNone(before)
                angle = np.deg2rad(3.)
                m.yaw_rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                           [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
                m.root_position += [0.04, 0.0, 0.03]
                m.timestamp = 0.04
                moved_points = (world-m.root_position) @ m.yaw_rotation
                snapshot = memory.extractor.extract(moved_points)
                moved = memory.update(moved_points, snapshot, m.root_position, m.yaw_rotation, m.timestamp)
                after = c.plan(moved, m)
                self.assertIsNotNone(after)
                self.assertEqual(before.track_id, after.track_id)
                self.assertLess(abs(before.heading-after.heading), np.deg2rad(0.5))
                self.assertLess(np.linalg.norm(after.targets-before.targets, axis=1).max(), 0.05)
                body_targets = (after.targets-m.root_position) @ m.yaw_rotation
                self.assertTrue(all(moved.footprint_supported(after.surface_id, p[:2], moved.heading_rad)
                                    for p in body_targets))
                np.testing.assert_allclose(after.targets[:, 2], direction*0.11, atol=0.001)
                self.assertGreater(np.linalg.norm(after.root_position-before.root_position), 0.04)

    def test_motion_makes_locked_target_unreachable_and_replans_before_first_lift(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                c, g, m = self.locked(direction)
                m.sole_positions[:, 0] -= 0.25
                m.root_position[0] -= 0.25
                c.update(None, m, 0.02)
                self.assertEqual(c.phase, StepPhase.OBSERVE)
                self.assertEqual(c.replan_reason, 'target_unreachable_before_lift')
                self.assertIsNone(c.lock)
                self.assertFalse(c.progress_event)

    def test_partial_unloading_blocks_unreachable_lift_without_retargeting(self):
        c, g, m = self.locked()
        targets = c.lock.targets.copy()
        m.sole_positions[:, 0] -= 0.25
        m.root_position[0] -= 0.25
        m.contact_forces[c.lead, 2] = 0.
        m.contact_forces[1-c.lead, 2] = 1000.
        for _ in range(4):
            c.update(None, m, 0.02)
        self.assertEqual(c.phase, StepPhase.SHIFT_LEAD)
        self.assertFalse(c.shift_requirements['target_reachable'])
        self.assertEqual(c.replan_reason, '')
        np.testing.assert_array_equal(c.lock.targets, targets)

    def test_stale_target_replans_only_with_stable_double_support(self):
        c, g, m = self.locked()
        m.timestamp += 0.30
        c.update(None, m, 0.02)
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        self.assertEqual(c.replan_reason, 'target_stale_before_lift')
        self.assertIsNone(c.lock)

    def test_trailing_foot_cannot_switch_tread_when_target_becomes_unreachable(self):
        c, g, m = self.locked()
        targets = c.lock.targets.copy()
        m.sole_positions[c.lead] = targets[c.lead]
        m.sole_positions[1-c.lead, 0] -= 0.25
        c._transition(StepPhase.SHIFT_TRAIL, m)
        c.update(None, m, 0.02)
        self.assertEqual(c.phase, StepPhase.SHIFT_TRAIL)
        self.assertFalse(c.shift_requirements['target_reachable'])
        self.assertEqual(c.replan_reason, '')
        np.testing.assert_array_equal(c.lock.targets, targets)

    def test_moving_robot_updates_target_error_without_moving_world_swing_goal(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                c, g, m = self.locked(direction)
                m.contact_forces[c.lead, 2] = 0.
                m.contact_forces[1-c.lead, 2] = 1000.
                self.tick(c, g, m, 2)
                self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
                targets = c.lock.targets.copy()
                old_error = c.features(m)[12:18].copy()
                angle = np.deg2rad(4.)
                m.yaw_rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                           [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
                m.root_position += [0.06, 0.01, 0.03]
                m.sole_positions[:, 0] += 0.03
                c.update(None, m, 0.02)
                np.testing.assert_array_equal(c.lock.targets, targets)
                expected = ((targets-m.sole_positions) @ m.yaw_rotation).ravel()
                np.testing.assert_allclose(c.features(m)[12:18], expected, atol=1.e-7)
                self.assertFalse(np.allclose(c.features(m)[12:18], old_error))
                self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
                self.assertEqual(c.replan_reason, '')

    def test_target_does_not_change_or_release_on_first_contact(self):
        c, g, m = self.locked()
        targets = c.lock.targets.copy()
        self.tick(c, g, m, 3)
        self.assertEqual(c.phase, StepPhase.SHIFT_LEAD)
        np.testing.assert_array_equal(c.lock.targets, targets)
        self.assertIsNotNone(c.lock)

    def test_timeout_fails_instead_of_advancing(self):
        c, g, m = self.locked()
        c.elapsed = 6.1
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.RECOVER)
        self.assertFalse(c.success_event)
        self.assertEqual(c.failure_reason, 'phase_timeout')

    def test_epoch_change_invalidates_locked_target(self):
        c, g, m = self.locked()
        m.generation += 1
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.RECOVER)

    def test_edge_contact_and_sliding_do_not_count_as_support(self):
        c, g, m = self.locked()
        m.sole_positions = c.lock.targets.copy()
        self.assertTrue(c._physical_support(m, 0))
        m.sole_positions[1, 0] = 0.25
        self.assertFalse(c._physical_support(m, 1))
        m.sole_positions[0, 0] += 0.02
        self.assertFalse(c._physical_support(m, 0))

    def test_region_support_accepts_off_reference_landing_but_not_wrong_height_or_unknown_cells(self):
        c, g, m = self.locked()
        m.sole_positions = c.lock.targets.copy()
        m.sole_positions[:, 1] += .04
        self.assertTrue(c._physical_support(m, 0, record_plant=False))
        m.sole_positions[0, 2] += .05
        self.assertFalse(c._physical_support(m, 0, record_plant=False))
        m.sole_positions[0, 2] -= .05
        geometry = c.lock.geometry
        position = (m.sole_positions[0]-c.lock.root_position) @ c.lock.rotation
        distances = np.linalg.norm(geometry.grid_xy-position[:2], axis=-1)
        cell = np.unravel_index(np.argmin(distances), distances.shape)
        geometry.surfaces[c.lock.surface_id].observed_mask[cell] = False
        self.assertFalse(c._physical_support(m, 0, record_plant=False))

    def test_region_support_preserves_left_right_separation(self):
        c, g, m = self.locked()
        m.sole_positions = c.lock.targets.copy()
        m.sole_positions[:, 1] = [.05, -.05]
        self.assertFalse(c._physical_support(m, 0))
        self.assertFalse(c._physical_support(m, 1))

    def test_confirmed_plant_anchors_body_and_foot_at_actual_landing(self):
        c, g, m = self.locked()
        lead = c.lead
        m.sole_positions[lead] = c.lock.targets[lead]+[0., .04, 0.]
        self.assertTrue(c._physical_support(m, lead))
        c.confirmed_plants[lead] = True
        actual = m.sole_positions[lead].copy()
        c._transition(StepPhase.SHIFT_TRAIL, m)
        np.testing.assert_allclose(c.reference_feet[lead], actual)
        m.hip_offsets = np.array([[0., .136, -.3825], [0., -.136, -.3825]])
        m.com_offset = np.array([.02, .04, -.28])
        c.features(m)
        np.testing.assert_allclose(c.reference_root[:2]+m.com_offset[:2], actual[:2])
        np.testing.assert_allclose(c.features(m)[12:18].reshape(2, 3)[lead], 0.)
        # The support reference must not follow later slip.
        m.sole_positions[lead, 1] += .02
        np.testing.assert_allclose(c.execution_targets()[lead], actual)
        self.assertFalse(c._physical_support(m, lead))

    def test_low_force_or_tilt_is_not_valid_support(self):
        c, g, m = self.locked()
        m.sole_positions = c.lock.targets.copy()
        m.contact_forces[0, 2] = 5.
        self.assertFalse(c._physical_support(m, 0))
        m.contact_forces[0, 2] = 500.
        angle = np.deg2rad(10)
        m.foot_rotations[0] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        self.assertFalse(c._physical_support(m, 0))

    def test_support_metrics_classify_failures_without_recording_a_plant(self):
        c, _, initial = self.locked()
        initial.sole_positions = c.lock.targets.copy()
        valid = c.physical_support_metrics(initial, 0)
        self.assertTrue(valid["valid"])
        self.assertIsNone(c.planted_positions[0])
        json.dumps(valid, allow_nan=False)
        cases = {
            "region": lambda m: m.sole_positions.__setitem__((0, 0), .25),
            "plane": lambda m: m.sole_positions.__setitem__((0, 2), initial.sole_positions[0, 2]+.021),
            "load": lambda m: m.contact_forces.__setitem__((0, 2), .119*m.body_weight),
            "force_direction": lambda m: m.contact_forces.__setitem__((0, 0), 600.),
            "speed": lambda m: m.sole_velocities.__setitem__((0, 0), .081),
            "tilt": lambda m: m.foot_rotations.__setitem__(0,
                [[math.cos(.13), 0, math.sin(.13)], [0, 1, 0], [-math.sin(.13), 0, math.cos(.13)]]),
        }
        for reason, change in cases.items():
            with self.subTest(reason=reason):
                measured = copy.deepcopy(initial)
                change(measured)
                support = c.physical_support_metrics(measured, 0)
                self.assertEqual(support["failed_conditions"], [reason])
                self.assertFalse(c._physical_support(measured, 0))
                self.assertIsNone(c.planted_positions[0])
                json.dumps(support, allow_nan=False)
        # Slip compares with the fixed recorded plant, not the moving reference.
        c.planted_positions[0] = initial.sole_positions[0]-[0., .016, 0.]
        plant = c.planted_positions[0].copy()
        self.assertEqual(c.physical_support_metrics(initial, 0)["failed_conditions"], ["slip"])
        np.testing.assert_array_equal(c.planted_positions[0], plant)
        c.planted_positions[0] = None
        self.assertTrue(c._physical_support(initial, 0, record_plant=False))
        self.assertIsNone(c.planted_positions[0])
        self.assertTrue(c._physical_support(initial, 0))
        np.testing.assert_array_equal(c.planted_positions[0], initial.sole_positions[0])

    def test_support_metrics_preserve_legacy_predicate_at_boundaries_and_mixed_states(self):
        c, _, initial = self.locked()
        initial.sole_positions = c.lock.targets.copy()

        def legacy(measured):
            # Frozen pre-extraction acceptance expression; no diagnostic API.
            if measured.contact_forces is None or c.lock is None:
                return False
            lock = c.lock
            surface = lock.geometry.surfaces[lock.surface_id]
            position = (measured.sole_positions[0]-lock.root_position) @ lock.rotation
            rotation = lock.rotation.T @ measured.foot_rotations[0]
            corners = np.array([[-.12, -.042, 0], [.12, -.042, 0],
                                [.12, .042, 0], [-.12, .042, 0]]) @ rotation.T+position
            plane_error = np.abs(corners @ surface.normal+surface.offset).max()
            tilt = math.acos(float(np.clip(rotation[:, 2] @ surface.normal, -1, 1)))
            force, planted = measured.contact_forces[0], c.planted_positions[0]
            loaded = force[2] >= .12*measured.body_weight and force[2] >= .75*np.linalg.norm(force)
            slip = planted is not None and np.linalg.norm(measured.sole_positions[0, :2]-planted[:2]) > c.cfg.max_slip
            return (c._footprint_in_tread(measured, 0) and plane_error <= c.cfg.height_tolerance
                    and loaded and np.linalg.norm(measured.sole_velocities[0]) <= c.cfg.max_sole_speed
                    and tilt <= c.cfg.foot_tilt_tolerance_rad and not slip)

        rng = np.random.default_rng(4)
        for i in range(80):
            measured = copy.deepcopy(initial)
            if i < 8:
                # Inclusive boundaries, then the immediately adjacent float.
                limit = (.12*measured.body_weight if i < 2 else c.cfg.max_sole_speed if i < 4
                         else c.cfg.height_tolerance if i < 6 else c.cfg.max_slip)
                value = limit if i % 2 == 0 else np.nextafter(limit, np.inf)
                if i < 2:
                    measured.contact_forces[0, 2] = value
                elif i < 4:
                    measured.sole_velocities[0, 0] = value
                elif i < 6:
                    measured.sole_positions[0, 2] += value
                else:
                    c.planted_positions[0] = measured.sole_positions[0]-[0., value, 0.]
            else:
                c.planted_positions[0] = initial.sole_positions[0].copy() if i % 2 else None
                measured.sole_positions[0] += rng.uniform(-.025, .025, 3)
                measured.sole_velocities[0] = rng.uniform(-.07, .07, 3)
                measured.contact_forces[0] = rng.uniform([0., 0., 100.], [200., 200., 500.])
                angle = rng.uniform(-.15, .15)
                measured.foot_rotations[0] = [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                                             [-math.sin(angle), 0, math.cos(angle)]]
            self.assertEqual(c.physical_support_metrics(measured, 0)["valid"], bool(legacy(measured)))
            self.assertEqual(c._physical_support(measured, 0, record_plant=False), bool(legacy(measured)))
        measured.contact_forces = None
        self.assertFalse(c.physical_support_metrics(measured, 0)["valid"])
        self.assertFalse(c._physical_support(measured, 0))

    def test_lower_reference_speed_is_bounded(self):
        c, g, m = self.locked(-1)
        m.contact_forces[c.lead, 2] = 0
        m.contact_forces[1-c.lead, 2] = 1000
        self.tick(c, g, m, 2)
        self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
        m.sole_positions[c.lead] = c.lock.targets[c.lead]+[0, 0, 0.17]
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.LOWER_LEAD)
        previous = c.reference_feet[c.lead, 2]
        for _ in range(100):
            self.tick(c, g, m)
            current = c.reference_feet[c.lead, 2]
            self.assertLessEqual(abs(current-previous)/0.02, 0.201)
            previous = current
        self.assertFalse(c.success_event)

    def test_region_descent_waits_for_horizontal_motion_to_slow(self):
        c, g, m = self.locked()
        swing = c.lead
        m.contact_forces[swing, 2] = 0
        m.contact_forces[1-swing, 2] = 1000
        self.tick(c, g, m, 2)
        self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
        m.sole_positions[swing] = c.lock.targets[swing]+[0., .04, .06]
        m.sole_velocities[swing, 0] = .20
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
        self.assertIsNone(c.descent_targets[swing])
        m.sole_velocities[swing] = 0.
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.LOWER_LEAD)
        np.testing.assert_allclose(c.execution_targets()[swing, :2], m.sole_positions[swing, :2])

    def test_committed_landing_is_not_refreshed_by_observing_only_nominal_reference(self):
        c, g, m = self.locked()
        swing = c.lead
        m.contact_forces[swing, 2] = 0
        m.contact_forces[1-swing, 2] = 1000
        self.tick(c, g, m, 2)
        m.sole_positions[swing] = c.lock.targets[swing]+[0., .04, .06]
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.LOWER_LEAD)
        target = (c.execution_targets()[swing]-m.root_position) @ m.yaw_rotation
        # Hide a lateral strip under the committed sole, outside its old reference.
        surface = g.surfaces[c.lock.surface_id]
        surface.observed_mask[g.grid_xy[..., 1] > target[1]+.05] = False
        nominal = (c.lock.targets[swing]-m.root_position) @ m.yaw_rotation
        self.assertTrue(g.footprint_supported(surface.surface_id, nominal[:2]))
        self.assertFalse(g.footprint_supported(surface.surface_id, target[:2]))
        last_seen = c.lock.last_seen[swing]
        self.tick(c, g, m)
        self.assertEqual(c.lock.last_seen[swing], last_seen)
        # A short occlusion is allowed, but it cannot remain fresh forever.
        m.timestamp = last_seen+c.cfg.flight_occlusion_s+.01
        c.update(g, m, .02)
        self.assertEqual(c.phase, StepPhase.RECOVER)
        self.assertEqual(c.failure_reason, 'swing_target_expired')

    def test_horizontal_crossing_waits_for_measured_sole_clearance(self):
        c, g, m = self.locked()
        c.cfg.overlap_swing_lift = True
        m.contact_forces[c.lead, 2] = 0
        m.contact_forces[1-c.lead, 2] = 1000
        self.tick(c, g, m, 2)
        start_xy = m.sole_positions[c.lead, :2].copy()
        self.tick(c, g, m, 30)
        self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
        self.assertFalse(c.lift_clearance_confirmed)
        # Move forward while lifting, but the whole sole must remain before the
        # riser until measured clearance is available.
        self.assertGreater(c.reference_feet[c.lead, 0], start_xy[0])
        local = (c.reference_feet[c.lead]-c.lock.root_position) @ c.lock.rotation
        surface = c.lock.geometry.surfaces[c.lock.surface_id]
        self.assertLess(local[0]+.12, surface.near_edge.x_at(local[1]))
        m.sole_positions[c.lead, 2] = c.lock.targets[c.lead, 2]+.021
        self.tick(c, g, m, 2)
        self.assertTrue(c.lift_clearance_confirmed)
        self.assertGreater(c.reference_feet[c.lead, 0], start_xy[0])

    def test_interior_support_anchor_does_not_follow_swing_or_slipping_support(self):
        c, g, m = self.locked()
        c.cfg.support_com_half_length_m = .045
        c.cfg.support_com_half_width_m = .010
        lead = c.lead
        m.sole_positions[lead] = c.lock.targets[lead]
        self.assertTrue(c._physical_support(m, lead))
        c.confirmed_plants[lead] = True
        c._transition(StepPhase.SHIFT_TRAIL, m)
        anchor = c._stance_com_anchor(lead)
        offset = anchor-c.planted_positions[lead][:2]
        self.assertAlmostEqual(offset[0], -.045)
        self.assertAlmostEqual(abs(offset[1]), .010)
        self.assertLess(abs(offset[0]), .080)
        self.assertLess(abs(offset[1]), .018)
        c.reference_feet[1-lead] += [.3, .02, .17]
        m.sole_positions[lead, :2] += [.01, .01]
        np.testing.assert_array_equal(c._stance_com_anchor(lead), anchor)

    def test_high_sole_center_with_low_toe_cannot_start_crossing(self):
        c, g, m = self.locked()
        c.cfg.overlap_swing_lift = True
        swing = c.lead
        m.contact_forces[swing, 2] = 0.
        m.contact_forces[1-swing, 2] = 1000.
        self.tick(c, g, m, 2)
        m.sole_positions[swing, 2] = c.lock.targets[swing, 2]+.025
        angle = np.deg2rad(15.)
        m.foot_rotations[swing] = [[np.cos(angle), 0., np.sin(angle)],
                                   [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
        self.tick(c, g, m, 20)
        self.assertFalse(c.lift_clearance_confirmed)
        local = (c.reference_feet[swing]-c.lock.root_position) @ c.lock.rotation
        surface = c.lock.geometry.surfaces[c.lock.surface_id]
        self.assertLess(local[0]+.12, surface.near_edge.x_at(local[1]))

    def test_lift_progress_rewards_only_new_measured_whole_sole_height(self):
        c, g, m = self.locked()
        m.contact_forces[c.lead, 2] = 0
        m.contact_forces[1-c.lead, 2] = 1000
        self.tick(c, g, m, 2)
        m.sole_positions[c.lead, 2] = .02
        self.tick(c, g, m)
        first = c.reward_metrics(m, .02)['lift_progress']
        self.assertGreater(first, 0.)
        self.tick(c, g, m)
        self.assertEqual(c.reward_metrics(m, .02)['lift_progress'], 0.)
        m.sole_positions[c.lead, 2] = .01
        self.tick(c, g, m)
        m.sole_positions[c.lead, 2] = .02
        self.tick(c, g, m)
        self.assertEqual(c.reward_metrics(m, .02)['lift_progress'], 0.)
        self.assertFalse(c.lift_clearance_confirmed)

    def test_shift_requires_stable_body_not_only_transient_load_transfer(self):
        c, g, m = self.locked()
        m.contact_forces[c.lead, 2] = 0
        m.contact_forces[1-c.lead, 2] = 1000
        m.root_velocity[1] = .25
        self.tick(c, g, m, 10)
        self.assertEqual(c.phase, StepPhase.SHIFT_LEAD)
        self.assertFalse(c.shift_requirements['body_stable'])
        m.root_velocity[:] = 0
        self.tick(c, g, m, 2)
        self.assertEqual(c.phase, StepPhase.LIFT_LEAD)

    def test_body_damping_does_not_penalize_commanded_swing_foot_motion(self):
        c, g, m = self.locked()
        m.contact_forces[c.lead, 2] = 0
        m.contact_forces[1-c.lead, 2] = 1000
        self.tick(c, g, m, 2)
        m.root_velocity[0] = .1
        m.sole_velocities[c.lead, 2] = .1
        metrics = c.reward_metrics(m, .02)
        self.assertGreater(metrics['stop_root_motion'], 0.)
        self.assertEqual(metrics['stop_feet_motion'], 0.)

    def test_synthetic_up_and_down_sequence_requires_both_feet_and_emits_once(self):
        # Measured states are constructed here: this is not a physics success test.
        for direction, lateral_offset in ((1, 0.), (-1, 0.), (1, .04), (-1, .04)):
            c, g, m = self.locked(direction)
            nominal = c.lock.targets.copy()
            target = nominal+[0., lateral_offset, 0.]
            lead, trail = c.lead, 1-c.lead
            m.contact_forces[:, 2] = [0, 1000] if lead == 0 else [1000, 0]
            self.tick(c, g, m, 2)
            self.assertEqual(c.phase, StepPhase.LIFT_LEAD)
            m.sole_positions[lead] = target[lead]+[0, 0, max(0, -direction*0.11)+0.06]
            self.tick(c, g, m)
            self.assertEqual(c.phase, StepPhase.LOWER_LEAD)
            m.sole_positions[lead] = target[lead]
            m.contact_forces[:, 2] = 500
            self.tick(c, g, m, 2)
            self.assertEqual(c.phase, StepPhase.TRANSFER)
            m.contact_forces[lead, 2], m.contact_forces[trail, 2] = 800, 200
            self.tick(c, g, m, 2)
            self.assertEqual(c.phase, StepPhase.SHIFT_TRAIL)
            m.contact_forces[lead, 2], m.contact_forces[trail, 2] = 1000, 0
            self.tick(c, g, m, 2)
            self.assertEqual(c.phase, StepPhase.LIFT_TRAIL)
            m.sole_positions[trail] = target[trail]+[0, 0, max(0, -direction*0.11)+0.06]
            self.tick(c, g, m)
            self.assertEqual(c.phase, StepPhase.LOWER_TRAIL)
            m.sole_positions[trail] = target[trail]
            m.contact_forces[:, 2] = 500
            self.tick(c, g, m, 2)
            self.assertEqual(c.phase, StepPhase.SETTLE)
            self.tick(c, g, m, 4)
            self.assertEqual(c.phase, StepPhase.COMPLETE)
            self.assertTrue(c.success_event)
            np.testing.assert_array_equal(c.lock.targets, nominal)
            np.testing.assert_allclose(c.execution_targets(), target)
            np.testing.assert_allclose(c.reference_feet, target)
            self.tick(c, g, m)
            self.assertFalse(c.success_event)

    def test_completion_rejects_heading_error(self):
        c, g, m = self.locked()
        c.phase = StepPhase.SETTLE
        m.sole_positions = c.lock.targets.copy()
        angle = np.deg2rad(9)
        m.yaw_rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        # Prevent reprojection against the intentionally unmoved synthetic map.
        for _ in range(10):
            c.update(None, m, 0.02)
        self.assertEqual(c.phase, StepPhase.SETTLE)
        self.assertFalse(c.success_event)

    def test_intermittent_first_touch_does_not_confirm_or_lose_established_support(self):
        c, g, m = self.locked()
        c._transition(StepPhase.LOWER_LEAD, m)
        lead, stance = c.lead, 1-c.lead
        m.sole_positions[lead] = c.lock.targets[lead]
        m.contact_forces[lead, 2], m.contact_forces[stance, 2] = 150, 850
        self.tick(c, g, m)
        self.assertFalse(c.confirmed_plants[lead])
        m.contact_forces[lead, 2] = 5
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.LOWER_LEAD)
        self.assertEqual(c.confirmed, 0)
        m.contact_forces[lead, 2] = 150
        self.tick(c, g, m, 2)
        self.assertEqual(c.phase, StepPhase.TRANSFER)
        self.assertTrue(c.confirmed_plants[lead])

    def test_features_have_fixed_dimension_and_reset_clears_lock(self):
        c, g, m = self.locked()
        self.assertEqual(len(c.features(m)), c.num_features)
        self.assertTrue(np.isfinite(c.features(m)).all())
        c.reset()
        self.assertIsNone(c.lock)
        self.assertFalse(c.active)

    def test_observe_exposes_provisional_geometry_without_authorizing_a_step(self):
        c, g, m = self.make_case()
        self.tick(c, g, m, 3)
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        m.root_velocity[0] = 0.5
        self.tick(c, g, m)
        self.assertIsNone(c.lock)
        self.assertIsNotNone(c.preview)
        features = c.features(m)
        self.assertGreater(np.linalg.norm(features[12:18]), 0)
        self.assertEqual(features[30], 1)
        self.assertEqual(c.phase, StepPhase.OBSERVE)

    def test_missing_force_does_not_mean_unloaded_or_authorize_a_step(self):
        c, g, m = self.make_case()
        m.contact_forces = None
        self.tick(c, g, m, 20)
        self.assertEqual(c.phase, StepPhase.OBSERVE)
        self.assertIsNone(c.lock)
        self.assertIsNotNone(c.preview)
        self.assertEqual(c.support_block_reason, 'load_feedback_unavailable')
        self.assertFalse(c.success_event)
        self.assertEqual(c.reward_metrics(m, 0.02)['load_reference'], 0)
        self.assertTrue(np.isfinite(c.features(m)).all())

    def test_losing_load_feedback_mid_step_fails_without_advancing(self):
        c, g, m = self.locked()
        m.contact_forces = None
        self.tick(c, g, m)
        self.assertEqual(c.phase, StepPhase.RECOVER)
        self.assertEqual(c.failure_reason, 'load_feedback_unavailable')

    def test_actor_does_not_receive_contact_force_or_force_qualified_support(self):
        c, g, m = self.locked()
        c.support_valid[:] = True
        teacher = c.features(m, privileged=True)
        actor = c.features(m)
        np.testing.assert_array_equal(actor[24:26], [0, 0])
        np.testing.assert_array_equal(actor[34:36], [0, 0])
        self.assertTrue((teacher[24:26] > 0).all())
        np.testing.assert_array_equal(teacher[34:36], [1, 1])

    def test_stop_rewards_penalize_motion_without_penalizing_planned_swing(self):
        c, g, m = self.make_case()
        c.phase = StepPhase.OBSERVE
        m.root_velocity[:] = [0.2, 0, 0]
        m.root_angular_velocity[:] = [0, 0, 0.4]
        m.sole_velocities[:] = 0.1
        metrics = c.reward_metrics(m, 0.02)
        self.assertGreater(metrics['stop_root_motion'], 0)
        self.assertGreater(metrics['stop_feet_motion'], 0)
        self.assertGreater(metrics['stop_angular_motion'], 0)
        c.phase = StepPhase.LIFT_LEAD
        metrics = c.reward_metrics(m, 0.02)
        self.assertEqual(metrics['stop_feet_motion'], 0)
        self.assertGreater(metrics['stop_root_motion'], 0)


if __name__ == '__main__':
    unittest.main()
