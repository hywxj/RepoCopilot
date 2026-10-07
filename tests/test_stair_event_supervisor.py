"""Event/region acceptance regressions, separate from teacher motor execution."""

import copy
import json
import unittest

import numpy as np

from legged_lab.perception.stair_event_supervisor import EventDrivenStairEpisode, EventStairCfg, phase_clock


class PhaseClockTest(unittest.TestCase):
    def test_clock_stops_with_continuous_position_velocity_acceleration(self):
        duration, ramp = .8, .1
        self.assertEqual(phase_clock(0., duration, ramp), (0., 0., 0., False))
        self.assertEqual(phase_clock(4., duration, ramp), (duration, 0., 0., True))
        for boundary in (0., ramp, duration, duration+ramp):
            left = phase_clock(boundary-1.e-7, duration, ramp)
            right = phase_clock(boundary+1.e-7, duration, ramp)
            np.testing.assert_allclose(left[:3], right[:3], atol=3.e-5)
        for t in np.linspace(.01, .89, 40):
            position, velocity, acceleration, _ = phase_clock(t, duration, ramp)
            low, high = phase_clock(t-1.e-6, duration, ramp), phase_clock(t+1.e-6, duration, ramp)
            self.assertAlmostEqual((high[0]-low[0])/2.e-6, velocity, places=6)
            self.assertAlmostEqual((high[1]-low[1])/2.e-6, acceleration, places=5)
            self.assertGreaterEqual(velocity, 0.)
            self.assertLessEqual(position, duration)

    def test_finish_confirmation_cannot_disable_autonomous_hold(self):
        with self.assertRaises(ValueError):
            EventStairCfg(finish_hold_s=.5)
        with self.assertRaises(ValueError):
            EventStairCfg(stable_confirmation_s=.1)


class EventStairSupervisorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = EventDrivenStairEpisode()

    @classmethod
    def tearDownClass(cls):
        cls.template.close()

    def setUp(self):
        self.episode = copy.deepcopy(self.template, {id(self.template.model): self.template.model})
        self.measured = self.episode.measurement()
        self.measured.root_velocity[:] = 0.
        self.measured.root_angular_velocity[:] = 0.
        self.measured.sole_velocities[:] = 0.
        self.measured.foot_rotations[:] = np.eye(3)
        self.measured.contact_forces[:] = [0., 0., .5*self.measured.body_weight]

    def test_slow_unloading_waits_past_original_deadline(self):
        episode, measured = self.episode, self.measured
        measured.timestamp += 1.2  # Original teacher liftoff was at 0.6 seconds.
        episode._physical_events(measured, 1.2, clock_finished=True)
        self.assertEqual(episode.event_phase, 0)
        self.assertEqual(episode.event_count, 0)
        measured.contact_forces[:, 2] = measured.body_weight*np.array([.98, .02])
        measured.timestamp += .13
        episode._physical_events(measured, .13, clock_finished=True)
        self.assertEqual(episode.event_phase, 1)
        self.assertEqual(episode.last_event, "leading_unloaded")

    def test_light_region_contact_starts_transfer_before_load_confirmation(self):
        episode, measured = self.episode, self.measured
        episode.event_phase = 1
        episode.clearance_confirmed[1] = True
        measured.sole_positions[1] = episode.target[1]+[.005, 0., 0.]
        measured.contact_forces[:, 2] = measured.body_weight*np.array([.99, .01])
        measured.timestamp += 1.5
        self.assertTrue(episode.controller._physical_support(measured, 1,
                        min_load=episode.controller.cfg.touchdown_contact_fraction, record_plant=False))
        episode._physical_events(measured, .13, clock_finished=False)
        self.assertFalse(episode.confirmed[1])
        self.assertEqual(episode.event_phase, 1)  # Reference joins continuously.
        measured.timestamp += .01
        episode._physical_events(measured, .01, clock_finished=True)
        self.assertEqual(episode.event_phase, 2)
        np.testing.assert_allclose(episode.plants[1, :2], measured.sole_positions[1, :2])
        self.assertGreater(np.linalg.norm(episode.plants[1, :2]-episode.target[1, :2]), .004)
        expected_anchor = measured.sole_positions[1, :2]+episode.controller.lock.rotation[:2, :2] @ [-.045, .012]
        np.testing.assert_allclose(episode.support_com[1, :2], expected_anchor)
        np.testing.assert_allclose(episode.event_com_goal[:2], expected_anchor)

    def test_load_transfer_must_confirm_lead_before_trail_liftoff(self):
        episode, measured = self.episode, self.measured
        episode.event_phase = 2
        episode.touching[1] = True
        measured.sole_positions[1] = episode.target[1]
        measured.contact_forces[:, 2] = measured.body_weight*np.array([.02, .98])
        episode._physical_events(measured, .13, True, reference_loads=np.array([.95, .05]))
        self.assertFalse(episode.confirmed[1])  # A transient impact cannot confirm an unloaded reference.
        self.assertEqual(episode.event_phase, 2)
        episode._physical_events(measured, .13, True, reference_loads=np.array([.02, .98]))
        self.assertTrue(episode.confirmed[1])
        self.assertEqual(episode.event_phase, 3)

    def test_waitable_body_endpoints_are_in_stance_and_join_c2(self):
        episode = self.episode
        for phase, foot in ((0, 0), (2, 1)):
            point = episode.support_com[foot]
            sole = episode.source[0] if foot == 0 else episode.target[1]
            local = (point[:2]-sole[:2]) @ episode.controller.lock.rotation[:2, :2]
            self.assertLess(abs(local[0]), .080)
            self.assertLess(abs(local[1]), .018)
        for phase in range(4):
            if phase == 1:
                episode.plants[1] = episode.target[1]
            duration = episode.edges[phase+1]-episode.edges[phase]
            before = episode._reference_state(duration+1.)
            episode._begin_next_phase(self.measured)
            after = episode._reference_state(0.)
            for index in range(3):
                np.testing.assert_allclose(before[index], after[index], atol=1.e-12)
            np.testing.assert_array_equal(after[1], 0.)
            np.testing.assert_array_equal(after[2], 0.)

    def test_success_requires_continuous_confirmation_and_one_second_hold(self):
        episode, measured = self.episode, self.measured
        episode.event_phase = 4
        episode.confirmed[:] = True
        measured.sole_positions[:] = episode.target
        episode._physical_events(measured, .25, clock_finished=True)
        self.assertTrue(episode.step_completed)
        self.assertFalse(episode._event_success)
        episode._physical_events(measured, .99, clock_finished=True)
        self.assertFalse(episode._event_success)
        measured.root_velocity[0] = .06
        episode._physical_events(measured, .01, clock_finished=True)
        self.assertEqual(episode.stable_for, 0.)
        measured.root_velocity[0] = 0.
        episode._physical_events(measured, 1.25, clock_finished=True)
        self.assertTrue(episode._event_success)

    def test_original_support_load_is_not_relaxed(self):
        episode, measured = self.episode, self.measured
        episode.event_phase = 4
        episode.confirmed[:] = True
        measured.sole_positions[:] = episode.target
        measured.contact_forces[1, 2] = .11*measured.body_weight
        with self.assertRaisesRegex(RuntimeError, "Confirmed support lost"):
            episode._physical_events(measured, .01, clock_finished=True)
        self.assertEqual(episode.failure_diagnostics["support"]["failed_conditions"], ["load"])

    def test_confirmed_support_diagnostics_distinguish_speed_and_plane_from_load(self):
        for reason in ("speed", "plane"):
            with self.subTest(reason=reason):
                episode = copy.deepcopy(self.episode, {id(self.episode.model): self.episode.model})
                measured = copy.deepcopy(self.measured)
                episode.event_phase = 4
                episode.confirmed[:] = True
                measured.sole_positions[:] = episode.target
                if reason == "speed":
                    measured.sole_velocities[1, 0] = .089
                else:
                    measured.sole_positions[1, 2] += .021
                with self.assertRaisesRegex(RuntimeError, r"^Confirmed support lost on foot 1\.$"):
                    episode._physical_events(measured, .01, clock_finished=True)
                diagnostic = episode.failure_diagnostics
                self.assertEqual(diagnostic["foot"], 1)
                self.assertEqual(diagnostic["phase_index"], 4)
                self.assertEqual(diagnostic["support"]["failed_conditions"], [reason])
                self.assertTrue(diagnostic["support"]["conditions"]["load"])
                self.assertAlmostEqual(diagnostic["support"]["measurements"]["load_fraction"], .5)
                json.dumps(diagnostic, allow_nan=False)

    def test_whole_sole_clearance_still_requires_twenty_millimeters(self):
        episode, measured = self.episode, self.measured
        episode.event_phase = 1
        measured.contact_forces[:, 2] = measured.body_weight*np.array([.98, .02])
        measured.sole_positions[1] = [.22, episode.target[1, 1], episode.target[1, 2]+.019]
        with self.assertRaisesRegex(RuntimeError, "whole sole lacks stair-edge clearance"):
            episode._physical_events(measured, .01, clock_finished=False)
        measured.sole_positions[1, 2] = episode.target[1, 2]+.021
        episode._physical_events(measured, .01, clock_finished=False)
        self.assertTrue(episode.clearance_confirmed[1])

    def test_loaded_partial_foot_on_far_edge_is_a_real_failure(self):
        episode, measured = self.episode, self.measured
        episode.event_phase = 1
        measured.contact_forces[:, 2] = measured.body_weight*np.array([.8, .2])
        measured.sole_positions[1] = [.45, episode.target[1, 1], episode.target[1, 2]]
        self.assertFalse(episode.controller._footprint_in_tread(measured, 1))
        with self.assertRaisesRegex(RuntimeError, "Unsafe loaded contact"):
            episode._physical_events(measured, .01, clock_finished=False)

    def test_motor_execution_requires_callback_and_never_implicitly_calls_teacher(self):
        episode = self.episode
        with self.assertRaisesRegex(RuntimeError, "no teacher fallback"):
            episode._advance()
        def forbidden_teacher(*args, **kwargs):
            raise AssertionError("Motor teacher was called during explicit PD control")
        episode.teacher.control = forbidden_teacher
        time = episode.data.time
        sample, torque = episode._advance(lambda _, sample: np.zeros(episode.model.nu))
        self.assertGreater(episode.data.time, time)
        np.testing.assert_array_equal(torque, 0.)
        self.assertEqual(sample["reference_supervisor"], "event")


if __name__ == "__main__":
    unittest.main()
