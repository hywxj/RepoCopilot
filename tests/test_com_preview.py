"""Numerical preview invariants; these are not robot or contact-tracking tests."""
import unittest

import numpy as np

try:
    import qpsolvers
    from legged_lab.perception.com_preview import plan_com_preview
    from legged_lab.perception.stair_com_preview import plan_height_aware, quintic_height
    HAVE_SOLVER = "quadprog" in qpsolvers.available_solvers
except ImportError:
    HAVE_SOLVER = False


@unittest.skipUnless(HAVE_SOLVER, "NumPy, SciPy, qpsolvers and quadprog are required")
class COMPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = np.array([[.03, .136], [.03, -.136]])
        cls.target = np.array([[.38, .136], [.38, -.136]])
        cls.plan = plan_com_preview(cls.source, cls.target, initial_com=[.015, 0.], dt=.04)
        cls.source3 = np.column_stack((cls.source, [0., 0.]))
        cls.target3 = np.column_stack((cls.target, [.11, .11]))
        cls.initial3 = np.array([.01547, .000086, .74928])
        cls.height_plan = plan_height_aware(cls.source3, cls.target3,
                                           initial_com=cls.initial3, dt=.04)

    def test_linear_momentum_and_displacement_conservation(self):
        plan = self.plan
        dt = np.diff(plan.time)
        # Integrating force per unit mass must recover momentum change. For
        # constant acceleration, trapezoidal velocity integration is exact.
        np.testing.assert_allclose(np.diff(plan.velocity, axis=0),
                                   dt[:, None]*plan.acceleration, atol=1.e-12)
        integrated_displacement = np.sum(dt[:, None]*(plan.velocity[:-1]+plan.velocity[1:])/2, axis=0)
        np.testing.assert_allclose(integrated_displacement, plan.position[-1]-plan.position[0], atol=1.e-12)
        np.testing.assert_allclose(np.sum(dt[:, None]*plan.acceleration, axis=0), [0., 0.], atol=1.e-12)

    def test_sample_derivatives_and_terminal_hold(self):
        plan = self.plan
        for time in plan.time[2:-2:7]+.013:
            epsilon = 1.e-6
            p, v, a = plan.sample(time)
            before, vb, _ = plan.sample(time-epsilon)
            after, va, _ = plan.sample(time+epsilon)
            np.testing.assert_allclose((after-before)/(2*epsilon), v, atol=1.e-9)
            np.testing.assert_allclose((va-vb)/(2*epsilon), a, atol=1.e-9)
            self.assertTrue(np.isfinite(p).all())
        for time in (plan.time[-1], plan.time[-1]+3.):
            p, v, a = plan.sample(time)
            np.testing.assert_allclose(p, self.target.mean(axis=0), atol=1.e-12)
            np.testing.assert_array_equal(v, [0., 0.])
            np.testing.assert_array_equal(a, [0., 0.])
        p[:] = 100.
        self.assertLess(plan.position[-1, 0], 1.)

    def test_mixed_height_support_conserves_force_and_centroidal_moment(self):
        plan = self.height_plan
        mixed_count = 0
        for k, phase in enumerate(plan.planar.phase):
            # Different interior samples from the planner's own validation.
            for fraction in (.137, .613, .923):
                t = plan.time[k]+fraction*(plan.time[k+1]-plan.time[k])
                com, _, acceleration = plan.sample(t)
                wrench = plan.sample_wrenches(t)
                feet = plan.feet_for_phase(phase)
                total_force = wrench[:, :3].sum(axis=0)
                np.testing.assert_allclose(total_force, acceleration/9.81+[0., 0., 1.], atol=1.e-10)
                # Wrenches are about sole centers, so translate them to COM.
                net_moment = np.cross(feet-com, wrench[:, :3]).sum(axis=0)+wrench[:, 3:].sum(axis=0)
                np.testing.assert_allclose(net_moment, [0., 0., 0.], atol=1.e-10)
                self.assertTrue((wrench[:, 2] >= -1.e-10).all())
                self.assertTrue((np.abs(wrench[:, 0])+np.abs(wrench[:, 1]) <= .6*wrench[:, 2]+1.e-8).all())
                if phase == 2:
                    mixed_count += 1
                    self.assertGreater(np.ptp(feet[:, 2]), .10)
        self.assertGreater(mixed_count, 0)

    def test_unloads_swing_foot_before_liftoff_and_holds_final_state(self):
        plan = self.height_plan
        dt = plan.time[1]-plan.time[0]
        for edge, swing in ((1, 1), (3, 0)):
            t = plan.phase_edges[edge]-.35*dt
            k = np.searchsorted(plan.time, t, side="right")-1
            self.assertTrue(plan.planar.contacts[k].all())
            loads = plan.sample_loads(t)
            self.assertLessEqual(loads[swing], .04+1.e-8)
            self.assertGreater(loads[1-swing], .8)
        for edge, swing in ((1, 1), (3, 0)):
            loads = plan.sample_loads(plan.phase_edges[edge]+.35*dt)
            self.assertAlmostEqual(loads[swing], 0., places=12)
        for t in (plan.time[-1], plan.time[-1]+1.):
            p, v, a = plan.sample(t)
            np.testing.assert_allclose(p, [self.target[:, 0].mean(), 0., self.initial3[2]+.11], atol=1.e-12)
            np.testing.assert_array_equal(v, [0., 0., 0.])
            np.testing.assert_array_equal(a, [0., 0., 0.])
            self.assertAlmostEqual(plan.sample_loads(t).sum(), 1., places=12)

    def test_rejects_malformed_or_nonfinite_planar_inputs(self):
        invalid = [
            {"initial_com": [0.]}, {"initial_velocity": [0., 0., 0.]},
            {"final_com": [np.inf, 0.]}, {"dt": 0.}, {"dt": np.nan},
            {"phase_durations": [.5]*4}, {"phase_durations": [.5, .5, 0., .5, .5]},
            {"half_extents": [.06, 0.]}, {"com_height": -1.}, {"gravity": np.inf},
            {"max_velocity": -.1}, {"max_acceleration": np.nan}, {"max_jerk": 0.},
            {"position_weights": [1.]}, {"velocity_weights": [-1., 1.]},
            {"acceleration_weight": -.1}, {"jerk_weight": np.nan},
            {"unload_lead_time": -1.}, {"unload_lead_time": 2.},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_com_preview(self.source, self.target, **kwargs)
        with self.assertRaises(ValueError):
            plan_com_preview(self.source*np.nan, self.target)
        with self.assertRaises(ValueError):
            self.plan.sample(np.nan)

    def test_rejects_invalid_height_load_and_constraint_inputs(self):
        invalid = [
            {"initial_velocity": [0., 0.]}, {"dt": 0.}, {"phase_durations": [.5]*4},
            {"friction": 0.}, {"max_iterations": 0}, {"max_iterations": 1.5},
            {"unload_max_load_bw": -1.}, {"contact_min_load_bw": .6},
            {"height_profile": 1.}, {"height_profile": lambda t: [1., 0.]},
            {"height_profile": lambda t: [np.nan, 0., 0.]},
            {"height_profile": lambda t: [.75, 0., -9.81]},
            {"position_waypoints": [(0., [0., 0.], [-1., 1.])]},
            {"position_bounds": [(0., 2, -1., 1.)]},
            {"position_bounds": [(0., 0, 1., -1.)]},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_height_aware(self.source3, self.target3, initial_com=self.initial3,
                                  **{"dt": .04, **kwargs})
        for args in ((0., 1., 1., 1.), (0., np.nan, 0., 1.)):
            with self.assertRaises(ValueError):
                quintic_height(*args)
        with self.assertRaises(ValueError):
            self.height_plan.sample(np.inf)


if __name__ == "__main__":
    unittest.main()
