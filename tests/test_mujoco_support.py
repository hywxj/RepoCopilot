import unittest

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class MujocoSupportTest(unittest.TestCase):
    def make_case(self):
        from legged_lab.perception.mujoco_support import MujocoSupportReader
        # Two soles, a free root, and no foot-force/touch sensor elements.
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
            <worldbody><geom name="floor" type="plane" size="3 3 0.1"/>
              <body name="torso_link" pos="0 0 0.8"><freejoint/>
                <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                <body name="l_ankle_x_link" pos="0 0.15 -0.76">
                  <geom type="box" pos="0.03 0 -0.02" size="0.12 0.042 0.02" mass="0.5"/>
                </body>
                <body name="r_ankle_x_link" pos="0 -0.15 -0.76">
                  <geom type="box" pos="0.03 0 -0.02" size="0.12 0.042 0.02" mass="0.5"/>
                </body>
              </body>
            </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        reader = MujocoSupportReader(model)
        return model, data, reader

    def settle(self, model, data):
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

    def test_contact_truth_without_any_foot_sensors(self):
        model, data, reader = self.make_case()
        self.assertEqual(model.nsensor, 0)
        self.settle(model, data)
        force = reader.contact_forces_world(data)
        self.assertTrue((force[:, 2] > 0).all())
        np.testing.assert_allclose(force[:, :2], 0, atol=0.02)
        self.assertAlmostEqual(force[:, 2].sum(), reader.body_weight, delta=0.02)
        measured = reader.measurement(data, include_contact_truth=True)
        np.testing.assert_allclose(measured.sole_positions[:, 0], 0.03, atol=0.001)
        self.assertTrue(np.isfinite(measured.contact_forces).all())

    def test_no_truth_mode_preserves_unknown_load_not_zero(self):
        model, data, reader = self.make_case()
        self.settle(model, data)
        self.assertIsNone(reader.measurement(data).contact_forces)

    def test_point_velocity_matches_model_jacobian_not_link_com_velocity(self):
        model, data, reader = self.make_case()
        data.qvel[:] = [0.1, 0.2, 0.3, 0, 0, 0.4]
        mujoco.mj_forward(model, data)
        measured = reader.measurement(data)
        expected = np.array([0.1, 0.2, 0.3])+np.cross([0, 0, 0.4],
                                                                     measured.sole_positions-measured.root_position)
        np.testing.assert_allclose(measured.sole_velocities, expected, atol=1.e-8)

    def test_airborne_soles_have_no_external_contact_force(self):
        model, data, reader = self.make_case()
        data.qpos[2] += 1.
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(reader.contact_forces_world(data), np.zeros((2, 3)))

    def test_invalid_body_names_fail_explicitly(self):
        from legged_lab.perception.mujoco_support import MujocoSupportReader
        model, _, _ = self.make_case()
        with self.assertRaisesRegex(ValueError, 'Unknown MuJoCo body'):
            MujocoSupportReader(model, root_body="missing")


if __name__ == '__main__':
    unittest.main()
