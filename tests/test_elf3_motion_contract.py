"""ELF3 motion ordering, root-frame FK and physically meaningful mirroring."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.motion.elf3_contract import (
    AMP_INDICES, AMP_JOINT_NAMES, END_EFFECTOR_BODIES, END_EFFECTOR_OFFSETS,
    JOINT_NAMES, PROJECT_XML, Elf3Kinematics,
    mirror_joint_positions, mirror_root_position, mirror_root_rotation,
)


class Elf3MotionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fk = Elf3Kinematics()

    def poses(self, count=8):
        rng = np.random.default_rng(42)
        limits = self.fk.joint_limits
        return rng.uniform(np.maximum(limits[:, 0], -.5), np.minimum(limits[:, 1], .5), (count, 29))

    def test_joint_order_and_amp_named_permutation(self):
        model = self.fk.model
        self.assertEqual(tuple(model.joint(i).name for i in range(1, model.njnt)), JOINT_NAMES)
        expected = ("r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint",
                    "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint")
        self.assertEqual(AMP_JOINT_NAMES[:7], expected)
        self.assertEqual(tuple(np.array(JOINT_NAMES)[AMP_INDICES]), AMP_JOINT_NAMES)
        self.assertEqual(len(set(AMP_INDICES.tolist())), 29)
        np.testing.assert_allclose(self.fk.joint_limits[1], [-.2618, .2618])

    def test_mirrored_fk_is_reflection_for_all_bodies_and_endpoints(self):
        poses = self.poses(32)
        actual = self.fk.forward(mirror_joint_positions(poses))
        original = self.fk.forward(poses)
        names = original["body_names"]
        def opposite(name):
            return "r_" + name[2:] if name.startswith("l_") else "l_" + name[2:] if name.startswith("r_") else name
        body_indices = [names.index(opposite(name)) for name in names]
        np.testing.assert_allclose(actual["body_positions_local"],
                                   original["body_positions_local"][:, body_indices] * [1., -1., 1.], atol=1.e-12)
        np.testing.assert_allclose(actual["end_effectors_local"],
                                   original["end_effectors_local"][:, [1, 0, 3, 2]] * [1., -1., 1.], atol=1.e-12)

    def test_double_mirror_restores_input_and_preserves_joint_limits(self):
        poses = self.poses()
        np.testing.assert_array_equal(mirror_joint_positions(mirror_joint_positions(poses)), poses)
        limits = self.fk.joint_limits
        mirrored_bounds = mirror_joint_positions(limits.T).T
        np.testing.assert_array_equal(np.sort(mirrored_bounds, axis=1), limits)
        positions = np.array([[.3, -.4, .9], [-.1, .2, 1.3]])
        rotations = Rotation.from_euler("xyz", [[.2, -.4, 1.1], [-.7, .1, -.2]]).as_quat()
        np.testing.assert_array_equal(mirror_root_position(mirror_root_position(positions)), positions)
        np.testing.assert_array_equal(mirror_root_rotation(mirror_root_rotation(rotations)), rotations)
        reflection = np.diag([1., -1., 1.])
        np.testing.assert_allclose(Rotation.from_quat(mirror_root_rotation(rotations)).as_matrix(),
                                   reflection @ Rotation.from_quat(rotations).as_matrix() @ reflection, atol=1.e-12)

    def test_fk_root_frame_matches_world_fk_with_arbitrary_root_pose(self):
        pose = self.poses(1)[0]
        result = self.fk.forward(pose)
        data = mujoco.MjData(self.fk.model)
        root = np.array([2.3, -1.2, .7])
        rotation = Rotation.from_euler("xyz", [.4, -.2, 1.7])
        data.qpos[:3] = root
        data.qpos[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
        data.qpos[7:] = pose
        mujoco.mj_kinematics(self.fk.model, data)
        local = (data.xpos[1:] - root) @ rotation.as_matrix()
        np.testing.assert_allclose(result["body_positions_local"][0], local, atol=1.e-12)
        self.assertEqual(result["body_positions_local"].shape, (1, 30, 3))
        np.testing.assert_array_equal(result["body_positions_local"][0, 0], np.zeros(3))

    def test_legacy_hand_offsets_rotate_with_elbows_and_feet_are_origins(self):
        result = self.fk.forward(self.poses(1))
        data = self.fk._data
        for i, name in enumerate(END_EFFECTOR_BODIES):
            body = self.fk.model.body(name).id
            expected = data.xpos[body] + data.xmat[body].reshape(3, 3) @ END_EFFECTOR_OFFSETS[i]
            np.testing.assert_allclose(result["end_effectors_local"][0, i], expected)
            self.assertAlmostEqual(np.linalg.norm(expected - data.xpos[body]), .3 if i < 2 else 0.)

    def test_alternate_model_root_name_and_initial_height_do_not_change_fk(self):
        # Reproduce the GMR/project kinematic convention difference using only
        # checked-in assets; no dependency on an external GMR checkout.
        tree = ET.parse(PROJECT_XML)
        root = tree.getroot()
        root.find("compiler").set("meshdir", str((PROJECT_XML.parent / "../meshes").resolve()))
        for include in root.findall("include"):
            include.set("file", str(PROJECT_XML.parent / include.get("file")))
        torso = root.find("worldbody/body[@name='torso_link']")
        torso.set("pos", "0 0 1.0")
        torso.find("joint[@type='free']").set("name", "torso_joint")
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "elf3_alternate_root.xml"
            tree.write(xml)
            alternate = Elf3Kinematics(xml)
        poses = self.poses()
        original = self.fk.forward(poses)
        other = alternate.forward(poses)
        np.testing.assert_array_equal(alternate.joint_limits, self.fk.joint_limits)
        self.assertEqual(other["body_names"], original["body_names"])
        np.testing.assert_allclose(other["body_positions_local"], original["body_positions_local"], atol=1.e-12)
        np.testing.assert_allclose(other["end_effectors_local"], original["end_effectors_local"], atol=1.e-12)

    def test_rejects_bad_input_and_does_not_step_physics(self):
        for values in (np.zeros(28), np.zeros((2, 3, 29)), np.full((1, 29), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                self.fk.forward(values)
        with self.assertRaises(ValueError):
            mirror_root_rotation([0., 0., 0., 0.])
        self.fk.forward(self.poses())
        self.assertEqual(self.fk._data.time, 0.)
        np.testing.assert_array_equal(self.fk._data.qvel, np.zeros(self.fk.model.nv))
        empty = self.fk.forward(np.empty((0, 29)))
        self.assertEqual(empty["body_positions_local"].shape, (0, 30, 3))
        self.assertEqual(empty["end_effectors_local"].shape, (0, 4, 3))


if __name__ == "__main__":
    unittest.main()
