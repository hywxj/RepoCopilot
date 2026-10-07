"""Named ELF3 motion and FK contracts for GMR and the existing AMP layout.

GMR stores root rotations as xyzw and joint angles as MuJoCo qpos[7:].
The 70-column AMP layout uses another joint order, followed by four endpoint
positions. Its hands are elbow-frame offsets and its feet are ankle origins;
these endpoints are not physical sole centers or collision contact points.
"""

from pathlib import Path

import mujoco
import numpy as np


PROJECT_XML = Path(__file__).resolve().parents[1] / "assets/elf3_lite/xml/elf3.xml"

JOINT_NAMES = (
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint", "l_elbow_y_joint",
    "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint",
    "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
)
AMP_JOINT_NAMES = JOINT_NAMES[22:29] + JOINT_NAMES[15:22] + JOINT_NAMES[:3] + JOINT_NAMES[9:15] + JOINT_NAMES[3:9]
AMP_INDICES = np.array([JOINT_NAMES.index(name) for name in AMP_JOINT_NAMES], dtype=np.int64)
AMP_INDICES.setflags(write=False)

# Matches Elf3Env.get_amp_obs_for_expert_trans; ordered LH, RH, LF, RF.
END_EFFECTOR_BODIES = ("l_elbow_y_link", "r_elbow_y_link", "l_ankle_x_link", "r_ankle_x_link")
END_EFFECTOR_OFFSETS = np.array(((0., 0., -.3), (0., 0., -.3), (0., 0., 0.), (0., 0., 0.)))
END_EFFECTOR_OFFSETS.setflags(write=False)


def _opposite_side(name):
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


MIRROR_INDICES = np.array([JOINT_NAMES.index(_opposite_side(name)) for name in JOINT_NAMES], dtype=np.int64)
# Reflection across the x/z plane preserves y-axis rotations and reverses x/z.
MIRROR_SIGNS = np.array([1. if "_y_joint" in name else -1. for name in JOINT_NAMES])
MIRROR_INDICES.setflags(write=False)
MIRROR_SIGNS.setflags(write=False)


def _finite_vectors(values, size, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != size:
        raise ValueError(f"{name} must have final dimension {size}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def mirror_joint_positions(values):
    """Reflect joint angles (or velocities) in JOINT_NAMES order; preserve shape."""
    array = _finite_vectors(values, len(JOINT_NAMES), "joint positions")
    return array[..., MIRROR_INDICES] * MIRROR_SIGNS


def mirror_root_position(values):
    """Reflect root positions or linear velocities across the world x/z plane."""
    return _finite_vectors(values, 3, "root position") * np.array([1., -1., 1.])


def mirror_root_rotation(values):
    """Reflect unit xyzw quaternions: R_mirror = S @ R @ S, S=diag(1,-1,1).

    No sign canonicalization or normalization is applied, so two reflections
    restore the original quaternion representation, including its sign.
    """
    array = _finite_vectors(values, 4, "root rotation (xyzw)")
    if not np.allclose(np.linalg.norm(array, axis=-1), 1., atol=1.e-6, rtol=0.):
        raise ValueError("root rotation (xyzw) must contain unit quaternions")
    return array * np.array([-1., 1., -1., 1.])


class Elf3Kinematics:
    """Pure MuJoCo FK in the torso frame, without stepping or contact inference.

    forward accepts (N,29), or one (29,) pose, in JOINT_NAMES order and always
    returns batched arrays. Angles are not clipped: validation/selection must
    retain and report source errors instead of silently changing the motion.
    A private MjData is reused, so one instance should not be called concurrently.
    """

    def __init__(self, xml_path=PROJECT_XML):
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        model = self.model
        free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        scalar_names = tuple(model.joint(i).name for i in range(model.njnt)
                             if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE)
        if len(free_ids) != 1 or len(scalar_names) != len(JOINT_NAMES) or set(scalar_names) != set(JOINT_NAMES):
            raise ValueError("ELF3 model must contain one free root and exactly the 29 named joints")
        self._joint_ids = np.array([model.joint(name).id for name in JOINT_NAMES])
        if not np.all(model.jnt_type[self._joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("ELF3 named joints must all be hinges")
        if model.body(model.jnt_bodyid[free_ids[0]]).name != "torso_link":
            raise ValueError("ELF3 free root must belong to torso_link")
        self._root_qpos = int(model.jnt_qposadr[free_ids[0]])
        self._qpos_indices = model.jnt_qposadr[self._joint_ids].copy()
        self.joint_limits = model.jnt_range[self._joint_ids].copy()
        self.body_names = [model.body(i).name for i in range(1, model.nbody)]
        if len(self.body_names) != 30:
            raise ValueError("ELF3 motion contract requires exactly 30 robot bodies excluding world")
        self._body_ids = np.arange(1, model.nbody)
        self._endpoint_ids = np.array([model.body(name).id for name in END_EFFECTOR_BODIES])
        self._data = mujoco.MjData(model)

    def forward(self, joint_positions):
        """Recompute all body origins and legacy AMP endpoints in root coordinates."""
        poses = _finite_vectors(joint_positions, len(JOINT_NAMES), "joint positions")
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.ndim != 2:
            raise ValueError(f"joint positions must have shape (N,29), got {poses.shape}")
        bodies = np.empty((len(poses), len(self.body_names), 3), dtype=np.float64)
        endpoints = np.empty((len(poses), len(END_EFFECTOR_BODIES), 3), dtype=np.float64)
        data = self._data
        for index, pose in enumerate(poses):
            data.qpos[:] = self.model.qpos0
            data.qpos[self._root_qpos:self._root_qpos + 7] = (0., 0., 0., 1., 0., 0., 0.)
            data.qpos[self._qpos_indices] = pose
            mujoco.mj_kinematics(self.model, data)
            bodies[index] = data.xpos[self._body_ids]
            rotations = data.xmat[self._endpoint_ids].reshape(-1, 3, 3)
            endpoints[index] = data.xpos[self._endpoint_ids] + np.einsum("bij,bj->bi", rotations, END_EFFECTOR_OFFSETS)
        return {"body_positions_local": bodies, "body_names": list(self.body_names),
                "end_effectors_local": endpoints}
