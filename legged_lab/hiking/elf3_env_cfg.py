"""ELF3 body adaptation of the official InstinctLab Hiking parkour task.

The environment, observation histories, FlatPatch commands, terrain curriculum,
reward functions and AMP groups are inherited from ``ParkourEnvCfg``. This file
only binds those mechanisms to ELF3 and sets a modest single-GPU scene size.
It intentionally imports no ``legged_lab.envs`` or legacy ``rsl_rl`` modules.
"""

from copy import deepcopy
import re

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.parkour_env_cfg import ParkourEnvCfg

from legged_lab.assets.elf3_lite.elf3 import ELF3LITE_CFG
from legged_lab.motion.elf3_contract import JOINT_NAMES

from .actions import PhysicsDelayedJointPositionActionCfg
from .amp_history import dataset_exhausted_with_reference_history_reset
from .motion_cfg import make_motion_reference_cfg


ELF3_FEET = ["l_ankle_x_link", "r_ankle_x_link"]
ELF3_LINKS = ["torso_link"] + [name.removesuffix("_joint") + "_link" for name in JOINT_NAMES]
ELF3_LEG_JOINTS = [name for name in JOINT_NAMES if any(
    part in name for part in ("_hip_", "_knee_", "_ankle_"))]
ELF3_UPPER_JOINTS = [name for name in JOINT_NAMES if name not in ELF3_LEG_JOINTS]
ELF3_ACTION_SCALE_OVERRIDES = {
    "l_hip_y_joint": .38, "r_hip_y_joint": .38,
    "l_knee_y_joint": .40, "r_knee_y_joint": .40,
}


def _named_parameter(value, joint_name, parameter):
    """Resolve a body parameter by name rather than a USD-dependent joint index."""
    if not isinstance(value, dict):
        if value is None:
            raise ValueError(f"ELF3 {parameter} must be explicit for {joint_name}")
        return float(value)
    matches = [v for pattern, v in value.items() if re.fullmatch(pattern, joint_name)]
    if len(matches) != 1:
        raise ValueError(f"ELF3 {parameter} must match {joint_name} exactly once, got {len(matches)}")
    return float(matches[0])


def make_elf3_hiking_robot():
    """Retain ELF3's implicit drives and derive named action amplitudes.

    ELF3's low-inertia ankles/wrists are unstable when its original implicit
    gains are copied into explicit PD at 5ms. Target delay belongs to the action
    term; feedback remains with the original PhysX implicit actuators. Action
    amplitudes start with the official ``0.25 * effort_limit / stiffness`` rule.
    Larger named hip-pitch/knee amplitudes are an ELF3 stepping experiment;
    they are not the retained blind policy's original action mapping.
    """
    robot = deepcopy(ELF3LITE_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    action_scale = {}
    for group_name, original in robot.actuators.items():
        names = [name for name in JOINT_NAMES if any(
            re.fullmatch(pattern, name) for pattern in original.joint_names_expr)]
        if not names:
            raise ValueError(f"ELF3 actuator group {group_name} has no named joints")
        effort = {name: _named_parameter(original.effort_limit_sim, name, "effort_limit") for name in names}
        stiffness = {name: _named_parameter(original.stiffness, name, "stiffness") for name in names}
        for name in names:
            if name in action_scale or stiffness[name] <= 0 or effort[name] <= 0:
                raise ValueError(f"ELF3 actuator assignment/gain is invalid for {name}")
            action_scale[name] = 0.25 * effort[name] / stiffness[name]
    if set(action_scale) != set(JOINT_NAMES):
        raise ValueError("ELF3 Hiking must actuate all 29 joints exactly once")
    action_scale.update(ELF3_ACTION_SCALE_OVERRIDES)
    return robot, action_scale


def _feet_entity(entity_name):
    # feet_at_plane assumes that the left foot is row 0 and the right foot row 1.
    return SceneEntityCfg(entity_name, body_names=list(ELF3_FEET), preserve_order=True)


@configclass
class Elf3HikingEnvCfg(ParkourEnvCfg):
    """Official mixed-terrain visual Hiking task, using the 29-DOF ELF3."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 128
        self.scene.robot, action_scale = make_elf3_hiking_robot()
        self.actions.joint_pos = PhysicsDelayedJointPositionActionCfg(
            asset_name="robot", joint_names=list(JOINT_NAMES), preserve_order=True,
            scale=action_scale, use_default_offset=True, min_delay=0, max_delay=2)
        self.scene.motion_reference = make_motion_reference_cfg()
        self.terminations.dataset_exhausted.func = dataset_exhausted_with_reference_history_reset

        # Retain all official terrain families and their proportions. Twenty
        # columns allocate at least one column to the 5% families as well.
        terrain = deepcopy(self.scene.terrain.terrain_generator)
        terrain.num_rows, terrain.num_cols = 4, 20
        for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
            terrain.sub_terrains[terrain_name].step_height_range = (.05, .18)
            terrain.sub_terrains[terrain_name].step_width = .32
        self.scene.terrain.terrain_generator = terrain
        self.scene.terrain.max_init_terrain_level = 1

        # The 24cm ELF3 foot has only 8cm of total margin on a 32cm tread.
        # Keeping the G1's 5cm edge cylinders would make every full-foot landing
        # intersect an edge. Preserve the official soft edge-penetration term
        # with a 2cm radius appropriate to this footprint.
        self.scene.terrain.virtual_obstacles["edges"].cylinder_radius = .02

        self.scene.contact_forces.history_length = max(self.decimation, self.scene.contact_forces.history_length)
        self.scene.left_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/l_ankle_x_link"
        self.scene.right_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/r_ankle_x_link"
        for scanner in (self.scene.left_height_scanner, self.scene.right_height_scanner):
            scanner.offset.pos = (.03, 0., 20.)
            scanner.pattern_cfg.resolution = .12
            scanner.pattern_cfg.size = [.24, 0.]

        self.scene.leg_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/[lr]_ankle_x_link"
        volume = self.scene.leg_volume_points.points_generator
        # Bounds measured from both ELF3 ankle STL collision meshes.
        volume.x_min, volume.x_max = -.09, .15
        volume.y_min, volume.y_max = -.042, .042
        volume.z_min, volume.z_max = -.04, .0134

        # Existing project ELF3 D435i mount pose, relative to torso_link. Keep
        # the official ray resolution/FOV/noise/37-frame history and 8 outputs.
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.camera.offset.pos = (.1152, .0175, -.1358)
        self.scene.camera.offset.rot = (.122796911, -.696361316, .696363874, -.122797362)
        self.scene.camera.offset.convention = "ros"
        self.scene.camera.mesh_prim_paths = ["/World/ground", *get_link_prim_targets(ELF3_LINKS)]
        # This mount looks 70 degrees down. The official G1 crop removes the
        # upper 18 rows, which would discard ELF3's useful forward view. Retain
        # the entire 36x64 image and resize to the original 18x32 encoder input.
        # Crop order is top/bottom/left/right; resize order is height/width.
        crop = self.scene.camera.noise_pipeline["crop_and_resize"]
        crop.crop_region = (0, 0, 0, 0)
        crop.resize_shape = (18, 32)

        # Policy/critic retain every official observation term and history.
        # Explicit names also align AMP policy and reference states despite
        # PhysX and retargeted motion using different native joint orders.
        for group_name in ("policy", "critic"):
            group = getattr(self.observations, group_name)
            for term_name in ("joint_pos", "joint_vel"):
                getattr(group, term_name).params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=list(JOINT_NAMES), preserve_order=True)
        for group_name, asset_name in (("amp_policy", "robot"), ("amp_reference", "motion_reference")):
            group = getattr(self.observations, group_name)
            for term_name in ("joint_pos_rel", "joint_vel"):
                getattr(group, term_name).params["asset_cfg"] = SceneEntityCfg(
                    asset_name, joint_names=list(JOINT_NAMES), preserve_order=True)

        # All original reward functions/weights remain; only robot selectors and
        # the physical ankle-to-sole height are adapted.
        rewards = self.rewards.rewards
        rewards.feet_air_time.params["sensor_cfg"] = _feet_entity("contact_forces")
        for term_name in ("feet_slide", "feet_flat_ori"):
            getattr(rewards, term_name).params["sensor_cfg"] = _feet_entity("contact_forces")
            getattr(rewards, term_name).params["asset_cfg"] = _feet_entity("robot")
        rewards.feet_at_plane.params["contact_sensor_cfg"] = _feet_entity("contact_forces")
        rewards.feet_at_plane.params["asset_cfg"] = _feet_entity("robot")
        rewards.feet_at_plane.params["height_offset"] = .04
        rewards.feet_close_xy.params["asset_cfg"] = _feet_entity("robot")
        rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["[lr]_hip_z_joint", "[lr]_hip_x_joint"])
        for term_name in ("dof_torques_l2", "energy"):
            getattr(rewards, term_name).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(ELF3_LEG_JOINTS))
        rewards.freeze_upper_body.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(ELF3_UPPER_JOINTS))
        rewards.pelvis_orientation_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_z_link")
        rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="(?![lr]_ankle_x_link$).*")

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="torso_link")
        # Same root-height termination mechanism, scaled to ELF3's 1.08m
        # nominal root height (the official G1 starts at 0.90m with a 0.50m floor).
        self.terminations.root_height.params["minimum_height"] = .60
