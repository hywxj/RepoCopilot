"""Small continuous ELF3 locomotion baseline; not a complete Hiking port."""

from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.elf3.walk_cfg import Elf3WalkAgentCfg, Elf3WalkFlatEnvCfg
from legged_lab.motion.elf3_contract import JOINT_NAMES
from legged_lab.terrains.terrain_generator_cfg import AtecTerrainGenerator
from legged_lab.utils.continuous_course import (
    ContinuousCourse, continuous_course_mesh, course_completion, course_edge_risk, course_failure,
    course_lateral_error, course_loaded_slip, course_progress,
)


@configclass
class ContinuousTerrainCfg(SubTerrainBaseCfg):
    function = continuous_course_mesh
    course: ContinuousCourse = ContinuousCourse()


@configclass
class ContinuousRewardCfg:
    velocity = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2., params={"std": .35})
    yaw_velocity = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=.5, params={"std": .5})
    progress = RewTerm(func=course_progress, weight=1.)
    lateral = RewTerm(func=course_lateral_error, weight=-.4)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.)
    angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-.05)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-.01)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.)
    energy = RewTerm(func=mdp.energy, weight=-2.e-4)
    slip = RewTerm(func=course_loaded_slip, weight=-.5)
    edge_margin = RewTerm(func=course_edge_risk, weight=-2.)
    failure = RewTerm(func=course_failure, weight=-5.)
    completion = RewTerm(func=course_completion, weight=5.)


@configclass
class Elf3ContinuousFlatEnvCfg(Elf3WalkFlatEnvCfg):
    continuous: ContinuousCourse = ContinuousCourse()
    # New baseline's explicit symmetric contract; never infer a joint order from a list.
    continuous_action_scales: dict[str, float] = dict(zip(JOINT_NAMES, [
        .231, .154, .213,
        .380, .220, .240, .400, .350, .220,
        .380, .220, .240, .400, .350, .220,
        .420, .070, .180, .340, .373, .373, .373,
        .420, .070, .180, .340, .373, .373, .373,
    ]))
    reward = ContinuousRewardCfg()

    def __post_init__(self):
        self.scene.num_envs = 128
        self.scene.contact_history_length = 4
        self.scene.max_episode_length_s = 14.
        self.scene.max_init_terrain_level = 0
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = TerrainGeneratorCfg(
            class_type=AtecTerrainGenerator, seed=42, curriculum=False,
            size=(self.continuous.length, self.continuous.width), border_width=2.,
            num_rows=1, num_cols=8, use_cache=False,
            sub_terrains={"continuous": ContinuousTerrainCfg(proportion=1., course=self.continuous)},
        )
        self.scene.height_scanner.enable_height_scan = False
        self.scene.depth_camera.enable_depth_camera = False
        self.scene.depth_camera.geometry.enabled = False
        self.scene.depth_camera.geometry.step_control_enabled = False
        self.scene.depth_camera.geometry.bootstrap_stairs = False
        self.scene.lidar.enable_lidar = False
        self.robot.actor_obs_history_length = 4
        self.robot.critic_obs_history_length = 4
        self.robot.terminate_contacts_body_names = ["^(?![lr]_ankle_x_link$).*"]
        self.commands.resampling_time_range = (20., 20.)
        self.commands.rel_standing_envs = 0.
        self.commands.rel_heading_envs = 1.
        self.commands.heading_command = True
        self.commands.debug_vis = False
        self.commands.ranges.lin_vel_x = (.25, .45)
        self.commands.ranges.lin_vel_y = (0., 0.)
        self.commands.ranges.ang_vel_z = (-.5, .5)
        self.commands.ranges.heading = (0., 0.)
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.add_base_mass = None
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (-.03, .03), "y": (-.03, .03), "yaw": (-.05, .05)}
        self.domain_rand.events.reset_base.params["velocity_range"] = {}
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (.98, 1.02)
        self.domain_rand.action_delay.enable = False
        self.normalization.clip_actions = 5.
        self.sim.dt, self.sim.decimation = .005, 4
        # Display-only loader is mandatory in the shared parent; one file is enough.
        self.amp_motion_files_display = self.amp_motion_files_display[:1]


@configclass
class Elf3ContinuousUpEnvCfg(Elf3ContinuousFlatEnvCfg):
    continuous: ContinuousCourse = ContinuousCourse(direction="up")


@configclass
class Elf3ContinuousDownEnvCfg(Elf3ContinuousFlatEnvCfg):
    continuous: ContinuousCourse = ContinuousCourse(direction="down")


@configclass
class Elf3ContinuousAgentCfg(Elf3WalkAgentCfg):
    experiment_name = "elf3_continuous"
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 1000
    amp_motion_manifest = str(Path(__file__).resolve().parents[3] / "logs/data_preparation/elf3_gmr_v1/manifest.json")
    amp_reward_coef = 2.
    amp_task_reward_lerp = .7
    amp_reward_time_scaled = True
    amp_motion_mask_enabled = True
    amp_discr_hidden_dims = [256, 128]
    amp_num_preload_transitions = 0
    resume = False

    def __post_init__(self):
        self.policy.actor_hidden_dims = [256, 128]
        self.policy.critic_hidden_dims = [256, 128]
        self.policy.init_noise_std = .6
        self.algorithm.num_learning_epochs = 3
        self.algorithm.num_mini_batches = 4
        self.algorithm.learning_rate = 3.e-4
