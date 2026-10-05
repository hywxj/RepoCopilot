"""Small-scale second-stage tasks: one verified 32 cm tread transfer per episode."""

import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from .walk_terrain_teacher_cfg import (
    Elf3WalkGeometryStairsBootstrapEnvCfg, Elf3WalkGeometryStairsDownFullControlAgentCfg,
)


@configclass
class Elf3SingleStepUpEnvCfg(Elf3WalkGeometryStairsBootstrapEnvCfg):
    scene = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg().scene)
    scene.num_envs = 1
    scene.seed = 42
    scene.max_episode_length_s = 30.
    scene.terrain_generator.curriculum = False
    scene.terrain_generator.num_rows = 4
    scene.terrain_generator.num_cols = 4
    scene.terrain_generator.sub_terrains = {
        name: terrain for name, terrain in scene.terrain_generator.sub_terrains.items()
        if name in ("plane_start", "stairs_up", "stairs_down", "plane_finish")
    }
    scene.terrain_generator.difficulty_range = (0., 0.)
    for name in ("stairs_up", "stairs_down"):
        # Two physical risers keep the FIRST destination a bounded 32 cm tread.
        # The episode completes on that tread, before the second transfer.
        scene.terrain_generator.sub_terrains[name].num_steps = 2
        scene.terrain_generator.sub_terrains[name].step_width = 0.32
    scene.depth_camera.width = 1280
    scene.depth_camera.height = 720
    scene.depth_camera.geometry.surface_validation_enabled = True
    scene.depth_camera.geometry.surface_memory_enabled = True
    scene.depth_camera.geometry.step_control_enabled = True
    scene.depth_camera.geometry.bootstrap_x_offset = 1.80
    scene.depth_camera.geometry.append_course_alignment = False
    scene.depth_camera.geometry.append_foothold_targets = False
    scene.depth_camera.geometry.foothold_target_lock_enabled = False
    scene.depth_camera.geometry.foothold_ik_enabled = False
    scene.depth_camera.geometry.foothold_overshoot_guard_enabled = False
    scene.depth_camera.geometry.stair_step_settle_enabled = False
    scene.depth_camera.geometry.stair_heading_feedback_enabled = False
    scene.depth_camera.geometry.alignment_safety_gate_enabled = False

    commands = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg().commands)
    commands.ranges.lin_vel_x = (0.20, 0.20)
    commands.ranges.lin_vel_y = (0., 0.)
    commands.ranges.ang_vel_z = (0., 0.)
    commands.debug_vis = False
    domain_rand = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg().domain_rand)
    domain_rand.events.push_robot = None
    domain_rand.events.reset_base.params["pose_range"] = {}
    domain_rand.events.reset_base.params["velocity_range"] = {}
    domain_rand.events.reset_robot_joints.params["position_range"] = (1., 1.)
    noise = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg().noise)
    noise.add_noise = False

    reward = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg().reward)
    # Keep physical regularization but remove fixed-period gait, forward-speed,
    # legacy one-foot completion and flat-walk posture objectives.
    _regularizers = {
        "energy", "dof_acc_l2", "action_rate_l2", "action_rate_smooth", "ankle_torque",
        "undesired_contacts", "feet_slide", "feet_stumble", "dof_pos_limits", "feet_too_near",
    }
    for _name, _term in vars(reward).items():
        if isinstance(_term, RewTerm) and _name not in _regularizers:
            _term.weight = 0.
    reward.termination_penalty = RewTerm(func=mdp.single_step_failure, weight=-50.)
    reward.step_feet_reference = RewTerm(func=mdp.stair_step_metric, weight=4.,
                                       params={"metric": "feet_reference", "quadratic_error": True})
    reward.step_body_reference = RewTerm(func=mdp.stair_step_metric, weight=2.,
                                       params={"metric": "body_reference", "quadratic_error": True})
    reward.step_load_reference = RewTerm(func=mdp.stair_step_metric, weight=2., params={"metric": "load_reference"})
    reward.step_heading = RewTerm(func=mdp.stair_step_metric, weight=1., params={"metric": "heading"})
    reward.body_orientation_l2.weight = -2.
    reward.step_sole_tilt = RewTerm(func=mdp.stair_step_metric, weight=-5., params={"metric": "sole_tilt"})
    reward.step_unsafe_contact = RewTerm(func=mdp.stair_step_metric, weight=-5., params={"metric": "unsafe_contact"})
    reward.step_downward_speed = RewTerm(func=mdp.stair_step_metric, weight=-20., params={"metric": "downward_speed"})
    reward.step_phase_progress = RewTerm(func=mdp.stair_step_metric, weight=10., params={"metric": "progress"})
    reward.step_lift_progress = RewTerm(func=mdp.stair_step_metric, weight=5., params={"metric": "lift_progress"})
    reward.step_completion = RewTerm(func=mdp.stair_step_metric, weight=100., params={"metric": "success"})
    reward.step_stop_root_motion = RewTerm(func=mdp.stair_step_metric, weight=-20., params={"metric": "stop_root_motion"})
    reward.step_stop_feet_motion = RewTerm(func=mdp.stair_step_metric, weight=-15., params={"metric": "stop_feet_motion"})
    reward.step_stop_angular_motion = RewTerm(func=mdp.stair_step_metric, weight=-2., params={"metric": "stop_angular_motion"})


@configclass
class Elf3SingleStepDownEnvCfg(Elf3SingleStepUpEnvCfg):
    scene = copy.deepcopy(Elf3SingleStepUpEnvCfg().scene)
    scene.max_init_terrain_level = 2
    scene.depth_camera.geometry.bootstrap_stair_row = 2
    scene.depth_camera.geometry.bootstrap_x_offset = 1.48


@configclass
class Elf3SingleStepAgentCfg(Elf3WalkGeometryStairsDownFullControlAgentCfg):
    run_name = "elf3_single_step_v2_stair_actor"
    max_iterations = 3000
    save_interval = 16
    amp_task_reward_lerp = 1.0
    # Flat walking AMP is not a valid reference for a deliberate stair pause.
    amp_reward_coef = 0.0
    amp_num_preload_transitions = 8192
    policy = copy.deepcopy(Elf3WalkGeometryStairsDownFullControlAgentCfg().policy)
    # Copy the existing gait into a trainable full stair actor. Flat ground keeps
    # the frozen original; stairs can change the gait instead of cancelling it.
    policy.active_base_action_scale = 0.0
    policy.residual_scale = 1.0
    policy.residual_hidden_dims = list(policy.actor_hidden_dims)
    policy.squash_residual = False
    policy.initialize_stair_actor_from_blind = True
    policy.residual_observation_scales = (
        [1.]*960 + [1.]*12 + [1/.32]*6 + [1/.08]*6 + [1.]*2 + [1/.12]*3 + [1.]*11)
    policy.init_noise_std = 0.03
    policy.min_action_std = 0.01
    policy.max_action_std = 0.10
    policy.low_noise_obs_indices = [961, 969]
    policy.low_noise_scale = .10
    algorithm = copy.deepcopy(Elf3WalkGeometryStairsDownFullControlAgentCfg().algorithm)
    algorithm.learning_rate = 1.e-5
    algorithm.desired_kl = 0.005
    algorithm.gamma = .998
    algorithm.num_mini_batches = 2
