# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--command_x", type=float, default=0.3, help="Forward velocity command in m/s.")
parser.add_argument("--command_y", type=float, default=0.0, help="Lateral velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Yaw velocity command in rad/s.")
parser.add_argument(
    "--terrain_mode",
    type=str,
    choices=("plane", "configured"),
    default="plane",
    help="Use a flat plane or the terrain generator configured by the task.",
)
parser.add_argument(
    "--terrain_difficulty",
    type=float,
    default=0.0,
    help="Fixed terrain-generator difficulty used during playback (0.0 to 1.0).",
)
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="Use only one configured sub-terrain, for example stairs_up_32 or pebbles.",
)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many policy steps; zero runs forever.")
parser.add_argument(
    "--trace_course_every",
    type=int,
    default=0,
    help="Print route position and stair detections every N policy steps; zero disables tracing.",
)
parser.add_argument(
    "--trace_footsteps",
    action="store_true",
    help="Print each confirmed tread landing with foot position, overlap, and height error.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=40.0,
    help="Episode duration in seconds during playback.",
)
parser.add_argument("--skip_export", action="store_true", help="Do not export JIT and ONNX policies during playback.")
parser.add_argument(
    "--verify_course_geometry",
    action="store_true",
    help="Check that consecutive route meshes meet at the same height and exit.",
)
parser.add_argument(
    "--camera_debug_vis",
    action="store_true",
    help="Show depth-camera point cloud markers in the Isaac Sim viewport.",
)
parser.add_argument(
    "--show_rgb",
    action="store_true",
    help="Show the realtime RGB image from the simulated D435i camera.",
)
parser.add_argument(
    "--show_depth",
    action="store_true",
    help="Show a realtime OpenCV window with the selected depth-camera image.",
)
parser.add_argument(
    "--save_depth_dir",
    type=str,
    default=None,
    help="Directory for saving debug depth-camera PNG frames.",
)
parser.add_argument(
    "--save_rgb_dir",
    type=str,
    default=None,
    help="Directory for saving RGB frames synchronized with depth diagnostics.",
)
parser.add_argument(
    "--save_depth_every",
    type=int,
    default=10,
    help="Save one depth frame every N policy steps when --save_depth_dir is set.",
)
parser.add_argument(
    "--save_depth_max",
    type=int,
    default=80,
    help="Maximum number of depth frames to save; zero means unlimited.",
)
parser.add_argument(
    "--save_depth_env",
    type=int,
    default=0,
    help="Environment index used when saving depth-camera frames.",
)
parser.add_argument(
    "--show_geometry",
    action="store_true",
    help="Show the D435i depth image with its extracted stair height profile.",
)
parser.add_argument(
    "--validate_surfaces", action="store_true",
    help="Add diagnostic 2-D plane validation and full-foot support visualization (CPU path).",
)
parser.add_argument("--surface_memory", action="store_true",
                    help="Show pose-compensated observed surfaces with expiry; diagnostic only.")
parser.add_argument(
    "--disable_geometry_action",
    action="store_true",
    help="Ablation: force the geometry gate and residual-policy inputs to zero.",
)
parser.add_argument(
    "--disable_foothold_targets",
    action="store_true",
    help="Ablation: hide per-foot tread targets while keeping stair mode and geometry history.",
)
parser.add_argument(
    "--foothold_actor_scale",
    type=float,
    default=None,
    help="Scale only the policy's per-foot depth-derived target observations (0 to 1).",
)
parser.add_argument("--foothold_target_distance_scale", type=float, default=None)
parser.add_argument("--descending_edge_offset_m", type=float, default=None)
parser.add_argument("--leg_residual_multiplier", type=float, default=None)
parser.add_argument(
    "--measure_target_sensitivity",
    action="store_true",
    help="Measure the policy action change when per-foot target inputs are masked, without changing executed actions.",
)
parser.add_argument(
    "--enable_foothold_ik",
    action="store_true",
    help="Experiment: add a bounded Jacobian correction toward detected upper stair treads.",
)
parser.add_argument(
    "--enable_foothold_overshoot_guard",
    action="store_true",
    help="Experiment: cap a descending swing foot's predicted forward reach at the tread center band.",
)
parser.add_argument(
    "--enable_foothold_target_lock",
    action="store_true",
    help="Keep a selected world-frame tread target until its swing foot lands.",
)
parser.add_argument(
    "--enable_foothold_swing_trajectory",
    action="store_true",
    help="Guide a swing foot along a world-fixed depth-derived tread trajectory (also enables IK and target locking).",
)
parser.add_argument("--foothold_ik_gain", type=float, default=None)
parser.add_argument(
    "--enable_foothold_preview",
    action="store_true",
    help="Expose the next stair tread target before a foot starts swinging.",
)
parser.add_argument("--foothold_ik_max_joint_step", type=float, default=None)
parser.add_argument(
    "--enable_stair_heading_feedback",
    action="store_true",
    help="Use proprioceptive yaw feedback to stay centered while stair mode is active.",
)
parser.add_argument("--stair_yaw_lateral_gain", type=float, default=None)
parser.add_argument("--stair_yaw_rate_limit", type=float, default=None)
parser.add_argument("--verified_stair_progress_weight", type=float, default=None)
parser.add_argument("--stair_goal_weight", type=float, default=None)
parser.add_argument("--stair_failure_penalty", type=float, default=None)
parser.add_argument("--hip_clearance_penalty", type=float, default=None)
parser.add_argument("--stair_collision_force_penalty", type=float, default=None)
parser.add_argument(
    "--disable_stair_heading_feedback",
    action="store_true",
    help="Ablation: remove the stair-only lane-centering yaw command.",
)
parser.add_argument(
    "--disable_stair_settle",
    action="store_true",
    help="Ablation: keep the walking command active after stair touchdowns.",
)
parser.add_argument(
    "--enable_stair_settle",
    action="store_true",
    help="Pause briefly after a confirmed lower-step touchdown on descent.",
)
parser.add_argument(
    "--blind_during_stair_settle",
    action="store_true",
    help="Temporarily gate off stair residual actions during the post-touchdown settling state.",
)
parser.add_argument(
    "--save_geometry_dir",
    type=str,
    default=None,
    help="Directory for saving stair-geometry diagnostic PNG frames.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if (
    args_cli.task is not None
    and ("sensor" in args_cli.task or "geometry" in args_cli.task)
) or args_cli.camera_debug_vis or args_cli.show_rgb or args_cli.show_depth or args_cli.save_depth_dir is not None or args_cli.show_geometry or args_cli.save_geometry_dir is not None:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.math import quat_apply

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    if args_cli.verify_course_geometry:
        import numpy as np

        terrain_cfg = env_cfg.scene.terrain_generator
        for difficulty in (0.0, 0.8, 1.0):
            previous_end = None
            for name, original_cfg in terrain_cfg.sub_terrains.items():
                sub_cfg = original_cfg.copy()
                sub_cfg.size = terrain_cfg.size
                if hasattr(sub_cfg, "horizontal_scale"):
                    sub_cfg.horizontal_scale = terrain_cfg.horizontal_scale
                    sub_cfg.vertical_scale = terrain_cfg.vertical_scale
                    sub_cfg.slope_threshold = terrain_cfg.slope_threshold
                meshes, _ = sub_cfg.function(difficulty, sub_cfg)
                vertices = np.concatenate([mesh.vertices for mesh in meshes])
                start_height = vertices[np.isclose(vertices[:, 0], 0.0), 2].max()
                end_height = vertices[np.isclose(vertices[:, 0], terrain_cfg.size[0]), 2].max()
                if previous_end is not None and abs(previous_end - start_height) > 0.011:
                    raise AssertionError(
                        f"Terrain seam before {name} at difficulty {difficulty:.1f}: "
                        f"{previous_end:.3f}m -> {start_height:.3f}m"
                    )
                print(
                    f"[COURSE] difficulty={difficulty:.1f} {name}: "
                    f"{start_height:.3f}m -> {end_height:.3f}m"
                )
                previous_end = end_height
        return

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = args_cli.episode_length_s
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    # env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    # env_cfg.commands.ranges.lin_vel_x = (-1.0, 1.0)
    # env_cfg.commands.ranges.lin_vel_x = (-0.5, 0.5)
    env_cfg.commands.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
    # env_cfg.commands.ranges.lin_vel_x = (-0.5, 2.3)
    # env_cfg.commands.ranges.lin_vel_x = (-0.5, 3.0)
    # env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
    env_cfg.commands.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    # env_cfg.commands.ranges.lin_vel_y = (-0.5, 0.5)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    geometry_cfg = getattr(getattr(env_cfg.scene, "depth_camera", None), "geometry", None)
    if args_cli.validate_surfaces or args_cli.surface_memory:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--validate_surfaces requires a geometry task.")
        geometry_cfg.surface_validation_enabled = True
        geometry_cfg.surface_memory_enabled = args_cli.surface_memory or geometry_cfg.step_control_enabled
        env_cfg.scene.depth_camera.width = 1280
        env_cfg.scene.depth_camera.height = 720
    if args_cli.camera_debug_vis:
        env_cfg.scene.depth_camera.debug_vis = True
    if args_cli.disable_stair_settle and hasattr(env_cfg.scene.depth_camera, "geometry"):
        env_cfg.scene.depth_camera.geometry.stair_step_settle_enabled = False
    if args_cli.enable_stair_settle and hasattr(env_cfg.scene.depth_camera, "geometry"):
        env_cfg.scene.depth_camera.geometry.stair_step_settle_enabled = True
    if args_cli.blind_during_stair_settle:
        env_cfg.scene.depth_camera.geometry.blind_during_stair_settle = True
    if args_cli.enable_foothold_ik:
        env_cfg.scene.depth_camera.geometry.foothold_ik_enabled = True
    if args_cli.enable_foothold_overshoot_guard:
        env_cfg.scene.depth_camera.geometry.foothold_overshoot_guard_enabled = True
    if args_cli.enable_foothold_target_lock:
        env_cfg.scene.depth_camera.geometry.foothold_target_lock_enabled = True
    if args_cli.enable_foothold_swing_trajectory:
        geometry_cfg = env_cfg.scene.depth_camera.geometry
        geometry_cfg.foothold_swing_trajectory_enabled = True
        geometry_cfg.foothold_ik_enabled = True
        geometry_cfg.foothold_target_lock_enabled = True
        if args_cli.foothold_ik_gain is None:
            geometry_cfg.foothold_ik_gain = 0.35
        if args_cli.foothold_ik_max_joint_step is None:
            geometry_cfg.foothold_ik_max_joint_step = 0.06
    if args_cli.enable_foothold_preview:
        env_cfg.scene.depth_camera.geometry.foothold_preview_stance = True
    if args_cli.foothold_actor_scale is not None:
        if not 0.0 <= args_cli.foothold_actor_scale <= 1.0:
            raise ValueError("--foothold_actor_scale must be between 0 and 1.")
        env_cfg.scene.depth_camera.geometry.foothold_actor_scale = args_cli.foothold_actor_scale
    if args_cli.foothold_target_distance_scale is not None:
        geometry_cfg = env_cfg.scene.depth_camera.geometry
        if not 0.1 <= args_cli.foothold_target_distance_scale <= geometry_cfg.max_forward:
            raise ValueError("--foothold_target_distance_scale must be 0.1 to max_forward metres.")
        geometry_cfg.foothold_target_distance_scale = args_cli.foothold_target_distance_scale
    if args_cli.descending_edge_offset_m is not None:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--descending_edge_offset_m requires a geometry task.")
        if not -0.15 <= args_cli.descending_edge_offset_m <= 0.15:
            raise ValueError("--descending_edge_offset_m must be between -0.15 and 0.15 metres.")
        geometry_cfg.descending_edge_offset_m = args_cli.descending_edge_offset_m
    if args_cli.foothold_ik_gain is not None:
        env_cfg.scene.depth_camera.geometry.foothold_ik_gain = args_cli.foothold_ik_gain
    if args_cli.foothold_ik_max_joint_step is not None:
        env_cfg.scene.depth_camera.geometry.foothold_ik_max_joint_step = args_cli.foothold_ik_max_joint_step
    if args_cli.enable_stair_heading_feedback:
        env_cfg.scene.depth_camera.geometry.stair_heading_feedback_enabled = True
    if args_cli.stair_yaw_lateral_gain is not None:
        env_cfg.scene.depth_camera.geometry.stair_yaw_lateral_gain = args_cli.stair_yaw_lateral_gain
    if args_cli.stair_yaw_rate_limit is not None:
        env_cfg.scene.depth_camera.geometry.stair_yaw_rate_limit = args_cli.stair_yaw_rate_limit
    reward_weights = (
        ("verified_stair_progress_weight", "verified_stair_step_progress", 1.0),
        ("stair_goal_weight", "stair_goal_completion", 1.0),
        ("stair_failure_penalty", "termination_penalty", -1.0),
        ("hip_clearance_penalty", "stair_hip_clearance_deficit", -1.0),
        ("stair_collision_force_penalty", "stair_nonfoot_collision_risk", -1.0),
    )
    for argument, term_name, sign in reward_weights:
        value = getattr(args_cli, argument)
        if value is not None:
            if value <= 0 or not hasattr(env_cfg.reward, term_name):
                raise ValueError(f"--{argument} requires a positive value and a geometry stair reward.")
            getattr(env_cfg.reward, term_name).weight = sign * value
    if args_cli.disable_stair_heading_feedback:
        env_cfg.scene.depth_camera.geometry.stair_heading_feedback_enabled = False

    if args_cli.terrain_mode == "plane":
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"
    elif env_cfg.scene.terrain_generator is None:
        raise ValueError(f"Task '{args_cli.task}' does not configure a terrain generator.")

    if env_cfg.scene.terrain_generator is not None:
        if args_cli.terrain_type is not None:
            available_terrains = env_cfg.scene.terrain_generator.sub_terrains
            if args_cli.terrain_type not in available_terrains:
                raise ValueError(
                    f"Unknown terrain type '{args_cli.terrain_type}'. Available: {sorted(available_terrains)}"
                )
            env_cfg.scene.terrain_generator.sub_terrains = {
                args_cli.terrain_type: available_terrains[args_cli.terrain_type]
            }
            env_cfg.scene.terrain_generator.num_cols = 1
        generator_name = getattr(env_cfg.scene.terrain_generator.class_type, "__name__", "")
        if generator_name == "AtecObstacleCourseTerrainGenerator":
            env_cfg.scene.max_init_terrain_level = 0
        elif args_cli.terrain_type is None:
            env_cfg.scene.terrain_generator.num_rows = 5
            env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        difficulty = max(0.0, min(1.0, args_cli.terrain_difficulty))
        env_cfg.scene.terrain_generator.difficulty_range = (difficulty, difficulty)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    elif args_cli.validate_surfaces or args_cli.surface_memory:
        env_cfg.scene.num_envs = 1
    if (
        env_cfg.scene.terrain_generator is not None
        and getattr(env_cfg.scene.terrain_generator.class_type, "__name__", "")
        == "AtecObstacleCourseTerrainGenerator"
        and args_cli.terrain_type is None
    ):
        # Playback uses one lane per robot; training retains its parallel
        # curriculum lanes in the task configuration.
        env_cfg.scene.terrain_generator.num_cols = env_cfg.scene.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if hasattr(agent_cfg, "amp_num_preload_transitions"):
        agent_cfg.amp_num_preload_transitions = 1
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    if args_cli.leg_residual_multiplier is not None:
        if not hasattr(env, "left_leg_ids") or not 1.0 <= args_cli.leg_residual_multiplier <= 4.0:
            raise ValueError("--leg_residual_multiplier requires ELF3 geometry and a value from 1 to 4.")
        scales = [1.0] * env.num_actions
        for joint_id in (*env.left_leg_ids, *env.right_leg_ids):
            scales[joint_id] = args_cli.leg_residual_multiplier
        agent_cfg.policy.residual_action_scales = scales
    if hasattr(env, "foothold_ik_active"):
        robot_names = [env.robot.body_names[index] for index in env.feet_body_ids]
        contact_names = [env.contact_sensor.body_names[index] for index in env.feet_cfg.body_ids]
        print(
            f"[DIAG] robot_feet={list(zip(env.feet_body_ids, robot_names))} "
            f"contact_feet={list(zip(env.feet_cfg.body_ids, contact_names))}"
        )
        if robot_names != contact_names:
            raise ValueError("Robot foot bodies and contact-sensor foot bodies must have matching order.")
    if (
        args_cli.camera_debug_vis
        or args_cli.show_rgb
        or args_cli.show_depth
        or args_cli.save_depth_dir is not None
        or args_cli.show_geometry
        or args_cli.save_geometry_dir is not None
    ) and not hasattr(
        env, "depth_camera"
    ):
        raise ValueError(
            f"Task '{args_cli.task}' does not enable a depth camera. Use a sensor task such as "
            "'walk_elf3_depth_sensor'."
        )

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    if not args_cli.skip_export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    target_obs_slice = None
    if hasattr(runner.alg.policy, "base_actor_obs_dim"):
        geometry_cfg = env.cfg.scene.depth_camera.geometry
        if geometry_cfg.append_foothold_targets:
            geometry_dim = geometry_cfg.history_length * (geometry_cfg.max_treads * 6 + 4)
            target_start = runner.alg.policy.base_actor_obs_dim + geometry_dim + (
                4 if geometry_cfg.append_course_alignment else 0
            )
            target_obs_slice = slice(target_start, target_start + 8)
    if (args_cli.disable_foothold_targets or args_cli.measure_target_sensitivity) and target_obs_slice is None:
        raise ValueError("This policy does not expose per-foot foothold target observations.")
    if args_cli.measure_target_sensitivity and (
        args_cli.disable_geometry_action or args_cli.disable_foothold_targets
    ):
        raise ValueError("Target sensitivity requires the unmasked geometry policy.")

    step_count = 0
    reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    goal_reached_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    distance_goal_count = 0
    stable_goal_count = 0
    goal_verified_level_histogram = torch.zeros(9, dtype=torch.long, device=env.device)
    goal_verified_contact_histogram = torch.zeros(9, dtype=torch.long, device=env.device)
    goal_physical_coverage_histogram = torch.zeros(9, dtype=torch.long, device=env.device)
    goal_broad_coverage_histogram = torch.zeros(9, dtype=torch.long, device=env.device)
    physical_level_count = env.stair_physical_covered.shape[1] if hasattr(env, "stair_physical_covered") else 8
    goal_physical_level_histogram = torch.zeros(physical_level_count, dtype=torch.long, device=env.device)
    goal_broad_level_histogram = torch.zeros(physical_level_count, dtype=torch.long, device=env.device)
    reward_sum = torch.zeros(env.num_envs, device=env.device)
    forward_speed_sum = torch.zeros(env.num_envs, device=env.device)
    lateral_offset_sum = torch.zeros(env.num_envs, device=env.device)
    max_forward_distance = torch.full((env.num_envs,), -torch.inf, device=env.device)
    action_magnitude_sum = 0.0
    geometry_gate_sum = 0.0
    geometry_residual_abs_sum = 0.0
    target_action_sensitivity_sum = 0.0
    foothold_target_valid_sum = 0.0
    detector_stair_sum = 0.0
    stair_settle_sum = 0.0
    stair_settle_yaw_max = 0.0
    foothold_ik_active_sum = 0.0
    foothold_guard_active_sum = 0.0
    foothold_lock_active_sum = 0.0
    stair_confidence_sum = 0.0
    raw_stair_detection_sum = 0.0
    gate_evidence_sum = 0.0
    stair_tread_audit_count = 0
    stair_tread_center_error_sum = 0.0
    stair_tread_center_signed_error_sum = 0.0
    stair_tread_width_error_sum = 0.0
    stair_tread_center_error_large = 0
    stair_tread_center_error_forward = 0
    safe_tread_positive_sum = 0.0
    safe_tread_reward_sum = 0.0
    safe_tread_first_contact_sum = 0.0
    tread_candidate_contact_count = 0
    tread_near_candidate_contact_count = 0
    safe_tread_contact_count = 0
    unsafe_tread_contact_count = 0
    confirmed_tread_contact_count = 0
    capture_confirmed_count = 0
    capture_forward_sum = 0.0
    capture_lateral_abs_sum = 0.0
    capture_envelope_count = 0
    verified_stair_step_count = 0
    raw_stair_step_count = 0
    physical_stair_step_count = 0
    stair_contact_failure_count = 0
    stair_lane_failure_count = 0
    stair_unsafe_end_failure_count = 0
    stair_timeout_failure_count = 0
    hip_clearance_sum = 0.0
    hip_clearance_contact_failure_sum = 0.0
    hip_clearance_low_count = 0
    stair_failure_level_histogram = torch.zeros(9, dtype=torch.long, device=env.device)
    termination_body_names = [
        env.contact_sensor.body_names[index]
        for index in env.termination_contact_cfg.body_ids
    ]
    stair_contact_body_histogram = torch.zeros(
        len(termination_body_names), dtype=torch.long, device=env.device
    )
    safe_tread_candidate_sum = 0.0
    safe_tread_inside_sum = 0.0
    safe_tread_min_distance = torch.tensor(torch.inf, device=env.device)
    candidate_foot_x_min = torch.tensor(torch.inf, device=env.device)
    candidate_foot_x_max = torch.tensor(-torch.inf, device=env.device)
    candidate_near_min = torch.tensor(torch.inf, device=env.device)
    candidate_far_max = torch.tensor(-torch.inf, device=env.device)
    safe_tread_max_overlap = torch.tensor(-torch.inf, device=env.device)
    safe_tread_contact_overlap = torch.tensor(-torch.inf, device=env.device)
    safe_tread_term_index = (
        env.reward_manager.active_terms.index("safe_tread_landing")
        if ("safe_tread_landing" in env.reward_manager.active_terms
            and env.reward_manager.get_term_cfg("safe_tread_landing").weight != 0.)
        else None
    )
    verified_step_term_index = (
        env.reward_manager.active_terms.index("verified_stair_step_progress")
        if "verified_stair_step_progress" in env.reward_manager.active_terms
        else None
    )
    depth_save_count = 0
    depth_save_dir = None
    rgb_save_count = 0
    rgb_save_dir = None
    geometry_save_count = 0
    geometry_save_dir = None
    rgb_debug_enabled = args_cli.show_rgb or args_cli.save_rgb_dir is not None
    depth_debug_enabled = args_cli.show_depth or args_cli.save_depth_dir is not None
    geometry_debug_enabled = args_cli.show_geometry or args_cli.save_geometry_dir is not None
    camera_debug_enabled = rgb_debug_enabled or depth_debug_enabled or geometry_debug_enabled
    if camera_debug_enabled:
        import cv2
        import numpy as np

        print(f"[DEPTH] Camera prim body: {env.cfg.scene.depth_camera.prim_body_name}")
        if rgb_debug_enabled:
            if "rgb" not in env.depth_camera.data.output:
                raise ValueError("RGB visualization requires 'rgb' in depth_camera.data_types.")
        if args_cli.show_rgb:
            cv2.namedWindow("ELF3 D435i RGB", cv2.WINDOW_NORMAL)
        if args_cli.save_rgb_dir is not None:
            rgb_save_dir = os.path.abspath(args_cli.save_rgb_dir)
            os.makedirs(rgb_save_dir, exist_ok=True)
            print(f"[RGB] Saving camera frames to: {rgb_save_dir}")
        if args_cli.show_depth:
            cv2.namedWindow("ELF3 waist depth", cv2.WINDOW_NORMAL)
        if args_cli.save_depth_dir is not None:
            depth_save_dir = os.path.abspath(args_cli.save_depth_dir)
            os.makedirs(depth_save_dir, exist_ok=True)
            print(f"[DEPTH] Saving camera frames to: {depth_save_dir}")
        if geometry_debug_enabled:
            if not env.cfg.scene.depth_camera.geometry.enabled:
                raise ValueError("Geometry visualization requires depth_camera.geometry.enabled=True.")
            from legged_lab.perception import draw_stair_geometry_debug

            if args_cli.show_geometry:
                cv2.namedWindow("ELF3 stair geometry", cv2.WINDOW_NORMAL)
            if args_cli.save_geometry_dir is not None:
                geometry_save_dir = os.path.abspath(args_cli.save_geometry_dir)
                os.makedirs(geometry_save_dir, exist_ok=True)
                print(f"[GEOMETRY] Saving diagnostic frames to: {geometry_save_dir}")

    while simulation_app.is_running():

        with torch.inference_mode():
            policy_obs = obs
            if target_obs_slice is not None:
                target_confidence = obs[:, target_obs_slice][:, [3, 7]]
                foothold_target_valid_sum += (target_confidence > 0.25).float().mean().item()
            if args_cli.disable_geometry_action:
                policy_obs = obs.clone()
                base_obs_dim = getattr(runner.alg.policy, "base_actor_obs_dim", policy_obs.shape[1])
                policy_obs[:, base_obs_dim:] = 0.0
            elif args_cli.disable_foothold_targets:
                policy_obs = obs.clone()
                policy_obs[:, target_obs_slice] = 0.0
            actions = policy(policy_obs)
            actor = getattr(runner.alg.policy, "actor", None)
            if actor is not None and hasattr(actor, "last_applied_residual"):
                geometry_gate_sum += actor.last_gate.mean().item()
                geometry_residual_abs_sum += actor.last_applied_residual.abs().mean().item()
            if args_cli.measure_target_sensitivity:
                masked_obs = obs.clone()
                masked_obs[:, target_obs_slice] = 0.0
                masked_actions = policy(masked_obs)
                target_action_sensitivity_sum += (actions - masked_actions).abs().mean().item()
            obs, rewards, dones, _ = env.step(actions)

        step_count += 1
        reset_counts += dones.long()
        if hasattr(env, "last_goal_reached"):
            goal_reached_count += env.last_goal_reached.long()
            distance_goal_count += int(env.last_goal_distance_reached.sum().item())
            stable_goal_count += int(env.last_goal_stable.sum().item())
            goal_verified_level_histogram += torch.bincount(
                env.last_goal_verified_level[env.last_goal_distance_reached], minlength=9
            )[:9]
            goal_verified_contact_histogram += torch.bincount(
                env.last_goal_verified_contacts[env.last_goal_distance_reached].clamp_max(8), minlength=9
            )[:9]
            goal_physical_coverage_histogram += torch.bincount(
                env.last_distance_goal_coverage[env.last_goal_distance_reached].clamp_max(8), minlength=9
            )[:9]
            goal_broad_coverage_histogram += torch.bincount(
                env.last_distance_goal_broad_coverage[env.last_goal_distance_reached].clamp_max(8), minlength=9
            )[:9]
            goal_physical_level_histogram += env.last_distance_goal_physical_levels.sum(dim=0)
            goal_broad_level_histogram += env.last_distance_goal_broad_levels.sum(dim=0)
            raw_stair_step_count += int(env.last_raw_stair_progress_increment.sum().item())
            physical_stair_step_count += int(env.last_stair_physical_step_increment.sum().item())
            stair_contact_failure_count += int(env.last_stair_failure_contact.sum().item())
            stair_lane_failure_count += int(env.last_stair_failure_lane.sum().item())
            stair_unsafe_end_failure_count += int(env.last_stair_failure_unsafe_end.sum().item())
            stair_timeout_failure_count += int(env.last_stair_failure_timeout.sum().item())
            hip_clearance_sum += env.stair_hip_clearance.mean().item()
            hip_clearance_low_count += (env.stair_hip_clearance < 0.45).sum().item()
            if torch.any(env.last_stair_failure_contact):
                hip_clearance_contact_failure_sum += env.stair_hip_clearance[
                    env.last_stair_failure_contact
                ].sum().item()
            failures = (
                env.last_stair_failure_contact
                | env.last_stair_failure_lane
                | env.last_stair_failure_unsafe_end
                | env.last_stair_failure_timeout
            )
            stair_failure_level_histogram += torch.bincount(
                env.last_stair_failure_level[failures], minlength=9
            )[:9]
            failed_body = env.last_stair_failure_body_index[env.last_stair_failure_contact]
            stair_contact_body_histogram += torch.bincount(
                failed_body, minlength=len(termination_body_names)
            )[: len(termination_body_names)]
            if verified_step_term_index is not None:
                verified_stair_step_count += int(
                    (env.reward_manager._step_reward[:, verified_step_term_index] > 0.0).sum().item()
                )
        reward_sum += rewards
        forward_speed_sum += env.robot.data.root_lin_vel_b[:, 0]
        lateral_offset_sum += (env.robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]).abs()
        forward_distance = env.robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        max_forward_distance = torch.maximum(max_forward_distance, forward_distance)
        if args_cli.trace_course_every > 0 and (
            step_count % args_cli.trace_course_every == 0 or torch.any(dones)
        ):
            relative_root = env.robot.data.root_pos_w - env.scene.env_origins
            traced_env = 0
            root_quat = env.robot.data.root_quat_w[traced_env]
            root_yaw = torch.atan2(
                2.0 * (root_quat[0] * root_quat[3] + root_quat[1] * root_quat[2]),
                1.0 - 2.0 * (root_quat[2] * root_quat[2] + root_quat[3] * root_quat[3]),
            ).item()
            valid_treads = env.terrain_geometry.treads[traced_env, :, 5] > 0.5
            nearest = (
                env.terrain_geometry.treads[traced_env, valid_treads, 0].min().item()
                if torch.any(valid_treads)
                else float("nan")
            )
            foothold = env._foothold_target_features()[traced_env].reshape(-1, 4)
            print(
                f"[TRACE] step={step_count} reset={int(dones[traced_env].item())} "
                f"x={relative_root[traced_env, 0].item():.3f} "
                f"y={relative_root[traced_env, 1].item():.3f} "
                f"z={relative_root[traced_env, 2].item():.3f} "
                f"yaw={root_yaw:.3f} "
                f"max_x={max_forward_distance[traced_env].item():.3f} "
                f"mode={int(env.stair_mode[traced_env].item())} "
                f"gate={env.stair_gate_strength[traced_env].item():.2f} "
                f"foot_dx={[round(x, 3) for x in foothold[:, 0].detach().cpu().tolist()]} "
                f"foot_conf={[round(x, 2) for x in foothold[:, 3].detach().cpu().tolist()]} "
                f"safe_inside={int(env.safe_tread_foot_inside[traced_env].any().item())} "
                f"safe_overlap={env.safe_tread_max_overlap[traced_env].item():.3f} "
                f"feet_x={[round(x, 3) for x in env.safe_tread_feet_x[traced_env].detach().cpu().tolist()]} "
                f"direction={env.terrain_geometry.direction[traced_env].item():+.0f} "
                f"confidence={env.terrain_geometry.stair_confidence[traced_env].item():.2f} "
                f"nearest_tread={nearest:.3f} "
                f"ik={[int(x) for x in env.foothold_ik_active[traced_env].tolist()]} "
                f"target_lock={[int(x) for x in env.foothold_lock_valid[traced_env].tolist()]} "
                f"settle={int(env.stair_settle_active[traced_env].item())}"
            )
        action_magnitude_sum += actions.abs().mean().item()
        if safe_tread_term_index is not None:
            safe_tread_reward = env.reward_manager._step_reward[:, safe_tread_term_index]
            safe_tread_positive_sum += (safe_tread_reward > 0.0).float().mean().item()
            safe_tread_reward_sum += safe_tread_reward.mean().item()
            safe_tread_first_contact_sum += env.safe_tread_first_contact.any(dim=1).float().mean().item()
            tread_candidate_contact_count += (
                env.safe_tread_first_contact & env.safe_tread_has_candidate.unsqueeze(1)
            ).sum().item()
            tread_near_candidate_contact_count += (
                env.safe_tread_first_contact & env.safe_tread_near_candidate
            ).sum().item()
            safe_tread_contact_count += env.safe_tread_supported_first_contact.sum().item()
            unsafe_tread_contact_count += env.safe_tread_unsafe_first_contact.sum().item()
            confirmed_tread_contact_count += env.safe_tread_confirmed_touchdown.sum().item()
            confirmed = env.safe_tread_confirmed_touchdown
            if torch.any(confirmed):
                capture_forward = env.safe_tread_capture_forward[confirmed]
                capture_lateral = env.safe_tread_capture_lateral[confirmed]
                capture_confirmed_count += capture_forward.numel()
                capture_forward_sum += capture_forward.sum().item()
                capture_lateral_abs_sum += capture_lateral.abs().sum().item()
                capture_envelope_count += (
                    (capture_forward >= -0.15)
                    & (capture_forward <= 0.35)
                    & (capture_lateral.abs() <= 0.20)
                ).sum().item()
            if args_cli.trace_footsteps:
                for event in env.safe_tread_confirmed_touchdown.nonzero(as_tuple=False):
                    event_env, foot_index = event.tolist()
                    sole_pos = env.robot.data.body_pos_w[event_env, env.feet_body_ids[foot_index]].clone()
                    sole_pos[2] -= 0.04
                    course_pos = sole_pos - env.scene.env_origins[event_env]
                    print(
                        f"[FOOTSTEP] step={step_count} env={event_env} foot={'L' if foot_index == 0 else 'R'} "
                        f"course_xyz={course_pos.detach().cpu().tolist()} "
                        f"safe_edges_body_x={env.safe_tread_confirmed_safe_edges[event_env, foot_index].detach().cpu().tolist()} "
                        f"tread_z={env.safe_tread_confirmed_tread_height[event_env, foot_index].item():.3f}m "
                        f"overlap={env.safe_tread_confirmed_overlap[event_env, foot_index].item():.3f}m "
                        f"height_error={env.safe_tread_confirmed_height_error[event_env, foot_index].item():.3f}m"
                    )
            safe_tread_candidate_sum += env.safe_tread_has_candidate.float().mean().item()
            safe_tread_inside_sum += env.safe_tread_foot_inside.any(dim=1).float().mean().item()
            candidate_envs = env.safe_tread_has_candidate
            if torch.any(candidate_envs):
                safe_tread_min_distance = torch.minimum(
                    safe_tread_min_distance, env.safe_tread_min_distance[candidate_envs].amin()
                )
                candidate_foot_x_min = torch.minimum(
                    candidate_foot_x_min, env.safe_tread_feet_x[candidate_envs].amin()
                )
                candidate_foot_x_max = torch.maximum(
                    candidate_foot_x_max, env.safe_tread_feet_x[candidate_envs].amax()
                )
                candidate_near_min = torch.minimum(
                    candidate_near_min, env.safe_tread_near[candidate_envs].amin()
                )
                candidate_far_max = torch.maximum(
                    candidate_far_max, env.safe_tread_far[candidate_envs].amax()
                )
                safe_tread_max_overlap = torch.maximum(
                    safe_tread_max_overlap, env.safe_tread_max_overlap[candidate_envs].amax()
                )
                safe_tread_contact_overlap = torch.maximum(
                    safe_tread_contact_overlap, env.safe_tread_contact_overlap[candidate_envs].amax()
                )
        if hasattr(env, "stair_mode"):
            detector_stair_sum += (env.stair_mode != 0).float().mean().item()
        if hasattr(env, "stair_settle_active"):
            stair_settle_sum += env.stair_settle_active.float().mean().item()
            if torch.any(env.stair_settle_active):
                stair_settle_yaw_max = max(
                    stair_settle_yaw_max,
                    env.command_generator.command[env.stair_settle_active, 2].abs().amax().item(),
                )
        if hasattr(env, "foothold_ik_active"):
            foothold_ik_active_sum += env.foothold_ik_active.any(dim=1).float().mean().item()
            foothold_guard_active_sum += env.foothold_guard_active.any(dim=1).float().mean().item()
            foothold_lock_active_sum += env.foothold_lock_valid.any(dim=1).float().mean().item()
        if getattr(env, "terrain_geometry", None) is not None:
            stair_confidence_sum += env.terrain_geometry.stair_confidence.mean().item()
            raw_stair_detection_sum += (env.terrain_geometry.direction != 0).float().mean().item()
            geometry_cfg = env.cfg.scene.depth_camera.geometry
            valid_treads = (env.terrain_geometry.treads[..., 5] > 0.5).sum(dim=1)
            nearest_tread = torch.where(
                env.terrain_geometry.treads[..., 5] > 0.5,
                env.terrain_geometry.treads[..., 0],
                torch.full_like(env.terrain_geometry.treads[..., 0], torch.inf),
            ).amin(dim=1)
            gate_evidence = (
                (env.terrain_geometry.stair_confidence >= geometry_cfg.gate_enter_confidence)
                & (valid_treads >= geometry_cfg.gate_min_valid_treads)
                & (nearest_tread >= geometry_cfg.gate_min_nearest_tread)
                & (nearest_tread <= geometry_cfg.gate_max_nearest_tread)
                & (env.terrain_geometry.direction != 0)
            )
            gate_evidence_sum += gate_evidence.float().mean().item()
            if hasattr(env, "stair_step_width") and env.cfg.scene.depth_camera.geometry.bootstrap_stair_row == 2:
                detected = env.terrain_geometry.treads
                detected_z = (
                    env.robot.data.root_pos_w[:, None, 2]
                    + env.terrain_geometry.reference_height[:, None]
                    + detected[..., 2]
                )
                relative_level = (
                    env.scene.env_origins[:, None, 2] - detected_z
                ) / env.stair_step_height[:, None].clamp_min(1.0e-6)
                level = relative_level.round().long()
                heading_x = env._root_heading_xy()[:, 0:1]
                root_x = env.robot.data.root_pos_w[:, 0:1] - env.scene.env_origins[:, 0:1]
                detected_center = root_x + heading_x * 0.5 * (detected[..., 0] + detected[..., 1])
                expected_center = env.stair_first_riser_distance[:, None] + (
                    level.float() - 0.5
                ) * env.stair_step_width[:, None]
                center_error = (detected_center - expected_center).abs()
                width_error = (detected[..., 3] - env.stair_step_width[:, None]).abs()
                audit = (
                    (detected[..., 5] > 0.5)
                    & (detected[..., 4] >= 0.70)
                    & (env.stair_mode[:, None] < 0)
                    & ~dones[:, None]
                    & (level >= 1)
                    & (level < env.stair_physical_covered.shape[1])
                    & ((relative_level - level).abs() <= 0.25)
                )
                stair_tread_audit_count += int(audit.sum().item())
                stair_tread_center_error_sum += center_error[audit].sum().item()
                signed_error = detected_center - expected_center
                stair_tread_center_signed_error_sum += signed_error[audit].sum().item()
                stair_tread_width_error_sum += width_error[audit].sum().item()
                stair_tread_center_error_large += int((center_error[audit] > 0.03).sum().item())
                stair_tread_center_error_forward += int((signed_error[audit] > 0.03).sum().item())

        if camera_debug_enabled:
            depth_cfg = env.cfg.scene.depth_camera
            env_idx = max(0, min(args_cli.save_depth_env, env.num_envs - 1))
            if rgb_debug_enabled:
                rgb = env.depth_camera.data.output["rgb"][env_idx].detach().cpu().numpy()
                if rgb.shape[-1] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                else:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if args_cli.show_rgb:
                    cv2.imshow("ELF3 D435i RGB", rgb)
                if rgb_save_dir is not None and args_cli.save_depth_every > 0:
                    should_save = step_count % args_cli.save_depth_every == 0
                    under_limit = args_cli.save_depth_max <= 0 or rgb_save_count < args_cli.save_depth_max
                    if should_save and under_limit:
                        rgb_path = os.path.join(rgb_save_dir, f"rgb_env{env_idx:03d}_step{step_count:06d}.png")
                        cv2.imwrite(rgb_path, rgb)
                        rgb_save_count += 1
                        if rgb_save_count == 1 or rgb_save_count % 10 == 0:
                            print(f"[RGB] Saved {rgb_save_count} frame(s); latest: {rgb_path}")
            depth_tensor = env.depth_camera.data.output["distance_to_image_plane"][env_idx, ..., 0]
            depth = depth_tensor.detach().float().cpu().numpy()
            depth = np.nan_to_num(
                depth,
                nan=depth_cfg.max_range,
                posinf=depth_cfg.max_range,
                neginf=depth_cfg.min_range,
            )
            depth = np.clip(depth, depth_cfg.min_range, depth_cfg.max_range)
            depth_norm = (depth - depth_cfg.min_range) / (depth_cfg.max_range - depth_cfg.min_range)
            depth_gray = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)
            if args_cli.show_depth:
                cv2.imshow("ELF3 waist depth", depth_color)
            if depth_save_dir is not None and args_cli.save_depth_every > 0:
                should_save = step_count % args_cli.save_depth_every == 0
                under_limit = args_cli.save_depth_max <= 0 or depth_save_count < args_cli.save_depth_max
                if should_save and under_limit:
                    depth_path = os.path.join(depth_save_dir, f"depth_env{env_idx:03d}_step{step_count:06d}.png")
                    cv2.imwrite(depth_path, depth_color)
                    depth_save_count += 1
                    if depth_save_count == 1 or depth_save_count % 10 == 0:
                        print(f"[DEPTH] Saved {depth_save_count} frame(s); latest: {depth_path}")

            if geometry_debug_enabled and env.terrain_geometry is not None:
                foot_overlay = {}
                if safe_tread_term_index is not None:
                    foot_bottom_w = env.robot.data.body_pos_w[env_idx, env.feet_body_ids, 2] - 0.04
                    reference_w = (
                        env.robot.data.root_pos_w[env_idx, 2]
                        + env.terrain_geometry.reference_height[env_idx]
                    )
                    foot_contact = env.contact_sensor.data.net_forces_w[
                        env_idx, env.feet_cfg.body_ids, 2
                    ] > 5.0
                    foot_overlay = {
                        "feet_x": env.safe_tread_feet_x[env_idx].detach().cpu().numpy(),
                        "feet_z": (foot_bottom_w - reference_w).detach().cpu().numpy(),
                        "foot_contact": foot_contact.detach().cpu().numpy(),
                        "foot_confirmed": (env.safe_tread_support_count[env_idx] >= 3).detach().cpu().numpy(),
                        "safe_edge_margin": env.reward_manager.get_term_cfg("safe_tread_landing").params["edge_margin"],
                    }
                geometry_image = draw_stair_geometry_debug(
                    depth_tensor,
                    env.terrain_geometry,
                    env_idx,
                    depth_cfg.min_range,
                    depth_cfg.max_range,
                    stair_mode=getattr(env, "stair_mode", None),
                    **foot_overlay,
                )
                if env.terrain_geometry.surface_geometry is not None:
                    from legged_lab.perception.tread_surfaces import draw_surface_geometry_debug

                    heading = env._root_heading_xy()[env_idx]
                    lateral = torch.stack((-heading[1], heading[0]))
                    feet = env.robot.data.body_pos_w[env_idx, env.feet_body_ids]
                    quat = env.robot.data.body_quat_w[env_idx, env.feet_body_ids]
                    forward = quat_apply(quat, torch.tensor([[1., 0., 0.]], device=env.device).expand(2, -1))
                    sole_center = feet[:, :2] + 0.03 * forward[:, :2] - env.robot.data.root_pos_w[env_idx, :2]
                    feet_xy = torch.stack((sole_center @ heading, sole_center @ lateral), dim=-1)
                    feet_yaw = torch.atan2(forward[:, :2] @ lateral, forward[:, :2] @ heading)
                    surface_rgb = env.depth_camera.data.output.get("rgb")
                    snapshot = env.terrain_geometry.surface_geometry[env_idx]
                    memories = env.terrain_geometry.surface_memory
                    display = snapshot if memories is None else memories[env_idx]
                    geometry_image = draw_surface_geometry_debug(
                        display,
                        None if surface_rgb is None else surface_rgb[env_idx], depth_tensor,
                        env.terrain_geometry.processed_image_shape,
                        feet_xy=feet_xy.detach().cpu().numpy(), feet_yaw=feet_yaw.detach().cpu().numpy(),
                        camera_result=snapshot,
                    )
                    if getattr(env, "stair_step_enabled", False):
                        controller = env.step_controllers[env_idx]
                        geometry_image = cv2.copyMakeBorder(geometry_image, 60, 0, 0, 0,
                                                           cv2.BORDER_CONSTANT, value=(20, 20, 20))
                        title = f"STEP: {controller.phase.name}  support L/R: {controller.support_valid.astype(int).tolist()}"
                        cv2.putText(geometry_image, title, (15, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (240, 240, 240), 2)
                if args_cli.show_geometry:
                    cv2.imshow("ELF3 stair geometry", geometry_image)
                if geometry_save_dir is not None and args_cli.save_depth_every > 0:
                    should_save = step_count % args_cli.save_depth_every == 0
                    under_limit = args_cli.save_depth_max <= 0 or geometry_save_count < args_cli.save_depth_max
                    if should_save and under_limit:
                        geometry_path = os.path.join(
                            geometry_save_dir, f"geometry_env{env_idx:03d}_step{step_count:06d}.png"
                        )
                        cv2.imwrite(geometry_path, geometry_image)
                        geometry_save_count += 1
                        treads = env.terrain_geometry.treads[env_idx]
                        valid_treads = treads[treads[:, 5] > 0.5]
                        point_bounds = env.terrain_geometry.point_bounds[env_idx]
                        valid_points = env.terrain_geometry.valid_point_count[env_idx]
                        profile = env.terrain_geometry.profile_z[env_idx]
                        profile_valid = env.terrain_geometry.profile_valid[env_idx]
                        if torch.any(profile_valid):
                            profile_range = (
                                profile[profile_valid].min().item(),
                                profile[profile_valid].max().item(),
                            )
                            valid_profile_x = env.terrain_geometry.profile_x[profile_valid]
                            profile_span = (valid_profile_x.min().item(), valid_profile_x.max().item())
                        else:
                            profile_range = (float("nan"), float("nan"))
                            profile_span = (float("nan"), float("nan"))
                        print(
                            f"[GEOMETRY] Saved {geometry_path}; "
                            f"mode={int(env.stair_mode[env_idx].item()) if hasattr(env, 'stair_mode') else 0:+d}, "
                            f"direction={env.terrain_geometry.direction[env_idx].item():+.0f}, "
                            f"confidence={env.terrain_geometry.stair_confidence[env_idx].item():.2f}, "
                            f"roughness={env.terrain_geometry.roughness[env_idx].item():.3f}m, "
                            f"valid_points={valid_points.item()}, "
                            f"profile_bins={profile_valid.sum().item()}, "
                            f"profile_x={[round(x, 3) for x in profile_span]}, "
                            f"profile_z={[round(z, 3) for z in profile_range]}, "
                            f"point_bounds={point_bounds.detach().cpu().numpy().round(3).tolist()}, "
                            f"treads={valid_treads[:, :5].detach().cpu().numpy().round(3).tolist()}"
                        )
            if (rgb_debug_enabled or args_cli.show_depth or args_cli.show_geometry) and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            failure_resets = reset_counts - goal_reached_count
            survival_rate = (failure_resets == 0).float().mean().item()
            print(f"[EVAL] checkpoint={resume_path}")
            print(f"[EVAL] steps={step_count} duration={step_count * env.step_dt:.2f}s envs={env.num_envs}")
            print(f"[EVAL] survival_rate={survival_rate:.4f} total_resets={reset_counts.sum().item()}")
            if hasattr(env, "last_goal_reached"):
                print(f"[EVAL] stair_goal_reached={goal_reached_count.sum().item()}")
                print(f"[EVAL] stair_distance_end_reached={distance_goal_count}")
                print(f"[EVAL] stair_goal_stable_support={stable_goal_count}")
                print(f"[EVAL] stair_goal_verified_level={goal_verified_level_histogram.tolist()}")
                print(f"[EVAL] stair_goal_verified_contacts={goal_verified_contact_histogram.tolist()}")
                print(f"[EVAL] stair_goal_physical_coverage={goal_physical_coverage_histogram.tolist()}")
                print(f"[EVAL] stair_goal_broad_coverage={goal_broad_coverage_histogram.tolist()}")
                print(f"[EVAL] stair_goal_physical_by_level={goal_physical_level_histogram.tolist()}")
                print(f"[EVAL] stair_goal_broad_by_level={goal_broad_level_histogram.tolist()}")
                print(f"[EVAL] stair_failure_resets={failure_resets.sum().item()}")
                print(
                    f"[EVAL] stair_failure_causes="
                    f"contact:{stair_contact_failure_count},"
                    f"lane:{stair_lane_failure_count},"
                    f"unsafe_end:{stair_unsafe_end_failure_count},"
                    f"timeout:{stair_timeout_failure_count}"
                )
                print(f"[EVAL] stair_failure_levels={stair_failure_level_histogram.tolist()}")
                print(
                    f"[EVAL] hip_clearance_above_support="
                    f"mean:{hip_clearance_sum / step_count:.4f}m,"
                    f"contact_failure_mean:{hip_clearance_contact_failure_sum / max(stair_contact_failure_count, 1):.4f}m,"
                    f"below_0.45_fraction:{hip_clearance_low_count / (step_count * env.num_envs):.4f}"
                )
                print(
                    "[EVAL] stair_contact_failure_bodies="
                    + str({
                        name: count
                        for name, count in zip(termination_body_names, stair_contact_body_histogram.tolist())
                        if count
                    })
                )
                print(f"[EVAL] raw_stair_step_progress={raw_stair_step_count}")
                print(f"[EVAL] physical_stair_step_coverage={physical_stair_step_count}")
                print(f"[EVAL] verified_stair_step_progress={verified_stair_step_count}")
                print(
                    f"[EVAL] stair_episode_success_rate="
                    f"{goal_reached_count.sum().item() / max(reset_counts.sum().item(), 1):.4f}"
                )
                if env.num_envs <= 16:
                    print(f"[EVAL] stair_goal_by_env={goal_reached_count.tolist()}")
                    print(f"[EVAL] stair_failure_by_env={failure_resets.tolist()}")
            print(f"[EVAL] mean_forward_speed={forward_speed_sum.mean().item() / step_count:.4f}m/s")
            print(f"[EVAL] mean_max_forward_distance={max_forward_distance.mean().item():.4f}m")
            relative_root = env.robot.data.root_pos_w - env.scene.env_origins
            print(f"[EVAL] mean_final_forward_distance={relative_root[:, 0].mean().item():.4f}m")
            print(f"[EVAL] mean_abs_lateral_offset={lateral_offset_sum.mean().item() / step_count:.4f}m")
            print(f"[EVAL] mean_final_abs_lateral_offset={relative_root[:, 1].abs().mean().item():.4f}m")
            print(f"[EVAL] mean_final_root_height={relative_root[:, 2].mean().item():.4f}m")
            print(f"[EVAL] mean_reward_per_step={reward_sum.mean().item() / step_count:.6f}")
            print(f"[EVAL] mean_abs_action={action_magnitude_sum / step_count:.6f}")
            if safe_tread_term_index is not None:
                print(f"[EVAL] safe_tread_positive_fraction={safe_tread_positive_sum / step_count:.6f}")
                print(f"[EVAL] mean_safe_tread_reward={safe_tread_reward_sum / step_count:.6f}")
                print(f"[EVAL] foot_first_contact_fraction={safe_tread_first_contact_sum / step_count:.6f}")
                print(
                    f"[EVAL] safe_tread_touchdowns={safe_tread_contact_count}/"
                    f"{tread_candidate_contact_count} candidate_contact_events"
                )
                print(
                    f"[EVAL] safe_tread_near_touchdowns={safe_tread_contact_count}/"
                    f"{tread_near_candidate_contact_count} near_candidate_contact_events"
                )
                print(f"[EVAL] unsafe_tread_first_contacts={unsafe_tread_contact_count}")
                print(f"[EVAL] confirmed_safe_tread_touchdowns={confirmed_tread_contact_count}")
                if capture_confirmed_count:
                    print(
                        f"[EVAL] confirmed_capture_point="
                        f"forward_mean:{capture_forward_sum / capture_confirmed_count:.4f}m,"
                        f"lateral_abs_mean:{capture_lateral_abs_sum / capture_confirmed_count:.4f}m,"
                        f"broad_envelope:{capture_envelope_count}/{capture_confirmed_count}"
                    )
                print(f"[EVAL] tread_candidate_fraction={safe_tread_candidate_sum / step_count:.6f}")
                print(f"[EVAL] foot_inside_tread_fraction={safe_tread_inside_sum / step_count:.6f}")
                print(
                    "[EVAL] tread_alignment="
                    f"min_distance={safe_tread_min_distance.item():.4f}m, "
                    f"foot_x=[{candidate_foot_x_min.item():.4f}, {candidate_foot_x_max.item():.4f}]m, "
                    f"safe_edges=[{candidate_near_min.item():.4f}, {candidate_far_max.item():.4f}]m, "
                    f"max_overlap={safe_tread_max_overlap.item():.4f}m, "
                    f"contact_overlap={safe_tread_contact_overlap.item():.4f}m"
                )
            if hasattr(getattr(runner.alg.policy, "actor", None), "last_applied_residual"):
                print(f"[EVAL] geometry_gate_fraction={geometry_gate_sum / step_count:.6f}")
                print(f"[EVAL] geometry_residual_abs={geometry_residual_abs_sum / step_count:.6f}")
                print(f"[EVAL] geometry_action_disabled={args_cli.disable_geometry_action}")
                print(f"[EVAL] foothold_targets_disabled={args_cli.disable_foothold_targets}")
                print(f"[EVAL] foothold_actor_scale={env.cfg.scene.depth_camera.geometry.foothold_actor_scale:.3f}")
                print(f"[EVAL] foothold_target_distance_scale={env.cfg.scene.depth_camera.geometry.foothold_target_distance_scale or env.cfg.scene.depth_camera.geometry.max_forward:.3f}m")
                print(f"[EVAL] descending_edge_offset_m={env.cfg.scene.depth_camera.geometry.descending_edge_offset_m:.3f}m")
                print(f"[EVAL] leg_residual_multiplier={args_cli.leg_residual_multiplier or 1.0:.3f}")
                print(f"[EVAL] foothold_target_valid_fraction={foothold_target_valid_sum / step_count:.6f}")
                if args_cli.measure_target_sensitivity:
                    print(f"[EVAL] target_action_sensitivity={target_action_sensitivity_sum / step_count:.6f}")
            if hasattr(env, "stair_mode"):
                print(f"[EVAL] detector_stair_fraction={detector_stair_sum / step_count:.6f}")
                if stair_tread_audit_count:
                    print(
                        "[EVAL] detected_tread_geometry_error="
                        f"center_abs_mean:{stair_tread_center_error_sum / stair_tread_audit_count:.4f}m,"
                        f"center_signed_mean:{stair_tread_center_signed_error_sum / stair_tread_audit_count:.4f}m,"
                        f"width_abs_mean:{stair_tread_width_error_sum / stair_tread_audit_count:.4f}m,"
                        f"center_over_3cm:{stair_tread_center_error_large}/{stair_tread_audit_count},"
                        f"forward_over_3cm:{stair_tread_center_error_forward}/{stair_tread_audit_count}"
                    )
                print(f"[EVAL] mean_stair_confidence={stair_confidence_sum / step_count:.6f}")
                print(f"[EVAL] raw_stair_detection_fraction={raw_stair_detection_sum / step_count:.6f}")
                print(f"[EVAL] gate_evidence_fraction={gate_evidence_sum / step_count:.6f}")
            if hasattr(env, "stair_settle_active"):
                print(f"[EVAL] stair_settle_fraction={stair_settle_sum / step_count:.6f}")
                print(f"[EVAL] stair_settle_max_abs_yaw_command={stair_settle_yaw_max:.6f}")
                print(f"[EVAL] stair_settle_enabled={env.stair_settle_enabled}")
                print(f"[EVAL] blind_during_stair_settle={env.cfg.scene.depth_camera.geometry.blind_during_stair_settle}")
            if hasattr(env, "foothold_ik_active"):
                print(f"[EVAL] foothold_ik_enabled={env.cfg.scene.depth_camera.geometry.foothold_ik_enabled}")
                print(f"[EVAL] foothold_swing_trajectory_enabled={env.cfg.scene.depth_camera.geometry.foothold_swing_trajectory_enabled}")
                print(f"[EVAL] foothold_ik_active_fraction={foothold_ik_active_sum / step_count:.6f}")
                print(f"[EVAL] foothold_overshoot_guard_enabled={env.cfg.scene.depth_camera.geometry.foothold_overshoot_guard_enabled}")
                print(f"[EVAL] foothold_overshoot_guard_active_fraction={foothold_guard_active_sum / step_count:.6f}")
                print(f"[EVAL] foothold_target_lock_enabled={env.cfg.scene.depth_camera.geometry.foothold_target_lock_enabled}")
                print(f"[EVAL] foothold_target_lock_active_fraction={foothold_lock_active_sum / step_count:.6f}")
                print(f"[EVAL] stair_heading_feedback_enabled={env.cfg.scene.depth_camera.geometry.stair_heading_feedback_enabled}")
            if hasattr(env, "_current_depth_camera_pose"):
                live_camera_pos, live_camera_quat = env._current_depth_camera_pose()
                cached_camera_pos = env.depth_camera.data.pos_w
                cached_camera_quat = env.depth_camera.data.quat_w_ros
                pose_error = torch.linalg.norm(live_camera_pos - cached_camera_pos, dim=1).mean()
                quat_alignment = torch.abs((live_camera_quat * cached_camera_quat).sum(dim=1)).mean()
                print(f"[EVAL] camera_cached_pose_error={pose_error.item():.6f}m")
                print(f"[EVAL] camera_quaternion_alignment={quat_alignment.item():.6f}")
                optical_forward = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
                forward_world = quat_apply(live_camera_quat[:1], optical_forward)[0]
                down_angle = torch.rad2deg(torch.atan2(-forward_world[2], forward_world[:2].norm()))
                print(f"[EVAL] camera_forward_world={forward_world.tolist()}")
                print(f"[EVAL] camera_down_angle={down_angle.item():.2f}deg")
            break


if __name__ == "__main__":
    try:
        play()
    except BaseException:
        simulation_app._app.post_quit(1)
        simulation_app.close(skip_cleanup=args_cli.headless)
        raise
    simulation_app.close(skip_cleanup=args_cli.headless)
