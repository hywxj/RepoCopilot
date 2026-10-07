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
from copy import deepcopy
from dataclasses import replace
import math

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
parser.add_argument("--stair_height", "--stair-height", type=float, default=None,
                    help="Continuous-course riser height in metres; default 0.11, optional curriculum starts at 0.04.")
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="Load policy weights but initialize a fresh optimizer for fine-tuning.",
)
parser.add_argument(
    "--policy_noise_std",
    type=float,
    default=None,
    help="Reset the loaded policy exploration standard deviation to this value.",
)
parser.add_argument(
    "--geometry_terrain_difficulty",
    type=float,
    default=None,
    help="Fix stair-bootstrap terrain difficulty from 0.0 to 1.0 for targeted training.",
)
parser.add_argument(
    "--enable_foothold_ik",
    action="store_true",
    help="Train with the bounded geometry-guided swing-foot correction enabled.",
)
parser.add_argument(
    "--enable_foothold_overshoot_guard",
    action="store_true",
    help="Train with the descending swing-foot forward reach guard enabled.",
)
parser.add_argument(
    "--enable_foothold_target_lock",
    action="store_true",
    help="Keep each swing-foot tread target until that foot lands.",
)
parser.add_argument(
    "--enable_foothold_swing_trajectory",
    action="store_true",
    help="Train with a world-fixed depth-derived swing-foot trajectory (also enables IK and target locking).",
)
parser.add_argument("--foothold_ik_gain", type=float, default=None)
parser.add_argument("--foothold_ik_max_joint_step", type=float, default=None)
parser.add_argument(
    "--enable_foothold_preview",
    action="store_true",
    help="Expose the next stair tread target before a foot starts swinging.",
)
parser.add_argument("--foothold_actor_scale", type=float, default=None)
parser.add_argument("--foothold_target_distance_scale", type=float, default=None)
parser.add_argument("--descending_edge_offset_m", type=float, default=None)
parser.add_argument("--leg_residual_multiplier", type=float, default=None)
parser.add_argument("--stair_yaw_lateral_gain", type=float, default=None)
parser.add_argument("--stair_yaw_rate_limit", type=float, default=None)
parser.add_argument("--verified_stair_progress_weight", type=float, default=None)
parser.add_argument("--stair_goal_weight", type=float, default=None)
parser.add_argument("--stair_failure_penalty", type=float, default=None)
parser.add_argument("--hip_clearance_penalty", type=float, default=None)
parser.add_argument("--stair_collision_force_penalty", type=float, default=None)
parser.add_argument("--blind_during_stair_settle", action="store_true")
parser.add_argument("--standing_pretrain", action="store_true", help="Pretrain a stationary skill on walk_elf3 without cameras.")
parser.add_argument("--step_skill_pretrain", action="store_true", help="Train the stair motor actor with simulation tread targets and no cameras; depth transfer remains required.")
parser.add_argument("--stair_stop_checkpoint", type=str, default=None,
                    help="Protect OBSERVE actions using a frozen, previously learned stop actor (training only).")
parser.add_argument("--stair_stop_anchor_coef", type=float, default=50.)
parser.add_argument("--stair_conditioning_only", action="store_true",
                    help="Freeze the stop-feedback network and train only newly added stair input weights.")
parser.add_argument("--stair_action_teacher", action="store_true",
                    help="Add training-only inverse-kinematics action-improvement labels; inference stays policy-only.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering for policies that consume depth-derived observations.
if not args_cli.step_skill_pretrain and args_cli.task is not None and ("sensor" in args_cli.task or "geometry" in args_cli.task):
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
import legged_lab.mdp as mdp
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.stair_height is not None:
        if not hasattr(env_cfg, "continuous") or not math.isfinite(args_cli.stair_height) or args_cli.stair_height <= 0:
            raise ValueError("--stair_height requires a continuous task and a finite positive height.")
        env_cfg.continuous = replace(env_cfg.continuous, step_height=args_cli.stair_height)
        env_cfg.scene.terrain_generator.sub_terrains["continuous"].course = env_cfg.continuous
    geometry_cfg = getattr(getattr(env_cfg.scene, "depth_camera", None), "geometry", None)
    if args_cli.step_skill_pretrain:
        if not bool(getattr(geometry_cfg, "step_control_enabled", False)):
            raise ValueError("--step_skill_pretrain requires a geometry_step task.")
        if args_cli.geometry_terrain_difficulty not in (None, 0.):
            raise ValueError("Motor initialization currently uses fixed 11 cm risers.")
        geometry_cfg.step_skill_pretrain = True
        env_cfg.scene.depth_camera.enable_depth_camera = False
        env_cfg.scene.terrain_generator.num_cols = env_cfg.scene.num_envs
        env_cfg.domain_rand.events.reset_base.params["pose_range"]["x"] = (.04, .10)
        agent_cfg.algorithm.learning_rate = 1.e-5
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.gamma = .998
        agent_cfg.algorithm.desired_kl = .01
        agent_cfg.algorithm.num_mini_batches = 4
        agent_cfg.policy.min_action_std = .04
        agent_cfg.policy.max_action_std = .15
    if args_cli.stair_conditioning_only:
        if not args_cli.step_skill_pretrain or not args_cli.reset_optimizer or not args_cli.resume:
            raise ValueError("Conditioning-only training requires motor pretraining, resume and --reset_optimizer.")
        agent_cfg.policy.train_stair_conditioning_only = True
        agent_cfg.algorithm.learning_rate = 1.e-4
    if args_cli.stair_action_teacher:
        if not bool(getattr(geometry_cfg, "step_control_enabled", False)):
            raise ValueError("Action guidance requires a stair-step task.")
        agent_cfg.algorithm.stair_action_teacher_coef = 20.
        agent_cfg.algorithm.use_clipped_value_loss = False
        agent_cfg.policy.low_noise_obs_indices = list(range(961, 970))
    if args_cli.geometry_terrain_difficulty is not None:
        if geometry_cfg is None or not geometry_cfg.bootstrap_stairs:
            raise ValueError("--geometry_terrain_difficulty requires a geometry stair-bootstrap task.")
        difficulty = args_cli.geometry_terrain_difficulty
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("--geometry_terrain_difficulty must be between 0 and 1.")
        env_cfg.scene.terrain_generator.difficulty_range = (difficulty, difficulty)
        env_cfg.scene.terrain_generator.num_cols = env_cfg.scene.num_envs
    if args_cli.enable_foothold_ik:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--enable_foothold_ik requires a geometry task.")
        geometry_cfg.foothold_ik_enabled = True
    if args_cli.enable_foothold_overshoot_guard:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--enable_foothold_overshoot_guard requires a geometry task.")
        geometry_cfg.foothold_overshoot_guard_enabled = True
    if args_cli.enable_foothold_target_lock:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--enable_foothold_target_lock requires a geometry task.")
        geometry_cfg.foothold_target_lock_enabled = True
    if args_cli.enable_foothold_swing_trajectory:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--enable_foothold_swing_trajectory requires a geometry task.")
        geometry_cfg.foothold_swing_trajectory_enabled = True
        geometry_cfg.foothold_ik_enabled = True
        geometry_cfg.foothold_target_lock_enabled = True
        if args_cli.foothold_ik_gain is None:
            geometry_cfg.foothold_ik_gain = 0.35
        if args_cli.foothold_ik_max_joint_step is None:
            geometry_cfg.foothold_ik_max_joint_step = 0.06
    if args_cli.enable_foothold_preview:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--enable_foothold_preview requires a geometry task.")
        geometry_cfg.foothold_preview_stance = True
    if args_cli.foothold_actor_scale is not None:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--foothold_actor_scale requires a geometry task.")
        if not 0.0 <= args_cli.foothold_actor_scale <= 1.0:
            raise ValueError("--foothold_actor_scale must be between 0 and 1.")
        geometry_cfg.foothold_actor_scale = args_cli.foothold_actor_scale
    if args_cli.foothold_target_distance_scale is not None:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--foothold_target_distance_scale requires a geometry task.")
        if not 0.1 <= args_cli.foothold_target_distance_scale <= geometry_cfg.max_forward:
            raise ValueError("--foothold_target_distance_scale must be 0.1 to max_forward metres.")
        geometry_cfg.foothold_target_distance_scale = args_cli.foothold_target_distance_scale
    if args_cli.descending_edge_offset_m is not None:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--descending_edge_offset_m requires a geometry task.")
        if not -0.15 <= args_cli.descending_edge_offset_m <= 0.15:
            raise ValueError("--descending_edge_offset_m must be between -0.15 and 0.15 metres.")
        geometry_cfg.descending_edge_offset_m = args_cli.descending_edge_offset_m
    if args_cli.stair_yaw_lateral_gain is not None:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--stair_yaw_lateral_gain requires a geometry task.")
        geometry_cfg.stair_yaw_lateral_gain = args_cli.stair_yaw_lateral_gain
    if args_cli.stair_yaw_rate_limit is not None:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--stair_yaw_rate_limit requires a geometry task.")
        geometry_cfg.stair_yaw_rate_limit = args_cli.stair_yaw_rate_limit
    if args_cli.blind_during_stair_settle:
        if geometry_cfg is None or not geometry_cfg.enabled:
            raise ValueError("--blind_during_stair_settle requires a geometry task.")
        geometry_cfg.blind_during_stair_settle = True
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
    if args_cli.foothold_ik_gain is not None:
        geometry_cfg.foothold_ik_gain = args_cli.foothold_ik_gain
    if args_cli.foothold_ik_max_joint_step is not None:
        geometry_cfg.foothold_ik_max_joint_step = args_cli.foothold_ik_max_joint_step

    if args_cli.standing_pretrain:
        if args_cli.task != "walk_elf3":
            raise ValueError("--standing_pretrain requires --task=walk_elf3.")
        env_cfg.scene.terrain_type = "plane"
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.depth_camera.enable_depth_camera = False
        env_cfg.scene.max_episode_length_s = 8.0
        env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.rel_standing_envs = 1.0
        env_cfg.commands.heading_command = False
        env_cfg.domain_rand.events.push_robot = None
        env_cfg.domain_rand.events.reset_base.params["pose_range"] = {}
        env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
        env_cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.noise.add_noise = False
        for name in ("gait_feet_frc_perio", "gait_feet_spd_perio", "gait_feet_frc_perio_penalize",
                     "feet_clearance", "swing_foot_forward", "feet_air_time", "idle_penalty",
                     "stand_still", "arm_swing", "fast_walk_height", "stance_knee_extension"):
            if hasattr(env_cfg.reward, name):
                setattr(env_cfg.reward, name, None)
        env_cfg.reward.track_lin_vel_xy_exp.params["std"] = 0.12
        env_cfg.reward.track_ang_vel_z_exp.params["std"] = 0.20
        feet = ["l_ankle_x_link", "r_ankle_x_link"]
        env_cfg.reward.standing_support_quality = RewardTermCfg(
            func=mdp.standing_support_quality, weight=5.0,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=feet, preserve_order=True),
                    "asset_cfg": SceneEntityCfg("robot", body_names=feet, preserve_order=True)},
        )
        agent_cfg.amp_reward_coef = 0.0
        agent_cfg.amp_task_reward_lerp = 1.0
        agent_cfg.amp_num_preload_transitions = 8192
        agent_cfg.algorithm.learning_rate = 1.e-4
        agent_cfg.algorithm.desired_kl = 0.005
        agent_cfg.save_interval = 16

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)
    if args_cli.standing_pretrain:
        env.standing_body_weight = env.robot.root_physx_view.get_masses().sum(dim=1).to(env.device) * 9.81
    if args_cli.leg_residual_multiplier is not None:
        if not hasattr(env, "left_leg_ids") or not 1.0 <= args_cli.leg_residual_multiplier <= 4.0:
            raise ValueError("--leg_residual_multiplier requires ELF3 geometry and a value from 1 to 4.")
        scales = [1.0] * env.num_actions
        for joint_id in (*env.left_leg_ids, *env.right_leg_ids):
            scales[joint_id] = args_cli.leg_residual_multiplier
        agent_cfg.policy.residual_action_scales = scales

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    fresh_critic = None
    if (args_cli.standing_pretrain or (args_cli.stair_conditioning_only and not args_cli.stair_action_teacher)) and args_cli.reset_optimizer:
        fresh_critic = {key: value.clone() for key, value in runner.alg.policy.critic.state_dict().items()}

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)
        if fresh_critic is not None:
            runner.alg.policy.critic.load_state_dict(fresh_critic)
            print("[INFO]: Initialized critic for the new pretraining reward objective.")
        if args_cli.policy_noise_std is not None:
            if args_cli.policy_noise_std <= 0:
                raise ValueError("--policy_noise_std must be greater than zero.")
            with torch.no_grad():
                if hasattr(runner.alg.policy, "std"):
                    runner.alg.policy.std.fill_(args_cli.policy_noise_std)
                elif hasattr(runner.alg.policy, "log_std"):
                    runner.alg.policy.log_std.fill_(math.log(args_cli.policy_noise_std))
                else:
                    raise AttributeError("The loaded policy does not expose std or log_std.")
            print(f"[INFO]: Reset policy noise std to {args_cli.policy_noise_std}.")

    if args_cli.stair_stop_checkpoint is not None:
        if not bool(getattr(env, "stair_step_enabled", False)) or agent_cfg.empirical_normalization:
            raise ValueError("Stop protection requires a stair-step task without empirical normalization.")
        reference = deepcopy(runner.alg.policy)
        reference.load_state_dict(torch.load(args_cli.stair_stop_checkpoint, map_location=env.device,
                                            weights_only=False)["model_state_dict"])
        runner.alg.set_stop_actor_anchor(reference.actor, [runner.alg.policy.base_actor_obs_dim+1], args_cli.stair_stop_anchor_coef)
        del reference
        agent_cfg.stair_stop_checkpoint = os.path.abspath(args_cli.stair_stop_checkpoint)
        agent_cfg.stair_stop_anchor_coef = args_cli.stair_stop_anchor_coef
        print(f"[INFO] Protecting OBSERVE actions from {args_cli.stair_stop_checkpoint}.")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=not (bool(getattr(env, "stair_step_enabled", False)) or
                                           bool(getattr(env, "continuous_course_enabled", False)) or
                                           args_cli.standing_pretrain))


if __name__ == "__main__":
    try:
        train()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=bool(args_cli.headless and args_cli.task and
                                             ("geometry_step_" in args_cli.task or
                                              args_cli.task.startswith("elf3_continuous_") or args_cli.standing_pretrain)))
