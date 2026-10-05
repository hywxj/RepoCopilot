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

import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.base.base_config import CommandRangesCfg, HeightScannerCfg
from legged_lab.envs.elf3.walk_cfg import Elf3WalkAgentCfg, Elf3WalkFlatEnvCfg
from legged_lab.terrains import (
    ATEC_OBSTACLE_COURSE_TERRAINS_CFG,
    ELF3_GEOMETRY_COURSE_TERRAINS_CFG,
    ELF3_GEOMETRY_EVAL_COURSE_TERRAINS_CFG,
    ELF3_STAIRS_CURRICULUM_TERRAINS_CFG,
)
from legged_lab.sensors.camera import GeometryPerceptionCfg
from legged_lab.sensors.camera.camera_cfgs import TiledD435iCameraCfg, TiledD455CameraCfg


@configclass
class Elf3WalkTerrainTeacherEnvCfg(Elf3WalkFlatEnvCfg):
    """ELF3 terrain-aware teacher.

    The actor is allowed to see a local height scan in this task. This is not
    the deployment policy; it is the privileged policy that should first learn
    reliable footholds on stairs, slopes, gravel, and rough terrain.
    """

    scene = copy.deepcopy(Elf3WalkFlatEnvCfg().scene)
    scene.terrain_generator = ATEC_OBSTACLE_COURSE_TERRAINS_CFG
    scene.max_init_terrain_level = 0
    scene.height_scanner = HeightScannerCfg(
        enable_height_scan=True,
        prim_body_name="torso_link",
        resolution=0.08,
        size=(1.8, 1.2),
        debug_vis=False,
        drift_range=(0.0, 0.0),
        use_for_actor=True,
        use_for_critic=True,
    )
    commands = copy.deepcopy(Elf3WalkFlatEnvCfg().commands)
    commands.rel_standing_envs = 0.0
    commands.rel_heading_envs = 0.0
    commands.heading_command = False
    commands.ranges = CommandRangesCfg(
        lin_vel_x=(0.25, 0.85),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(0.0, 0.0),
    )
    reward = copy.deepcopy(Elf3WalkFlatEnvCfg().reward)
    reward.course_centerline_l2 = RewTerm(
        func=mdp.course_centerline_l2,
        weight=-2.0,
        params={"lane_half_width": 1.5, "deadband": 0.05},
    )
    reward.course_heading_exp = RewTerm(
        func=mdp.course_heading_exp,
        weight=1.0,
        params={"std": 0.35, "target_yaw": 0.0},
    )


@configclass
class Elf3WalkTerrainTeacherAgentCfg(Elf3WalkAgentCfg):
    experiment_name = "walk"
    run_name = "elf3_terrain_teacher"
    neptune_project = "walk_elf3_terrain_teacher"
    wandb_project = "walk_elf3_terrain_teacher"


@configclass
class Elf3WalkStairsCurriculumEnvCfg(Elf3WalkTerrainTeacherEnvCfg):
    """Privileged stair curriculum for the ELF3 locomotion policy.

    Curriculum rows increase riser height from 11 cm to 16 cm. Terrain columns
    use fixed 32 cm treads for both ascending and descending stairs.
    """

    scene = copy.deepcopy(Elf3WalkTerrainTeacherEnvCfg().scene)
    scene.terrain_generator = ELF3_STAIRS_CURRICULUM_TERRAINS_CFG
    scene.max_init_terrain_level = 1

    commands = copy.deepcopy(Elf3WalkTerrainTeacherEnvCfg().commands)
    commands.ranges = CommandRangesCfg(
        lin_vel_x=(0.20, 0.60),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(0.0, 0.0),
    )

    reward = copy.deepcopy(Elf3WalkTerrainTeacherEnvCfg().reward)
    reward.course_centerline_l2 = RewTerm(
        func=mdp.course_centerline_l2,
        weight=-2.0,
        params={"lane_half_width": 0.75, "deadband": 0.05},
    )
    reward.course_heading_exp = RewTerm(
        func=mdp.course_heading_exp,
        weight=1.5,
        params={"std": 0.30, "target_yaw": 0.0},
    )
    reward.feet_clearance = RewTerm(
        func=mdp.feet_clearance_relative,
        weight=1.55,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "target_height": 0.18,
            "std": 0.08,
            "command_threshold": 0.15,
        },
    )


@configclass
class Elf3WalkStairsCurriculumAgentCfg(Elf3WalkTerrainTeacherAgentCfg):
    run_name = "elf3_stairs_curriculum"
    neptune_project = "walk_elf3_stairs_curriculum"
    wandb_project = "walk_elf3_stairs_curriculum"


@configclass
class Elf3WalkGeometryFusionEnvCfg(Elf3WalkStairsCurriculumEnvCfg):
    """Deployment-shaped actor input: proprioception plus explicit geometry.

    The actor cannot see the ideal height scanner. The critic retains it as
    privileged information during training. The camera renders below native
    resolution for parallel simulation while preserving the D435i 87 x 58
    degree field of view. Simulation renders directly at the extractor's fixed
    160 x 90 processing resolution; real 1280 x 720 input is downsampled to it.
    """

    scene = copy.deepcopy(Elf3WalkStairsCurriculumEnvCfg().scene)
    scene.num_envs = 32
    scene.terrain_generator = copy.deepcopy(ELF3_GEOMETRY_COURSE_TERRAINS_CFG)
    # Three width stages, with one physical lane per parallel environment.
    # Sharing a lane makes cameras see other robots and corrupts depth.
    scene.terrain_generator.num_cols = 96
    scene.max_init_terrain_level = 0
    scene.max_episode_length_s = 90.0
    scene.height_scanner.use_for_actor = False
    scene.height_scanner.use_for_critic = True
    scene.depth_camera = TiledD435iCameraCfg(
        prim_body_name="torso_link/depth_camera",
        width=160,
        height=90,
        min_range=0.3,
        max_range=5.0,
        update_period=0.04,
        debug_vis=False,
        geometry=GeometryPerceptionCfg(
            enabled=True,
            use_for_actor=True,
            use_for_critic=True,
            history_length=10,
            processing_width=160,
            processing_height=90,
            # Simulated D435i at 160 x 90 sees descending risers about 6-7 cm
            # beyond their mesh edges; keep this calibration explicit.
            descending_edge_offset_m=-0.065,
            gate_enabled=True,
            gate_enter_confidence=0.65,
            gate_exit_confidence=0.35,
            gate_enter_frames=3,
            # Keep the stair branch alive across the short depth dropout at
            # a riser crest; the residual itself remains bounded and the
            # target history rejects stale surfaces behind the robot.
            gate_exit_frames=120,
            gate_switch_frames=3,
            # One tread already represents a horizontal surface bounded by
            # two same-direction riser edges; temporal hysteresis supplies the
            # remaining rejection of isolated depth artifacts.
            gate_min_valid_treads=1,
            gate_min_nearest_tread=0.15,
            gate_max_nearest_tread=1.25,
            append_stair_mode=True,
            append_course_alignment=True,
            append_foothold_targets=True,
            # Keep the deployment actor on the validated stair-only residual
            # path. The optional safety gate is disabled until a separate
            # slope/pebble curriculum has been trained for it.
            alignment_safety_gate_enabled=False,
            alignment_safety_lateral_threshold=0.20,
            alignment_safety_heading_threshold=0.18,
            foothold_target_offset=0.20,
            course_lane_half_width=0.75,
            course_lateral_velocity_scale=0.6,
            course_yaw_rate_scale=1.5,
            motion_compensate_history=True,
            landing_memory_length=40,
            stair_heading_feedback_enabled=True,
            stair_step_settle_enabled=True,
            settle_min_time_s=0.08,
            settle_max_time_s=0.24,
            settle_stable_frames=2,
            settle_edge_tolerance=0.04,
        ),
    )

    reward = copy.deepcopy(Elf3WalkStairsCurriculumEnvCfg().reward)
    reward.termination_penalty = RewTerm(func=mdp.stair_failure_termination, weight=-220.0)
    reward.stair_goal_completion = RewTerm(func=mdp.stair_goal_completion, weight=250.0)
    reward.stair_step_progress = RewTerm(func=mdp.stair_step_progress, weight=0.0)
    reward.stair_physical_step_coverage = RewTerm(
        func=mdp.stair_physical_step_coverage,
        weight=300.0,
    )
    reward.track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.5,
        params={"std": 0.30},
    )
    # The fusion curriculum commands yaw=0; do not pay a standing bonus for tracking zero yaw rate.
    reward.track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.0,
        params={"std": 0.5},
    )
    reward.idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-4.0,
        params={"cmd_threshold": 0.12, "vel_threshold": 0.08, "yaw_cmd_weight": 0.5, "yaw_vel_weight": 0.5},
    )
    reward.forward_velocity_floor = RewTerm(
        func=mdp.forward_velocity_floor,
        weight=1.5,
        params={"ratio": 0.70, "std": 0.12, "command_threshold": 0.15},
    )
    reward.safe_tread_landing = RewTerm(
        func=mdp.safe_tread_landing,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "edge_margin": 0.05,
            "foot_rear_extent": 0.074,
            "foot_front_extent": 0.132,
            "min_support_overlap": 0.08,
            "center_fraction": 0.40,
            "confirmation_frames": 3,
            "max_support_speed": 0.25,
            "min_support_vertical_ratio": 0.65,
            "sole_bottom_offset": 0.04,
            "height_tolerance": 0.06,
            "touchdown_bonus": 20.0,
        },
    )
    reward.verified_stair_step_progress = RewTerm(
        func=mdp.verified_stair_step_progress,
        weight=150.0,
    )
    reward.unsafe_tread_first_contact = RewTerm(
        func=mdp.unsafe_tread_first_contact,
        weight=-180.0,
    )
    reward.stair_hip_clearance_deficit = RewTerm(
        func=mdp.stair_hip_clearance_deficit,
        weight=0.0,
        params={"min_clearance": 0.45},
    )
    reward.stair_nonfoot_collision_risk = RewTerm(
        func=mdp.stair_nonfoot_collision_risk,
        weight=0.0,
        params={"threshold": 20.0, "force_span": 300.0},
    )
    reward.foothold_target_alignment_exp = RewTerm(
        func=mdp.foothold_target_alignment_exp,
        # This is the dense signal that teaches the residual branch to place
        # the next swing foot on the detected tread, rather than merely
        # classifying the terrain as stairs.
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "std": 0.12,
            "target_offset": 0.03,
            "command_threshold": 0.12,
        },
    )
    reward.course_centerline_l2 = RewTerm(
        func=mdp.course_centerline_l2,
        weight=-4.0,
        params={"lane_half_width": 0.75, "deadband": 0.04},
    )
    reward.course_heading_exp = RewTerm(
        func=mdp.course_heading_exp,
        weight=2.5,
        params={"std": 0.22, "target_yaw": 0.0},
    )
    reward.course_heading_l2 = RewTerm(
        func=mdp.course_heading_l2,
        weight=-0.0,
        params={"std": 0.35, "command_threshold": 0.12},
    )
    reward.course_lateral_velocity_l2 = RewTerm(
        func=mdp.course_lateral_velocity_l2,
        weight=-1.0,
        params={"std": 0.25, "command_threshold": 0.12},
    )


@configclass
class Elf3WalkGeometryFusionAgentCfg(Elf3WalkStairsCurriculumAgentCfg):
    run_name = "elf3_geometry_fusion"
    neptune_project = "walk_elf3_geometry_fusion"
    wandb_project = "walk_elf3_geometry_fusion"
    resume = True
    load_run = "2026-09-19_12-54-59_elf3_atec_blind_final_v18_1024env_3k"
    load_checkpoint = "model_43600.pt"

    policy = copy.deepcopy(Elf3WalkStairsCurriculumAgentCfg().policy)
    policy.class_name = "GatedResidualActorCritic"
    policy.init_noise_std = 0.15
    policy.base_actor_obs_dim = 960
    policy.gate_obs_index = -1
    policy.residual_hidden_dims = [256, 128]
    policy.residual_scale = 0.18
    policy.min_action_std = 0.05
    policy.max_action_std = 0.25

    algorithm = copy.deepcopy(Elf3WalkStairsCurriculumAgentCfg().algorithm)
    # The blind actor is frozen; only a bounded residual is optimized. A
    # smaller step keeps the pretrained gait stable while the stair target
    # signal is learned.
    algorithm.learning_rate = 5.0e-5
    algorithm.entropy_coef = 2.0e-4


@configclass
class Elf3WalkGeometryStairsBootstrapEnvCfg(Elf3WalkGeometryFusionEnvCfg):
    """Skill-acquisition stage that starts just before the first stair set."""

    scene = copy.deepcopy(Elf3WalkGeometryFusionEnvCfg().scene)
    scene.terrain_generator = copy.deepcopy(ELF3_GEOMETRY_COURSE_TERRAINS_CFG)
    scene.terrain_generator.num_cols = 96
    scene.terrain_generator.difficulty_range = (0.0, 1.0)
    scene.max_init_terrain_level = 1
    scene.max_episode_length_s = 30.0
    scene.depth_camera.geometry.bootstrap_stairs = True
    scene.depth_camera.geometry.bootstrap_stair_row = 1
    scene.depth_camera.geometry.bootstrap_x_offset = 2.0
    scene.depth_camera.geometry.bootstrap_goal_distance = 4.4
    scene.depth_camera.geometry.bootstrap_spawn_height = 0.0


@configclass
class Elf3WalkGeometryStairsBootstrapAgentCfg(Elf3WalkGeometryFusionAgentCfg):
    run_name = "elf3_geometry_stairs_bootstrap"
    algorithm = copy.deepcopy(Elf3WalkGeometryFusionAgentCfg().algorithm)
    algorithm.learning_rate = 1.0e-5


@configclass
class Elf3WalkGeometryStairsDownBootstrapEnvCfg(Elf3WalkGeometryStairsBootstrapEnvCfg):
    """Descending-stair stage starting on the upper platform."""

    scene = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg().scene)
    scene.max_init_terrain_level = 2
    scene.depth_camera.geometry.bootstrap_stair_row = 2
    scene.depth_camera.geometry.bootstrap_spawn_height = 0.88


@configclass
class Elf3WalkGeometryStairsDownBootstrapAgentCfg(Elf3WalkGeometryStairsBootstrapAgentCfg):
    run_name = "elf3_geometry_stairs_down_bootstrap"


@configclass
class Elf3WalkGeometryStairsDownFullControlAgentCfg(Elf3WalkGeometryStairsDownBootstrapAgentCfg):
    """Learn a stair gait while the frozen blind actor remains the flat-ground fallback."""

    run_name = "elf3_geometry_stairs_down_full_control"
    load_run = "2026-09-19_12-54-59_elf3_atec_blind_final_v18_1024env_3k"
    load_checkpoint = "model_43997.pt"
    policy = copy.deepcopy(Elf3WalkGeometryStairsDownBootstrapAgentCfg().policy)
    policy.residual_scale = 0.80
    algorithm = copy.deepcopy(Elf3WalkGeometryStairsDownBootstrapAgentCfg().algorithm)
    algorithm.learning_rate = 5.0e-5


@configclass
class Elf3WalkGeometryCourseEnvCfg(Elf3WalkGeometryFusionEnvCfg):
    """Single-lane evaluation course with ordered terrain transitions."""

    scene = copy.deepcopy(Elf3WalkGeometryFusionEnvCfg().scene)
    scene.num_envs = 1
    scene.terrain_generator = ELF3_GEOMETRY_EVAL_COURSE_TERRAINS_CFG
    scene.max_init_terrain_level = 0
    scene.max_episode_length_s = 160.0


@configclass
class Elf3WalkGeometryCourseAgentCfg(Elf3WalkGeometryFusionAgentCfg):
    run_name = "elf3_geometry_course"


@configclass
class Elf3WalkGeometryDebugEnvCfg(Elf3WalkStairsCurriculumEnvCfg):
    """Native-resolution D435i diagnostic task using an existing teacher."""

    scene = copy.deepcopy(Elf3WalkStairsCurriculumEnvCfg().scene)
    scene.depth_camera = TiledD435iCameraCfg(
        prim_body_name="torso_link/depth_camera",
        width=1280,
        height=720,
        min_range=0.3,
        max_range=5.0,
        update_period=0.04,
        debug_vis=False,
        geometry=GeometryPerceptionCfg(
            enabled=True,
            use_for_actor=False,
            use_for_critic=False,
            history_length=1,
            processing_width=160,
            processing_height=90,
        ),
    )


@configclass
class Elf3WalkGeometryDebugAgentCfg(Elf3WalkStairsCurriculumAgentCfg):
    run_name = "elf3_geometry_debug"


@configclass
class Elf3WalkTerrainTeacherSensorEnvCfg(Elf3WalkTerrainTeacherEnvCfg):
    """Terrain teacher with an extra waist-mounted depth camera for debugging."""

    scene = copy.deepcopy(Elf3WalkTerrainTeacherEnvCfg().scene)
    scene.depth_camera = TiledD455CameraCfg(
        prim_body_name="torso_link/depth_camera",
        offset=TiledD455CameraCfg.OffsetCfg(
            pos=(0.1152, 0.0175, -0.1358),
            rot=(0.122796911, -0.696361316, 0.696363874, -0.122797362),
            convention="ros",
        ),
        width=64,
        height=48,
        min_range=0.3,
        max_range=5.0,
        feature_width=8,
        feature_height=12,
        depth_history_length=3,
        frame_hold_prob=0.05,
        depth_quantization=0.001,
        update_period=0.04,
        debug_vis=False,
    )
    scene.depth_camera.sensor_noise.dropout_prob = 0.005


@configclass
class Elf3WalkTerrainTeacherSensorAgentCfg(Elf3WalkTerrainTeacherAgentCfg):
    run_name = "elf3_terrain_teacher_sensor"
