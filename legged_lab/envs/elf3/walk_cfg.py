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

import glob
import math
import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.elf3_lite import ELF3LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import ATEC_OFFLINE_TERRAINS_CFG, GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.42
    gait_air_ratio_r: float = 0.42
    gait_phase_offset_l: float = 0.42
    gait_phase_offset_r: float = 0.92
    gait_cycle: float = 1.00


# #run gait
# @configclass
# class GaitCfg:
#     gait_air_ratio_l: float = 0.6
#     gait_air_ratio_r: float = 0.6
#     gait_phase_offset_l: float = 0.6
#     gait_phase_offset_r: float = 0.1
#     gait_cycle: float = 0.5
    

@configclass
class LiteRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.5, params={"std": 0.45})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=5.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_rate_smooth = RewTerm(func=mdp.action_smoothness, weight=-0.003)
    
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-0.35)
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-0.8)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*_knee_y.*", ".*_hip_z.*", ".*_hip_y.*", ".*_shoulder_y.*", ".*_shoulder_z.*", ".*_wrist_z.*", ".*_elbow_y.*", "waist_z.*", "torso_link"]
            ),
            "threshold": 3.0,
        },
    )
    
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=-3.0
    )
    
    body_orientation_euler = RewTerm(
        func=mdp.body_orientation_euler, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=1.0
    )
    
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.4)

    recovery_orientation = RewTerm(
        func=mdp.recovery_orientation_l2,
        weight=-8.0,
        params={"tilt_threshold": 0.08},
    )
    recovery_ang_vel = RewTerm(
        func=mdp.recovery_ang_vel_xy_l2,
        weight=-0.55,
        params={"tilt_threshold": 0.08, "ang_vel_threshold": 0.35},
    )
    recovery_action_rate = RewTerm(
        func=mdp.recovery_action_rate_l2,
        weight=-0.015,
        params={"tilt_threshold": 0.08, "ang_vel_threshold": 0.40},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-220.0)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.55,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_x.*"),
            "threshold": 500,   
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "threshold": 0.24,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_x.*"])},
    )
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-5.5, params={"target_width": 0.30, "yaw_width_gain": 0.04, "max_extra_width": 0.04, "y_vel_threshold": 0.1})

    stance_foot_ang_vel = RewTerm(
        func=mdp.stance_foot_ang_vel_l2,
        weight=-0.025,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
        },
    )

    fast_walk_height = RewTerm(
        func=mdp.fast_walk_height,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "target_height": 0.98,
            "std": 0.07,
            "speed_threshold": 0.35,
        },
    )

    stance_knee_extension = RewTerm(
        func=mdp.stance_knee_extension,
        weight=0.65,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "max_knee": 0.62,
            "std": 0.08,
            "speed_threshold": 0.35,
        },
    )
    
    # feet_orientation_l2 = RewTerm(
    #     func=mdp.feet_orientation_l2,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_sensor", 
    #             body_names=".*ankle_x_link",
    #         ),
    #     }
    # )
    
    # feet_orientation_euler = RewTerm(
    #     func=mdp.feet_orientation_euler, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x_link.*")}, weight=0.25
    # )
    
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.5, params={"soft_ratio": 0.9})
    
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        # weight=-0.2,
        weight=-0.24,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[ ".*shoulder_x.*", ".*shoulder_z.*", ".*_wrist.*"]
            )
        },
    )

    arm_swing = RewTerm(
        func=mdp.arm_swing_opposite_feet,
        weight=1.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "neutral_pitch": 0.0,
            "swing_gain": 0.38,
            "foot_delta_scale": 0.18,
            "pos_std": 0.28,
            "diff_std": 0.36,
            "command_threshold": 0.15,
        },
    )

    elbow_lateral_range = RewTerm(
        func=mdp.elbow_lateral_range,
        weight=0.45,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_elbow_y_link", "r_elbow_y_link"]),
            "min_abs_y": 0.20,
            "max_abs_y": 0.30,
            "std": 0.045,
            "command_threshold": 0.15,
        },
    )

    elbow_relaxed = RewTerm(
        func=mdp.elbow_relaxed_motion,
        weight=0.4,
        params={
            "min_flexion": -0.05,
            "max_flexion": 1.0,
            "target_flexion": 0.35,
            "phase_gain": 0.18,
            "vel_gain": 0.45,
            "activity_speed": 0.18,
            "command_threshold": 0.15,
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        # weight=-0.15,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                # "robot", joint_names=[".*hip_z.*", ".*hip_x.*", ".*shoulder_y.*", ".*elbow_y.*"],
                "robot", joint_names=[".*hip_z.*", ".*hip_x.*",],
                # joint_names=[
                #     # ".*_hip_y_joint",
                #     ".*_hip_x_joint",
                #     ".*_hip_z_joint",
                # ],
            )
        },
    )

    waist_roll_stability = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_x_joint"])},
    )

    waist_pitch_stability = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_y_joint"])},
    )
    # joint_deviation_hip_walk = RewTerm(
    #     func=mdp.joint_deviation_l2,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=[".*hip_z.*", ".*hip_x.*", ".*shoulder_y.*", ".*elbow_y.*"],
    #         )
    #     },
    # )
    # #避免一侧罚重一侧轻arm/waist/wrist关节偏差
    # joint_deviation_arms_0 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # func=mdp.joint_deviation_l1_always,
    #     # weight=-2.0,
    #     weight=-0.15,
    #     # weight=-0.5,
    #     # weight=-0.05,
    #     # weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_x_joint", ".*_shoulder_z_joint"])},
    #     # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_x_joint", ".*_shoulder_z_joint", ".*_wrist_.*_joint"])},
    #     # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_y_joint", ".*_shoulder_x_joint", ".*_shoulder_z_joint", ".*_elbow_y_joint", ".*_wrist_.*_joint"])},
    # )
    # joint_deviation_arms_1 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # func=mdp.joint_deviation_l1_always,
    #     # weight=-2.0,
    #     weight=-0.15,
    #     # weight=-0.5,
    #     # weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_y_joint", ".*_elbow_y_joint" , ".*_wrist_.*_joint"])},
    # )
    # #加入腰部关节偏差惩罚 - 覆盖所有腰部关节
    # joint_deviation_waists = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # func=mdp.joint_deviation_l1_always,
    #     # weight=-1.0,
    #     weight=-0.15,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_x_joint"])},

    # )
    # joint_deviation_waists1 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.15,
    #     # params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},  # 覆盖所有腰部关节
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_y_joint","waist_z_joint"])},
    # )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        # func=mdp.joint_deviation_l1_always,
        weight=-0.016,
        # weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*waist.*",
                    ".*_hip_y_joint",
                    ".*_knee_y_joint",
                    ".*_ankle_y_joint",
                    ".*_ankle_x_joint",
                ],
            )
        },
    )
    
    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=2.0, params={"delta_t": 0.02})# 增加权重,鼓励抬脚用力周期
    # gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=2.0, params={"delta_t": 0.02})
    # gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=1.2, params={"delta_t": 0.02})

    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio_smooth, weight=2.0, params={"delta_t": 0.015})
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio_smooth, weight=1.0, params={"delta_t": 0.015})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio_smooth, weight=1.0, params={"delta_t": 0.015})
    gait_feet_frc_perio_penalize = RewTerm(func=mdp.gait_feet_frc_perio_penalize, weight=-1.0, params={"delta_t": 0.015})

    fly = RewTerm(
        func=mdp.fly,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"), "threshold": 1.0},
    )
    
    # #zero stand 
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,  
        # weight=-4.0,  
        params={"command_threshold": 0.1, "yaw_command_weight": 0.5},
    )
    
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.15,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"), "threshold": 0.4},
    # )
    
    # ==================== DWAQ Core Rewards (from DreamWaQ) ====================
    # Survival bonus - 使用与 HumanoidDreamWaq 一致的权重
    # HumanoidDreamWaq 使用 alive = 0.15，较小的值避免偷懒站立
    # alive = RewTerm(func=mdp.alive, weight=0.15)
    
    # ==================== 偷懒惩罚 (DWAQ 专用) ====================
    # 核心问题: 机器人学会"收到移动命令但站着不动"的偷懒策略
    # 解决方案: 直接惩罚"被命令移动但实际静止"的行为
    # - cmd_threshold=0.12: 命令速度 > 0.12 m/s 时视为"需要移动"
    # - vel_threshold=0.06: 实际速度 < 0.06 m/s 时视为"静止"
    # - weight=-2.0: 每步惩罚 -2.0，与 termination_penalty=-200 形成对比
    #   (站着20秒 = 1000步 × 2.0 = -2000，远比摔倒惩罚高)
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.12, "vel_threshold": 0.06, "yaw_cmd_weight": 0.5, "yaw_vel_weight": 0.5},
    )
    
    # Swing-foot clearance relative to the support foot; works on flat ground and stairs.
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_relative,
        weight=1.55,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "target_height": 0.145,
            "std": 0.08,
            "command_threshold": 0.15,
        },
    )

    swing_foot_forward = RewTerm(
        func=mdp.swing_foot_forward_progress,
        weight=1.30,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
            "target_forward": 0.22,
            "std": 0.12,
            "command_threshold": 0.15,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.22,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x_link", "r_ankle_x_link"]), "threshold": 0.44},
    )
    
    # Gait phase matching for bipedal walking - 学习正确的两足步态
    # 奖励机器人在正确的相位进行触地/摆动
    # - stance phase (phase < 0.55): 脚应该接触地面
    # - swing phase (phase >= 0.55): 脚应该在空中
    # 注意: body_names 顺序必须是 [左脚, 右脚]，与 leg_phase 顺序一致
    # gait_phase_contact = RewTerm(
    #     func=mdp.gait_phase_contact,
    #     # weight=0.2,
    #     weight=1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x.*", "r_ankle_x.*"]), "stance_threshold": 0.55},
    # )
    
    


@configclass
class Elf3WalkFlatEnvCfg:
    amp_motion_files_display = sorted(
        glob.glob("legged_lab/envs/elf3/datasets/motion_visualization/elf3_loco_clean_gmr/**/*.txt", recursive=True)
    ) or ["legged_lab/envs/elf3/datasets/motion_visualization/walk.txt"]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ELF3LITE_CFG,
        terrain_type="generator",
        terrain_generator=ATEC_OFFLINE_TERRAINS_CFG,
        max_init_terrain_level=3,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        # critic_obs_history_length=1,
        action_scale=[
            0.231, 0.154, 0.213,
            0.380, 0.220, 0.240,
            0.400, 0.350, 0.220,
            0.380, 0.220, 0.240,
            0.400, 0.350, 0.220,
            0.420, 0.070, 0.180,
            0.340, 0.373, 0.373, 0.373,
            0.420, 0.070, 0.180,
            0.340, 0.373, 0.373, 0.373,
        ],
        terminate_contacts_body_names=[".*_hip_z.*",".*_shoulder_y.*", ".*_shoulder_z.*",".*_wrist_z.*",  "waist_z.*", "torso_link"],
        feet_body_names=[".*_ankle_x.*"],
    )
    reward = LiteRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        # resampling_time_range=(10.0, 20.0),
        # rel_standing_envs=0.3,
        # rel_standing_envs=0.2,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.20, 1.00), lin_vel_y=(-0.10, 0.10), ang_vel_z=(-0.80, 0.80), heading=(0.0, 0.0)
            # lin_vel_x=(-0.6, 3.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)#run
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=None,
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
                    "velocity_range": {
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (-0.05, 0.05),
                        "roll": (-0.05, 0.05),
                        "pitch": (-0.05, 0.05),
                        "yaw": (-0.05, 0.05),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.9, 1.1),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(8.0, 12.0),
                params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.20, 0.20)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Elf3WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    # num_steps_per_env = 32
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        # symmetry_cfg=RslRlSymmetryCfg(),  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg() 
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.35
    # amp_motion_files = ["legged_lab/envs/elf3/datasets/motion_amp_expert/walk.txt"]
    amp_motion_files = sorted(
        glob.glob("legged_lab/envs/elf3/datasets/motion_amp_expert/elf3_loco_clean_gmr/**/*.txt", recursive=True)
    ) or ["legged_lab/envs/elf3/datasets/motion_amp_expert/walk.txt"]
    amp_num_preload_transitions = 200000
    # amp_task_reward_lerp = 0.5#0.7
    amp_task_reward_lerp = 0.55#0.7
    # amp_task_reward_lerp = 0.65#0.7
    # amp_task_reward_lerp = 0.7#0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    # min_normalized_std = [0.05] * 20
    min_normalized_std = [0.05] * 29
 
