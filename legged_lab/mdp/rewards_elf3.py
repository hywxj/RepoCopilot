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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
import math
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.elf3.elf3_env import Elf3Env


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | Elf3Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Tracks the desired linear velocity (XY plane), calculated in the yaw coordinate system.
    
    Convert the robot's linear speed to the yaw coordinate system (rotate around the Z axis),
    Then compare with the expected speed (the first two dimensions of the command).
    Use an exponential function to map the error to a reward value in the range (0,1].
    
    Parameters:
        env: environment instance
        std: standard deviation, controls the decay rate of rewards
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: Line speed tracking reward, shape is [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 将全局线速度转换到偏航坐标系
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    # 计算XY平面速度误差的平方和
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    # 使用指数衰减函数：误差越小，奖励越接近1
    # return torch.exp(-lin_vel_error / std**2) * (zero_flag)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | Elf3Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Tracks the desired angular velocity (Z-axis), calculated in world coordinates.
    
    Directly compare the robot's angular velocity around the Z-axis to the desired yaw angular velocity (the third dimension of the command).
    Use an exponential function to map the error to a reward value in the range (0,1].
    
    Parameters:
        env: environment instance
        std: standard deviation, controls the decay rate of rewards
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: Angular velocity tracking reward, shape [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算Z轴角速度误差的平方
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    # 使用指数衰减函数
    # return torch.exp(-ang_vel_error / std**2) * (zero_flag)
    return torch.exp(-ang_vel_error / std**2) 


def lin_vel_z_l2(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the linear velocity in the vertical direction (Z-axis).
    
    Used to prevent the robot from unnecessary jumping or sinking and maintain a stable standing/walking height.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: vertical velocity penalty (squared value), shape [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the angular velocity in the body coordinate system (X and Y axes).
    
    Used to maintain body stability and reduce roll and pitch shaking.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: XY axis angular velocity penalty (sum of squares), shape is [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty energy consumption.
    
    Calculate the sum of the absolute values ​​of joint power (torque × speed) to encourage energy-saving movements.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: Energy consumption penalty, shape is [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes joint acceleration.
    
    Used to smooth movement and reduce sudden acceleration or deceleration of joints.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: joint acceleration penalty (sum of squares), shape [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | Elf3Env) -> torch.Tensor:
    """Penalty action change rate.
    
    Compare the difference between the current action and the action at the previous moment to encourage smooth action sequences.
    
    Parameters:
        env: environment instance
        
    Return:
        torch.Tensor: action change rate penalty (sum of squares), shape [num_envs]
    """
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

def action_smoothness(env: BaseEnv | Elf3Env) -> torch.Tensor:
    # Get the action buffer from the environment (stores the most recent series of actions)
    buf = env.action_buffer._circular_buffer.buffer
    
    # Extract actions from the last three time steps
    a_t   = buf[:, -1, :]   # 当前时刻动作
    a_t1  = buf[:, -2, :]   # 上一时刻动作
    a_t2  = buf[:, -3, :]   # 上上时刻动作
    
    # 计算三个平滑度指标：
    term_1 = torch.sum((a_t - a_t1)**2, dim=1)  # 相邻动作变化幅度（一阶差分）
    term_2 = torch.sum((a_t + a_t2 - 2*a_t1)**2, dim=1)  # 动作加速度（二阶差分）
    term_3 = 0.05 * torch.sum(torch.abs(a_t), dim=1)  # 动作幅度的正则化项
    
    # 返回总平滑度得分（值越小表示动作越平滑）
    return term_1 + term_2 + term_3

def undesired_contacts(env: BaseEnv | Elf3Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Punishes unwanted body parts for touching the ground.
    
    Check whether a specific body part is in contact with the ground. If the contact force exceeds a threshold, it is considered a violation.
    
    Parameters:
        env: environment instance
        threshold: contact force threshold, exceeding this value is considered contact
        sensor_cfg: Contact sensor configuration, specifying the body part to be checked
        
    Return:
        torch.Tensor: The number of illegal contacts, the shape is [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Check if any body part's contact force exceeds a threshold
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | Elf3Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check that the robot is "flying" (all designated body parts are not touching the ground).
    
    Used to detect whether the robot is completely off the ground, usually used to trigger termination conditions or penalties.
    
    Parameters:
        env: environment instance
        threshold: contact force threshold
        sensor_cfg: Contact sensor configuration
        
    Return:
        torch.Tensor: Boolean value, True means that all specified parts are not touching the ground, the shape is [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # Check that all specified parts are not in contact (i.e. "flying" state)
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Punishes non-horizontal body posture.
    
    Check whether the body remains level (upright) by projecting the gravity vector into the body coordinate system.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: body tilt penalty (sum of squares of gravity projection), shape [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def feet_orientation_l2(env: Elf3Env, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_rotate_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )

def feet_orientation_euler(env: Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    assert len(asset_cfg.body_ids) == 2
    feet_euler_xyz = get_euler_xyz_tensor(asset.data.body_quat_w[:, asset_cfg.body_ids, :])
    rotation = torch.sum(torch.square(feet_euler_xyz[:, :, 2:3]), dim=[1, 2])
    r = torch.exp(-rotation * 1)
    return r

def is_terminated(env: BaseEnv | Elf3Env) -> torch.Tensor:
    """Penalize early termination caused by non-timeout.
    
    Used to identify early termination due to a constraint violation (such as a fall) instead of the normal round timeout.
    
    Parameters:
        env: environment instance
        
    Return:
        torch.Tensor: Early termination penalty flag, shape [num_envs]
    """
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | Elf3Env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward bipedal robots for foot air time (gait periodicity).
    
    Count the time the foot is in the air (swing phase), but only reward during the single-leg support phase.
    Used to encourage natural gait patterns.
    
    Parameters:
        env: environment instance
        threshold: air time threshold for maximum reward
        sensor_cfg: Contact sensor configuration
        
    Return:
        torch.Tensor: Foot air time reward, shape is [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    # 计算当前处于接触还是空中模式的时间
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # 检查是否为单腿支撑阶段（只有一条腿接触地面）
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # 取两条腿中较短的时间作为奖励（确保协调）
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # 零速度命令时不给予奖励
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | Elf3Env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Punishing the foot for sliding on the ground.
    
    Calculate the horizontal speed of the foot when it touches the ground to prevent the foot from slipping.
    
    Parameters:
        env: environment instance
        sensor_cfg: Contact sensor configuration
        asset_cfg: asset configuration (default uses robot)
        
    Return:
        torch.Tensor: foot sliding penalty, shape is [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Detect whether the foot is in contact with the ground (contact force >1.0)
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    # Get the speed of the foot on the horizontal plane (xy)
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # Only penalizes sliding speed on contact
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | Elf3Env, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """Punishes excessive physical contact.
    
    Monitor vertical contact forces on specific body parts to prevent excessive impact forces.
    
    Parameters:
        env: environment instance
        sensor_cfg: Contact sensor configuration
        threshold: the force threshold at which punishment begins
        max_reward: maximum penalty value
        
    Return:
        torch.Tensor: body contact force penalty, shape is [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get the vertical contact force of the body part (z-axis)
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    # Only punish forces above a threshold
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the joint position to deviate from the default position (only takes effect at zero speed).
    
    Encourage the robot to maintain its default standing posture when stationary.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (robot is used by default)
        
    Return:
        torch.Tensor: Joint position deviation penalty (L1 norm), shape is [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # Only takes effect when the speed command is very small
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag

def joint_deviation_l2(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cond1 = torch.norm(env.command_generator.command[:, :2], dim=1) < 0.1
    cond2 = torch.norm(env.command_generator.command[:, 2:3], dim=1) > 0.05

    zero_flag = cond1 & cond2
    return torch.sum(torch.square(angle), dim=1) * ~zero_flag

def joint_deviation_l1_always(env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint deflection penalties that are always in effect (not affected by speed commands).
    
    Similar to the previous function, but takes effect regardless of the speed command.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (robot is used by default)
        
    Return:
        torch.Tensor: Joint position deviation penalty (L1 norm), shape is [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)  # 移除 zero_flag


def body_orientation_l2(
    env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalizes non-horizontal posture of specific body parts.
    
    Convert the gravity vector to the coordinate system of the specified body part and check whether it is vertical.
    
    Parameters:
        env: environment instance
        asset_cfg: asset configuration (robot is used by default), body_ids should contain the parts to be checked
        
    Return:
        torch.Tensor: Body part tilt penalty, shape is [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 将重力向量转换到身体部位的坐标系
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    # 检查水平分量（XY）的大小
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

def body_orientation_euler(env: Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    body_euler = get_euler_xyz_tensor(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :])
    # print(body_euler[0])
    quat_mismatch = torch.exp(-torch.sum(torch.abs(body_euler[:, 1:3]), dim=1) * 10)
    orientation = torch.exp(-torch.norm(body_orientation[:, :2], dim=1) * 20)
    
    return (quat_mismatch + orientation) / 2.

def feet_stumble(env: BaseEnv | Elf3Env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """检测脚部是否绊倒（水平力过大）。
    
    检查脚部水平接触力是否远大于垂直力，这通常表示绊倒或滑动。
    
    参数:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        
    返回:
        torch.Tensor: 布尔值，True表示检测到绊倒，形状为[num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检查水平力是否大于垂直力的5倍
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | Elf3Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.27
) -> torch.Tensor:
    """惩罚两脚距离过近（人形机器人专用）。
    
    防止双脚交叉或距离过近导致的不稳定步态。
    
    参数:
        env: 环境实例
        asset_cfg: 资产配置（默认使用机器人），body_ids应包含左右脚的索引
        threshold: 最小允许距离阈值
        
    返回:
        torch.Tensor: 双脚过近的惩罚，形状为[num_envs]
    """
    assert len(asset_cfg.body_ids) == 2  # 必须指定两只脚
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取双脚的世界坐标位置
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # 计算双脚之间的距离
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    # 距离小于阈值时给予惩罚
    return (threshold - distance).clamp(min=0)


# ==================== Elf3特定奖励函数（针对类人机器人）====================

def ankle_torque(env: Elf3Env) -> torch.Tensor:
    """惩罚脚踝关节扭矩（只在站立不动时生效）。
    
    减少静止站立时的能量消耗和关节压力。
    
    参数:
        env: Elf3Env实例
        
    返回:
        torch.Tensor: 脚踝扭矩惩罚，形状为[num_envs]
    """

    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1) 


def ankle_action(env: Elf3Env) -> torch.Tensor:
    """惩罚脚踝关节动作（只在站立不动时生效）。
    
    鼓励静止时保持脚踝中立位置。
    
    参数:
        env: Elf3Env实例
        
    返回:
        torch.Tensor: 脚踝动作惩罚，形状为[num_envs]
    """

    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1) 


def hip_roll_action(env: Elf3Env) -> torch.Tensor:
    """惩罚髋关节侧摆（roll）动作。
    
    减少不必要的髋部侧向摆动，保持稳定步态。
    
    参数:
        env: Elf3Env实例
        
    返回:
        torch.Tensor: 髋关节侧摆动作惩罚，形状为[num_envs]
    """
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[1], env.right_leg_ids[1]]]), dim=1)


def hip_yaw_action(env: Elf3Env) -> torch.Tensor:
    """惩罚髋关节偏航（yaw）动作。
    
    减少髋部旋转，保持前进方向的稳定性。
    
    参数:
        env: Elf3Env实例
        
    返回:
        torch.Tensor: 髋关节偏航动作惩罚，形状为[num_envs]
    """
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: Elf3Env) -> torch.Tensor:
    """惩罚脚部Y方向距离偏差（当Y向速度较小时）。
    
    保持双脚在侧向的适当距离，防止步宽过窄或过宽。
    
    参数:
        env: Elf3Env实例
        
    返回:
        torch.Tensor: 脚部Y距离偏差惩罚，形状为[num_envs]
    """
    # 计算双脚相对于身体的位置
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    # 转换到身体坐标系
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    # 计算Y方向距离与期望值（0.299）的偏差
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    # 只在Y向速度较小时生效
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# ==================== 步态周期性奖励函数 ===================

def gait_clock(phase, air_ratio, delta_t):
    """生成足部摆动和站立阶段的周期性步态时钟信号。
    
    该函数构造两个相位相关信号：
    - I_frc：在摆动阶段有效（用于惩罚地面反力）
    - I_spd：在站立阶段有效（用于惩罚脚部速度）
    
    摆动和站立之间的过渡在delta_t范围内平滑插值，创建可微分的过渡。
    
    参数:
        phase: 标准化步态相位，范围[0, 1]，形状：[num_envs]
        air_ratio: 步态周期中摆动阶段所占比例，形状：[num_envs]
        delta_t: 相边界周围的过渡宽度
        
    返回:
        I_frc: 基于步态的摆动相位时钟信号，范围[0, 1]，形状：[num_envs]
        I_spd: 基于步态的站立相位时钟信号，范围[0, 1]，形状：[num_envs]
    """
    # 定义各个阶段的布尔掩码
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))  # 纯摆动阶段
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))  # 纯站立阶段
    
    # 过渡阶段
    trans_flag1 = phase < delta_t  # 开始过渡到摆动
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))  # 摆动到站立的过渡
    trans_flag3 = phase > (1 - delta_t)  # 结束过渡
    
    # 计算摆动相位时钟信号（线性插值过渡）
    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc  # 站立相位时钟信号
    return I_frc, I_spd


def gait_feet_frc_perio(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """惩罚步态摆动阶段的足部地面反力。
    
    在摆动阶段，脚部应该在空中，因此地面反力应该接近零。
    
    参数:
        env: Elf3Env实例
        delta_t: 步态过渡宽度
        
    返回:
        torch.Tensor: 摆动阶段地面反力惩罚，形状为[num_envs]
    """
    # 获取左右脚的摆动阶段掩码
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    # 计算摆动阶段的地面反力奖励（力越小奖励越高）
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    # 只在非零速命令时生效
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    # return (left_frc_score + right_frc_score) * zero_flag
    return (left_frc_score + right_frc_score)


def gait_feet_spd_perio(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """在步态的支撑阶段惩罚脚部速度。
    
    在站立阶段，脚部应该相对地面静止，因此速度应该接近零。
    
    参数:
        env: Elf3Env实例
        delta_t: 步态过渡宽度
        
    返回:
        torch.Tensor: 支撑阶段脚部速度惩罚，形状为[num_envs]
    """
    # 获取左右脚的站立阶段掩码
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    # 计算站立阶段的脚速奖励（速度越小奖励越高）
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    # 只在非零速命令时生效
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    # return (left_spd_score + right_spd_score) * zero_flag
    return (left_spd_score + right_spd_score)


def gait_feet_frc_support_perio(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """在站立（支撑）阶段促进适当支撑力的奖励。
    
    在站立阶段，脚部应该提供足够的支撑力来承重。
    
    参数:
        env: Elf3Env实例
        delta_t: 步态过渡宽度
        
    返回:
        torch.Tensor: 支撑阶段地面反力奖励，形状为[num_envs]
    """
    # 获取左右脚的站立阶段掩码
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    # 计算站立阶段的地面反力奖励（力越大奖励越高，但有饱和）
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    # 只在非零速命令时生效
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    # return (left_frc_score + right_frc_score) * zero_flag
    return (left_frc_score + right_frc_score)

# ==================== 步态周期性奖励函数（平滑版）====================

def _gauss_cdf(x: torch.Tensor) -> torch.Tensor:
    # 标准正态分布 CDF：Φ(x) = 0.5*(1+erf(x/√2))
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gait_clock_smooth(phase: torch.Tensor, air_ratio: torch.Tensor, delta_t: float):
    """
    平滑版步态时钟：
      - I_frc: swing 权重（~1 在 swing）
      - I_spd: stance 权重（= 1 - I_frc）
    使用高斯 CDF 构造"软矩形"，并做周期扩展保证相位 0/1 处连续。

    参数
    ----
    phase:     [N]，相位∈[0,1]
    air_ratio: [N] 或标量，swing 占比（0..1）
    delta_t:   平滑强度（作为 σ 使用；越大越平滑，越小越接近硬切换）

    返回
    ----
    I_frc, I_spd: [N]，范围约 [0,1]
    """
    # 广播到相同 device/dtype
    if not torch.is_tensor(air_ratio):
        air_ratio = torch.tensor(air_ratio, device=phase.device, dtype=phase.dtype)

    sigma = torch.as_tensor(delta_t, device=phase.device, dtype=phase.dtype).clamp(min=1e-6)

    # swing 区间 [start, end]，这里 start=0，end=air_ratio
    start = torch.zeros_like(phase)
    end   = torch.clamp(air_ratio, 1e-6, 1.0 - 1e-6)

    # 软矩形（周期扩展：k∈{-1,0,+1}）
    # 基本窗：Φ((φ - start)/σ) - Φ((φ - end)/σ)
    def win(phi):
        return _gauss_cdf((phi - start) / sigma) - _gauss_cdf((phi - end) / sigma)

    I_swing = win(phase) + win(phase - 1.0) + win(phase + 1.0)  # 周期复制，确保 0/1 连续
    I_swing = I_swing.clamp(0.0, 1.0)  # 数值安全

    I_frc = I_swing
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio_smooth(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-25 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-25 * torch.square(env.avg_feet_force_per_step[:, 1])))
    
    # left_frc_score = left_frc_swing_mask * (torch.exp(-100 * torch.square(env.avg_feet_force_per_step[:, 0])))
    # right_frc_score = right_frc_swing_mask * (torch.exp(-100 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_frc_perio_penalize(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """惩罚步态摆动阶段的足部力量."""
    left_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_force = env.avg_feet_force_per_step[:, 0]
    right_force = env.avg_feet_force_per_step[:, 1]
    left_frc_score = left_frc_swing_mask * (torch.abs(left_force) > 5.0).float()
    right_frc_score = right_frc_swing_mask * (torch.abs(right_force) > 5.0).float()
    
    return left_frc_score + right_frc_score


def gait_feet_spd_perio_smooth(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """惩罚步态支撑阶段的足部速度."""
    left_spd_support_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-200 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-200 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    # left_spd_score = left_spd_support_mask * (torch.exp(-300 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    # right_spd_score = right_spd_support_mask * (torch.exp(-300 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio_smooth(env: Elf3Env, delta_t: float = 0.02) -> torch.Tensor:
    """奖励步态支撑阶段的足部力量."""
    left_frc_support_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


# ==================== 站立稳定性奖励函数 ===================

def stand_still(
    env: Elf3Env, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_generator.command
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize motion when command is nearly zero.
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.sum(torch.abs(angle), dim=1) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def idle_when_commanded(
    env: Elf3Env,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a velocity command is given.
    
    This reward function detects "lazy standing" behavior where the robot receives
    a movement command but remains stationary. It returns 1.0 when the robot should
    be moving but is not, enabling a negative weight penalty.
    
    Args:
        env: Environment instance.
        cmd_threshold: Minimum command magnitude to be considered "commanded to move".
            Commands below this threshold are ignored (robot is allowed to stand).
        vel_threshold: Maximum velocity magnitude to be considered "idle/stationary".
            If actual velocity is below this, the robot is considered not moving.
        asset_cfg: Robot configuration.
    
    Returns:
        Tensor of shape (num_envs,) with values:
        - 1.0 if commanded to move but idle (should be penalized)
        - 0.0 otherwise (no penalty)
    
    Example:
        idle_penalty = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-2.0,
            params={"cmd_threshold": 0.2, "vel_threshold": 0.1}
        )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取速度命令（xy 分量）
    cmd_xy = env.command_generator.command[:, :2]
    cmd_magnitude = torch.linalg.norm(cmd_xy, dim=-1)
    
    # 获取实际根速度（偏航坐标系，与 track_lin_vel_xy 使用的相同）
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_magnitude = torch.linalg.norm(vel_yaw[:, :2], dim=-1)
    
    # 检测“已命令但空闲”状况
    is_commanded = cmd_magnitude > cmd_threshold  # Should be moving
    is_idle = vel_magnitude < vel_threshold       # But not moving
    
    return (is_commanded & is_idle).float()

# ======================== DWAQ Rewards ========================
# These rewards are adapted from the DreamWaQ project for blind walking.


def alive(env: Elf3Env) -> torch.Tensor:
    """Reward for staying alive.
    
    A simple constant reward that encourages the robot to not terminate early.
    Reference: DreamWaQ (HumanoidDreamWaq/legged_gym/envs/g1/g1_env.py)
    """
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)


def gait_phase_contact(
    env: Elf3Env, sensor_cfg: SceneEntityCfg, stance_threshold: float = 0.55
) -> torch.Tensor:
    """与预期步态阶段相匹配的足部接触的奖励。
    
    当脚部接触状态与预期的站立/摆动阶段相匹配时奖励机器人。
    在站立阶段（阶段<立场阈值），脚应该接触。
    在摆动阶段（phase >=tance_threshold），脚应该在空中。
    
    参数：
        env：具有步态阶段信息的环境。
        sensor_cfg：脚的接触传感器配置。
        tance_threshold：阶段阈值，低于该阈值脚应处于站立状态。
        
    参考：DreamWaQ _reward_contact()
    
    注意：该函数使用 env.leg_phase ，它应该是 [num_envs, num_feet] 张量
    其中leg_phase[:, 0]=phase_left，leg_phase[:,1]=phase_right。
    Sensor_cfg.body_ids 应匹配相同的顺序（左脚在前，右脚在后）。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # Check contact for each foot (use z-component like original DreamWaQ)
    # Original: contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    contact = net_contact_forces[:, :, 2] > 1.0  # (num_envs, num_feet), z-direction force
    
    # Use leg_phase directly from environment
    # leg_phase shape: (num_envs, 2) where [:, 0] = left, [:, 1] = right
    # leg_phase = env.leg_phase
    leg_phase = env.gait_phase
    
    # Expected stance: phase < stance_threshold
    is_stance = leg_phase < stance_threshold
    
    # Reward: 1 if contact matches expected phase, 0 otherwise
    # XOR gives True when they don't match, so we negate it
    phase_match = ~(contact ^ is_stance)  # (num_envs, num_feet)
    
    return torch.sum(phase_match.float(), dim=-1)  # Sum over feet

def feet_swing_height(
    env: Elf3Env, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08
) -> torch.Tensor:
    """
    简单版本：惩罚摆动脚高度偏离固定目标的情况。
    
    这是使用绝对 z 坐标的原始简单实现。
    使用 foot_swing_height() 作为地形感知版本。
    
    参数：
        env：环境。
        sensor_cfg：脚的接触传感器配置。
        asset_cfg：脚部带有 body_ids 的机器人配置。
        target_height：摆动脚的目标高度（默认0.08m）。
    参考：DreamWaQ _reward_feet_swing_height()
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, num_feet)
    
    # Get feet positions (z-coordinate)
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
    
    # Penalize height error only during swing phase (not in contact)
    pos_error = torch.square(feet_pos_z - target_height) * (~contact).float()
    
    return torch.sum(pos_error, dim=-1)



def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
    return euler_xyz

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        torch.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*torch.pi), pitch % (2*torch.pi), yaw % (2*torch.pi)

def copysign_new(a, b):
    
    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)