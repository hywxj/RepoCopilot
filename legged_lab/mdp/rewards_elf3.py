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
from legged_lab.perception.stair_geometry import (
    centered_tread_support,
    match_sole_to_treads,
    unsafe_tread_touchdown,
)
from legged_lab.perception.foothold_control import (
    capture_point_offset,
    confirmed_support_transition,
    nonfoot_collision_risk,
    standing_support_quality as _standing_support_quality,
    verified_stair_progress_transition,
)

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.elf3.elf3_env import Elf3Env


def standing_support_quality(
    env: BaseEnv | Elf3Env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Use simulation contacts as reward labels, never as actor observations."""
    data = env.scene[asset_cfg.name].data
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    quat = data.body_quat_w[:, asset_cfg.body_ids]
    offset = torch.tensor([0.03, 0.0, -0.04], device=env.device).expand(*quat.shape[:-1], 3)
    offset_w = math_utils.quat_apply(quat, offset)
    velocity = data.body_link_lin_vel_w[:, asset_cfg.body_ids] + torch.cross(
        data.body_ang_vel_w[:, asset_cfg.body_ids], offset_w, dim=-1,
    )
    return _standing_support_quality(forces, velocity, env.standing_body_weight)


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


def course_centerline_l2(
    env: BaseEnv | Elf3Env,
    lane_half_width: float = 1.5,
    deadband: float = 0.05,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral drift from each environment's course centerline."""
    asset: Articulation = env.scene[asset_cfg.name]
    lateral_error = torch.abs(asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    normalized_error = torch.clamp(lateral_error - deadband, min=0.0) / max(lane_half_width, 1.0e-6)
    return torch.clamp(torch.square(normalized_error), max=max_penalty)


def course_heading_exp(
    env: BaseEnv | Elf3Env,
    std: float = 0.35,
    target_yaw: float = 0.0,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment with the course's positive x direction while moving."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_error = math_utils.wrap_to_pi(yaw - target_yaw)
    moving = torch.abs(env.command_generator.command[:, 0]) > command_threshold
    return torch.exp(-torch.square(yaw_error) / std**2) * moving.float()


def course_heading_l2(
    env: BaseEnv | Elf3Env,
    std: float = 0.35,
    command_threshold: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a dense, bounded penalty for turning away from the route."""

    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_error = math_utils.wrap_to_pi(yaw)
    moving = torch.abs(env.command_generator.command[:, 0]) > command_threshold
    normalized = torch.clamp(yaw_error / max(std, 1.0e-6), min=-3.0, max=3.0)
    return torch.square(normalized) * moving.float()


def course_lateral_velocity_l2(
    env: BaseEnv | Elf3Env,
    std: float = 0.25,
    command_threshold: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sideways velocity while the route command is active.

    Position-only centerline rewards arrive too late to stop a drift on a
    stair riser. This term supplies the residual policy with a dense damping
    signal while leaving standing and commanded turns unaffected.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    moving = torch.abs(env.command_generator.command[:, 0]) > command_threshold
    lateral_velocity = asset.data.root_lin_vel_w[:, 1]
    return torch.square(lateral_velocity / max(std, 1.0e-6)) * moving.float()


def foothold_target_alignment_exp(
    env: Elf3Env,
    sensor_cfg: SceneEntityCfg,
    std: float = 0.12,
    target_offset: float = 0.03,
    command_threshold: float = 0.12,
) -> torch.Tensor:
    """Guide swing feet onto detected tread centers, not just toward them."""

    if not hasattr(env, "_foothold_target_features") or not hasattr(env, "stair_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    features = env._foothold_target_features().reshape(env.num_envs, -1, 4)
    geometry_cfg = env.cfg.scene.depth_camera.geometry
    target_dx = features[..., 0] * (
        geometry_cfg.foothold_target_distance_scale or geometry_cfg.max_forward
    )
    confidence = features[..., 3]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.norm(contact_forces, dim=-1).amax(dim=1) > 1.0
    moving = torch.abs(env.command_generator.command[:, 0:1]) > command_threshold
    active = moving & ~contact & (confidence > 0.25)
    target_error = target_dx - target_offset
    score = torch.exp(-torch.square(target_error) / max(std, 1.0e-6) ** 2) * confidence
    return (score * active.float()).sum(dim=1) / active.float().sum(dim=1).clamp_min(1.0)


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

def action_smoothness(
    env: BaseEnv | Elf3Env,
    first_order_scale: float = 1.0,
    second_order_scale: float = 1.0,
    action_magnitude_scale: float = 0.05,
) -> torch.Tensor:
    # Get the action buffer from the environment (stores the most recent series of actions)
    buf = env.action_buffer._circular_buffer.buffer
    
    # Extract actions from the last three time steps
    a_t   = buf[:, -1, :]   # 当前时刻动作
    a_t1  = buf[:, -2, :]   # 上一时刻动作
    a_t2  = buf[:, -3, :]   # 上上时刻动作
    
    # 计算三个平滑度指标：
    term_1 = first_order_scale * torch.sum((a_t - a_t1)**2, dim=1)  # 相邻动作变化幅度（一阶差分）
    term_2 = second_order_scale * torch.sum((a_t + a_t2 - 2*a_t1)**2, dim=1)  # 动作加速度（二阶差分）
    term_3 = action_magnitude_scale * torch.sum(torch.abs(a_t), dim=1)  # 动作幅度的正则化项
    
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


def stair_failure_termination(env: Elf3Env) -> torch.Tensor:
    """Keep a successful stair finish out of the fall/violation penalty."""

    terminated = env.reset_buf & ~env.time_out_buf
    if hasattr(env, "last_goal_reached"):
        terminated = terminated & ~env.last_goal_reached
    return terminated.float()


def stair_goal_completion(env: Elf3Env) -> torch.Tensor:
    if hasattr(env, "last_goal_reached"):
        return env.last_goal_reached.float()
    return torch.zeros(env.num_envs, device=env.device)


def stair_step_progress(env: Elf3Env) -> torch.Tensor:
    if hasattr(env, "stair_progress_increment"):
        return env.stair_progress_increment
    return torch.zeros(env.num_envs, device=env.device)


def stair_physical_step_coverage(env: Elf3Env) -> torch.Tensor:
    """Credit each physical stair level only after centered stable foot support."""

    if hasattr(env, "stair_physical_step_increment"):
        return env.stair_physical_step_increment
    return torch.zeros(env.num_envs, device=env.device)


def verified_stair_step_progress(env: Elf3Env) -> torch.Tensor:
    """Credit only a confirmed tread landing at the next stair height."""

    if not hasattr(env, "stair_verified_progress_level"):
        return torch.zeros(env.num_envs, device=env.device)
    if getattr(env, "safe_tread_evaluated_step", -1) != env.sim_step_counter:
        raise RuntimeError("safe_tread_landing must run before verified_stair_step_progress")
    descending = env.cfg.scene.depth_camera.geometry.bootstrap_stair_row == 2
    verified, increment = verified_stair_progress_transition(
        env.stair_progress_level,
        env.stair_verified_progress_level,
        env.safe_tread_confirmed_touchdown,
        env.safe_tread_confirmed_tread_height,
        env.scene.env_origins[:, 2],
        env.stair_step_height,
        direction=-1 if descending else 1,
    )
    env.stair_verified_progress_level = verified
    env.stair_verified_contact_count += increment.long()
    return increment


def unsafe_tread_first_contact(env: Elf3Env) -> torch.Tensor:
    """Penalize an edge or wall touchdown before it becomes a fall."""

    if not hasattr(env, "safe_tread_unsafe_first_contact"):
        return torch.zeros(env.num_envs, device=env.device)
    if getattr(env, "safe_tread_evaluated_step", -1) != env.sim_step_counter:
        raise RuntimeError("safe_tread_landing must run before unsafe_tread_first_contact")
    return env.safe_tread_unsafe_first_contact.float().sum(dim=1)


def stair_hip_clearance_deficit(env: Elf3Env, min_clearance: float = 0.45) -> torch.Tensor:
    """Provide early feedback when the hips sink toward the supported stair."""

    if not hasattr(env, "stair_mode") or not hasattr(env, "_hip_clearance_above_support"):
        return torch.zeros(env.num_envs, device=env.device)
    contacts = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2] > 5.0
    active = (env.stair_mode != 0) & contacts.any(dim=1)
    return (min_clearance - env._hip_clearance_above_support()).clamp_min(0.0) * active


def stair_nonfoot_collision_risk(
    env: Elf3Env,
    threshold: float = 20.0,
    force_span: float = 300.0,
) -> torch.Tensor:
    """Penalize an incipient body impact while the depth gate sees stairs."""

    if not hasattr(env, "stair_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    force = env.contact_sensor.data.net_forces_w_history[
        :, :, env.termination_contact_cfg.body_ids, :
    ]
    return nonfoot_collision_risk(force, threshold, force_span) * (env.stair_mode != 0)


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
    """Penalize insufficient lateral foot separation in the robot frame.
    
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
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_rel_w = feet_pos_w - asset.data.root_link_pos_w[:, None, :]
    num_envs, num_feet = feet_rel_w.shape[:2]
    root_quat_inv = math_utils.quat_conjugate(asset.data.root_link_quat_w)
    root_quat_inv = root_quat_inv[:, None, :].expand(num_envs, num_feet, 4).reshape(-1, 4)
    feet_pos_b = math_utils.quat_apply(root_quat_inv, feet_rel_w.reshape(-1, 3)).reshape(num_envs, num_feet, 3)

    lateral_separation = feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1]
    return (threshold - lateral_separation).clamp(min=0)


def stance_foot_ang_vel_l2(
    env: BaseEnv | Elf3Env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize support-foot rocking after contact on uneven terrain."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0
    foot_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]

    support = contacts.float()
    support_count = support.sum(dim=1).clamp_min(1.0)
    roll_pitch_speed = torch.sum(torch.square(foot_ang_vel[:, :, :2]), dim=-1)
    return torch.sum(roll_pitch_speed * support, dim=1) / support_count


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


def feet_y_distance(
    env: Elf3Env,
    target_width: float = 0.31,
    yaw_width_gain: float = 0.035,
    max_extra_width: float = 0.035,
    y_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize lateral foot-width error, with a slightly wider target while turning."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)

    yaw_extra = torch.clamp(torch.abs(env.command_generator.command[:, 2]) * yaw_width_gain, max=max_extra_width)
    target = target_width + yaw_extra
    y_distance_b = torch.abs(torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1]) - target)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < y_vel_threshold
    return y_distance_b * y_vel_flag


# ==================== 步态周期性奖励函数 ===================

def fast_walk_height(
    env: BaseEnv | Elf3Env,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.98,
    std: float = 0.08,
    speed_threshold: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward torso height relative to the supporting feet while moving forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_count = contacts.sum(dim=1)
    support_z = torch.sum(feet_z * contacts.float(), dim=1) / support_count.clamp_min(1)

    cmd_x = env.command_generator.command[:, 0]
    fast = cmd_x > speed_threshold
    height = asset.data.root_link_pos_w[:, 2] - support_z
    deficit = torch.clamp(target_height - height, min=0.0)
    score = torch.exp(-torch.square(deficit) / std**2)
    has_support = support_count > 0
    return score * fast.float() * has_support.float()


def stance_knee_extension(
    env: BaseEnv | Elf3Env,
    sensor_cfg: SceneEntityCfg,
    max_knee: float = 0.62,
    std: float = 0.08,
    speed_threshold: float = 0.75,
) -> torch.Tensor:
    """Reward the stance knee for not over-flexing during fast forward walking."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0

    knee_ids = torch.tensor([env.left_leg_ids[3], env.right_leg_ids[3]], device=env.device)
    knee_pos = env.robot.data.joint_pos[:, knee_ids]
    over_flex = torch.clamp(knee_pos - max_knee, min=0.0)

    support = contacts.float()
    support_count = support.sum(dim=1).clamp_min(1.0)
    error = torch.sum(torch.square(over_flex) * support, dim=1) / support_count
    score = torch.exp(-error / std**2)

    fast = env.command_generator.command[:, 0] > speed_threshold
    has_support = support.sum(dim=1) > 0.0
    return score * fast.float() * has_support.float()


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
    env: Elf3Env,
    command_threshold: float = 0.06,
    yaw_command_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from default joints only when all velocity commands are small."""
    command = env.command_generator.command
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command_magnitude = torch.norm(command[:, :2], dim=1) + yaw_command_weight * torch.abs(command[:, 2])

    return torch.sum(torch.abs(angle), dim=1) * (command_magnitude < command_threshold)

def idle_when_commanded(
    env: Elf3Env,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    yaw_cmd_weight: float = 0.5,
    yaw_vel_weight: float = 0.5,
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
    
    # Linear and yaw commands both mean the robot should not stay idle.
    command = env.command_generator.command
    cmd_magnitude = torch.linalg.norm(command[:, :2], dim=-1) + yaw_cmd_weight * torch.abs(command[:, 2])
    
    # 获取实际根速度（偏航坐标系，与 track_lin_vel_xy 使用的相同）
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_magnitude = torch.linalg.norm(vel_yaw[:, :2], dim=-1) + yaw_vel_weight * torch.abs(
        asset.data.root_ang_vel_w[:, 2]
    )
    
    # 检测“已命令但空闲”状况
    is_commanded = cmd_magnitude > cmd_threshold  # Should be moving
    is_idle = vel_magnitude < vel_threshold       # But not moving
    
    return (is_commanded & is_idle).float()


def forward_velocity_floor(
    env: BaseEnv | Elf3Env,
    ratio: float = 0.75,
    std: float = 0.12,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward commanded forward motion for not dropping below a speed floor."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    cmd_x = env.command_generator.command[:, 0]
    target_floor = ratio * cmd_x
    deficit = torch.clamp(target_floor - vel_yaw[:, 0], min=0.0)
    score = torch.exp(-torch.square(deficit) / std**2)
    moving_forward = cmd_x > command_threshold
    return score * moving_forward


def arm_swing_opposite_hips(
    env: Elf3Env,
    pos_gain: float = 0.6,
    vel_gain: float = 0.5,
    pos_std: float = 0.35,
    vel_std: float = 1.5,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward human-like contralateral arm swing during locomotion.

    The left shoulder pitch follows the right hip pitch, and the right shoulder
    pitch follows the left hip pitch. This gives AMP a direct handle on arm
    swing without forcing a fixed arm pose.
    """
    joint_pos = env.robot.data.joint_pos
    joint_vel = env.robot.data.joint_vel
    default_pos = env.robot.data.default_joint_pos

    left_shoulder_id = env.left_arm_ids[0]
    right_shoulder_id = env.right_arm_ids[0]
    left_hip_id = env.left_leg_ids[0]
    right_hip_id = env.right_leg_ids[0]

    left_shoulder = joint_pos[:, left_shoulder_id] - default_pos[:, left_shoulder_id]
    right_shoulder = joint_pos[:, right_shoulder_id] - default_pos[:, right_shoulder_id]
    left_hip = joint_pos[:, left_hip_id] - default_pos[:, left_hip_id]
    right_hip = joint_pos[:, right_hip_id] - default_pos[:, right_hip_id]

    left_shoulder_vel = joint_vel[:, left_shoulder_id]
    right_shoulder_vel = joint_vel[:, right_shoulder_id]
    left_hip_vel = joint_vel[:, left_hip_id]
    right_hip_vel = joint_vel[:, right_hip_id]

    pos_error = torch.square(left_shoulder - pos_gain * right_hip) + torch.square(
        right_shoulder - pos_gain * left_hip
    )
    vel_error = torch.square(left_shoulder_vel - vel_gain * right_hip_vel) + torch.square(
        right_shoulder_vel - vel_gain * left_hip_vel
    )

    pos_score = torch.exp(-pos_error / pos_std**2)
    vel_score = torch.exp(-vel_error / vel_std**2)
    moving = torch.norm(env.command_generator.command[:, :2], dim=1) > command_threshold
    return (0.7 * pos_score + 0.3 * vel_score) * moving


def arm_swing_opposite_feet(
    env: Elf3Env,
    asset_cfg: SceneEntityCfg,
    neutral_pitch: float = 0.0,
    swing_gain: float = 0.28,
    foot_delta_scale: float = 0.18,
    pos_std: float = 0.24,
    diff_std: float = 0.30,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward contralateral arm swing from the actual foot fore-aft phase.

    When the left foot is ahead of the right foot, the right shoulder pitch is
    rewarded for moving forward; when the right foot is ahead, the left shoulder
    pitch is rewarded for moving forward. For ELF3, expert data shows forward
    arm swing corresponds to a smaller shoulder_y value.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = env.robot.data.joint_pos

    left_shoulder = joint_pos[:, env.left_arm_ids[0]]
    right_shoulder = joint_pos[:, env.right_arm_ids[0]]

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_rel_w = feet_pos_w - asset.data.root_link_pos_w[:, None, :]
    num_envs, num_feet = feet_rel_w.shape[:2]
    root_quat_inv = math_utils.quat_conjugate(asset.data.root_link_quat_w)
    root_quat_inv = root_quat_inv[:, None, :].expand(num_envs, num_feet, 4).reshape(-1, 4)
    feet_pos_b = math_utils.quat_apply(root_quat_inv, feet_rel_w.reshape(-1, 3)).reshape(num_envs, num_feet, 3)

    left_foot_x = feet_pos_b[:, 0, 0]
    right_foot_x = feet_pos_b[:, 1, 0]
    foot_phase = torch.tanh((left_foot_x - right_foot_x) / foot_delta_scale)

    left_target = neutral_pitch + swing_gain * foot_phase
    right_target = neutral_pitch - swing_gain * foot_phase
    pos_error = torch.square(left_shoulder - left_target) + torch.square(right_shoulder - right_target)
    pos_score = torch.exp(-pos_error / pos_std**2)

    shoulder_diff_target = 2.0 * swing_gain * foot_phase
    shoulder_diff = left_shoulder - right_shoulder
    diff_score = torch.exp(-torch.square(shoulder_diff - shoulder_diff_target) / diff_std**2)

    moving_command = torch.norm(env.command_generator.command[:, :2], dim=1) + 0.5 * torch.abs(
        env.command_generator.command[:, 2]
    )
    moving = moving_command > command_threshold
    return (0.75 * pos_score + 0.25 * diff_score) * moving


def elbow_lateral_range(
    env: BaseEnv | Elf3Env,
    asset_cfg: SceneEntityCfg,
    min_abs_y: float = 0.20,
    max_abs_y: float = 0.30,
    std: float = 0.04,
    command_threshold: float = 0.0,
) -> torch.Tensor:
    """Reward elbows for staying in a natural lateral band around the torso."""
    asset: Articulation = env.scene[asset_cfg.name]
    elbows_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    elbows_rel_w = elbows_pos_w - asset.data.root_link_pos_w[:, None, :]
    num_envs, num_elbows = elbows_rel_w.shape[:2]
    root_quat_inv = math_utils.quat_conjugate(asset.data.root_link_quat_w)
    root_quat_inv = root_quat_inv[:, None, :].expand(num_envs, num_elbows, 4).reshape(-1, 4)
    elbows_pos_b = math_utils.quat_apply(root_quat_inv, elbows_rel_w.reshape(-1, 3)).reshape(num_envs, num_elbows, 3)

    abs_y = torch.abs(elbows_pos_b[:, :, 1])
    low_error = torch.clamp(min_abs_y - abs_y, min=0.0)
    high_error = torch.clamp(abs_y - max_abs_y, min=0.0)
    score = torch.exp(-torch.sum(torch.square(low_error) + torch.square(high_error), dim=1) / std**2)

    if command_threshold > 0.0:
        moving_command = torch.norm(env.command_generator.command[:, :2], dim=1) + 0.5 * torch.abs(
            env.command_generator.command[:, 2]
        )
        score = score * (moving_command > command_threshold)
    return score


def elbow_relaxed_motion(
    env: BaseEnv | Elf3Env,
    min_flexion: float = 0.15,
    max_flexion: float = 1.25,
    target_flexion: float = 0.95,
    phase_gain: float = 0.0,
    vel_gain: float = 0.35,
    range_std: float = 0.25,
    target_std: float = 0.45,
    vel_std: float = 1.2,
    activity_speed: float = 0.35,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward elbows that stay relaxed and move a little with shoulder swing."""
    joint_pos = env.robot.data.joint_pos
    joint_vel = env.robot.data.joint_vel

    left_shoulder_id = env.left_arm_ids[0]
    right_shoulder_id = env.right_arm_ids[0]
    left_elbow_id = env.left_arm_ids[3]
    right_elbow_id = env.right_arm_ids[3]

    left_elbow = joint_pos[:, left_elbow_id]
    right_elbow = joint_pos[:, right_elbow_id]
    left_elbow_vel = joint_vel[:, left_elbow_id]
    right_elbow_vel = joint_vel[:, right_elbow_id]
    left_shoulder_vel = joint_vel[:, left_shoulder_id]
    right_shoulder_vel = joint_vel[:, right_shoulder_id]

    low_error = torch.square(torch.clamp(min_flexion - left_elbow, min=0.0)) + torch.square(
        torch.clamp(min_flexion - right_elbow, min=0.0)
    )
    high_error = torch.square(torch.clamp(left_elbow - max_flexion, min=0.0)) + torch.square(
        torch.clamp(right_elbow - max_flexion, min=0.0)
    )
    range_score = torch.exp(-(low_error + high_error) / range_std**2)

    if phase_gain != 0.0 and hasattr(env, "gait_phase"):
        left_target = target_flexion + phase_gain * torch.sin(2 * torch.pi * env.gait_phase[:, 1])
        right_target = target_flexion + phase_gain * torch.sin(2 * torch.pi * env.gait_phase[:, 0])
    else:
        left_target = target_flexion
        right_target = target_flexion

    target_error = torch.square(left_elbow - left_target) + torch.square(right_elbow - right_target)
    target_score = torch.exp(-target_error / target_std**2)

    left_vel_error = torch.square(torch.abs(left_elbow_vel) - vel_gain * torch.abs(left_shoulder_vel))
    right_vel_error = torch.square(torch.abs(right_elbow_vel) - vel_gain * torch.abs(right_shoulder_vel))
    vel_score = torch.exp(-(left_vel_error + right_vel_error) / vel_std**2)
    activity_score = torch.tanh((torch.abs(left_elbow_vel) + torch.abs(right_elbow_vel)) / (2 * activity_speed))

    moving_command = torch.norm(env.command_generator.command[:, :2], dim=1) + 0.5 * torch.abs(
        env.command_generator.command[:, 2]
    )
    moving = moving_command > command_threshold
    return (0.15 * range_score + 0.30 * target_score + 0.35 * vel_score + 0.20 * activity_score) * moving

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


def feet_clearance_relative(
    env: BaseEnv | Elf3Env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.16,
    std: float = 0.05,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward swing-foot clearance relative to the currently supporting foot.

    World-frame foot height is not suitable on stairs because the terrain
    elevation changes. This term uses the contacted foot as the local ground
    reference and only scores the other foot while the robot is commanded to
    move.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    support_count = contacts.sum(dim=1, keepdim=True)
    support_z = torch.sum(feet_z * contacts.float(), dim=1, keepdim=True) / support_count.clamp_min(1)
    moving = torch.norm(env.command_generator.command[:, :2], dim=1, keepdim=True) > command_threshold
    swing_mask = (~contacts) & (support_count > 0) & moving

    clearance = feet_z - support_z
    score = torch.exp(-torch.square(clearance - target_height) / std**2)
    swing_count = swing_mask.sum(dim=1).clamp_min(1)
    return torch.sum(score * swing_mask.float(), dim=1) / swing_count


def swing_foot_forward_progress(
    env: BaseEnv | Elf3Env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_forward: float = 0.16,
    std: float = 0.10,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward the swing foot for moving ahead of the support foot.

    Clearance alone can produce a backward-kicking gait. This term measures foot
    x-position in the robot body frame and rewards the swing foot when it moves
    in the commanded travel direction relative to the stance foot.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    feet_rel_w = feet_pos_w - asset.data.root_link_pos_w[:, None, :]
    num_envs, num_feet = feet_rel_w.shape[:2]
    root_quat_inv = math_utils.quat_conjugate(asset.data.root_link_quat_w)
    root_quat_inv = root_quat_inv[:, None, :].expand(num_envs, num_feet, 4).reshape(-1, 4)
    feet_pos_b = math_utils.quat_apply(root_quat_inv, feet_rel_w.reshape(-1, 3)).reshape(num_envs, num_feet, 3)

    support_count = contacts.sum(dim=1, keepdim=True)
    support_x = torch.sum(feet_pos_b[:, :, 0] * contacts.float(), dim=1, keepdim=True) / support_count.clamp_min(1)

    cmd_x = env.command_generator.command[:, 0:1]
    direction = torch.sign(cmd_x)
    moving = torch.abs(cmd_x) > command_threshold
    swing_mask = (~contacts) & (support_count > 0) & moving

    progress = direction * (feet_pos_b[:, :, 0] - support_x)
    deficit = torch.clamp(target_forward - progress, min=0.0)
    score = torch.exp(-torch.square(deficit) / std**2)
    swing_count = swing_mask.sum(dim=1).clamp_min(1)
    return torch.sum(score * swing_mask.float(), dim=1) / swing_count


def safe_tread_landing(
    env: BaseEnv | Elf3Env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    edge_margin: float = 0.03,
    foot_rear_extent: float = 0.074,
    foot_front_extent: float = 0.132,
    min_support_overlap: float = 0.04,
    sole_bottom_offset: float = 0.04,
    height_tolerance: float = 0.10,
    touchdown_bonus: float = 1.0,
    center_fraction: float = 0.0,
    confirmation_frames: int = 3,
    max_support_speed: float = 0.25,
    min_support_vertical_ratio: float = 0.65,
) -> torch.Tensor:
    """Reward centered swing-foot overlap and confirmed tread support.

    The ELF3 sole is longer than its 20 cm curriculum treads and the ankle-link
    origin is not the sole center. Compare the actual fore-aft collision extent
    with the safe tread interval, allowing realistic toe or heel overhang.
    """

    if not hasattr(env, "terrain_geometry") or env.terrain_geometry is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = feet_pos_w.shape[:2]
    front_offset = torch.tensor(
        (foot_front_extent, 0.0, 0.0), device=env.device
    ).expand(num_envs, num_feet, -1)
    rear_offset = torch.tensor(
        (-foot_rear_extent, 0.0, 0.0), device=env.device
    ).expand(num_envs, num_feet, -1)
    foot_front_w = feet_pos_w + math_utils.quat_apply(
        feet_quat_w.reshape(-1, 4), front_offset.reshape(-1, 3)
    ).reshape(num_envs, num_feet, 3)
    foot_rear_w = feet_pos_w + math_utils.quat_apply(
        feet_quat_w.reshape(-1, 4), rear_offset.reshape(-1, 3)
    ).reshape(num_envs, num_feet, 3)

    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    yaw_quat = yaw_quat[:, None, :].expand(num_envs, num_feet, 4).reshape(-1, 4)
    foot_front_yaw = math_utils.quat_apply_inverse(
        yaw_quat, (foot_front_w - asset.data.root_pos_w[:, None, :]).reshape(-1, 3)
    ).reshape(num_envs, num_feet, 3)
    foot_rear_yaw = math_utils.quat_apply_inverse(
        yaw_quat, (foot_rear_w - asset.data.root_pos_w[:, None, :]).reshape(-1, 3)
    ).reshape(num_envs, num_feet, 3)
    foot_near_x = torch.minimum(foot_front_yaw[..., 0], foot_rear_yaw[..., 0]).unsqueeze(-1)
    foot_far_x = torch.maximum(foot_front_yaw[..., 0], foot_rear_yaw[..., 0]).unsqueeze(-1)

    # Use the same ego-motion-compensated history as the actor. By the time a
    # foot contacts a tread, the current camera frame often looks past that
    # surface to the next step; current-frame-only matching therefore misses
    # the landing that the policy just executed.
    if hasattr(env, "motion_compensated_treads"):
        treads = env.motion_compensated_treads()
    else:
        treads = env.terrain_geometry.treads
    tread_world_z = env.motion_compensated_tread_heights()
    sole_bottom_z = (feet_pos_w[..., 2] - sole_bottom_offset).unsqueeze(-1)
    supported_by_tread, support_overlap, height_error, dense_score = match_sole_to_treads(
        foot_near_x,
        foot_far_x,
        sole_bottom_z,
        treads,
        tread_world_z,
        edge_margin,
        min_support_overlap,
        height_tolerance,
    )
    safe_near = treads[..., 0].unsqueeze(1) + edge_margin
    safe_far = treads[..., 1].unsqueeze(1) - edge_margin
    tread_valid = treads[..., 5].unsqueeze(1) > 0.5
    centered_support, center_error, center_tolerance = centered_tread_support(
        foot_near_x, foot_far_x, treads, supported_by_tread, center_fraction
    )
    dense_score = dense_score * torch.exp(-torch.square(center_error / center_tolerance))
    near_tread = tread_valid & (support_overlap >= -0.06) & (height_error <= 0.20)
    foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    upward_support = (foot_force[..., 2] > 5.0) & (
        foot_force[..., 2] / torch.linalg.norm(foot_force, dim=-1).clamp_min(1.0)
        >= min_support_vertical_ratio
    )
    safe_contact = centered_support.any(dim=-1) & first_contact & upward_support
    stair_mode = getattr(env, "stair_mode", torch.zeros(num_envs, device=env.device))
    unsafe_contact = unsafe_tread_touchdown(
        first_contact,
        safe_contact,
        tread_valid,
        support_overlap,
        height_error,
        stair_mode,
        height_tolerance,
    )
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    stable_support = centered_support.any(dim=-1) & upward_support & (foot_speed < max_support_speed)
    if not hasattr(env, "safe_tread_support_count"):
        env.safe_tread_support_count = torch.zeros(
            num_envs, num_feet, dtype=torch.long, device=env.device
        )
    support_count, confirmed = confirmed_support_transition(
        env.safe_tread_support_count, stable_support, confirmation_frames
    )
    env.safe_tread_support_count = support_count

    # Expose compact tensors for playback diagnostics. They are detached from
    # the reward result and do not enter the policy observation.
    env.safe_tread_first_contact = first_contact
    env.safe_tread_supported_first_contact = safe_contact
    env.safe_tread_unsafe_first_contact = unsafe_contact
    env.safe_tread_confirmed_touchdown = confirmed
    env.safe_tread_evaluated_step = env.sim_step_counter
    if not hasattr(env, "_safe_tread_body_mass"):
        env._safe_tread_body_mass = asset.data.default_mass.to(feet_pos_w.device)
    mass = env._safe_tread_body_mass
    total_mass = mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    com_pos_w = (asset.data.body_com_pos_w * mass.unsqueeze(-1)).sum(dim=1) / total_mass
    com_vel_w = (asset.data.body_com_lin_vel_w * mass.unsqueeze(-1)).sum(dim=1) / total_mass
    capture_offset = capture_point_offset(
        com_pos_w, com_vel_w, feet_pos_w, env._root_heading_xy()
    )
    env.safe_tread_capture_forward = capture_offset[..., 0]
    env.safe_tread_capture_lateral = capture_offset[..., 1]
    env.safe_tread_stable_support = stable_support
    matched_overlap = torch.where(
        centered_support, support_overlap, -torch.inf
    )
    matched_index = matched_overlap.argmax(dim=-1, keepdim=True)
    env.safe_tread_confirmed_overlap = matched_overlap.gather(-1, matched_index).squeeze(-1)
    env.safe_tread_confirmed_height_error = height_error.gather(-1, matched_index).squeeze(-1)
    env.safe_tread_confirmed_safe_edges = torch.stack((safe_near, safe_far), dim=-1).expand(
        -1, num_feet, -1, -1
    ).gather(-2, matched_index.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(-2)
    env.safe_tread_confirmed_tread_height = tread_world_z.unsqueeze(1).expand(
        -1, num_feet, -1
    ).gather(-1, matched_index).squeeze(-1)
    env.safe_tread_has_candidate = tread_valid.any(dim=(1, 2))
    env.safe_tread_near_candidate = near_tread.any(dim=-1)
    env.safe_tread_foot_inside = centered_support.any(dim=-1)
    env.safe_tread_max_overlap = torch.where(
        tread_valid, support_overlap, -torch.inf
    ).amax(dim=(1, 2))
    best_overlap_per_foot = torch.where(
        tread_valid, support_overlap, -torch.inf
    ).amax(dim=2)
    env.safe_tread_contact_overlap = torch.where(
        first_contact, best_overlap_per_foot, -torch.inf
    ).amax(dim=1)
    env.safe_tread_min_height_error = torch.where(
        tread_valid & (support_overlap > 0.0), height_error, torch.inf
    ).amin(dim=(1, 2))
    interval_distance = torch.maximum(safe_near - foot_far_x, foot_near_x - safe_far).clamp_min(0.0)
    interval_distance = torch.where(tread_valid, interval_distance, torch.inf)
    env.safe_tread_min_distance = interval_distance.amin(dim=(1, 2))
    env.safe_tread_feet_x = 0.5 * (foot_near_x.squeeze(-1) + foot_far_x.squeeze(-1))
    env.safe_tread_near = torch.where(treads[..., 5] > 0.5, treads[..., 0] + edge_margin, torch.inf)
    env.safe_tread_far = torch.where(treads[..., 5] > 0.5, treads[..., 1] - edge_margin, -torch.inf)

    # Keep the first-contact diagnostics above, but provide a dense overlap
    # signal for PPO. Requiring both a first-contact event and 4 cm of overlap
    # made this term almost always zero, so the policy could see a valid target
    # without receiving feedback while the foot was moving onto it.
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.linalg.norm(contact_forces, dim=-1).amax(dim=1) > 1.0
    swing_with_support = (~contacts) & contacts.any(dim=1, keepdim=True)
    overlap_score = (dense_score * swing_with_support.unsqueeze(-1)).amax(dim=(1, 2))
    confirmed_score = confirmed.float().mean(dim=1)
    return overlap_score + touchdown_bonus * confirmed_score


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


def recovery_orientation_l2(
    env: BaseEnv | Elf3Env,
    tilt_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body tilt only after it has entered a recovery band."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return torch.square(torch.clamp(tilt - tilt_threshold, min=0.0))


def recovery_ang_vel_xy_l2(
    env: BaseEnv | Elf3Env,
    tilt_threshold: float = 0.10,
    ang_vel_threshold: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity when the body is already wobbling."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    ang_vel_xy = torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
    recovering = (tilt > tilt_threshold) | (ang_vel_xy > ang_vel_threshold)
    return recovering.float() * torch.square(torch.clamp(ang_vel_xy - ang_vel_threshold, min=0.0))


def recovery_action_rate_l2(
    env: BaseEnv | Elf3Env,
    tilt_threshold: float = 0.10,
    ang_vel_threshold: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Discourage abrupt action changes while the robot is recovering balance."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    ang_vel_xy = torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
    recovering = (tilt > tilt_threshold) | (ang_vel_xy > ang_vel_threshold)

    buf = env.action_buffer._circular_buffer.buffer
    action_rate = torch.sum(torch.square(buf[:, -1, :] - buf[:, -2, :]), dim=1)
    return recovering.float() * action_rate



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
