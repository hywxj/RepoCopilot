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

from dataclasses import dataclass
from typing import Literal

import torch
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg as BaseCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.spawners import PreviewSurfaceCfg, SphereCfg
from isaaclab.utils import configclass

from .camera import Camera


@dataclass
class SensorNoiseCfg:
    """Configuration for sensor noise."""

    enable: bool = False
    mode: Literal["gaussian", "dropout", "combined"] = "gaussian"

    # Gaussian noise parameters
    depth_std: float = 0.01
    depth_std_multiplier: float = 0.01

    # Dropout noise parameters
    dropout_prob: float = 0.01
    dropout_value: float = 0.0


@configclass
class GeometryPerceptionCfg:
    """Fixed-size terrain geometry extracted from depth before policy fusion."""

    enabled: bool = False
    use_for_actor: bool = True
    use_for_critic: bool = True
    history_length: int = 2

    processing_width: int = 160
    processing_height: int = 90
    min_forward: float = 0.15
    max_forward: float = 2.0
    lateral_half_width: float = 0.35
    min_height_from_root: float = -1.5
    max_height_from_root: float = 0.35
    profile_bin_size: float = 0.025
    max_treads: int = 4
    descending_edge_offset_m: float = 0.0

    # The detector is deliberately slightly wider than the 11-16 cm training
    # range so noisy real measurements are not rejected at the boundary.
    min_riser_height: float = 0.08
    max_riser_height: float = 0.21
    min_tread_depth: float = 0.16
    max_tread_depth: float = 0.36
    max_tread_height_std: float = 0.035
    min_tread_confidence: float = 0.25
    safe_edge_margin: float = 0.03
    roughness_reference: float = 0.04

    # CPU diagnostic validation; not part of the legacy actor observation.
    surface_validation_enabled: bool = False
    surface_processing_width: int = 640
    surface_processing_height: int = 360
    surface_normals_enabled: bool = True
    surface_local_tilt_deg: float = 45.0
    surface_normal_window_size: int = 7
    surface_normal_radius: int = 4
    surface_memory_enabled: bool = False
    surface_edge_margin: float = 0.02
    surface_uncertainty_margin: float = 0.01
    # Opt-in single-step supervisor; legacy fusion checkpoints remain unchanged.
    step_control_enabled: bool = False
    step_skill_pretrain: bool = False

    # A temporal state machine keeps isolated depth artifacts from enabling
    # the stair-specific policy branch.
    gate_enabled: bool = False
    gate_enter_confidence: float = 0.65
    gate_exit_confidence: float = 0.35
    gate_enter_frames: int = 3
    gate_exit_frames: int = 15
    gate_switch_frames: int = 3
    gate_min_valid_treads: int = 1
    gate_min_nearest_tread: float = 0.15
    gate_max_nearest_tread: float = 1.25
    append_stair_mode: bool = False
    append_course_alignment: bool = False
    append_foothold_targets: bool = False
    foothold_actor_scale: float = 1.0
    alignment_safety_gate_enabled: bool = False
    alignment_safety_lateral_threshold: float = 0.20
    alignment_safety_heading_threshold: float = 0.18
    foothold_target_offset: float = 0.20
    foothold_target_min_forward_gap: float = -0.12
    foothold_target_distance_scale: float | None = None
    foothold_preview_stance: bool = False
    course_lane_half_width: float = 1.5
    course_lateral_velocity_scale: float = 0.6
    course_yaw_rate_scale: float = 1.5
    bootstrap_stairs: bool = False
    bootstrap_stair_row: int = 1
    bootstrap_x_offset: float = 1.5
    bootstrap_spawn_height: float = 0.0
    bootstrap_goal_distance: float = 3.4
    bootstrap_lateral_limit: float = 0.80

    # Keep detected treads in a short, ego-motion-compensated history so a
    # fast camera motion does not immediately erase a valid landing region.
    motion_compensate_history: bool = False
    landing_memory_length: int = 40
    landing_history_rear_limit: float = 0.60
    foothold_target_lock_enabled: bool = False
    foothold_ik_enabled: bool = False
    foothold_ik_gain: float = 0.6
    foothold_ik_max_joint_step: float = 0.12
    foothold_overshoot_guard_enabled: bool = False
    foothold_overshoot_guard_max_joint_step: float = 0.05
    foothold_swing_trajectory_enabled: bool = False
    foothold_swing_duration_s: float = 0.28
    foothold_swing_clearance_m: float = 0.10
    foothold_control_min_confidence: float = 0.70
    foothold_control_min_gate: float = 0.60
    foothold_control_center_tolerance_m: float = 0.08
    stair_heading_feedback_enabled: bool = False
    stair_yaw_lateral_gain: float = 0.6
    stair_yaw_rate_limit: float = 0.45

    # After a foot lands on a stair, briefly hold the walking command until
    # consecutive fresh depth frames agree on the tread edges.
    stair_step_settle_enabled: bool = False
    blind_during_stair_settle: bool = False
    settle_min_time_s: float = 0.08
    settle_max_time_s: float = 0.24
    settle_stable_frames: int = 2
    settle_edge_tolerance: float = 0.04
    settle_contact_force: float = 5.0
    settle_step_drop: float = 0.055


@configclass
class CameraCfg(BaseCameraCfg):
    class_type: type = Camera

    enable_depth_camera: bool = False
    prim_body_name: str = "pelvis/depth_camera"

    # Camera parameters
    width: int = 480
    height: int = 270
    max_range: float = 15.0
    min_range: float = 0.2
    feature_width: int = 8
    feature_height: int = 12
    depth_history_length: int = 3
    frame_hold_prob: float = 0.0
    depth_quantization: float = 0.0
    geometry: GeometryPerceptionCfg = GeometryPerceptionCfg()

    data_types: list[str] = ["distance_to_image_plane"]
    offset: BaseCameraCfg.OffsetCfg = BaseCameraCfg.OffsetCfg()
    spawn: PinholeCameraCfg = PinholeCameraCfg()
    sensor_noise: SensorNoiseCfg = SensorNoiseCfg()

    # Camera Visualization Configuration
    debug_vis: bool = False
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/CameraPointCloud",
        markers={
            "point": SphereCfg(
                radius=0.02,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            )
        },
    )
    visualizer_cfg.decimation = 10

    far_out_of_range_value = torch.inf
    near_out_of_range_value = torch.inf
