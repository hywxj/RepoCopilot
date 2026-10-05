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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
# from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, quat_from_euler_xyz
from scipy.spatial.transform import Rotation

# from legged_lab.envs.elf3.run_cfg import Elf3RunFlatEnvCfg
# from legged_lab.envs.elf3.run_with_sensor_cfg import Elf3RunWithSensorFlatEnvCfg
from legged_lab.envs.elf3.walk_cfg import Elf3WalkFlatEnvCfg
from legged_lab.perception import StairGeometryExtractor, StairModeGate, StairModeGateCfg, compensate_tread_history
from legged_lab.perception.stair_step_controller import StairStepController, StepMeasurement, StepPhase
from legged_lab.perception.stair_skill_teacher import StairSkillTeacher
from legged_lab.perception.stair_geometry import _matrix_from_quat, _yaw_matrix
from legged_lab.perception.foothold_control import (
    bounded_overshoot_joint_step,
    damped_joint_step,
    current_tread_match,
    foothold_lock_transition,
    terminal_platform_support,
    terminal_platform_height,
    select_next_treads,
    stair_support_coverage_transition,
    stair_width_promotion,
    stair_yaw_feedback,
    supported_stair_progress,
    swing_foot_trajectory,
)
# from legged_lab.envs.elf3.walk_with_sensor_cfg import (
#     Elf3WalkWithSensorFlatEnvCfg,
# )
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class _MaskedHistoryBuffer:
    """Small batched history whose rows can advance independently."""

    def __init__(self, max_len: int, batch_size: int, feature_shape: tuple[int, ...], device: str):
        self.max_len = max_len
        self.batch_size = batch_size
        self._buffer = torch.zeros(batch_size, max_len, *feature_shape, device=device)
        self._count = torch.zeros(batch_size, dtype=torch.long, device=device)

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    def append(self, data: torch.Tensor, update_mask: torch.Tensor | None = None) -> None:
        if update_mask is None:
            update_mask = torch.ones(self.batch_size, dtype=torch.bool, device=data.device)
        first = update_mask & (self._count == 0)
        continuing = update_mask & ~first
        if torch.any(continuing):
            self._buffer[continuing, :-1] = self._buffer[continuing, 1:].clone()
            self._buffer[continuing, -1] = data[continuing]
        if torch.any(first):
            self._buffer[first] = data[first].unsqueeze(1)
        self._count[update_mask] += 1

    def reset(self, env_ids: torch.Tensor) -> None:
        self._buffer[env_ids] = 0.0
        self._count[env_ids] = 0


class Elf3Env(VecEnv):
    def __init__(
        self,
        # cfg: (
        #     Elf3RunFlatEnvCfg
        #     | Elf3WalkFlatEnvCfg
        #     | Elf3WalkWithSensorFlatEnvCfg
        #     | Elf3RunWithSensorFlatEnvCfg
        # ),
        cfg: (
            Elf3WalkFlatEnvCfg
        ),
        headless,
    ):
        # self.cfg: (
        #     Elf3RunFlatEnvCfg
        #     | Elf3WalkFlatEnvCfg
        #     | Elf3WalkWithSensorFlatEnvCfg
        #     | Elf3RunWithSensorFlatEnvCfg
        # )
        self.cfg: (
            Elf3WalkFlatEnvCfg
        )

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        terrain_cfg = self.cfg.scene.terrain_generator
        geometry_cfg = getattr(getattr(self.cfg.scene, "depth_camera", None), "geometry", None)
        self.stair_step_enabled = bool(getattr(geometry_cfg, "step_control_enabled", False))
        self.step_skill_pretrain = bool(getattr(geometry_cfg, "step_skill_pretrain", False))
        if self.step_skill_pretrain and (not self.stair_step_enabled or self.cfg.scene.depth_camera.enable_depth_camera):
            raise ValueError("The simulation motor teacher requires step control and cameras disabled.")
        if self.stair_step_enabled:
            if not (geometry_cfg.enabled and geometry_cfg.surface_validation_enabled
                    and geometry_cfg.surface_memory_enabled and geometry_cfg.append_stair_mode):
                raise ValueError("Step control requires 2-D observed surface memory and a final actor gate.")
            if any((geometry_cfg.foothold_ik_enabled, geometry_cfg.foothold_overshoot_guard_enabled,
                    geometry_cfg.foothold_target_lock_enabled, geometry_cfg.stair_step_settle_enabled,
                    geometry_cfg.stair_heading_feedback_enabled)):
                raise ValueError("Single-step supervision cannot be mixed with legacy stair controllers.")
            if self.num_envs > 4 and not self.step_skill_pretrain:
                raise ValueError("CPU 2-D single-step validation currently supports at most 4 environments.")
        if (
            geometry_cfg is not None
            and geometry_cfg.enabled
            and getattr(getattr(terrain_cfg, "class_type", None), "__name__", "")
            == "AtecObstacleCourseTerrainGenerator"
            and self.num_envs > terrain_cfg.num_cols
        ):
            raise ValueError(
                f"Geometry course needs a separate terrain lane per camera: "
                f"num_envs={self.num_envs}, num_cols={terrain_cfg.num_cols}."
            )
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # self.robot: Articulation = self.scene["robot"]
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]
            camera_parent_name = self.cfg.scene.depth_camera.prim_body_name.split("/", 1)[0]
            camera_parent_ids, _ = self.robot.find_bodies(name_keys=[camera_parent_name], preserve_order=True)
            if len(camera_parent_ids) != 1:
                raise ValueError(
                    f"Depth-camera parent body '{camera_parent_name}' must resolve to exactly one robot body."
                )
            self.depth_camera_parent_body_id = camera_parent_ids[0]
            self.depth_camera_offset_pos = torch.tensor(
                self.cfg.scene.depth_camera.offset.pos, dtype=torch.float, device=self.device
            ).expand(self.num_envs, -1)
            self.depth_camera_offset_quat = torch.tensor(
                self.cfg.scene.depth_camera.offset.rot, dtype=torch.float, device=self.device
            ).expand(self.num_envs, -1)

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()
        self._initialize_course_width_curriculum()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        if self.stair_step_enabled:
            self.step_controllers = [StairStepController() for _ in range(self.num_envs)]
            self.step_body_masses = self.robot.root_physx_view.get_masses().to(self.device)
            self.step_body_weight = self.step_body_masses.sum(dim=1).cpu().numpy()*9.81
            self.step_contact_ids, _ = self.contact_sensor.find_bodies(
                ["l_ankle_x_link", "r_ankle_x_link"], preserve_order=True)
            self.step_features = torch.zeros(self.num_envs, StairStepController.num_features, device=self.device)
            self.step_critic_features = torch.zeros_like(self.step_features)
            self.step_gate = torch.zeros(self.num_envs, 1, device=self.device)
            self.step_success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.step_failure = torch.zeros_like(self.step_success_event)
            self.step_last_phases = [StepPhase.BLIND]*self.num_envs
            self.step_last_failures = [""]*self.num_envs
            self.step_reward_metrics = {name: torch.zeros(self.num_envs, device=self.device) for name in (
                "feet_reference", "body_reference", "load_reference", "heading", "sole_tilt", "unsafe_contact",
                "downward_speed", "progress", "lift_progress", "success", "stop_root_motion", "stop_feet_motion",
                "stop_angular_motion")}
            if self.step_skill_pretrain:
                stair_cfg = self.cfg.scene.terrain_generator.sub_terrains[
                    "stairs_down" if geometry_cfg.bootstrap_stair_row == 2 else "stairs_up"]
                direction = -1 if geometry_cfg.bootstrap_stair_row == 2 else 1
                near_edge = stair_cfg.approach_length-.5*self.cfg.scene.terrain_generator.size[0]+geometry_cfg.bootstrap_x_offset
                if direction < 0:
                    near_edge += stair_cfg.step_width
                self.step_skill_teacher = StairSkillTeacher(
                    self.scene.env_origins, near_edge, stair_cfg.step_width, stair_cfg.step_height_range[0],
                    direction, torch.as_tensor(self.step_body_weight, device=self.device))
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        # 支持标量或逐关节列表的action_scale
        if isinstance(self.cfg.robot.action_scale, (list, tuple)):
            self.action_scale = torch.tensor(self.cfg.robot.action_scale, dtype=torch.float, device=self.device)
        else:
            self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        
        ###########################################################################add joint and body ids
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_y_joint",
                "waist_x_joint",
                "waist_z_joint",
            ],
            preserve_order=True,
        )
        self.left_wrist_ids, _ = self.robot.find_joints(
            name_keys=[
                "l_wrist_x_joint",
                "l_wrist_y_joint",
                "l_wrist_z_joint",
            ],
            preserve_order=True,
        )
        self.right_wrist_ids, _ = self.robot.find_joints(
            name_keys=[
                "r_wrist_x_joint",
                "r_wrist_y_joint",
                "r_wrist_z_joint",
            ],
            preserve_order=True,
        )
        ###########################################################################
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_ankle_x_link", "r_ankle_x_link"], preserve_order=True
        )
        self.hip_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_hip_z_link", "r_hip_z_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_elbow_y_link", "r_elbow_y_link"], preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "l_hip_y_joint",
                "l_hip_x_joint",
                "l_hip_z_joint",
                "l_knee_y_joint",
                "l_ankle_y_joint",
                "l_ankle_x_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "r_hip_y_joint",
                "r_hip_x_joint",
                "r_hip_z_joint",
                "r_knee_y_joint",
                "r_ankle_y_joint",
                "r_ankle_x_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "l_shoulder_y_joint",
                "l_shoulder_x_joint",
                "l_shoulder_z_joint",
                "l_elbow_y_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "r_shoulder_y_joint",
                "r_shoulder_x_joint",
                "r_shoulder_z_joint",
                "r_elbow_y_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["l_ankle_y_joint", "r_ankle_y_joint", "l_ankle_x_joint", "r_ankle_x_joint"],
            preserve_order=True,
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        
        # # Init gait parameter
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        
        # self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
        
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

    def visualize_motion(self, time, root_z_offset=0.3, root_x_offset=0.0):
        """
        根据给定时间的 AMP 运动捕捉数据更新机器人模拟状态。

        该函数设置关节位置和速度、根位置和方向，
        以及根据指定时间的 AMP 运动框架的线/角速度，
        然后逐步进行模拟并更新场景。

        参数：
            time（浮点数）：获取 AMP 运动帧的时间（以秒为单位）。

        返回：
            无
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device
        
        # # ====== 新增：关键调试信息 ======
        # print(f'[DEBUG] 运动数据总维度: {visual_motion_frame.shape}') # 应为70
        # print(f'[DEBUG] 机器人关节数: {self.robot.num_joints}') # 应为29

        # # 打印前35维（关节位置）和后35维（关节速度）的样例，了解数据范围
        # print(f'[DEBUG] 关节位置数据样例 (前10维): {visual_motion_frame[:10].cpu().numpy().round(3)}')
        # print(f'[DEBUG] 关节速度数据样例 (索引35-44): {visual_motion_frame[35:45].cpu().numpy().round(3)}')
        # # 打印最后几维，确认根信息
        # print(f'[DEBUG] 数据最后几维 (索引-12): {visual_motion_frame[-12:].cpu().numpy().round(3)}')
        # # ====== 调试结束 ======
        #数据输入

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)#35
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)#35
        # root pos: 0-3
        # root euler: 3-6
        dof_pos[:, self.waist_ids] = visual_motion_frame[6:9]#
        dof_pos[:, self.left_leg_ids] = visual_motion_frame[9:15]#3+3+3+6
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[15:21]#3+3+3+6+6
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[21:25]#3+3+3+6+6+4
        dof_pos[:, self.left_wrist_ids] = visual_motion_frame[25:28]#3+3+3+6+6+4+3
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[28:32]#3+3+3+6+6+4+3+4
        dof_pos[:, self.right_wrist_ids] = visual_motion_frame[32:35]#3+3+3+6+6+4+3+4+3
        #root_lin_vel: 35-38 
        #root_ang_vel: 38-41
        dof_vel[:, self.waist_ids] = visual_motion_frame[41:44]#35+3+3+3+6
        dof_vel[:, self.left_leg_ids] = visual_motion_frame[44:50]#35+3+3+3+6
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[50:56]#35+3+3+6+6
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[56:60]#35+3+3+6+6+4
        dof_vel[:, self.left_wrist_ids] = visual_motion_frame[60:63]#35+3+3+6+6+4+3
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[63:67]#35+3+3+6+6+4+3+4
        dof_vel[:, self.right_wrist_ids] = visual_motion_frame[67:70]#35+3+3+6+6+4+3+4+3

        self.robot.write_joint_position_to_sim(dof_pos)#35
        self.robot.write_joint_velocity_to_sim(dof_vel)#35

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[0] += root_x_offset
        root_pos[2] += root_z_offset

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[35:38].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)  # gmr转motion_data数据写入模拟器
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.waist_dof_pos =  dof_pos[:, self.waist_ids] 
        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids] 
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids +self.left_wrist_ids] 
        # self.left_wrist_dof_pos = dof_pos[:, self.left_wrist_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids + self.right_wrist_ids]
        # self.right_wrist_dof_pos = dof_pos[:, self.right_wrist_ids]
        
        
        self.waist_dof_vel =  dof_vel[:, self.waist_ids]
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids] 
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids + self.left_wrist_ids] 
        # self.left_wrist_dof_vel = dof_vel[:, self.left_wrist_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids + self.right_wrist_ids]
        # self.right_wrist_dof_vel = dof_vel[:, self.right_wrist_ids]

        return torch.cat(
            (   
                
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.waist_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                
                
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.waist_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                
                
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )


    def compute_current_observations(self):
        robot = self.robot
        # print("robot data default joint pos:", robot.data.default_joint_pos)#默认关节位置
        # print(robot.data.joint_names)#关节名称
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 29
                joint_vel * self.obs_scales.joint_vel,  # 29
                action * self.obs_scales.actions,  # 29
                # torch.sin(2 * torch.pi * self.gait_phase),  # 2
                # torch.cos(2 * torch.pi * self.gait_phase),  # 2
                # self.phase_ratio,  # 2
            ],
            dim=-1,
        )
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def _update_terrain_geometry(self) -> bool:
        """Extract and buffer the latest geometry frame before reward evaluation."""
        if not (
            self.cfg.scene.depth_camera.enable_depth_camera
            and self.cfg.scene.depth_camera.geometry.enabled
        ):
            return False
        depth_image = self.depth_camera.data.output["distance_to_image_plane"]
        camera_frame = self.depth_camera.frame.clone()
        if self._last_geometry_camera_frame is None:
            fresh_frame = torch.ones_like(camera_frame, dtype=torch.bool)
            self._last_geometry_camera_frame = camera_frame
        else:
            fresh_frame = camera_frame != self._last_geometry_camera_frame
            self._last_geometry_camera_frame.copy_(camera_frame)
            if not torch.any(fresh_frame):
                memories = self.terrain_geometry_extractor.surface_memories
                if memories is not None:
                    positions = self.robot.data.root_pos_w.detach().cpu().numpy()
                    from legged_lab.perception.stair_geometry import _yaw_matrix
                    rotations = _yaw_matrix(self.robot.data.root_quat_w).detach().cpu().numpy()
                    times = self.depth_camera.current_timestamp.detach().cpu().numpy()
                    self.terrain_geometry.surface_memory = [memory.refresh(positions[i], rotations[i], times[i])
                                                            for i, memory in enumerate(memories)]
                return False

        camera_pos_w, camera_quat_w_ros = self._current_depth_camera_pose()
        self.terrain_geometry = self.terrain_geometry_extractor.extract(
            depth_image=depth_image,
            intrinsic_matrices=self.depth_camera.data.intrinsic_matrices,
            camera_pos_w=camera_pos_w,
            camera_quat_w_ros=camera_quat_w_ros,
            root_pos_w=self.robot.data.root_pos_w,
            root_quat_w=self.robot.data.root_quat_w,
            foot_positions_w=self.robot.data.body_pos_w[:, self.feet_body_ids],
            foot_quaternions_w=self.robot.data.body_quat_w[:, self.feet_body_ids],
            frame_timestamp=(self.depth_camera.frame_timestamp
                             if self.cfg.scene.depth_camera.geometry.surface_memory_enabled else None),
        )
        self.geometry_obs_buffer.append(self.terrain_geometry.features, fresh_frame)
        if self.cfg.scene.depth_camera.geometry.motion_compensate_history:
            self.geometry_position_buffer.append(self.robot.data.root_pos_w[:, :2], fresh_frame)
            self.geometry_heading_buffer.append(self._root_heading_xy(), fresh_frame)
            self.landing_geometry_buffer.append(self.terrain_geometry.features, fresh_frame)
            self.landing_position_buffer.append(self.robot.data.root_pos_w[:, :2], fresh_frame)
            self.landing_heading_buffer.append(self._root_heading_xy(), fresh_frame)
            world_reference_height = (
                self.robot.data.root_pos_w[:, 2] + self.terrain_geometry.reference_height
            ).unsqueeze(1)
            self.landing_reference_height_buffer.append(world_reference_height, fresh_frame)

        self._update_geometry_stability(fresh_frame)
        if self.cfg.scene.depth_camera.geometry.gate_enabled:
            valid_tread_mask = self.terrain_geometry.treads[..., 5] > 0.5
            valid_tread_count = valid_tread_mask.sum(dim=1)
            nearest_tread = torch.where(
                valid_tread_mask,
                self.terrain_geometry.treads[..., 0],
                torch.full_like(self.terrain_geometry.treads[..., 0], torch.inf),
            ).amin(dim=1)
            self.stair_mode = self.stair_mode_gate.update(
                confidence=self.terrain_geometry.stair_confidence,
                valid_treads=valid_tread_count,
                nearest_tread_m=nearest_tread,
                direction=self.terrain_geometry.direction,
                update_mask=fresh_frame,
            )
            # Shape the residual permission continuously. A hard 0 -> 1
            # switch can inject a yaw impulse exactly at the first riser.
            geometry_cfg = self.cfg.scene.depth_camera.geometry
            confidence_span = max(
                geometry_cfg.gate_enter_confidence - geometry_cfg.gate_exit_confidence,
                1.0e-6,
            )
            confidence_gate = (
                (self.terrain_geometry.stair_confidence - geometry_cfg.gate_exit_confidence)
                / confidence_span
            ).clamp(0.0, 1.0)
            valid_gate = (valid_tread_count >= geometry_cfg.gate_min_valid_treads) & (
                nearest_tread >= geometry_cfg.gate_min_nearest_tread
            ) & (nearest_tread <= geometry_cfg.gate_max_nearest_tread)
            target_gate = torch.where(
                self.stair_mode != 0,
                torch.where(valid_gate, confidence_gate.clamp_min(0.25), torch.full_like(confidence_gate, 0.25)),
                torch.zeros_like(confidence_gate),
            )
            fresh_active = fresh_frame & (self.stair_mode != 0)
            self.stair_gate_strength = torch.where(
                fresh_active,
                0.8 * self.stair_gate_strength + 0.2 * target_gate,
                torch.where(
                    fresh_frame & (self.stair_mode == 0),
                    torch.zeros_like(self.stair_gate_strength),
                    self.stair_gate_strength * 0.995,
                ),
            )
        return True

    def _current_depth_camera_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose the live parent-body pose with the calibrated ROS camera offset."""

        parent_pos_w = self.robot.data.body_pos_w[:, self.depth_camera_parent_body_id]
        parent_quat_w = self.robot.data.body_quat_w[:, self.depth_camera_parent_body_id]
        camera_pos_w = parent_pos_w + quat_apply(parent_quat_w, self.depth_camera_offset_pos)
        camera_quat_w_ros = quat_mul(parent_quat_w, self.depth_camera_offset_quat)
        return camera_pos_w, camera_quat_w_ros

    def _root_heading_xy(self) -> torch.Tensor:
        heading = quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(self.num_envs, -1),
        )[:, :2]
        return heading / torch.linalg.norm(heading, dim=1, keepdim=True).clamp_min(1.0e-6)

    def _apply_stair_heading_feedback(self) -> None:
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        if not geometry_cfg.stair_heading_feedback_enabled:
            return
        heading = self._root_heading_xy()
        heading_error = torch.atan2(heading[:, 1], heading[:, 0])
        lateral_error = self.robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
        lateral_velocity = self.robot.data.root_lin_vel_w[:, 1]
        correction = stair_yaw_feedback(
            lateral_error,
            heading_error,
            lateral_velocity,
            max_yaw_rate=geometry_cfg.stair_yaw_rate_limit,
            lateral_gain=geometry_cfg.stair_yaw_lateral_gain,
        )
        nominal_yaw = self.cfg.commands.ranges.ang_vel_z[0]
        active = self.stair_mode != 0
        self.command_generator.command[:, 2] = torch.where(
            active, nominal_yaw + correction, nominal_yaw
        )

    def _course_alignment_features(self) -> torch.Tensor:
        """Return lane error and its damping signals in the course frame.

        These values stay in the observation even in BLIND mode. The residual
        action is still multiplied by the stair-mode gate, so the frozen blind
        actor remains exactly unchanged on flat terrain.
        """

        geometry_cfg = self.cfg.scene.depth_camera.geometry
        lateral_error = (
            self.robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
        ) / max(geometry_cfg.course_lane_half_width, 1.0e-6)
        heading = self._root_heading_xy()
        heading_error = torch.atan2(heading[:, 1], heading[:, 0]) / torch.pi
        lateral_velocity = self.robot.data.root_lin_vel_w[:, 1] / max(
            geometry_cfg.course_lateral_velocity_scale, 1.0e-6
        )
        yaw_rate = self.robot.data.root_ang_vel_w[:, 2] / max(
            geometry_cfg.course_yaw_rate_scale, 1.0e-6
        )
        return torch.stack(
            (
                lateral_error.clamp(-1.0, 1.0),
                heading_error.clamp(-1.0, 1.0),
                lateral_velocity.clamp(-1.0, 1.0),
                yaw_rate.clamp(-1.0, 1.0),
            ),
            dim=1,
        )

    def _foothold_target_features(self) -> torch.Tensor:
        """Return an explicit next-tread target for each ankle.

        Each foot receives ``target_dx, target_dz, target_width, confidence``.
        The target is selected from motion-compensated tread history, so a
        newly landed foot can still use the tread that was visible before the
        camera moved past its riser.
        """

        geometry_cfg = self.cfg.scene.depth_camera.geometry
        if self.terrain_geometry is None:
            return torch.zeros(self.num_envs, 8, device=self.device)
        treads = self.motion_compensated_treads()
        tread_heights = self.motion_compensated_tread_heights()
        root_position = self.robot.data.root_pos_w.unsqueeze(1)
        feet_position = self.robot.data.body_pos_w[:, self.feet_body_ids, :] - root_position
        heading = self._root_heading_xy().unsqueeze(1)
        foot_x = (feet_position[..., :2] * heading).sum(dim=-1)
        sole_z = self.robot.data.body_pos_w[:, self.feet_body_ids, 2] - 0.04
        contacts = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2] > 5.0
        target, target_height, has_target = self._next_foothold_targets(
            treads,
            tread_heights,
            foot_x,
            sole_z,
            contacts,
            use_locked_target=not geometry_cfg.foothold_swing_trajectory_enabled,
        )
        target_center = 0.5 * (target[..., 0] + target[..., 1])
        distance_scale = geometry_cfg.foothold_target_distance_scale or geometry_cfg.max_forward
        features = torch.stack(
            (
                ((target_center - foot_x) / distance_scale).clamp(-1.0, 1.0),
                ((target_height - sole_z) / 0.5).clamp(-1.0, 1.0),
                (target[..., 3] / 0.3).clamp(0.0, 1.0),
                target[..., 4].clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        return torch.where(has_target.unsqueeze(-1), features, torch.zeros_like(features)).flatten(1)

    def _motion_compensated_geometry_history(self) -> torch.Tensor:
        """Express buffered tread edges in the current robot yaw frame."""

        features = self.geometry_obs_buffer.buffer
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        if not geometry_cfg.motion_compensate_history:
            return features

        positions = self.geometry_position_buffer.buffer
        headings = self.geometry_heading_buffer.buffer
        return compensate_tread_history(
            features,
            positions,
            headings,
            self.robot.data.root_pos_w[:, :2],
            self._root_heading_xy(),
            geometry_cfg.max_treads,
            geometry_cfg.max_forward,
            geometry_cfg.landing_history_rear_limit,
        )

    def motion_compensated_treads(self) -> torch.Tensor:
        """Return all buffered tread candidates with metric edges in the current yaw frame."""

        if self.terrain_geometry is None:
            geometry_cfg = self.cfg.scene.depth_camera.geometry
            return torch.zeros(
                self.num_envs,
                geometry_cfg.history_length * geometry_cfg.max_treads,
                6,
                device=self.device,
            )
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        if not geometry_cfg.motion_compensate_history:
            return self.terrain_geometry.treads

        features = compensate_tread_history(
            self.landing_geometry_buffer.buffer,
            self.landing_position_buffer.buffer,
            self.landing_heading_buffer.buffer,
            self.robot.data.root_pos_w[:, :2],
            self._root_heading_xy(),
            geometry_cfg.max_treads,
            geometry_cfg.max_forward,
            geometry_cfg.landing_history_rear_limit,
        )
        treads = features[..., : geometry_cfg.max_treads * 6].reshape(
            self.num_envs, -1, 6
        ).clone()
        treads[..., 0:2] *= geometry_cfg.max_forward
        return treads

    def motion_compensated_tread_heights(self) -> torch.Tensor:
        """World heights aligned with the landing-memory tread candidates."""

        geometry_cfg = self.cfg.scene.depth_camera.geometry
        if not geometry_cfg.motion_compensate_history:
            return self.robot.data.root_pos_w[:, 2:3] + (
                self.terrain_geometry.reference_height.unsqueeze(1)
                + self.terrain_geometry.treads[..., 2]
            )
        tread_heights = self.landing_geometry_buffer.buffer[
            ..., : geometry_cfg.max_treads * 6
        ].reshape(self.num_envs, -1, geometry_cfg.max_treads, 6)[..., 2] * 0.5
        reference_height = self.landing_reference_height_buffer.buffer
        return (reference_height + tread_heights).reshape(self.num_envs, -1)

    def _next_foothold_targets(
        self,
        treads: torch.Tensor,
        tread_heights: torch.Tensor,
        feet_x: torch.Tensor,
        sole_z: torch.Tensor,
        contacts: torch.Tensor,
        use_locked_target: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        selected, heights, valid = select_next_treads(
            treads,
            tread_heights,
            feet_x,
            sole_z,
            contacts,
            self.stair_mode,
            geometry_cfg.foothold_target_offset,
            preview_stance=geometry_cfg.foothold_preview_stance,
            min_forward_gap=geometry_cfg.foothold_target_min_forward_gap,
        )
        if not geometry_cfg.foothold_target_lock_enabled:
            return selected, heights, valid

        mode = self.stair_mode.unsqueeze(1).expand_as(contacts)
        self.foothold_lock_valid, acquire = foothold_lock_transition(
            self.foothold_lock_valid,
            self.foothold_lock_mode,
            contacts,
            self.stair_mode,
            valid,
        )
        if geometry_cfg.foothold_swing_trajectory_enabled:
            self.foothold_swing_elapsed = torch.where(
                self.foothold_lock_valid,
                self.foothold_swing_elapsed,
                torch.zeros_like(self.foothold_swing_elapsed),
            )
            self.foothold_lock_control_valid &= self.foothold_lock_valid
        if torch.any(acquire):
            heading = self._root_heading_xy().unsqueeze(1)
            lateral = torch.stack((-heading[..., 1], heading[..., 0]), dim=-1)
            center_x = 0.5 * (selected[..., 0] + selected[..., 1])
            foot_world = self.robot.data.body_pos_w[:, self.feet_body_ids]
            foot_lateral = (
                (foot_world[..., :2] - self.robot.data.root_pos_w[:, None, :2]) * lateral
            ).sum(dim=-1)
            world_xy = (
                self.robot.data.root_pos_w[:, None, :2]
                + heading * center_x.unsqueeze(-1)
                + lateral * foot_lateral.unsqueeze(-1)
            )
            self.foothold_lock_world_xy[acquire] = world_xy[acquire]
            self.foothold_lock_world_z[acquire] = heights[acquire]
            self.foothold_lock_width[acquire] = selected[..., 3][acquire]
            self.foothold_lock_confidence[acquire] = selected[..., 4][acquire]
            self.foothold_lock_mode[acquire] = mode[acquire]
            if geometry_cfg.foothold_swing_trajectory_enabled:
                self.foothold_swing_start_w[acquire] = foot_world[acquire]
                self.foothold_swing_elapsed[acquire] = 0.0
                fresh = current_tread_match(
                    selected,
                    self.terrain_geometry.treads,
                    geometry_cfg.foothold_control_min_confidence,
                    geometry_cfg.foothold_control_center_tolerance_m,
                )
                control_valid = (
                    fresh
                    & (selected[..., 4] >= geometry_cfg.foothold_control_min_confidence)
                    & (self.stair_gate_strength.unsqueeze(1) >= geometry_cfg.foothold_control_min_gate)
                )
                self.foothold_lock_control_valid[acquire] = control_valid[acquire]

        heading = self._root_heading_xy().unsqueeze(1)
        center_x = (
            (self.foothold_lock_world_xy - self.robot.data.root_pos_w[:, None, :2]) * heading
        ).sum(dim=-1)
        locked = selected.clone()
        locked[..., 0] = center_x - 0.5 * self.foothold_lock_width
        locked[..., 1] = center_x + 0.5 * self.foothold_lock_width
        locked[..., 3] = self.foothold_lock_width
        locked[..., 4] = self.foothold_lock_confidence
        if not use_locked_target:
            return selected, heights, valid
        return locked, self.foothold_lock_world_z, self.foothold_lock_valid

    def _apply_foothold_ik(self, joint_targets: torch.Tensor) -> torch.Tensor:
        """Nudge a swing foot toward the next detected stair tread."""

        self.foothold_ik_active.zero_()
        if self.terrain_geometry is None:
            return joint_targets
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        treads = self.motion_compensated_treads()
        tread_heights = self.motion_compensated_tread_heights()
        feet_pos = self.robot.data.body_pos_w[:, self.feet_body_ids]
        root_pos = self.robot.data.root_pos_w
        heading = self._root_heading_xy()
        feet_x = ((feet_pos[..., :2] - root_pos[:, None, :2]) * heading[:, None]).sum(dim=-1)
        sole_z = feet_pos[..., 2] - 0.04
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids]
        contacts = contact_forces[..., 2] > 5.0
        jacobians = self.robot.root_physx_view.get_jacobians()
        gate_threshold = 0.20 if geometry_cfg.foothold_target_lock_enabled else 0.35
        stair_active = (self.stair_mode != 0) & (self.stair_gate_strength > gate_threshold)
        targets, target_heights, target_valid = self._next_foothold_targets(
            treads,
            tread_heights,
            feet_x,
            sole_z,
            contacts,
        )

        if geometry_cfg.foothold_swing_trajectory_enabled:
            target_world = torch.cat(
                (self.foothold_lock_world_xy, (self.foothold_lock_world_z + 0.04).unsqueeze(-1)),
                dim=-1,
            )
            elapsed = self.foothold_swing_elapsed + self.step_dt
            desired_world = swing_foot_trajectory(
                self.foothold_swing_start_w,
                target_world,
                elapsed,
                geometry_cfg.foothold_swing_duration_s,
                geometry_cfg.foothold_swing_clearance_m,
            )

        for foot_index, joint_ids in enumerate((self.left_leg_ids, self.right_leg_ids)):
            target_center = 0.5 * (targets[:, foot_index, 0] + targets[:, foot_index, 1])
            target_height = target_heights[:, foot_index]
            active = (
                stair_active
                & target_valid[:, foot_index]
                & ~contacts[:, foot_index]
                & contacts[:, 1 - foot_index]
            )
            if geometry_cfg.foothold_swing_trajectory_enabled:
                active &= self.foothold_lock_control_valid[:, foot_index]
            self.foothold_ik_active[:, foot_index] = active

            if geometry_cfg.foothold_swing_trajectory_enabled:
                ramp = (elapsed[:, foot_index] / 0.08).clamp(0.0, 1.0)
                position_error = (
                    desired_world[:, foot_index] - feet_pos[:, foot_index]
                ).clamp(-0.10, 0.10) * ramp.unsqueeze(-1)
                self.foothold_swing_elapsed[:, foot_index] = torch.where(
                    active, elapsed[:, foot_index], self.foothold_swing_elapsed[:, foot_index]
                )
            else:
                forward_gap = target_center - feet_x[:, foot_index]
                min_forward_error = -0.08 if geometry_cfg.foothold_target_lock_enabled else 0.0
                forward_error = (forward_gap - 0.03).clamp(min_forward_error, 0.08)
                clearance = 0.08 * (forward_gap / 0.25).clamp(0.0, 1.0)
                up_error = (target_height + clearance - sole_z[:, foot_index]).clamp(0.0, 0.08)
                descent_phase = ((0.18 - forward_gap) / 0.14).clamp(0.0, 1.0)
                down_error = (target_height + 0.04 - sole_z[:, foot_index]).clamp(-0.08, 0.04)
                vertical_error = torch.where(self.stair_mode > 0, up_error, down_error * descent_phase)
                position_error = torch.stack(
                    (forward_error * heading[:, 0], forward_error * heading[:, 1], vertical_error), dim=1
                )

            body_index = self.feet_body_ids[foot_index] - int(self.robot.is_fixed_base)
            jacobian_joint_ids = [joint_id + (0 if self.robot.is_fixed_base else 6) for joint_id in joint_ids]
            jacobian = jacobians[:, body_index, :3, :][:, :, jacobian_joint_ids]
            joint_step = damped_joint_step(
                jacobian,
                position_error * geometry_cfg.foothold_ik_gain,
                max_joint_step=geometry_cfg.foothold_ik_max_joint_step,
            )
            joint_targets[:, joint_ids] += joint_step * active.unsqueeze(1)
        return joint_targets

    def _apply_foothold_overshoot_guard(self, joint_targets: torch.Tensor) -> torch.Tensor:
        """Limit only a descending swing foot's predicted forward edge overrun."""

        self.foothold_guard_active.zero_()
        if self.terrain_geometry is None:
            return joint_targets
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        foot_pos = self.robot.data.body_pos_w[:, self.feet_body_ids]
        heading = self._root_heading_xy()
        foot_x = (
            (foot_pos[..., :2] - self.robot.data.root_pos_w[:, None, :2])
            * heading[:, None]
        ).sum(dim=-1)
        sole_z = foot_pos[..., 2] - 0.04
        contacts = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2] > 5.0
        targets, heights, valid = select_next_treads(
            self.motion_compensated_treads(),
            self.motion_compensated_tread_heights(),
            foot_x,
            sole_z,
            contacts,
            self.stair_mode,
            geometry_cfg.foothold_target_offset,
            min_forward_gap=-0.06,
        )
        jacobians = self.robot.root_physx_view.get_jacobians()
        for foot_index, joint_ids in enumerate((self.left_leg_ids, self.right_leg_ids)):
            target_center = 0.5 * (targets[:, foot_index, 0] + targets[:, foot_index, 1])
            height_above = sole_z[:, foot_index] - heights[:, foot_index]
            forward_gap = target_center - foot_x[:, foot_index]
            active = (
                (self.stair_mode < 0)
                & (self.stair_gate_strength >= 0.60)
                & valid[:, foot_index]
                & (targets[:, foot_index, 4] >= 0.70)
                & ~contacts[:, foot_index]
                & contacts[:, 1 - foot_index]
                & (height_above >= 0.01)
                & (height_above <= 0.28)
                & (forward_gap >= -0.06)
                & (forward_gap <= 0.18)
            )
            body_index = self.feet_body_ids[foot_index] - int(self.robot.is_fixed_base)
            jacobian_joint_ids = [joint_id + (0 if self.robot.is_fixed_base else 6) for joint_id in joint_ids]
            jacobian = jacobians[:, body_index, :3, :][:, :, jacobian_joint_ids]
            forward_jacobian = (
                jacobian[:, 0] * heading[:, 0:1]
                + jacobian[:, 1] * heading[:, 1:2]
            )
            commanded_delta = joint_targets[:, joint_ids] - self.robot.data.joint_pos[:, joint_ids]
            max_ankle_x = target_center - 0.029 + 0.10 * targets[:, foot_index, 3]
            joint_step = bounded_overshoot_joint_step(
                forward_jacobian,
                commanded_delta,
                foot_x[:, foot_index],
                max_ankle_x,
                max_joint_step=geometry_cfg.foothold_overshoot_guard_max_joint_step,
            )
            correcting = active & (joint_step.abs().amax(dim=1) > 1.0e-5)
            self.foothold_guard_active[:, foot_index] = correcting
            joint_targets[:, joint_ids] += joint_step * correcting.unsqueeze(1)
        return joint_targets

    def _update_geometry_stability(self, update_mask: torch.Tensor) -> None:
        """Count consecutive fresh frames whose stair edges agree."""

        if self.terrain_geometry is None:
            return
        treads = self.terrain_geometry.treads
        current_edges = treads[..., :2]
        current_valid = treads[..., 5] > 0.5
        overlap = current_valid & self.previous_geometry_valid
        edge_delta = torch.where(
            overlap.unsqueeze(-1),
            torch.abs(current_edges - self.previous_geometry_edges),
            torch.zeros_like(current_edges),
        )
        max_delta = edge_delta.flatten(1).amax(dim=1)
        stable = (
            (overlap.sum(dim=1) >= self.cfg.scene.depth_camera.geometry.gate_min_valid_treads)
            & (max_delta <= self.cfg.scene.depth_camera.geometry.settle_edge_tolerance)
        )
        next_stable_count = torch.where(
            stable,
            self.geometry_stable_count + 1,
            torch.zeros_like(self.geometry_stable_count),
        )
        self.geometry_stable_count = torch.where(update_mask, next_stable_count, self.geometry_stable_count)
        self.previous_geometry_edges[update_mask] = current_edges[update_mask]
        self.previous_geometry_valid[update_mask] = current_valid[update_mask]

    def _restore_settle_commands(self) -> None:
        if self.stair_settle_enabled and torch.any(self.stair_settle_active):
            self.command_generator.command[self.stair_settle_active] = self.nominal_command[
                self.stair_settle_active
            ]

    def _update_stair_settle(self) -> None:
        """Pause only after support moves down one stair, not at every footfall."""

        if not self.stair_settle_enabled:
            return
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        foot_contact = self.avg_feet_force_per_step > geometry_cfg.settle_contact_force
        touchdown = torch.any(foot_contact & ~self.previous_foot_contact, dim=1)
        self.previous_foot_contact.copy_(foot_contact)
        sole_z = self.robot.data.body_pos_w[:, self.feet_body_ids, 2] - 0.04
        supported_z = torch.where(foot_contact, sole_z, torch.inf).amin(dim=1)
        stepped_down = (
            touchdown
            & torch.isfinite(self.previous_supported_height)
            & (supported_z < self.previous_supported_height - geometry_cfg.settle_step_drop)
        )
        self.previous_supported_height = torch.where(
            torch.isfinite(supported_z), supported_z, self.previous_supported_height
        )

        start = stepped_down & (self.stair_mode < 0) & ~self.stair_settle_active
        self.stair_settle_active |= start
        self.stair_settle_elapsed[start] = 0.0
        self.geometry_stable_count[start] = 0

        self.stair_settle_elapsed += self.stair_settle_active.to(torch.float) * self.step_dt
        stable_release = (
            (self.stair_settle_elapsed >= geometry_cfg.settle_min_time_s)
            & (self.geometry_stable_count >= geometry_cfg.settle_stable_frames)
        )
        timeout_release = self.stair_settle_elapsed >= geometry_cfg.settle_max_time_s
        release = self.stair_settle_active & (stable_release | timeout_release | (self.stair_mode == 0))
        self.stair_settle_active[release] = False
        self.stair_settle_elapsed[release] = 0.0

        self.command_generator.command[self.stair_settle_active] = 0.0

    def compute_observations(self, update_geometry: bool = True):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            if self.cfg.scene.height_scanner.use_for_critic:
                critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan = height_scan + (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            if self.cfg.scene.height_scanner.use_for_actor:
                actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        if self.stair_step_enabled:
            if update_geometry and not self.step_skill_pretrain:
                self._update_terrain_geometry()
            # The first 960 proprioceptive entries and final gate retain the
            # frozen blind actor contract. Only the new supervisor drives fusion.
            actor_obs = torch.cat((actor_obs, self.step_features, self.step_gate), dim=-1)
            critic_obs = torch.cat((critic_obs, self.step_critic_features, self.step_gate), dim=-1)
            return actor_obs.clamp(-self.clip_obs, self.clip_obs), critic_obs.clamp(-self.clip_obs, self.clip_obs)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]
            if self.cfg.scene.depth_camera.geometry.enabled:
                if update_geometry:
                    self._update_terrain_geometry()
                geometry_features = self._motion_compensated_geometry_history().reshape(self.num_envs, -1)
                geometry_cfg = self.cfg.scene.depth_camera.geometry
                if geometry_cfg.gate_enabled:
                    gate = self.stair_gate_strength.to(geometry_features.dtype).unsqueeze(1)
                    actor_geometry_features = geometry_features * gate
                else:
                    actor_geometry_features = geometry_features
                if self.cfg.scene.depth_camera.geometry.use_for_actor:
                    actor_obs = torch.cat([actor_obs, actor_geometry_features], dim=-1)
                if self.cfg.scene.depth_camera.geometry.use_for_critic:
                    critic_obs = torch.cat([critic_obs, geometry_features], dim=-1)
                if geometry_cfg.append_course_alignment:
                    alignment_features = self._course_alignment_features()
                    if geometry_cfg.use_for_actor:
                        actor_obs = torch.cat([actor_obs, alignment_features], dim=-1)
                    if geometry_cfg.use_for_critic:
                        critic_obs = torch.cat([critic_obs, alignment_features], dim=-1)
                if geometry_cfg.append_foothold_targets:
                    foothold_features = self._foothold_target_features()
                    if geometry_cfg.gate_enabled:
                        foothold_actor_features = foothold_features * gate * geometry_cfg.foothold_actor_scale
                    else:
                        foothold_actor_features = foothold_features * geometry_cfg.foothold_actor_scale
                    if geometry_cfg.use_for_actor:
                        actor_obs = torch.cat([actor_obs, foothold_actor_features], dim=-1)
                    if geometry_cfg.use_for_critic:
                        critic_obs = torch.cat([critic_obs, foothold_features], dim=-1)
                if geometry_cfg.append_stair_mode:
                    stair_mode = (
                        self.stair_mode.to(actor_obs.dtype)
                        * self.stair_gate_strength.to(actor_obs.dtype)
                    ).unsqueeze(1)
                    if (
                        geometry_cfg.alignment_safety_gate_enabled
                        and geometry_cfg.append_course_alignment
                    ):
                        # Leave the blind actor untouched inside the lane.
                        # If it has already drifted on a non-stair segment,
                        # permit only a bounded residual to recover the route.
                        lateral_excess = (
                            alignment_features[:, 0].abs()
                            - geometry_cfg.alignment_safety_lateral_threshold
                        ) / max(1.0 - geometry_cfg.alignment_safety_lateral_threshold, 1.0e-6)
                        heading_excess = (
                            alignment_features[:, 1].abs()
                            - geometry_cfg.alignment_safety_heading_threshold
                        ) / max(1.0 - geometry_cfg.alignment_safety_heading_threshold, 1.0e-6)
                        safety_gate = torch.maximum(lateral_excess, heading_excess).clamp(0.0, 1.0)
                        stair_mode = torch.where(
                            stair_mode.abs() > 1.0e-4,
                            stair_mode,
                            safety_gate.unsqueeze(1),
                        )
                    if geometry_cfg.use_for_actor:
                        actor_stair_mode = stair_mode
                        if geometry_cfg.blind_during_stair_settle:
                            actor_stair_mode = torch.where(
                                self.stair_settle_active.unsqueeze(1),
                                torch.zeros_like(stair_mode),
                                stair_mode,
                            )
                        actor_obs = torch.cat([actor_obs, actor_stair_mode], dim=-1)
                    if geometry_cfg.use_for_critic:
                        critic_obs = torch.cat([critic_obs, stair_mode], dim=-1)
            else:
                depth_features = self.process_depth_observations(depth_image)
                actor_obs = torch.cat([actor_obs, depth_features], dim=-1)
                critic_obs = torch.cat([critic_obs, depth_features], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                if self.course_width_curriculum_enabled:
                    terrain_levels = self.update_course_width_curriculum(env_ids)
                else:
                    terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        if self.stair_step_enabled:
            self.extras["log"]["Step/completed"] = self.step_success_event[env_ids].float().mean()
            self.extras["log"]["Step/failed"] = self.step_failure[env_ids].float().mean()
            for env_id in env_ids.tolist():
                self.step_controllers[env_id].reset()
            self.step_features[env_ids] = 0
            self.step_features[env_ids, int(StepPhase.BLIND)] = 1
            self.step_critic_features[env_ids] = self.step_features[env_ids]
            self.step_gate[env_ids] = 0
            if self.step_skill_pretrain:
                self.step_skill_teacher.reset(env_ids)
                self.step_features[env_ids] = 0
                self.step_features[env_ids, int(StepPhase.OBSERVE)] = 1
                self.step_critic_features[env_ids] = self.step_features[env_ids]
                self.step_gate[env_ids] = self.step_skill_teacher.direction
                self.command_generator.command[env_ids] = 0
        if hasattr(self, "safe_tread_support_count"):
            self.safe_tread_support_count[env_ids] = 0
        if self.stair_settle_enabled:
            self.nominal_command[env_ids] = self.command_generator.command[env_ids]
            self.stair_settle_active[env_ids] = False
            self.stair_settle_elapsed[env_ids] = 0.0
            self.previous_supported_height[env_ids] = torch.nan
            self.previous_foot_contact[env_ids] = False
            self.geometry_stable_count[env_ids] = 0
            self.previous_geometry_edges[env_ids] = 0.0
            self.previous_geometry_valid[env_ids] = False
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        if hasattr(self, "stair_progress_level"):
            self.stair_progress_level[env_ids] = 0
            self.stair_progress_increment[env_ids] = 0.0
            self.stair_verified_progress_level[env_ids] = 0
            self.stair_verified_contact_count[env_ids] = 0
            self.stair_physical_support_count[env_ids] = 0
            self.stair_physical_covered[env_ids] = False
            self.stair_broad_support_count[env_ids] = 0
            self.stair_broad_covered[env_ids] = False
            self.stair_physical_step_increment[env_ids] = 0.0
            self.terminal_support_count[env_ids] = 0
        if hasattr(self, "foothold_lock_valid"):
            self.foothold_lock_valid[env_ids] = False
            self.foothold_lock_mode[env_ids] = 0
            if self.cfg.scene.depth_camera.geometry.foothold_swing_trajectory_enabled:
                self.foothold_swing_elapsed[env_ids] = 0.0
                self.foothold_swing_start_w[env_ids] = 0.0
                self.foothold_lock_control_valid[env_ids] = False
        if self.cfg.scene.depth_camera.enable_depth_camera:
            if self.cfg.scene.depth_camera.geometry.enabled:
                self.geometry_obs_buffer.reset(env_ids)
                self.terrain_geometry_extractor.reset_surface_memory(env_ids)
                if self.cfg.scene.depth_camera.geometry.motion_compensate_history:
                    self.geometry_position_buffer.reset(env_ids)
                    self.geometry_heading_buffer.reset(env_ids)
                    self.landing_geometry_buffer.reset(env_ids)
                    self.landing_position_buffer.reset(env_ids)
                    self.landing_heading_buffer.reset(env_ids)
                    self.landing_reference_height_buffer.reset(env_ids)
                if self.cfg.scene.depth_camera.geometry.gate_enabled:
                    self.stair_mode_gate.reset(env_ids)
                    self.stair_gate_strength[env_ids] = 0.0
            else:
                self.depth_obs_buffer.reset(env_ids)
                self.last_depth_features[env_ids] = 0.0
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos
        if (
            self.cfg.scene.depth_camera.enable_depth_camera
            and self.cfg.scene.depth_camera.geometry.enabled
            and self.cfg.scene.depth_camera.geometry.foothold_ik_enabled
        ):
            processed_actions = self._apply_foothold_ik(processed_actions)
        if (
            self.cfg.scene.depth_camera.enable_depth_camera
            and self.cfg.scene.depth_camera.geometry.enabled
            and self.cfg.scene.depth_camera.geometry.foothold_overshoot_guard_enabled
        ):
            processed_actions = self._apply_foothold_overshoot_guard(processed_actions)

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        # RTX sensors still require an explicit render in headless training;
        # otherwise the camera keeps returning its initialization frame.
        if not self.headless or self.cfg.scene.depth_camera.enable_depth_camera:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        self._restore_settle_commands()
        self.command_generator.compute(self.step_dt)
        if self.stair_settle_enabled:
            self.nominal_command.copy_(self.command_generator.command)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        if (
            self.cfg.scene.depth_camera.enable_depth_camera
            and self.cfg.scene.depth_camera.geometry.enabled
        ):
            self._update_terrain_geometry()
            self._update_stair_settle()
            self._apply_stair_heading_feedback()
            if self.stair_step_enabled:
                self._update_stair_step()
        if self.step_skill_pretrain:
            self._update_stair_skill()

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        if hasattr(self, "last_goal_verified_level"):
            self.last_goal_verified_level = torch.where(
                self.last_goal_distance_reached,
                self.stair_verified_progress_level,
                torch.zeros_like(self.stair_verified_progress_level),
            )
            self.last_goal_verified_contacts = torch.where(
                self.last_goal_distance_reached,
                self.stair_verified_contact_count,
                torch.zeros_like(self.stair_verified_contact_count),
            )
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations(update_geometry=False)
        self.extras["observations"] = {"critic": critic_obs}
        if (
            self.cfg.scene.depth_camera.enable_depth_camera
            and self.cfg.scene.depth_camera.geometry.enabled
            and self.cfg.scene.depth_camera.geometry.gate_enabled
            and not self.stair_step_enabled
        ):
            self.extras.setdefault("log", {})["Perception/stair_mode_active"] = (self.stair_mode != 0).float().mean()
            self.extras["log"]["Perception/stair_gate_strength"] = self.stair_gate_strength.mean()
            self.extras["log"]["Perception/stairs_up"] = (self.stair_mode > 0).float().mean()
            self.extras["log"]["Perception/stairs_down"] = (self.stair_mode < 0).float().mean()
            self.extras["log"]["Perception/stair_confidence"] = self.terrain_geometry.stair_confidence.mean()
            if self.stair_settle_enabled:
                self.extras["log"]["Perception/stair_settle_active"] = self.stair_settle_active.float().mean()
            if self.cfg.scene.depth_camera.geometry.foothold_ik_enabled:
                self.extras["log"]["Perception/foothold_ik_active"] = (
                    self.foothold_ik_active.any(dim=1).float().mean()
                )

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        termination_force = torch.linalg.norm(
            net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1
        ).amax(dim=1)
        reset_buf = (termination_force > 500.0).any(dim=1)
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        if self.stair_step_enabled:
            lane_failure = (self.robot.data.root_pos_w[:, 1]-self.scene.env_origins[:, 1]).abs() > 0.80
            # A completed supervisor sequence is not allowed to override an
            # independent simultaneous physical termination.
            self.step_success_event &= ~(reset_buf | time_out_buf | lane_failure)
            self.step_reward_metrics["success"] = self.step_success_event.float()/self.step_dt
            self.step_failure |= reset_buf | time_out_buf | lane_failure
            return reset_buf | time_out_buf | lane_failure | self.step_failure | self.step_success_event, time_out_buf
        geometry_cfg = getattr(getattr(self.cfg.scene, "depth_camera", None), "geometry", None)
        if geometry_cfg is not None and geometry_cfg.bootstrap_stairs:
            relative_root = self.robot.data.root_pos_w - self.scene.env_origins
            reached_stair_end = relative_root[:, 0] >= geometry_cfg.bootstrap_goal_distance
            left_stair_lane = relative_root[:, 1].abs() >= geometry_cfg.bootstrap_lateral_limit
            contact_failure = reset_buf.clone()
            self.stair_hip_clearance.copy_(self._hip_clearance_above_support())
            if geometry_cfg.bootstrap_stair_row in (1, 2):
                self._update_supported_stair_progress(relative_root[:, 0])
                self._update_physical_stair_coverage()
                self.last_raw_stair_progress_increment.copy_(self.stair_progress_increment)
                self.last_stair_physical_step_increment.copy_(self.stair_physical_step_increment)
            descending = geometry_cfg.bootstrap_stair_row == 2
            stair_name = "stairs_down" if descending else "stairs_up"
            platform_height = terminal_platform_height(
                self.scene.env_origins[:, 2],
                self.stair_step_height,
                self.cfg.scene.terrain_generator.sub_terrains[stair_name].num_steps,
                descending,
            )
            planted = terminal_platform_support(
                self.robot.data.body_pos_w[:, self.feet_body_ids, 2] - 0.04,
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :],
                torch.linalg.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :2], dim=-1),
                platform_height,
            )
            near_end = relative_root[:, 0] >= geometry_cfg.bootstrap_goal_distance - 0.40
            self.terminal_support_count = torch.where(
                near_end & planted,
                (self.terminal_support_count + 1).clamp_max(3),
                torch.zeros_like(self.terminal_support_count),
            )
            self.last_goal_distance_reached = (
                reached_stair_end & ~reset_buf & ~time_out_buf & ~left_stair_lane
            )
            self.last_goal_stable = (
                self.last_goal_distance_reached
                & (self.terminal_support_count >= 3)
                & (self.stair_hip_clearance >= 0.45)
            )
            self.last_goal_reached = self.last_goal_stable & self.stair_physical_covered.all(dim=1)
            self.last_distance_goal_coverage = torch.where(
                self.last_goal_distance_reached,
                self.stair_physical_covered.sum(dim=1),
                torch.zeros_like(self.stair_progress_level),
            )
            self.last_distance_goal_broad_coverage = torch.where(
                self.last_goal_distance_reached,
                self.stair_broad_covered.sum(dim=1),
                torch.zeros_like(self.stair_progress_level),
            )
            self.last_distance_goal_physical_levels = (
                self.stair_physical_covered & self.last_goal_distance_reached[:, None]
            )
            self.last_distance_goal_broad_levels = (
                self.stair_broad_covered & self.last_goal_distance_reached[:, None]
            )
            reset_buf |= reached_stair_end | left_stair_lane
            lane_failure = left_stair_lane & ~contact_failure
            self.last_stair_failure_contact = contact_failure
            self.last_stair_failure_body_index = torch.where(
                contact_failure,
                termination_force.argmax(dim=1),
                torch.full_like(self.last_stair_failure_body_index, -1),
            )
            self.last_stair_failure_lane = lane_failure
            self.last_stair_failure_unsafe_end = (
                self.last_goal_distance_reached & ~self.last_goal_reached
            )
            self.last_stair_failure_timeout = (
                time_out_buf & ~contact_failure & ~lane_failure & ~self.last_goal_reached
            )
            self.last_stair_failure_level.copy_(self.stair_progress_level)
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def _update_stair_skill(self):
        data = self.robot.data
        rotation = _matrix_from_quat(data.body_quat_w[:, self.feet_body_ids])
        offset = torch.tensor([.03, 0., -.04], device=self.device)
        offset = (rotation @ offset).squeeze(-1)
        soles = data.body_pos_w[:, self.feet_body_ids]+offset
        velocity = data.body_link_lin_vel_w[:, self.feet_body_ids]+torch.cross(
            data.body_ang_vel_w[:, self.feet_body_ids], offset, dim=-1)
        corners = torch.tensor([[-.12, -.042, 0.], [.12, -.042, 0.],
                                [.12, .042, 0.], [-.12, .042, 0.]], device=self.device)
        corners = corners @ rotation.transpose(-1, -2)+soles[:, :, None]
        teacher = self.step_skill_teacher
        features, critic, metrics = teacher.update(
            data.root_pos_w, _yaw_matrix(data.root_quat_w), soles, velocity, corners, rotation[..., :, 2],
            self.contact_sensor.data.net_forces_w[:, self.step_contact_ids], data.root_lin_vel_w,
            data.root_ang_vel_w, self.step_dt,
            hip_offsets=data.body_pos_w[:, self.hip_body_ids]-data.root_pos_w[:, None],
            ankle_offsets=-offset, com_offset=self._step_com_offset())
        self.step_features.copy_(features)
        self.step_critic_features.copy_(critic)
        delta = self.step_dt/teacher.cfg.action_transition_s
        self.step_gate += (teacher.direction-self.step_gate).clamp(-delta, delta)
        self.step_success_event.copy_(teacher.success)
        self.step_failure.copy_(teacher.failure)
        for name, value in metrics.items():
            self.step_reward_metrics[name].copy_(value)
        self.command_generator.command[:] = 0
        self.extras.setdefault("log", {}).update({
            "Step/phase": teacher.phase.float().mean(),
            "Step/motor_teacher": 1.,
            "Step/lift_active": ((teacher.phase == int(StepPhase.LIFT_LEAD)) |
                                 (teacher.phase == int(StepPhase.LIFT_TRAIL))).float().mean(),
            "Step/clearance_confirmed": teacher.clearance.float().mean(),
            "Step/shift_active": ((teacher.phase == int(StepPhase.SHIFT_LEAD)) |
                                  (teacher.phase == int(StepPhase.SHIFT_TRAIL))).float().mean(),
            "Step/root_speed": data.root_lin_vel_w.norm(dim=-1).mean(),
            "Step/sole_speed": velocity.norm(dim=-1).mean(),
            "Step/lift_reference_limit": teacher.lift_limit.mean(),
            "Step/lift_reference_limit_max": teacher.lift_limit.max(),
        })

    def _step_com_offset(self):
        return ((self.step_body_masses[..., None]*self.robot.data.body_com_pos_w).sum(dim=1)
                /self.step_body_masses.sum(dim=1, keepdim=True)-self.robot.data.root_pos_w)

    def stair_action_guidance(self, action_mean):
        """Simulation-only action-improvement labels; never writes joint targets."""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        phase = self.step_features[:, :len(StepPhase)].argmax(dim=1)
        valid = (phase >= int(StepPhase.SHIFT_LEAD)) & (phase <= int(StepPhase.SETTLE))
        reference = action_mean.detach().clone()
        if not valid.any():
            return reference, valid
        if not hasattr(self, "_step_ik_teachers"):
            cfg = DifferentialIKControllerCfg(command_type="pose", ik_method="dls", ik_params={"lambda_val": .08})
            self._step_ik_teachers = [DifferentialIKController(cfg, self.num_envs, self.device) for _ in range(2)]
        data = self.robot.data
        if self.step_skill_pretrain:
            feet_reference = self.step_skill_teacher.reference_feet
            root_reference = self.step_skill_teacher.reference_root
        else:
            feet_reference = torch.as_tensor(np.stack([c.reference_feet for c in self.step_controllers]),
                                              device=self.device, dtype=data.root_pos_w.dtype)
            root_reference = torch.as_tensor(np.stack([c.reference_root for c in self.step_controllers]),
                                              device=self.device, dtype=data.root_pos_w.dtype)
        root_rotation = _matrix_from_quat(data.root_quat_w)
        yaw_rotation = _yaw_matrix(data.root_quat_w)
        desired_yaw = torch.atan2(yaw_rotation[:, 1, 0], yaw_rotation[:, 0, 0])+self.step_features[:, 29]
        desired_root_quat = quat_from_euler_xyz(torch.zeros_like(desired_yaw), torch.zeros_like(desired_yaw), desired_yaw)
        flat_quat = torch.zeros(self.num_envs, 4, device=self.device)
        flat_quat[:, 0] = 1.
        ankle_offset = quat_apply(desired_root_quat, torch.tensor([-.03, 0., .04], device=self.device).expand(self.num_envs, 3))
        jacobians = self.robot.root_physx_view.get_jacobians()
        sole_offsets = quat_apply(data.body_quat_w[:, self.feet_body_ids],
                                 torch.tensor([.03, 0., -.04], device=self.device).expand(self.num_envs, 2, 3))
        soles = data.body_pos_w[:, self.feet_body_ids]+sole_offsets
        com = data.root_pos_w+self._step_com_offset()
        span = soles[:, 0, :2]-soles[:, 1, :2]
        fraction = ((com[:, :2]-soles[:, 1, :2])*span).sum(dim=1)/span.square().sum(dim=1).clamp_min(1.e-4)
        fraction = fraction.clamp(0., 1.)
        loads = torch.stack((fraction, 1-fraction), dim=1)
        lead = self.step_features[:, 32:34].argmax(dim=1)
        rows = torch.arange(self.num_envs, device=self.device)
        for lift_phase, lower_phase, swing in ((StepPhase.LIFT_LEAD, StepPhase.LOWER_LEAD, lead),
                                               (StepPhase.LIFT_TRAIL, StepPhase.LOWER_TRAIL, 1-lead)):
            flying, lowering = phase == int(lift_phase), phase == int(lower_phase)
            loads[rows[flying], swing[flying]] = 0.
            loads[rows[flying], 1-swing[flying]] = 1.
            loads[rows[lowering], swing[lowering]] = .15
            loads[rows[lowering], 1-swing[lowering]] = .85
        cop = (loads[..., None]*soles).sum(dim=1)
        cop_offset = torch.zeros_like(com)
        cop_offset[:, 0] = (com[:, 0]-cop[:, 0]).clamp(-.09, .09)
        cop_offset[:, 1] = (com[:, 1]-cop[:, 1]).clamp(-.03, .03)
        forces = torch.zeros_like(soles)
        forces[..., 2] = loads*self.step_body_masses.sum(dim=1, keepdim=True)*9.81
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
        if not self.robot.is_fixed_base:
            gravity = gravity[:, 6:]
        for foot, joint_ids in enumerate((self.left_leg_ids, self.right_leg_ids)):
            body_id = self.feet_body_ids[foot]
            row = body_id-1 if self.robot.is_fixed_base else body_id
            columns = [joint_id+(0 if self.robot.is_fixed_base else 6) for joint_id in joint_ids]
            world_jacobian = jacobians[:, row][:, :, columns]
            jacobian = world_jacobian.clone()
            inverse_rotation = root_rotation.transpose(1, 2)
            jacobian[:, :3] = inverse_rotation @ jacobian[:, :3]
            jacobian[:, 3:] = inverse_rotation @ jacobian[:, 3:]
            actual_position = quat_apply_inverse(data.root_quat_w, data.body_pos_w[:, body_id]-data.root_pos_w)
            actual_rotation = quat_mul(quat_conjugate(data.root_quat_w), data.body_quat_w[:, body_id])
            desired_position = quat_apply_inverse(desired_root_quat, feet_reference[:, foot]+ankle_offset-root_reference)
            controller = self._step_ik_teachers[foot]
            controller.set_command(torch.cat((desired_position, flat_quat), dim=1))
            joints = data.joint_pos[:, joint_ids]
            target = controller.compute(actual_position, actual_rotation, jacobian, joints)
            limits = data.soft_joint_pos_limits[:, joint_ids]
            delta = (target.clamp(limits[..., 0], limits[..., 1])-joints).clamp(-.04, .04)
            scale = self.action_scale[joint_ids] if isinstance(self.action_scale, torch.Tensor) else self.action_scale
            contact_offset = sole_offsets[:, foot]+cop_offset
            point_jacobian = world_jacobian[:, :3]+torch.cross(
                world_jacobian[:, 3:].transpose(1, 2), contact_offset[:, None].expand(-1, len(joint_ids), -1),
                dim=-1).transpose(1, 2)
            support_torque = (point_jacobian.transpose(1, 2) @ forces[:, foot, :, None]).squeeze(-1)
            static_torque = gravity[:, joint_ids]-support_torque
            effort = data.joint_effort_limits[:, joint_ids]
            static_torque = static_torque.clamp(-effort, effort)
            command = joints+delta+static_torque/data.joint_stiffness[:, joint_ids].clamp_min(1.)
            normalized = (command-data.default_joint_pos[:, joint_ids])/scale
            reference[:, joint_ids] = action_mean[:, joint_ids]+(normalized-action_mean[:, joint_ids]).clamp(-.5, .5)
        reference[~valid] = action_mean[~valid]
        if not torch.isfinite(reference).all():
            raise RuntimeError("Nonfinite stair action-guidance label.")
        return reference, valid

    def _update_stair_step(self):
        data = self.robot.data
        feet_quat = data.body_quat_w[:, self.feet_body_ids]
        offsets = quat_apply(feet_quat, torch.tensor([0.03, 0., -0.04], device=self.device).expand(self.num_envs, 2, 3))
        sole_pos = data.body_pos_w[:, self.feet_body_ids] + offsets
        sole_vel = data.body_link_lin_vel_w[:, self.feet_body_ids] + torch.cross(
            data.body_ang_vel_w[:, self.feet_body_ids], offsets, dim=-1)
        arrays = [value.detach().cpu().numpy() for value in (
            data.root_pos_w, _yaw_matrix(data.root_quat_w), sole_pos, _matrix_from_quat(feet_quat),
            sole_vel, self.contact_sensor.data.net_forces_w[:, self.step_contact_ids],
            data.root_lin_vel_w, data.root_ang_vel_w, self.depth_camera.current_timestamp,
            self.depth_camera.frame_timestamp)]
        features, critic_features, metrics = [], [], {name: [] for name in self.step_reward_metrics}
        for env_id, controller in enumerate(self.step_controllers):
            memory = self.terrain_geometry_extractor.surface_memories[env_id]
            measurement = StepMeasurement(float(arrays[8][env_id]), memory.generation,
                *(array[env_id] for array in arrays[:8]), float(self.step_body_weight[env_id]),
                float(arrays[9][env_id]),
                (data.body_pos_w[env_id, self.hip_body_ids]-data.root_pos_w[env_id]).detach().cpu().numpy(),
                self._step_com_offset()[env_id].detach().cpu().numpy())
            geometry = self.terrain_geometry.surface_memory[env_id]
            controller.update(geometry, measurement, self.step_dt)
            features.append(controller.features(measurement))
            critic_features.append(controller.features(measurement, privileged=True))
            for name, value in controller.reward_metrics(measurement, self.step_dt).items():
                metrics[name].append(value)
            gate_target = controller.action_gate_target(measurement)
            gate_delta = self.step_dt/controller.cfg.action_transition_s
            self.step_gate[env_id] += torch.clamp(
                gate_target-self.step_gate[env_id], -gate_delta, gate_delta)
            self.step_success_event[env_id] = controller.success_event
            self.step_failure[env_id] = controller.phase == StepPhase.RECOVER
            self.step_last_phases[env_id] = controller.phase
            self.step_last_failures[env_id] = controller.failure_reason
            if controller.active:
                self.command_generator.command[env_id] = torch.as_tensor(
                    controller.walking_command(measurement), device=self.device, dtype=self.command_generator.command.dtype)
        self.step_features.copy_(torch.as_tensor(np.stack(features), device=self.device))
        self.step_critic_features.copy_(torch.as_tensor(np.stack(critic_features), device=self.device))
        for name, values in metrics.items():
            self.step_reward_metrics[name].copy_(torch.tensor(values, device=self.device))
        self.extras.setdefault("log", {}).update({
            "Step/active": self.step_gate.abs().mean(),
            "Step/phase": sum(int(c.phase) for c in self.step_controllers)/self.num_envs,
            "Step/target_locked": sum(c.lock is not None for c in self.step_controllers)/self.num_envs,
            "Step/alignment_command_active": (self.command_generator.command[:, 2].abs() > 0).float().mean(),
        })

    def _hip_clearance_above_support(self) -> torch.Tensor:
        sole_z = self.robot.data.body_pos_w[:, self.feet_body_ids, 2] - 0.04
        contacts = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2] > 5.0
        high_support_z = torch.where(contacts, sole_z, -torch.inf).amax(dim=1)
        high_support_z = torch.where(
            torch.isfinite(high_support_z), high_support_z, sole_z.amax(dim=1)
        )
        hip_z = self.robot.data.body_pos_w[:, self.hip_body_ids, 2].amin(dim=1)
        return hip_z - high_support_z

    def _update_supported_stair_progress(self, forward_distance: torch.Tensor) -> None:
        terrain_cfg = self.cfg.scene.terrain_generator
        descending = self.cfg.scene.depth_camera.geometry.bootstrap_stair_row == 2
        stair_cfg = terrain_cfg.sub_terrains["stairs_down" if descending else "stairs_up"]
        terrain_columns = self.scene.terrain.terrain_types.float()
        lower, upper = terrain_cfg.difficulty_range
        difficulty = lower + (upper - lower) * terrain_columns / max(terrain_cfg.num_cols - 1, 1)
        step_height = stair_cfg.step_height_range[0] + difficulty * (
            stair_cfg.step_height_range[1] - stair_cfg.step_height_range[0]
        )
        self.stair_step_height.copy_(step_height)
        if stair_cfg.step_width_range is None:
            step_width = torch.full_like(step_height, stair_cfg.step_width)
        else:
            step_width = stair_cfg.step_width_range[0] + difficulty * (
                stair_cfg.step_width_range[1] - stair_cfg.step_width_range[0]
            )
        contacts = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2] > 5.0
        sole_height = self.robot.data.body_pos_w[:, self.feet_body_ids, 2] - 0.04
        if descending:
            supported_height = torch.where(contacts, sole_height, torch.inf).amin(dim=1)
        else:
            supported_height = torch.where(contacts, sole_height, -torch.inf).amax(dim=1)
        first_riser_distance = (
            stair_cfg.approach_length
            - 0.5 * terrain_cfg.size[0]
            + self.cfg.scene.depth_camera.geometry.bootstrap_x_offset
        )
        if descending:
            first_riser_distance += step_width
        new_level, increment = supported_stair_progress(
            forward_distance,
            supported_height,
            self.scene.env_origins[:, 2],
            step_width,
            step_height,
            self.stair_progress_level,
            first_riser_distance,
            stair_cfg.num_steps,
            direction=-1 if descending else 1,
        )
        self.stair_progress_level = new_level
        self.stair_progress_increment = increment.float()
        self.stair_step_width.copy_(step_width)
        self.stair_first_riser_distance[:] = first_riser_distance

    def _update_physical_stair_coverage(self) -> None:
        """Evaluate stable support on each known simulation stair tread."""

        foot_pos = self.robot.data.body_pos_w[:, self.feet_body_ids]
        foot_quat = self.robot.data.body_quat_w[:, self.feet_body_ids]
        foot_forward = quat_apply(
            foot_quat.reshape(-1, 4),
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(self.num_envs * 2, -1),
        ).reshape(self.num_envs, 2, 3)
        sole_center_x = (
            foot_pos[..., 0] + 0.029 * foot_forward[..., 0]
            - self.scene.env_origins[:, None, 0]
        )
        foot_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids]
        foot_speed = torch.linalg.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :2], dim=-1)
        counts, covered, increment = stair_support_coverage_transition(
            self.stair_physical_support_count,
            self.stair_physical_covered,
            sole_center_x,
            foot_pos[..., 2] - 0.04,
            foot_force,
            foot_speed,
            self.stair_first_riser_distance,
            self.stair_step_width,
            self.scene.env_origins[:, 2],
            self.stair_step_height,
            descending=self.cfg.scene.depth_camera.geometry.bootstrap_stair_row == 2,
        )
        self.stair_physical_support_count = counts
        self.stair_physical_covered = covered
        self.stair_physical_step_increment = increment.float()
        broad_counts, broad_covered, _ = stair_support_coverage_transition(
            self.stair_broad_support_count,
            self.stair_broad_covered,
            sole_center_x,
            foot_pos[..., 2] - 0.04,
            foot_force,
            foot_speed,
            self.stair_first_riser_distance,
            self.stair_step_width,
            self.scene.env_origins[:, 2],
            self.stair_step_height,
            descending=self.cfg.scene.depth_camera.geometry.bootstrap_stair_row == 2,
            center_tolerance_fraction=0.45,
            max_center_tolerance=0.12,
        )
        self.stair_broad_support_count = broad_counts
        self.stair_broad_covered = broad_covered

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            # noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel         #3
            # noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel        #3   
            # noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity     #3
            # noise_vec[9:12] = 0      #commands no noise        #3   
            # noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos      #29
            # noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = ( #29
            #     noise_scales.joint_vel * self.obs_scales.joint_vel
            # )
            # noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0  #actions no noise   #29
            # noise_vec[12 + self.num_actions * 3 : 18 + self.num_actions * 3] = 0.0  #gait phase sin no noise   #6
            
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel        #3   
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity     #3
            noise_vec[6:9] = 0      #commands no noise        #3   
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos      #29
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = ( #29
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0  #actions no noise   #29
            # noise_vec[9 + self.num_actions * 3 : 15 + self.num_actions * 3] = 0.0  #gait phase sin no noise   #6
            
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.stair_settle_enabled = False
        if self.cfg.scene.depth_camera.enable_depth_camera:
            if self.cfg.scene.depth_camera.geometry.enabled:
                geometry_cfg = self.cfg.scene.depth_camera.geometry
                self.terrain_geometry_extractor = StairGeometryExtractor(geometry_cfg)
                self.geometry_obs_buffer = _MaskedHistoryBuffer(
                    max_len=geometry_cfg.history_length,
                    batch_size=self.num_envs,
                    feature_shape=(self.terrain_geometry_extractor.num_features,),
                    device=self.device,
                )
                self._last_geometry_camera_frame = None
                if geometry_cfg.motion_compensate_history:
                    self.geometry_position_buffer = _MaskedHistoryBuffer(
                        max_len=geometry_cfg.history_length,
                        batch_size=self.num_envs,
                        feature_shape=(2,),
                        device=self.device,
                    )
                    self.geometry_heading_buffer = _MaskedHistoryBuffer(
                        max_len=geometry_cfg.history_length,
                        batch_size=self.num_envs,
                        feature_shape=(2,),
                        device=self.device,
                    )
                    landing_length = geometry_cfg.landing_memory_length
                    self.landing_geometry_buffer = _MaskedHistoryBuffer(
                        max_len=landing_length,
                        batch_size=self.num_envs,
                        feature_shape=(self.terrain_geometry_extractor.num_features,),
                        device=self.device,
                    )
                    self.landing_position_buffer = _MaskedHistoryBuffer(
                        max_len=landing_length,
                        batch_size=self.num_envs,
                        feature_shape=(2,),
                        device=self.device,
                    )
                    self.landing_heading_buffer = _MaskedHistoryBuffer(
                        max_len=landing_length,
                        batch_size=self.num_envs,
                        feature_shape=(2,),
                        device=self.device,
                    )
                    self.landing_reference_height_buffer = _MaskedHistoryBuffer(
                        max_len=landing_length,
                        batch_size=self.num_envs,
                        feature_shape=(1,),
                        device=self.device,
                    )
                self.terrain_geometry = None
                self.foothold_ik_active = torch.zeros(
                    self.num_envs, len(self.feet_body_ids), dtype=torch.bool, device=self.device
                )
                self.foothold_guard_active = torch.zeros(
                    self.num_envs, len(self.feet_body_ids), dtype=torch.bool, device=self.device
                )
                lock_shape = (self.num_envs, len(self.feet_body_ids))
                self.foothold_lock_valid = torch.zeros(lock_shape, dtype=torch.bool, device=self.device)
                self.foothold_lock_mode = torch.zeros(lock_shape, dtype=torch.int8, device=self.device)
                self.foothold_lock_world_xy = torch.zeros(*lock_shape, 2, device=self.device)
                self.foothold_lock_world_z = torch.zeros(lock_shape, device=self.device)
                self.foothold_lock_width = torch.zeros(lock_shape, device=self.device)
                self.foothold_lock_confidence = torch.zeros(lock_shape, device=self.device)
                self.foothold_swing_elapsed = torch.zeros(lock_shape, device=self.device)
                self.foothold_swing_start_w = torch.zeros(*lock_shape, 3, device=self.device)
                self.foothold_lock_control_valid = torch.zeros(lock_shape, dtype=torch.bool, device=self.device)
                self.stair_mode = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
                self.stair_gate_strength = torch.zeros(self.num_envs, device=self.device)
                self.previous_geometry_edges = torch.zeros(
                    self.num_envs, geometry_cfg.max_treads, 2, device=self.device
                )
                self.previous_geometry_valid = torch.zeros(
                    self.num_envs, geometry_cfg.max_treads, dtype=torch.bool, device=self.device
                )
                self.geometry_stable_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                self.stair_settle_enabled = geometry_cfg.gate_enabled and geometry_cfg.stair_step_settle_enabled
                self.stair_settle_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self.stair_settle_elapsed = torch.zeros(self.num_envs, device=self.device)
                self.previous_supported_height = torch.full(
                    (self.num_envs,), torch.nan, device=self.device
                )
                self.previous_foot_contact = torch.zeros(
                    self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.bool, device=self.device
                )
                self.nominal_command = torch.zeros(self.num_envs, 3, device=self.device)
                if geometry_cfg.gate_enabled:
                    self.stair_mode_gate = StairModeGate(
                        num_envs=self.num_envs,
                        device=self.device,
                        cfg=StairModeGateCfg(
                            enter_confidence=geometry_cfg.gate_enter_confidence,
                            exit_confidence=geometry_cfg.gate_exit_confidence,
                            enter_frames=geometry_cfg.gate_enter_frames,
                            exit_frames=geometry_cfg.gate_exit_frames,
                            switch_frames=geometry_cfg.gate_switch_frames,
                            min_valid_treads=geometry_cfg.gate_min_valid_treads,
                            min_nearest_tread_m=geometry_cfg.gate_min_nearest_tread,
                            max_nearest_tread_m=geometry_cfg.gate_max_nearest_tread,
                        ),
                    )
            else:
                self.depth_obs_buffer = CircularBuffer(
                    max_len=self.cfg.scene.depth_camera.depth_history_length,
                    batch_size=self.num_envs,
                    device=self.device,
                )
                depth_feature_dim = (
                    self.cfg.scene.depth_camera.feature_height * self.cfg.scene.depth_camera.feature_width
                )
                self.last_depth_features = torch.zeros(self.num_envs, depth_feature_dim, device=self.device)

    def _initialize_course_width_curriculum(self) -> None:
        """Start ordered-course environments in the 30 cm stair lane."""

        terrain_cfg = self.cfg.scene.terrain_generator
        geometry_cfg = getattr(self.cfg.scene.depth_camera, "geometry", None)
        generator_name = getattr(getattr(terrain_cfg, "class_type", None), "__name__", "")
        self.course_width_curriculum_enabled = bool(
            terrain_cfg is not None
            and (terrain_cfg.curriculum or bool(getattr(geometry_cfg, "bootstrap_stairs", False)))
            and generator_name == "AtecObstacleCourseTerrainGenerator"
            and (terrain_cfg.num_cols > 1 or bool(getattr(geometry_cfg, "bootstrap_stairs", False)))
        )
        if not self.course_width_curriculum_enabled:
            return
        terrain = self.scene.terrain
        bootstrap_stairs = bool(getattr(geometry_cfg, "bootstrap_stairs", False))
        if bootstrap_stairs:
            num_steps = terrain_cfg.sub_terrains[
                "stairs_down" if geometry_cfg.bootstrap_stair_row == 2 else "stairs_up"
            ].num_steps
            self.last_goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_goal_distance_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_goal_stable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_goal_verified_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.last_goal_verified_contacts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.last_distance_goal_coverage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.last_distance_goal_broad_coverage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.last_distance_goal_physical_levels = torch.zeros(
                self.num_envs, num_steps, dtype=torch.bool, device=self.device
            )
            self.last_distance_goal_broad_levels = torch.zeros(
                self.num_envs, num_steps, dtype=torch.bool, device=self.device
            )
            self.terminal_support_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.stair_progress_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.stair_progress_increment = torch.zeros(self.num_envs, device=self.device)
            self.last_raw_stair_progress_increment = torch.zeros(self.num_envs, device=self.device)
            self.last_stair_physical_step_increment = torch.zeros(self.num_envs, device=self.device)
            self.last_stair_failure_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_stair_failure_lane = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_stair_failure_unsafe_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_stair_failure_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_stair_failure_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.last_stair_failure_body_index = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self.stair_hip_clearance = torch.zeros(self.num_envs, device=self.device)
            self.stair_verified_progress_level = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.stair_verified_contact_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.stair_physical_support_count = torch.zeros(
                self.num_envs, 2, num_steps, dtype=torch.long, device=self.device
            )
            self.stair_physical_covered = torch.zeros(
                self.num_envs, num_steps, dtype=torch.bool, device=self.device
            )
            self.stair_broad_support_count = torch.zeros(
                self.num_envs, 2, num_steps, dtype=torch.long, device=self.device
            )
            self.stair_broad_covered = torch.zeros(
                self.num_envs, num_steps, dtype=torch.bool, device=self.device
            )
            self.stair_physical_step_increment = torch.zeros(self.num_envs, device=self.device)
            self.stair_step_height = torch.zeros(self.num_envs, device=self.device)
            self.stair_step_width = torch.zeros(self.num_envs, device=self.device)
            self.stair_first_riser_distance = torch.zeros(self.num_envs, device=self.device)
        initial_row = int(getattr(geometry_cfg, "bootstrap_stair_row", 1)) if bootstrap_stairs else 0
        initial_row = min(initial_row, terrain.terrain_origins.shape[0] - 1)
        terrain.terrain_levels.fill_(initial_row)
        self.course_width_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        lane_slot = torch.arange(self.num_envs, device=self.device)
        terrain.terrain_types[:] = lane_slot
        terrain.env_origins[:] = terrain.terrain_origins[initial_row, lane_slot]
        if bootstrap_stairs:
            terrain.env_origins[:, 0] -= float(getattr(geometry_cfg, "bootstrap_x_offset", 1.5))
            # The generated stair origin stores the height at the center of
            # the sub-terrain.  Bootstrap starts before the first riser, so
            # retaining that center height would spawn the robot in mid-air
            # and make the base policy fail before perception is exercised.
            terrain.env_origins[:, 2] = self._bootstrap_spawn_height(lane_slot)

    def _bootstrap_spawn_height(self, lane_column: torch.Tensor) -> torch.Tensor:
        geometry_cfg = self.cfg.scene.depth_camera.geometry
        terrain_cfg = self.cfg.scene.terrain_generator
        if geometry_cfg.bootstrap_stair_row != 2:
            return torch.full_like(lane_column, float(geometry_cfg.bootstrap_spawn_height), dtype=torch.float)
        stair_cfg = terrain_cfg.sub_terrains["stairs_down"]
        lower, upper = terrain_cfg.difficulty_range
        difficulty = lower + (upper - lower) * lane_column.float() / max(terrain_cfg.num_cols - 1, 1)
        riser = stair_cfg.step_height_range[0] + difficulty * (
            stair_cfg.step_height_range[1] - stair_cfg.step_height_range[0]
        )
        return stair_cfg.num_steps * riser

    def update_course_width_curriculum(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance course difficulty after completion; tread width stays 32 cm."""

        terrain = self.scene.terrain
        segment_length = self.cfg.scene.terrain_generator.size[0]
        geometry_cfg = getattr(self.cfg.scene.depth_camera, "geometry", None)
        bootstrap_stairs = bool(getattr(geometry_cfg, "bootstrap_stairs", False))
        # Origins are at the start of each segment; completion is the end of
        # the final segment, not the start of the last row.
        finish_distance = segment_length * (self.cfg.scene.terrain_generator.num_rows - 0.5) - 0.5
        forward_distance = self.robot.data.root_pos_w[env_ids, 0] - self.scene.env_origins[env_ids, 0]
        # During the first reset the articulation can still carry the terrain
        # importer's pre-curriculum world pose. Never interpret that stale pose
        # as a completed route.
        has_run_episode = self.episode_length_buf[env_ids] > 1
        completed = has_run_episode & (forward_distance >= finish_distance)
        failed_early = has_run_episode & (forward_distance < 0.5 * finish_distance)

        level = self.course_width_level[env_ids]
        if bootstrap_stairs:
            # Physical coverage can promote a height stage before perfect 8/8
            # completion, without weakening the strict success criterion.
            completed = has_run_episode & self.last_goal_reached[env_ids]
            promotable = has_run_episode & stair_width_promotion(
                self.last_goal_stable[env_ids], self.stair_physical_covered[env_ids]
            )
            failed_early = has_run_episode & (forward_distance < 0.5 * geometry_cfg.bootstrap_goal_distance)
        max_stage = terrain.terrain_origins.shape[1] // self.num_envs - 1
        promotion = promotable if bootstrap_stairs else completed
        level = torch.clamp(level + promotion.long() - failed_early.long(), 0, max_stage)
        self.course_width_level[env_ids] = level
        lane_column = level * self.num_envs + env_ids
        terrain.terrain_types[env_ids] = lane_column
        row = torch.full_like(level, int(getattr(geometry_cfg, "bootstrap_stair_row", 1))) if bootstrap_stairs else torch.zeros_like(level)
        row = row.clamp_max(terrain.terrain_origins.shape[0] - 1)
        terrain.terrain_levels[env_ids] = row
        terrain.env_origins[env_ids] = terrain.terrain_origins[row, lane_column]
        if bootstrap_stairs:
            terrain.env_origins[env_ids, 0] -= float(getattr(geometry_cfg, "bootstrap_x_offset", 1.5))
            terrain.env_origins[env_ids, 2] = self._bootstrap_spawn_height(lane_column)
        return {
            "Curriculum/stair_width_level": self.course_width_level.float().mean(),
            "Curriculum/course_completion": completed.float().mean(),
            "Curriculum/stair_width_promotion": promotion.float().mean(),
        }

    def process_depth_observations(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_cfg = self.cfg.scene.depth_camera
        depth = depth_image.squeeze(-1)
        depth = torch.nan_to_num(
            depth,
            nan=depth_cfg.max_range,
            posinf=depth_cfg.max_range,
            neginf=depth_cfg.min_range,
        )
        depth = depth.clamp(depth_cfg.min_range, depth_cfg.max_range)
        depth = F.adaptive_avg_pool2d(
            depth.unsqueeze(1),
            (depth_cfg.feature_height, depth_cfg.feature_width),
        ).flatten(1)
        quantization = depth_cfg.depth_quantization
        if quantization > 0.0:
            depth = torch.round(depth / quantization) * quantization
        depth = (depth - depth_cfg.min_range) / (depth_cfg.max_range - depth_cfg.min_range)
        if depth_cfg.frame_hold_prob > 0.0:
            hold = torch.rand(self.num_envs, 1, device=self.device) < depth_cfg.frame_hold_prob
            depth = torch.where(hold, self.last_depth_features, depth)
        self.last_depth_features.copy_(depth)
        self.depth_obs_buffer.append(depth)
        return self.depth_obs_buffer.buffer.reshape(self.num_envs, -1)

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_motion_mask(self, lin_threshold: float = 0.1, yaw_threshold: float = 0.1) -> torch.Tensor:
        """
        Returns a mask indicating whether AMP reward should be applied.
        Returns 1.0 for moving commands, 0.0 for near-zero commands.
        This allows disabling AMP's periodic motion encouragement when standing still.
        """
        cmd = self.command_generator.command
        is_moving = (torch.norm(cmd[:, :2], dim=-1) >= lin_threshold) | (torch.abs(cmd[:, 2]) >= yaw_threshold)
        return is_moving.float()

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy"""
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        
        self.waist_dof_pos = self.robot.data.joint_pos[:, self.waist_ids]
        self.left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        self.left_arm_dof_pos = self.robot.data.joint_pos[:, self.left_arm_ids + self.left_wrist_ids]
        # self.left_wrist_dof_pos = self.robot.data.joint_pos[:, self.left_wrist_ids]
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids + self.right_wrist_ids]
        # self.right_wrist_dof_pos = self.robot.data.joint_pos[:, self.right_wrist_ids]
        
        self.waist_dof_vel = self.robot.data.joint_vel[:, self.waist_ids]
        self.left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        self.left_arm_dof_vel = self.robot.data.joint_vel[:, self.left_arm_ids + self.left_wrist_ids]
        # self.left_wrist_dof_vel = self.robot.data.joint_vel[:, self.left_wrist_ids]
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids + self.right_wrist_ids]
        # self.right_wrist_dof_vel = self.robot.data.joint_vel[:, self.right_wrist_ids]
        
        return torch.cat(
            (   

                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.waist_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,

                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.waist_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )
        

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
