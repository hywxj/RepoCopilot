"""Shared position-PD actor with analytic local geometry and whole-course evaluation."""

import torch
from isaaclab.utils.math import quat_apply

from legged_lab.envs.elf3.elf3_env import Elf3Env
from legged_lab.utils.continuous_course import (
    GEOMETRY_SCHEMA, course_outcomes, footprint_edge_margin, local_height_features,
)


class ContinuousElf3Env(Elf3Env):
    geometry_schema = GEOMETRY_SCHEMA
    continuous_course_enabled = True

    def __init__(self, cfg, headless):
        if cfg.scene.depth_camera.enable_depth_camera or cfg.scene.depth_camera.geometry.step_control_enabled:
            raise ValueError("Continuous baseline requires analytic geometry without a step supervisor")
        if abs(cfg.sim.dt * cfg.sim.decimation - .02) > 1.e-9:
            raise ValueError("Continuous ELF3 contract requires a 20ms action period")
        if cfg.scene.contact_history_length < cfg.sim.decimation:
            raise ValueError("Contact history must cover every physics substep")
        super().__init__(cfg, headless)
        if self.num_actions != 29:
            raise ValueError("Continuous ELF3 requires all 29 joints")

    def init_buffers(self):
        names = tuple(self.robot.joint_names)
        named_scales = self.cfg.continuous_action_scales
        if len(names) != 29 or set(names) != set(named_scales):
            raise ValueError("Named continuous action scales must cover exactly the 29 physical joints")
        self.cfg.robot.action_scale = [named_scales[name] for name in names]
        self.continuous_action_joint_names = names
        super().init_buffers()
        self.course_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.course_failed = torch.zeros_like(self.course_success)
        self.course_fallen = torch.zeros_like(self.course_success)
        self.course_out_of_bounds = torch.zeros_like(self.course_success)
        self.course_collision = torch.zeros_like(self.course_success)
        self.course_slip_sum = torch.zeros(self.num_envs, device=self.device)
        self.course_load_count = torch.zeros_like(self.course_slip_sum)
        self.course_min_edge_margin = torch.full_like(self.course_slip_sum, 10.)
        self.course_max_tilt = torch.zeros_like(self.course_slip_sum)
        self.course_collision_count = torch.zeros_like(self.course_slip_sum)
        self.course_peak_foot_force = torch.zeros_like(self.course_slip_sum)
        self.course_peak_loaded_slip = torch.zeros_like(self.course_slip_sum)
        self.course_edge_violation_time = torch.zeros_like(self.course_slip_sum)
        self.course_slip_violation_time = torch.zeros_like(self.course_slip_sum)
        self.course_current_edge_margin = torch.zeros(self.num_envs, 2, device=self.device)
        self.course_current_sole_speed = torch.zeros_like(self.course_current_edge_margin)
        self.course_loaded_feet = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

    def _calculate_gait_para(self):
        # Parent retains compatibility buffers; no periodic phase enters control/reward/observations.
        pass

    def compute_observations(self, update_geometry=True):
        actor, critic = super().compute_observations(update_geometry=False)
        q = self.robot.data.root_quat_w
        yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2].square()+q[:, 3].square()))
        geometry = local_height_features(self.robot.data.root_pos_w-self.scene.env_origins, yaw, self.cfg.continuous)
        return torch.cat((actor, geometry), dim=-1), torch.cat((critic, geometry), dim=-1)

    def _foot_corners(self):
        # Conservative ankle-frame URDF footprint, shared with tread_surfaces config.
        local = torch.tensor([[-.09, -.042, -.04], [-.09, .042, -.04],
                              [.15, -.042, -.04], [.15, .042, -.04]], device=self.device)
        q = self.robot.data.body_quat_w[:, self.feet_body_ids]
        rotated = quat_apply(q[:, :, None, :].expand(-1, -1, 4, -1), local.expand(self.num_envs, 2, -1, -1))
        return rotated + self.robot.data.body_pos_w[:, self.feet_body_ids, None, :] - self.scene.env_origins[:, None, None, :]

    def check_reset(self):
        relative = self.robot.data.root_pos_w-self.scene.env_origins
        forces = self.contact_sensor.data.net_forces_w_history
        collision_force = forces[:, :, self.termination_contact_cfg.body_ids].norm(dim=-1).amax(dim=(1, 2))
        self.course_collision = collision_force > 40.
        timed_out = self.episode_length_buf >= self.max_episode_length
        loads = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2].clamp_min(0.)
        corners = self._foot_corners()
        self.course_success, self.course_failed, self.course_fallen, self.course_out_of_bounds = course_outcomes(
            relative, self.robot.data.projected_gravity_b, corners, loads, self.course_collision,
            timed_out, self.cfg.continuous,
        )
        loaded = loads > 20.
        sole_offset = quat_apply(self.robot.data.body_quat_w[:, self.feet_body_ids],
                                torch.tensor([.03, 0., -.04], device=self.device).expand(self.num_envs, 2, 3))
        sole_velocity = (self.robot.data.body_link_lin_vel_w[:, self.feet_body_ids]
                         + torch.cross(self.robot.data.body_ang_vel_w[:, self.feet_body_ids], sole_offset, dim=-1))
        speed = sole_velocity[..., :2].norm(dim=-1)
        self.course_current_sole_speed = speed
        self.course_slip_sum += (speed*loaded).sum(-1)*self.step_dt
        self.course_load_count += loaded.sum(-1)*self.step_dt
        self.course_peak_loaded_slip = torch.maximum(self.course_peak_loaded_slip, (speed*loaded).amax(-1))
        self.course_slip_violation_time += ((speed > .2) & loaded).any(-1)*self.step_dt
        foot_x = self.robot.data.body_pos_w[:, self.feet_body_ids, 0]-self.scene.env_origins[:, :1]
        margin = footprint_edge_margin(corners, foot_x, self.cfg.continuous)
        self.course_current_edge_margin = margin
        self.course_loaded_feet = loaded
        self.course_edge_violation_time += ((margin < -.02) & loaded).any(-1)*self.step_dt
        self.course_min_edge_margin = torch.minimum(self.course_min_edge_margin, torch.where(loaded, margin, 10.).amin(-1))
        tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.))
        self.course_max_tilt = torch.maximum(self.course_max_tilt, tilt)
        self.course_collision_count += self.course_collision.float()
        peak_force = forces[:, :, self.feet_cfg.body_ids].norm(dim=-1).amax(dim=(1, 2))
        self.course_peak_foot_force = torch.maximum(self.course_peak_foot_force, peak_force)
        return self.course_success | self.course_failed | timed_out, timed_out

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        stats = {}
        if len(completed):
            ids = completed
            loaded_ids = ids[self.course_load_count[ids] > 0]
            stats = {
                "Continuous/endpoint_completion": self.course_success[ids].float().mean(),
                "Continuous/fall": self.course_fallen[ids].float().mean(),
                "Continuous/collision": (self.course_collision_count[ids] > 0).float().mean(),
                "Continuous/out_of_bounds": self.course_out_of_bounds[ids].float().mean(),
                "Continuous/timeout": self.time_out_buf[ids].float().mean(),
                "Continuous/max_tilt_rad": self.course_max_tilt[ids].mean(),
                "Continuous/peak_foot_force_n": self.course_peak_foot_force[ids].mean(),
                "Continuous/peak_loaded_slip_m_s": self.course_peak_loaded_slip[ids].mean(),
                "Continuous/loaded_edge_overhang_gt2cm_s": self.course_edge_violation_time[ids].mean(),
                "Continuous/loaded_slip_gt02m_s_s": self.course_slip_violation_time[ids].mean(),
                "Continuous/distance_m": (self.robot.data.root_pos_w[ids, 0]-self.scene.env_origins[ids, 0]).mean(),
            }
            if len(loaded_ids):
                stats["Continuous/loaded_foot_slip_m_s"] = (self.course_slip_sum[loaded_ids]/self.course_load_count[loaded_ids]).mean()
                stats["Continuous/min_loaded_edge_margin_m"] = self.course_min_edge_margin[loaded_ids].mean()
        super().reset(env_ids)
        self.extras["log"].update(stats)
        for name in ("course_slip_sum", "course_load_count", "course_max_tilt", "course_collision_count", "course_peak_foot_force",
                     "course_peak_loaded_slip", "course_edge_violation_time", "course_slip_violation_time"):
            getattr(self, name)[env_ids] = 0.
        self.course_min_edge_margin[env_ids] = 10.
