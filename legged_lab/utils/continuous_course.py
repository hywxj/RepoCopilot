"""Analytic continuous-course geometry and safety queries, independent of Isaac.

Coordinates are relative to the spawn origin; z is relative to its support plane.
The actor contract is 45 root/yaw-frame heights followed by 45 validity bits.
Future perception can project measured support planes into this same grid.
"""

from dataclasses import dataclass

import torch


GEOMETRY_SCHEMA = "elf3_local_yaw_height45_valid45_v1"
GEOMETRY_DIM = 90


@dataclass
class ContinuousCourse:
    direction: str = "flat"
    num_steps: int = 4
    step_height: float = 0.11
    step_depth: float = 0.32
    first_edge: float = 0.70
    spawn_x: float = 1.0
    length: float = 6.0
    width: float = 3.0
    goal_x: float = 2.50
    lateral_limit: float = 1.15

    def __post_init__(self):
        if self.direction not in ("flat", "up", "down"):
            raise ValueError("Course direction must be flat, up or down")
        if not 3 <= self.num_steps <= 5 or self.step_height <= 0 or self.step_depth <= 0:
            raise ValueError("Course requires 3–5 positive-height, positive-depth steps")
        if self.goal_x <= self.final_edge + 0.25 or self.goal_x >= self.length - self.spawn_x - .25:
            raise ValueError("Goal must lie inside the final platform, beyond the last riser")

    @property
    def final_edge(self):
        return self.first_edge + (self.num_steps - 1) * self.step_depth

    @property
    def spawn_height(self):
        return self.num_steps * self.step_height if self.direction == "down" else 0.0


def support_height(x: torch.Tensor, course: ContinuousCourse) -> torch.Tensor:
    if course.direction == "flat":
        return torch.zeros_like(x)
    level = (torch.floor((x - course.first_edge) / course.step_depth) + 1).clamp(0, course.num_steps)
    return level * course.step_height * (1.0 if course.direction == "up" else -1.0)


def local_height_features(root_position, root_yaw, course: ContinuousCourse):
    """Return N×90 features; positions/yaw use the fixed course frame.

    x samples -0.25…1.35m (0.20m apart), y samples -0.4…0.4m.
    Invalid samples contain zero height and a zero validity bit.
    """
    x = torch.linspace(-.25, 1.35, 9, device=root_position.device, dtype=root_position.dtype)
    y = torch.linspace(-.4, .4, 5, device=root_position.device, dtype=root_position.dtype)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    c, s = root_yaw.cos()[:, None], root_yaw.sin()[:, None]
    world_x = root_position[:, :1] + c * gx.flatten() - s * gy.flatten()
    world_y = root_position[:, 1:2] + s * gx.flatten() + c * gy.flatten()
    valid = ((world_x >= -course.spawn_x) & (world_x <= course.length - course.spawn_x)
             & (world_y.abs() <= course.width / 2))
    heights = support_height(world_x, course) - root_position[:, 2:3]
    return torch.cat((torch.where(valid, heights, 0.).clamp(-2., 2.), valid.to(heights.dtype)), dim=1)


def continuous_course_mesh(difficulty, cfg):
    """Isaac terrain callback; descending first edge drops immediately one level."""
    import numpy as np
    import trimesh

    c = cfg.course
    if tuple(cfg.size) != (c.length, c.width):
        raise ValueError("Terrain size and continuous-course dimensions must agree")
    if c.direction == "flat":
        boundaries, heights = [0., c.length], [0.]
    else:
        boundaries = [0.] + [c.spawn_x + c.first_edge + i*c.step_depth for i in range(c.num_steps)] + [c.length]
        heights = [i*c.step_height for i in range(c.num_steps + 1)]
        if c.direction == "down":
            heights.reverse()
    meshes = []
    for left, right, height in zip(boundaries[:-1], boundaries[1:], heights):
        # All segments share the -2cm underside; no thin elevated plates.
        transform = trimesh.transformations.translation_matrix(((left+right)/2, c.width/2, (height-.02)/2))
        meshes.append(trimesh.creation.box((right-left, c.width, height+.02), transform))
    return meshes, np.array([c.spawn_x, c.width/2, c.spawn_height], dtype=float)


def footprint_edge_margin(corners, foot_x, course):
    """Signed x/y support-plane margin for each full rotated foot rectangle.

    corners: N×2×4×3, foot_x: N×2. A straddling loaded foot has negative margin.
    This is a geometric diagnostic, not a pressure/CoP stability estimate.
    """
    left = torch.full_like(foot_x, -course.spawn_x)
    right = torch.full_like(foot_x, course.length - course.spawn_x)
    if course.direction != "flat":
        for i in range(course.num_steps):
            edge = course.first_edge + i*course.step_depth
            left = torch.where(foot_x >= edge, edge, left)
        for i in reversed(range(course.num_steps)):
            edge = course.first_edge + i*course.step_depth
            right = torch.where(foot_x < edge, edge, right)
    x_margin = torch.minimum(corners[..., 0].amin(-1)-left, right-corners[..., 0].amax(-1))
    y_margin = course.width/2-corners[..., 1].abs().amax(-1)
    return torch.minimum(x_margin, y_margin)


def course_outcomes(root_position, projected_gravity, corners, foot_loads,
                    collision, timed_out, course, min_clearance=.55):
    """Whole-course endpoint test without a gait, per-step gate or dwell timer."""
    floor = support_height(root_position[:, 0], course)
    fallen = ((root_position[:, 2]-floor < min_clearance)
              | (-projected_gravity[:, 2] < .60))
    out_of_bounds = ((root_position[:, 1].abs() > course.lateral_limit)
                     | (root_position[:, 0] < -.55)
                     | (root_position[:, 0] > course.length-course.spawn_x-.2))
    failed = fallen | collision | out_of_bounds
    corner_floor = support_height(corners[..., 0], course)
    supported = (foot_loads > 20.) & ((corners[..., 2]-corner_floor).abs().amax(-1) < .05)
    endpoint = course.final_edge + .02 if course.direction != "flat" else course.goal_x-.5
    feet_past_end = (corners[..., 0].amin(-1) > endpoint).all(-1)
    feet_inside = (corners[..., 1].abs().amax(dim=(-1, -2)) < course.width/2-.02)
    success = ((root_position[:, 0] >= course.goal_x) & feet_past_end & feet_inside & supported.all(-1)
               & ~failed & ~timed_out)
    return success, failed, fallen, out_of_bounds


# RewardManager supplies dt. Event terms divide by dt once to keep event value fixed.
def course_progress(env):
    return env.robot.data.root_lin_vel_w[:, 0].clamp(-.5, .8)


def course_failure(env):
    return env.course_failed.float() / env.step_dt


def course_completion(env):
    return env.course_success.float() / env.step_dt


def course_lateral_error(env):
    return (env.robot.data.root_pos_w[:, 1]-env.scene.env_origins[:, 1]).square()


def course_edge_risk(env, margin=.015):
    """Soft loaded-foot edge cost; never chooses a foot or gates the action."""
    return ((margin-env.course_current_edge_margin).clamp(0., .25)*env.course_loaded_feet).sum(-1)


def course_loaded_slip(env):
    return (env.course_current_sole_speed*env.course_loaded_feet).sum(-1)
