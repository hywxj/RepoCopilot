"""Small task-space corrections for geometry-guided swing feet."""

import torch


def standing_support_quality(
    contact_forces: torch.Tensor,
    sole_velocities: torch.Tensor,
    body_weight: torch.Tensor,
) -> torch.Tensor:
    """Dense simulation reward for stationary, evenly loaded feet."""
    load = contact_forces[..., 2].clamp_min(0.0) / body_weight[:, None].clamp_min(1.0)
    balance = 1.0 / (1.0 + (load - 0.5).square().sum(dim=1) / 0.25**2)
    motion = sole_velocities.square().sum(dim=-1).mean(dim=1)
    return balance / (1.0 + motion / 0.08**2)


def confirmed_support_transition(
    previous_count: torch.Tensor,
    valid_support: torch.Tensor,
    confirmation_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit one landing event after consecutive valid support frames."""

    next_count = torch.where(
        valid_support, previous_count + 1, torch.zeros_like(previous_count)
    ).clamp_max(confirmation_frames)
    confirmed = valid_support & (next_count == confirmation_frames) & (
        previous_count < confirmation_frames
    )
    return next_count, confirmed


def terminal_platform_support(
    sole_height: torch.Tensor,
    foot_force: torch.Tensor,
    foot_speed_xy: torch.Tensor,
    platform_height: torch.Tensor,
    height_tolerance: float = 0.10,
    max_speed: float = 0.25,
    min_vertical_ratio: float = 0.65,
) -> torch.Tensor:
    """Require a planted sole on the terminal platform, not merely root progress."""

    vertical = foot_force[..., 2]
    upward = (vertical > 5.0) & (
        vertical / torch.linalg.norm(foot_force, dim=-1).clamp_min(1.0)
        >= min_vertical_ratio
    )
    return (
        upward
        & ((sole_height - platform_height.unsqueeze(1)).abs() <= height_tolerance)
        & (foot_speed_xy < max_speed)
    ).any(dim=1)


def terminal_platform_height(
    start_height: torch.Tensor,
    step_height: torch.Tensor,
    num_steps: int,
    descending: bool,
) -> torch.Tensor:
    """Resolve the final platform from the bootstrap spawn platform height."""

    direction = -1 if descending else 1
    return start_height + direction * num_steps * step_height


def stair_support_coverage_transition(
    support_count: torch.Tensor,
    covered: torch.Tensor,
    sole_center_x: torch.Tensor,
    sole_height: torch.Tensor,
    foot_force: torch.Tensor,
    foot_speed_xy: torch.Tensor,
    first_riser_x: torch.Tensor,
    step_width: torch.Tensor,
    start_height: torch.Tensor,
    step_height: torch.Tensor,
    descending: bool,
    confirmation_frames: int = 3,
    center_tolerance_fraction: float = 0.10,
    max_center_tolerance: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Audit stable, centered support on each physical stair level in simulation."""

    num_steps = covered.shape[1]
    level = torch.arange(1, num_steps + 1, device=covered.device)
    tread_center = first_riser_x[:, None] + (level[None] - 0.5) * step_width[:, None]
    center_tolerance = (center_tolerance_fraction * step_width).clamp_max(max_center_tolerance)
    centered = (
        sole_center_x[:, :, None] - tread_center[:, None, :]
    ).abs() <= center_tolerance[:, None, None]
    # The last level continues onto the final platform after the last riser.
    last_start = first_riser_x + (num_steps - 1) * step_width
    centered[..., -1] |= sole_center_x >= (last_start + 0.05).unsqueeze(1)
    direction = -1 if descending else 1
    target_height = start_height[:, None] + direction * level[None] * step_height[:, None]
    same_height = (
        sole_height[:, :, None] - target_height[:, None, :]
    ).abs() <= 0.06
    vertical_force = foot_force[..., 2]
    support = (vertical_force > 5.0) & (
        vertical_force / torch.linalg.norm(foot_force, dim=-1).clamp_min(1.0) >= 0.65
    ) & (foot_speed_xy < 0.25)
    stable = centered & same_height & support.unsqueeze(-1)
    next_count = torch.where(stable, support_count + 1, torch.zeros_like(support_count))
    next_count = next_count.clamp_max(confirmation_frames)
    next_covered = covered | (next_count >= confirmation_frames).any(dim=1)
    increment = (next_covered & ~covered).sum(dim=1)
    return next_count, next_covered, increment


def stair_width_promotion(
    stable_terminal_support: torch.Tensor,
    physical_coverage: torch.Tensor,
    allowed_missed_levels: int = 2,
) -> torch.Tensor:
    """Advance curriculum on strong partial coverage; full success still requires every level."""

    required = max(1, physical_coverage.shape[1] - allowed_missed_levels)
    return stable_terminal_support & (physical_coverage.sum(dim=1) >= required)


def nonfoot_collision_risk(
    contact_force: torch.Tensor,
    threshold: float = 20.0,
    force_span: float = 300.0,
) -> torch.Tensor:
    """Continuous warning for a non-foot impact before termination force."""

    peak_force = torch.linalg.norm(contact_force, dim=-1).amax(dim=(1, 2))
    return ((peak_force - threshold) / force_span).clamp(0.0, 1.0)


def damped_joint_step(
    jacobian: torch.Tensor,
    position_error: torch.Tensor,
    damping: float = 0.08,
    max_joint_step: float = 0.12,
) -> torch.Tensor:
    """Return a bounded damped least-squares joint correction."""

    identity = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype).expand(jacobian.shape[0], -1, -1)
    gram = jacobian @ jacobian.transpose(1, 2) + damping**2 * identity
    task_step = torch.linalg.solve(gram, position_error.unsqueeze(-1))
    joint_step = jacobian.transpose(1, 2) @ task_step
    return joint_step.squeeze(-1).clamp(-max_joint_step, max_joint_step)


def bounded_overshoot_joint_step(
    forward_jacobian: torch.Tensor,
    commanded_joint_delta: torch.Tensor,
    ankle_x: torch.Tensor,
    max_ankle_x: torch.Tensor,
    max_joint_step: float = 0.05,
    damping: float = 0.05,
) -> torch.Tensor:
    """Project a swing-foot joint command back from a detected tread edge."""

    predicted_x = ankle_x + (forward_jacobian * commanded_joint_delta).sum(dim=1)
    overshoot = (predicted_x - max_ankle_x).clamp_min(0.0)
    denominator = forward_jacobian.square().sum(dim=1).clamp_min(0.0) + damping**2
    correction = -forward_jacobian * (overshoot / denominator).unsqueeze(1)
    return correction.clamp(-max_joint_step, max_joint_step)


def select_next_treads(
    treads: torch.Tensor,
    tread_world_z: torch.Tensor,
    feet_x: torch.Tensor,
    sole_z: torch.Tensor,
    contacts: torch.Tensor,
    stair_mode: torch.Tensor,
    target_offset: float = 0.20,
    preview_stance: bool = False,
    min_forward_gap: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a swing-foot landing on the support level or the next stair level."""

    centers = 0.5 * (treads[..., 0] + treads[..., 1])
    support_z = sole_z.flip(1)
    height_advance = stair_mode[:, None, None] * (
        tread_world_z[:, None, :] - support_z[:, :, None]
    )
    forward_gap = centers[:, None, :] - feet_x[:, :, None]
    valid = (
        (treads[:, None, :, 5] > 0.5)
        & (treads[:, None, :, 4] > 0.25)
        & (preview_stance | ~contacts[:, :, None])
        & contacts.flip(1)[:, :, None]
        & (stair_mode[:, None, None] != 0)
        & (height_advance >= -0.035)
        & (height_advance <= 0.22)
        & (forward_gap >= min_forward_gap)
        & (forward_gap <= 0.65)
    )
    next_level = height_advance >= 0.05
    distance = ((forward_gap - target_offset).abs() + (~next_level).float() * 0.40).masked_fill(
        ~valid, torch.inf
    )
    index = distance.argmin(dim=-1, keepdim=True)
    selected = treads[:, None].expand(-1, feet_x.shape[1], -1, -1).gather(
        2, index.unsqueeze(-1).expand(-1, -1, -1, treads.shape[-1])
    ).squeeze(2)
    selected_height = tread_world_z[:, None].expand(-1, feet_x.shape[1], -1).gather(
        2, index
    ).squeeze(2)
    return selected, selected_height, valid.any(dim=-1)


def stair_yaw_feedback(
    lateral_error: torch.Tensor,
    heading_error: torch.Tensor,
    lateral_velocity: torch.Tensor,
    max_yaw_rate: float = 0.45,
    lateral_gain: float = 0.6,
    heading_gain: float = 1.0,
    velocity_gain: float = 0.15,
) -> torch.Tensor:
    """Turn toward the lane center while damping sideways motion on stairs."""

    return (-lateral_gain * lateral_error - heading_gain * heading_error - velocity_gain * lateral_velocity).clamp(
        -max_yaw_rate, max_yaw_rate
    )


def foothold_lock_transition(
    locked_valid: torch.Tensor,
    locked_mode: torch.Tensor,
    contacts: torch.Tensor,
    stair_mode: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a swing target until contact or mode change; acquire only when unlocked."""

    mode = stair_mode.unsqueeze(1).expand_as(contacts)
    keep = locked_valid & ~contacts & (mode != 0) & (mode == locked_mode)
    acquire = candidate_valid & ~contacts & (mode != 0) & ~keep
    return keep | acquire, acquire


def current_tread_match(
    selected_treads: torch.Tensor,
    current_treads: torch.Tensor,
    min_confidence: float = 0.7,
    center_tolerance_m: float = 0.08,
) -> torch.Tensor:
    """Check whether a historical target is still visible as a reliable current tread."""

    selected_center = 0.5 * (selected_treads[..., 0] + selected_treads[..., 1])
    current_center = 0.5 * (current_treads[..., 0] + current_treads[..., 1])
    visible = (current_treads[..., 5] > 0.5) & (current_treads[..., 4] >= min_confidence)
    aligned = (
        (selected_center.unsqueeze(-1) - current_center.unsqueeze(1)).abs()
        <= center_tolerance_m
    )
    return (aligned & visible.unsqueeze(1)).any(dim=-1)


def capture_point_offset(
    com_pos_w: torch.Tensor,
    com_vel_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    heading_xy: torch.Tensor,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Approximate horizontal capture point relative to each foot in heading coordinates."""

    height = (com_pos_w[:, None, 2] - foot_pos_w[..., 2]).clamp(0.10, 1.50)
    capture_xy = com_pos_w[:, None, :2] + com_vel_w[:, None, :2] * torch.sqrt(height / gravity).unsqueeze(-1)
    offset = capture_xy - foot_pos_w[..., :2]
    forward = (offset * heading_xy[:, None]).sum(dim=-1)
    lateral = offset[..., 1] * heading_xy[:, None, 0] - offset[..., 0] * heading_xy[:, None, 1]
    return torch.stack((forward, lateral), dim=-1)


def swing_foot_trajectory(
    start_world: torch.Tensor,
    target_world: torch.Tensor,
    elapsed_s: torch.Tensor,
    duration_s: float,
    clearance_m: float,
) -> torch.Tensor:
    """Interpolate a world-fixed foothold with zero endpoint velocity and toe clearance."""

    progress = (elapsed_s / duration_s).clamp(0.0, 1.0)
    blend = progress.square() * (3.0 - 2.0 * progress)
    arch = 16.0 * progress.square() * (1.0 - progress).square()
    desired = start_world + blend.unsqueeze(-1) * (target_world - start_world)
    desired[..., 2] += clearance_m * arch
    return desired


def supported_stair_progress(
    forward_distance: torch.Tensor,
    supported_sole_height: torch.Tensor,
    start_height: torch.Tensor,
    step_width: torch.Tensor,
    step_height: torch.Tensor,
    previous_level: torch.Tensor,
    first_riser_distance: float | torch.Tensor,
    num_steps: int,
    direction: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count a new step only after forward movement and raised foot support."""

    horizontal_level = torch.floor(
        (forward_distance - first_riser_distance) / step_width.clamp_min(1.0e-6)
    ).long() + 1
    vertical_distance = direction * (supported_sole_height - start_height)
    vertical_level = torch.floor(
        (vertical_distance + 0.03) / step_height.clamp_min(1.0e-6)
    ).long()
    observed_level = torch.minimum(horizontal_level, vertical_level).clamp(0, num_steps)
    new_level = torch.maximum(previous_level, observed_level)
    return new_level, new_level - previous_level


def verified_stair_progress_transition(
    observed_level: torch.Tensor,
    verified_level: torch.Tensor,
    confirmed_touchdown: torch.Tensor,
    matched_tread_height: torch.Tensor,
    start_height: torch.Tensor,
    step_height: torch.Tensor,
    direction: int,
    height_tolerance: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Credit each newly confirmed stair level once, without paying for skipped steps."""

    relative_height = direction * (
        matched_tread_height - start_height.unsqueeze(1)
    ) / step_height.unsqueeze(1).clamp_min(1.0e-6)
    matched_level = relative_height.round().long()
    expected_height = start_height.unsqueeze(1) + direction * matched_level * step_height.unsqueeze(1)
    tolerance = torch.minimum(
        torch.full_like(step_height, height_tolerance), 0.35 * step_height
    ).unsqueeze(1)
    valid = (
        confirmed_touchdown
        & (matched_level > verified_level.unsqueeze(1))
        & (matched_level <= observed_level.unsqueeze(1))
        & ((matched_tread_height - expected_height).abs() <= tolerance)
    )
    next_level = torch.where(valid, matched_level, verified_level.unsqueeze(1)).amax(dim=1)
    increment = next_level > verified_level
    return next_level, increment.float()
