# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _matrix_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices without simulator dependencies."""

    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _yaw_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the body-to-world rotation containing only quaternion yaw."""

    w, x, y, z = quaternion.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    zeros = torch.zeros_like(cosine)
    ones = torch.ones_like(cosine)
    return torch.stack(
        (cosine, -sine, zeros, sine, cosine, zeros, zeros, zeros, ones), dim=-1
    ).reshape(quaternion.shape[:-1] + (3, 3))


def project_surface_points(depth_image, intrinsic_matrices, camera_pos_w, camera_quat_w_ros,
                           root_pos_w, root_quat_w, cfg, foot_positions_w=None,
                           foot_quaternions_w=None):
    """Project diagnostic depth independently of the legacy actor feature path."""
    if depth_image.ndim == 4:
        depth_image = depth_image[..., 0]
    if depth_image.ndim != 3:
        raise ValueError("Diagnostic depth must have shape [N,H,W] or [N,H,W,1].")
    batch_size, image_height, image_width = depth_image.shape
    height = min(getattr(cfg, "surface_processing_height", 360), image_height)
    width = min(getattr(cfg, "surface_processing_width", 640), image_width)
    if height <= 0 or width <= 0:
        raise ValueError("Diagnostic processing dimensions must be positive.")
    depth = F.interpolate(depth_image[:, None], size=(height, width), mode="nearest")[:, 0]
    valid = torch.isfinite(depth) & (depth >= cfg.min_forward) & (depth <= cfg.max_forward * 2.0)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    v, u = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                          torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij")
    fx = intrinsic_matrices[:, 0, 0, None, None] * (width / image_width)
    fy = intrinsic_matrices[:, 1, 1, None, None] * (height / image_height)
    cx = intrinsic_matrices[:, 0, 2, None, None] * (width / image_width)
    cy = intrinsic_matrices[:, 1, 2, None, None] * (height / image_height)
    camera = torch.stack(((u - cx) * depth / fx, (v - cy) * depth / fy, depth), dim=-1)
    world = torch.einsum("bij,bpj->bpi", _matrix_from_quat(camera_quat_w_ros),
                         camera.reshape(batch_size, -1, 3)) + camera_pos_w[:, None]
    body = torch.einsum("bji,bpj->bpi", _yaw_matrix(root_quat_w), world - root_pos_w[:, None])
    valid = valid.reshape(batch_size, -1)
    valid &= (body[..., 2] >= cfg.min_height_from_root) & (body[..., 2] <= cfg.max_height_from_root)
    if getattr(cfg, "surface_normals_enabled", True):
        import numpy as np
        from .tread_surfaces import depth_horizontal_mask

        intrinsic = intrinsic_matrices.detach().cpu().numpy().copy()
        intrinsic[:, 0] *= width / image_width
        intrinsic[:, 1] *= height / image_height
        rotation = _matrix_from_quat(camera_quat_w_ros).detach().cpu().numpy()
        depth_np = depth.detach().cpu().numpy()
        masks = [depth_horizontal_mask(depth_np[i], intrinsic[i], rotation[i],
                                        max_tilt_deg=getattr(cfg, "surface_local_tilt_deg", 45.0),
                                        window_size=getattr(cfg, "surface_normal_window_size", 7),
                                        radius=getattr(cfg, "surface_normal_radius", 4))
                 for i in range(batch_size)]
        valid &= torch.as_tensor(np.stack(masks).reshape(batch_size, -1), device=depth.device)
    if (foot_positions_w is None) != (foot_quaternions_w is None):
        raise ValueError("Foot positions and rotations must be provided together.")
    if foot_positions_w is not None:
        foot_local = torch.einsum("nfji,nfpj->nfpi", _matrix_from_quat(foot_quaternions_w),
                                  world[:, None] - foot_positions_w[:, :, None])
        lower = torch.tensor([-0.10, -0.052, -0.05], device=depth.device)
        upper = torch.tensor([0.16, 0.052, 0.025], device=depth.device)
        valid &= ~((foot_local >= lower) & (foot_local <= upper)).all(dim=-1).any(dim=1)
    return body, valid, (height, width)


@dataclass
class StairGeometryResult:
    """Metric geometry plus its normalized, fixed-size policy representation.

    ``treads`` stores ``near_x, far_x, relative_height, depth, confidence,
    valid`` for each candidate horizontal landing surface. Distances and
    heights are in the robot yaw frame and use metres.
    """

    treads: torch.Tensor
    direction: torch.Tensor
    stair_confidence: torch.Tensor
    roughness: torch.Tensor
    slope: torch.Tensor
    reference_height: torch.Tensor
    profile_x: torch.Tensor
    profile_z: torch.Tensor
    profile_valid: torch.Tensor
    features: torch.Tensor
    point_bounds: torch.Tensor | None = None
    valid_point_count: torch.Tensor | None = None
    surface_geometry: list | None = None
    surface_memory: list | None = None
    processed_image_shape: tuple[int, int] | None = None


def compensate_tread_history(
    features: torch.Tensor,
    positions_xy: torch.Tensor,
    headings_xy: torch.Tensor,
    current_position_xy: torch.Tensor,
    current_heading_xy: torch.Tensor,
    max_treads: int,
    max_forward: float,
    rear_limit: float,
) -> torch.Tensor:
    """Move buffered tread edges into the current yaw frame without losing landings."""

    compensated = features.clone()
    translation_x = ((positions_xy - current_position_xy.unsqueeze(1)) * current_heading_xy.unsqueeze(1)).sum(dim=-1)
    heading_alignment = (headings_xy * current_heading_xy.unsqueeze(1)).sum(dim=-1)
    tread_features = compensated[..., : max_treads * 6].reshape(*features.shape[:2], max_treads, 6)
    old_edges_m = tread_features[..., 0:2] * max_forward
    current_edges_m = translation_x[..., None, None] + heading_alignment[..., None, None] * old_edges_m
    valid = tread_features[..., 5] > 0.5
    valid &= current_edges_m[..., 1] >= -rear_limit
    valid &= current_edges_m[..., 0] <= max_forward
    tread_features[..., 0:2] = (current_edges_m / max_forward).clamp(-1.0, 1.0)
    tread_features[..., 5] = valid.to(tread_features.dtype)
    return compensated


def match_sole_to_treads(
    foot_near_x: torch.Tensor,
    foot_far_x: torch.Tensor,
    sole_bottom_z: torch.Tensor,
    treads: torch.Tensor,
    tread_world_z: torch.Tensor,
    edge_margin: float,
    min_support_overlap: float,
    height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match each sole to horizontal tread area at the same world height."""

    safe_near = treads[:, None, :, 0] + edge_margin
    safe_far = treads[:, None, :, 1] - edge_margin
    valid = treads[:, None, :, 5] > 0.5
    overlap = torch.minimum(foot_far_x, safe_far) - torch.maximum(foot_near_x, safe_near)
    height_error = (sole_bottom_z - tread_world_z[:, None, :]).abs()
    supported = valid & (overlap >= min_support_overlap) & (height_error <= height_tolerance)
    horizontal_score = (overlap.clamp_min(0.0) / max(min_support_overlap, 1.0e-6)).clamp_max(1.0)
    vertical_score = torch.exp(-torch.square(height_error / max(height_tolerance, 1.0e-6)))
    dense_score = torch.where(valid, horizontal_score * vertical_score, torch.zeros_like(overlap))
    return supported, overlap, height_error, dense_score


def centered_tread_support(
    foot_near_x: torch.Tensor,
    foot_far_x: torch.Tensor,
    treads: torch.Tensor,
    supported: torch.Tensor,
    center_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Require the projected sole center to stay inside the tread's central band."""

    sole_center = 0.5 * (foot_near_x + foot_far_x)
    tread_center = 0.5 * (treads[..., 0] + treads[..., 1]).unsqueeze(1)
    center_error = (sole_center - tread_center).abs()
    tolerance = (treads[..., 3] * (0.5 - center_fraction)).clamp_min(0.015).unsqueeze(1)
    return supported & (center_error <= tolerance), center_error, tolerance


def unsafe_tread_touchdown(
    first_contact: torch.Tensor,
    safe_contact: torch.Tensor,
    tread_valid: torch.Tensor,
    support_overlap: torch.Tensor,
    height_error: torch.Tensor,
    stair_mode: torch.Tensor,
    height_tolerance: float,
) -> torch.Tensor:
    """Identify a first contact close to a detected tread but outside safe support."""

    plausible_tread = (
        tread_valid & (support_overlap >= -0.03) & (height_error <= height_tolerance)
    ).any(dim=-1)
    return (
        first_contact
        & plausible_tread
        & ~safe_contact
        & (stair_mode != 0).unsqueeze(1)
    )


class StairGeometryExtractor:
    """Convert camera depth into stair edges, horizontal treads, and roughness.

    The conversion is deterministic: depth is deprojected with camera
    intrinsics, transformed into the robot yaw frame, collapsed into a robust
    forward height profile, then segmented at riser-sized discontinuities.
    Raw image pixels never enter the policy.
    """

    TREAD_FEATURES = 6
    GLOBAL_FEATURES = 4

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_bins = max(8, int(round((cfg.max_forward - cfg.min_forward) / cfg.profile_bin_size)))
        self.profile_x = torch.linspace(
            cfg.min_forward + 0.5 * cfg.profile_bin_size,
            cfg.max_forward - 0.5 * cfg.profile_bin_size,
            self.num_bins,
        )
        self.num_features = cfg.max_treads * self.TREAD_FEATURES + self.GLOBAL_FEATURES
        self._pixel_grid: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.surface_extractor = None
        self.surface_memories = None
        if getattr(cfg, "surface_validation_enabled", False):
            from .tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor

            self.surface_extractor = TreadSurfaceExtractor(SurfaceValidationCfg(
                min_forward=cfg.min_forward, max_forward=cfg.max_forward,
                lateral_half_width=cfg.lateral_half_width,
                edge_margin=getattr(cfg, "surface_edge_margin", 0.02),
                uncertainty_margin=getattr(cfg, "surface_uncertainty_margin", 0.01),
            ))

    def extract(
        self,
        depth_image: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w_ros: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        foot_positions_w: torch.Tensor | None = None,
        foot_quaternions_w: torch.Tensor | None = None,
        frame_timestamp: torch.Tensor | None = None,
    ) -> StairGeometryResult:
        """Extract geometry for a batch of depth images."""

        if depth_image.ndim == 4:
            depth_image = depth_image[..., 0]
        if depth_image.ndim != 3:
            raise ValueError(f"Expected depth shape [N,H,W] or [N,H,W,1], got {tuple(depth_image.shape)}")

        batch_size, image_height, image_width = depth_image.shape
        process_height = min(self.cfg.processing_height, image_height)
        process_width = min(self.cfg.processing_width, image_width)
        depth = F.interpolate(
            depth_image.unsqueeze(1), size=(process_height, process_width), mode="nearest"
        ).squeeze(1)
        depth_valid = torch.isfinite(depth) & (depth >= self.cfg.min_forward) & (depth <= self.cfg.max_forward * 2.0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        u, v = self._get_pixel_grid(process_height, process_width, depth.device, depth.dtype)
        scale_x = process_width / image_width
        scale_y = process_height / image_height
        fx = intrinsic_matrices[:, 0, 0].view(-1, 1, 1) * scale_x
        fy = intrinsic_matrices[:, 1, 1].view(-1, 1, 1) * scale_y
        cx = intrinsic_matrices[:, 0, 2].view(-1, 1, 1) * scale_x
        cy = intrinsic_matrices[:, 1, 2].view(-1, 1, 1) * scale_y

        # ROS optical frame: +x right, +y down, +z forward.
        points_camera = torch.stack(
            ((u - cx) * depth / fx, (v - cy) * depth / fy, depth), dim=-1
        ).reshape(batch_size, -1, 3)
        rotation_camera_to_world = _matrix_from_quat(camera_quat_w_ros)
        points_world = torch.einsum("bij,bpj->bpi", rotation_camera_to_world, points_camera)
        points_world = points_world + camera_pos_w.unsqueeze(1)

        rotation_body_to_world = _yaw_matrix(root_quat_w)
        points_body = torch.einsum(
            "bji,bpj->bpi", rotation_body_to_world, points_world - root_pos_w.unsqueeze(1)
        )

        sensor_valid = depth_valid.reshape(batch_size, -1).clone()
        point_valid = sensor_valid.clone()
        point_valid &= points_body[..., 0] >= self.cfg.min_forward
        point_valid &= points_body[..., 0] < self.cfg.max_forward
        point_valid &= points_body[..., 1].abs() <= self.cfg.lateral_half_width
        point_valid &= points_body[..., 2] >= self.cfg.min_height_from_root
        point_valid &= points_body[..., 2] <= self.cfg.max_height_from_root

        profile_z, profile_valid = self._build_height_profile(points_body, point_valid)
        result = self.extract_from_profile(profile_z, profile_valid)
        lower = torch.where(sensor_valid.unsqueeze(-1), points_body, torch.inf).amin(dim=1)
        upper = torch.where(sensor_valid.unsqueeze(-1), points_body, -torch.inf).amax(dim=1)
        result.point_bounds = torch.cat((lower, upper), dim=1)
        result.valid_point_count = point_valid.sum(dim=1)
        if self.surface_extractor is not None:
            # Sample the original image independently; never upsample the 1-D policy profile.
            surface_body, surface_valid, image_shape = project_surface_points(
                depth_image, intrinsic_matrices, camera_pos_w, camera_quat_w_ros,
                root_pos_w, root_quat_w, self.cfg, foot_positions_w, foot_quaternions_w,
            )
            points_cpu = surface_body.detach().cpu().numpy()
            valid_cpu = surface_valid.detach().cpu().numpy()
            result.surface_geometry = [self.surface_extractor.extract(points_cpu[i], valid_cpu[i])
                                       for i in range(batch_size)]
            result.processed_image_shape = image_shape
            if getattr(self.cfg, "surface_memory_enabled", False):
                from .surface_memory import SurfaceMemory
                if frame_timestamp is None:
                    raise ValueError("Surface memory requires camera acquisition timestamps.")
                if self.surface_memories is None or len(self.surface_memories) != batch_size:
                    self.surface_memories = [SurfaceMemory(self.surface_extractor.cfg) for _ in range(batch_size)]
                poses = root_pos_w.detach().cpu().numpy()
                rotations = rotation_body_to_world.detach().cpu().numpy()
                times = frame_timestamp.detach().cpu().numpy()
                result.surface_memory = [self.surface_memories[i].update(
                    points_cpu[i], result.surface_geometry[i], poses[i], rotations[i], times[i], valid_cpu[i])
                    for i in range(batch_size)]
        return result

    def reset_surface_memory(self, env_ids):
        if self.surface_memories is not None:
            for index in env_ids:
                self.surface_memories[int(index)].reset()

    def extract_from_profile(
        self, profile_z: torch.Tensor, profile_valid: torch.Tensor | None = None
    ) -> StairGeometryResult:
        """Segment an existing forward height profile, useful for tests and diagnostics."""

        if profile_z.ndim == 1:
            profile_z = profile_z.unsqueeze(0)
        if profile_valid is None:
            profile_valid = torch.isfinite(profile_z)
        elif profile_valid.ndim == 1:
            profile_valid = profile_valid.unsqueeze(0)
        profile_valid = profile_valid.bool() & torch.isfinite(profile_z)
        profile_z = torch.nan_to_num(profile_z, nan=0.0, posinf=0.0, neginf=0.0)

        profile_z, profile_valid = self._fill_small_gaps(profile_z, profile_valid)
        profile_smooth = self._median_filter(profile_z)
        profile_smooth = torch.where(profile_valid, profile_smooth, profile_z)

        profile_x = self.profile_x.to(device=profile_z.device, dtype=profile_z.dtype)
        reference_limit = min(self.cfg.min_forward + 0.25, self.cfg.max_forward)
        reference_mask = profile_valid & (profile_x.unsqueeze(0) <= reference_limit)
        reference_count = reference_mask.sum(dim=1).clamp_min(1)
        reference_height = (profile_smooth * reference_mask).sum(dim=1) / reference_count
        profile_relative = profile_smooth - reference_height.unsqueeze(1)

        treads, direction, stair_confidence = self._segment_treads(profile_relative, profile_valid)
        if self.cfg.descending_edge_offset_m:
            descending = (direction < 0).view(-1, 1, 1)
            valid_treads = treads[..., 5:6] > 0.5
            edge_offset = torch.where(
                descending & valid_treads,
                self.cfg.descending_edge_offset_m,
                0.0,
            )
            treads[..., :2] += edge_offset
        roughness = self._estimate_roughness(profile_relative, profile_valid)
        slope = self._estimate_slope(profile_relative, profile_valid)
        features = self._make_policy_features(treads, direction, stair_confidence, roughness, slope)

        return StairGeometryResult(
            treads=treads,
            direction=direction,
            stair_confidence=stair_confidence,
            roughness=roughness,
            slope=slope,
            reference_height=reference_height,
            profile_x=profile_x,
            profile_z=profile_relative,
            profile_valid=profile_valid,
            features=features,
        )

    def _get_pixel_grid(self, height: int, width: int, device, dtype):
        key = (height, width, str(device), dtype)
        if key not in self._pixel_grid:
            v, u = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )
            self._pixel_grid[key] = (u.unsqueeze(0), v.unsqueeze(0))
        return self._pixel_grid[key]

    def _build_height_profile(self, points_body: torch.Tensor, point_valid: torch.Tensor):
        batch_size = points_body.shape[0]
        bin_index = torch.floor(
            (points_body[..., 0] - self.cfg.min_forward) / self.cfg.profile_bin_size
        ).long()
        bin_index = bin_index.clamp(0, self.num_bins - 1)
        batch_offset = torch.arange(batch_size, device=points_body.device).unsqueeze(1) * self.num_bins
        flat_index = (bin_index + batch_offset).reshape(-1)

        source_height = torch.where(
            point_valid, points_body[..., 2], torch.full_like(points_body[..., 2], -torch.inf)
        ).reshape(-1)
        profile = torch.full(
            (batch_size * self.num_bins,), -torch.inf, device=points_body.device, dtype=points_body.dtype
        )
        profile.scatter_reduce_(0, flat_index, source_height, reduce="amax", include_self=True)

        counts = torch.zeros(batch_size * self.num_bins, device=points_body.device, dtype=torch.int32)
        counts.scatter_add_(0, flat_index, point_valid.reshape(-1).to(torch.int32))
        profile_valid = counts.reshape(batch_size, self.num_bins) >= 2
        profile = profile.reshape(batch_size, self.num_bins)
        profile = torch.where(profile_valid, profile, torch.zeros_like(profile))
        return profile, profile_valid

    @staticmethod
    def _median_filter(profile: torch.Tensor) -> torch.Tensor:
        padded = F.pad(profile.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
        return padded.unfold(1, 3, 1).median(dim=-1).values

    @staticmethod
    def _fill_small_gaps(profile: torch.Tensor, valid: torch.Tensor):
        filled = profile.clone()
        filled_valid = valid.clone()
        for _ in range(2):
            left_valid = F.pad(filled_valid[:, :-1], (1, 0), value=False)
            right_valid = F.pad(filled_valid[:, 1:], (0, 1), value=False)
            left_value = F.pad(filled[:, :-1], (1, 0))
            right_value = F.pad(filled[:, 1:], (0, 1))
            can_fill = (~filled_valid) & left_valid & right_valid
            filled = torch.where(can_fill, 0.5 * (left_value + right_value), filled)
            filled_valid |= can_fill
        return filled, filled_valid

    def _segment_treads(self, profile: torch.Tensor, valid: torch.Tensor):
        batch_size, num_bins = profile.shape
        span = max(1, int(round(0.06 / self.cfg.profile_bin_size)))
        center_offset = span // 2
        transition = profile[:, span:] - profile[:, :-span]
        transition_valid = valid[:, span:] & valid[:, :-span]

        edge_strength = torch.zeros_like(profile)
        edge_signed = torch.zeros_like(profile)
        target = slice(center_offset, center_offset + transition.shape[1])
        edge_strength[:, target] = torch.where(transition_valid, transition.abs(), 0.0)
        edge_signed[:, target] = torch.where(transition_valid, transition, 0.0)

        nms_width = max(3, int(round(0.10 / self.cfg.profile_bin_size)))
        if nms_width % 2 == 0:
            nms_width += 1
        local_max = F.max_pool1d(
            edge_strength.unsqueeze(1), kernel_size=nms_width, stride=1, padding=nms_width // 2
        ).squeeze(1)
        edge_mask = edge_strength >= self.cfg.min_riser_height
        edge_mask &= edge_strength <= self.cfg.max_riser_height
        edge_mask &= edge_strength >= local_max - 1.0e-6
        # A sharp riser can create a two-bin plateau in the span difference.
        # Keep only its first maximum so it is not mistaken for a tiny tread.
        previous_strength = F.pad(edge_strength[:, :-1], (1, 0))
        edge_mask &= edge_strength > previous_strength + 1.0e-6

        bin_ids = torch.arange(num_bins, device=profile.device).unsqueeze(0).expand(batch_size, -1)
        ordered_edges = torch.where(edge_mask, bin_ids, num_bins).sort(dim=1).values
        edge_count = self.cfg.max_treads + 1
        if ordered_edges.shape[1] < edge_count:
            ordered_edges = F.pad(ordered_edges, (0, edge_count - ordered_edges.shape[1]), value=num_bins)
        ordered_edges = ordered_edges[:, :edge_count]

        near_index = ordered_edges[:, :-1]
        far_index = ordered_edges[:, 1:]
        pair_valid = (near_index < num_bins) & (far_index < num_bins)
        safe_near = near_index.clamp(0, num_bins - 1)
        safe_far = far_index.clamp(0, num_bins - 1)

        x = self.profile_x.to(device=profile.device, dtype=profile.dtype)
        near_x = x[safe_near]
        far_x = x[safe_far]
        tread_depth = far_x - near_x

        prefix_count = torch.cat(
            (torch.zeros(batch_size, 1, device=profile.device), valid.float().cumsum(dim=1)), dim=1
        )
        prefix_sum = torch.cat(
            (torch.zeros(batch_size, 1, device=profile.device), (profile * valid).cumsum(dim=1)), dim=1
        )
        prefix_sq = torch.cat(
            (torch.zeros(batch_size, 1, device=profile.device), (profile.square() * valid).cumsum(dim=1)),
            dim=1,
        )
        segment_start = (safe_near + 1).clamp(max=num_bins)
        segment_end = safe_far.clamp(max=num_bins)
        count = prefix_count.gather(1, segment_end) - prefix_count.gather(1, segment_start)
        height_sum = prefix_sum.gather(1, segment_end) - prefix_sum.gather(1, segment_start)
        height_sq_sum = prefix_sq.gather(1, segment_end) - prefix_sq.gather(1, segment_start)
        tread_height = height_sum / count.clamp_min(1.0)
        tread_variance = (height_sq_sum / count.clamp_min(1.0) - tread_height.square()).clamp_min(0.0)
        tread_std = tread_variance.sqrt()

        near_delta = edge_signed.gather(1, safe_near)
        far_delta = edge_signed.gather(1, safe_far)
        same_direction = near_delta * far_delta > 0.0
        pair_valid &= same_direction
        pair_valid &= tread_depth >= self.cfg.min_tread_depth
        pair_valid &= tread_depth <= self.cfg.max_tread_depth
        pair_valid &= tread_std <= self.cfg.max_tread_height_std
        pair_valid &= count >= 2.0

        riser_score = (
            torch.minimum(near_delta.abs(), far_delta.abs()) / max(self.cfg.min_riser_height, 1.0e-6)
        ).clamp(0.0, 1.0)
        width_center = 0.5 * (self.cfg.min_tread_depth + self.cfg.max_tread_depth)
        width_radius = 0.5 * (self.cfg.max_tread_depth - self.cfg.min_tread_depth)
        width_score = (1.0 - (tread_depth - width_center).abs() / max(width_radius, 1.0e-6)).clamp(0.0, 1.0)
        flat_score = (1.0 - tread_std / max(self.cfg.max_tread_height_std, 1.0e-6)).clamp(0.0, 1.0)
        confidence = (0.4 * riser_score + 0.3 * width_score + 0.3 * flat_score) * pair_valid
        pair_valid &= confidence >= self.cfg.min_tread_confidence
        confidence *= pair_valid

        zeros = torch.zeros_like(near_x)
        treads = torch.stack(
            (
                torch.where(pair_valid, near_x, zeros),
                torch.where(pair_valid, far_x, zeros),
                torch.where(pair_valid, tread_height, zeros),
                torch.where(pair_valid, tread_depth, zeros),
                confidence,
                pair_valid.float(),
            ),
            dim=-1,
        )
        stair_confidence = confidence.max(dim=1).values
        signed_sum = (edge_signed * edge_mask).sum(dim=1)
        direction = torch.sign(signed_sum) * (stair_confidence >= self.cfg.min_tread_confidence)
        return treads, direction, stair_confidence

    def _estimate_roughness(self, profile: torch.Tensor, valid: torch.Tensor):
        delta = (profile[:, 1:] - profile[:, :-1]).abs()
        delta_valid = valid[:, 1:] & valid[:, :-1]
        non_riser = delta_valid & (delta < self.cfg.min_riser_height)
        return (delta * non_riser).sum(dim=1) / non_riser.sum(dim=1).clamp_min(1)

    def _estimate_slope(self, profile: torch.Tensor, valid: torch.Tensor):
        x = self.profile_x.to(device=profile.device, dtype=profile.dtype).unsqueeze(0)
        count = valid.sum(dim=1).clamp_min(1)
        mean_x = (x * valid).sum(dim=1) / count
        mean_z = (profile * valid).sum(dim=1) / count
        x_centered = x - mean_x.unsqueeze(1)
        z_centered = profile - mean_z.unsqueeze(1)
        numerator = (x_centered * z_centered * valid).sum(dim=1)
        denominator = (x_centered.square() * valid).sum(dim=1).clamp_min(1.0e-6)
        return numerator / denominator

    def _make_policy_features(self, treads, direction, stair_confidence, roughness, slope):
        normalized = treads.clone()
        normalized[..., 0:2] /= self.cfg.max_forward
        normalized[..., 2] /= 0.5
        normalized[..., 3] /= 0.3
        global_features = torch.stack(
            (
                direction,
                stair_confidence,
                (roughness / self.cfg.roughness_reference).clamp(0.0, 2.0),
                slope.clamp(-1.0, 1.0),
            ),
            dim=1,
        )
        return torch.cat((normalized.flatten(1), global_features), dim=1)


def draw_stair_geometry_debug(
    depth_image,
    result: StairGeometryResult,
    env_index: int,
    min_range: float,
    max_range: float,
    stair_mode=None,
    feet_x=None,
    feet_z=None,
    foot_contact=None,
    foot_confirmed=None,
    safe_edge_margin: float = 0.05,
):
    """Render a depth image and its extracted metric height profile."""

    import cv2
    import numpy as np

    depth = depth_image.detach().float().cpu().numpy() if torch.is_tensor(depth_image) else np.asarray(depth_image)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=max_range, posinf=max_range, neginf=min_range)
    depth = np.clip(depth, min_range, max_range)
    depth_norm = (depth - min_range) / max(max_range - min_range, 1.0e-6)
    depth_gray = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)

    width = max(depth_color.shape[1], 960)
    depth_height = max(1, int(depth_color.shape[0] * width / depth_color.shape[1]))
    depth_color = cv2.resize(depth_color, (width, depth_height), interpolation=cv2.INTER_NEAREST)
    panel_height = 340
    canvas = np.full((depth_height + panel_height, width, 3), 24, dtype=np.uint8)
    canvas[:depth_height] = depth_color

    x = result.profile_x.detach().cpu().numpy()
    z = result.profile_z[env_index].detach().cpu().numpy()
    valid = result.profile_valid[env_index].detach().cpu().numpy().astype(bool)
    treads = result.treads[env_index].detach().cpu().numpy()
    direction = float(result.direction[env_index].item())
    confidence = float(result.stair_confidence[env_index].item())
    roughness = float(result.roughness[env_index].item())
    slope = float(result.slope[env_index].item())

    left, right = 70, width - 35
    top, bottom = depth_height + 55, depth_height + panel_height - 45
    z_min = min(-0.35, float(z[valid].min()) - 0.05) if valid.any() else -0.35
    z_max = max(0.55, float(z[valid].max()) + 0.05) if valid.any() else 0.55

    display_min_x = -0.60 if feet_x is not None else float(x[0])

    def map_x(value):
        return int(left + (value - display_min_x) / max(x[-1] - display_min_x, 1.0e-6) * (right - left))

    def map_z(value):
        return int(bottom - (value - z_min) / max(z_max - z_min, 1.0e-6) * (bottom - top))

    cv2.rectangle(canvas, (left, top), (right, bottom), (70, 70, 70), 1)
    if feet_x is not None:
        cv2.line(canvas, (map_x(x[0]), top), (map_x(x[0]), bottom), (55, 55, 55), 1)
    cv2.line(canvas, (left, map_z(0.0)), (right, map_z(0.0)), (80, 80, 80), 1)
    for i in range(len(x) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(canvas, (map_x(x[i]), map_z(z[i])), (map_x(x[i + 1]), map_z(z[i + 1])), (230, 230, 230), 2)

    colors = ((0, 220, 255), (255, 130, 20), (80, 240, 80), (230, 80, 230))
    for tread_index, tread in enumerate(treads):
        near_x, far_x, height, tread_depth, tread_confidence, tread_valid = tread
        if tread_valid < 0.5:
            continue
        color = colors[tread_index % len(colors)]
        near_px, far_px, height_px = map_x(near_x), map_x(far_x), map_z(height)
        cv2.line(canvas, (near_px, top), (near_px, bottom), color, 1)
        cv2.line(canvas, (far_px, top), (far_px, bottom), color, 1)
        cv2.line(canvas, (near_px, height_px), (far_px, height_px), color, 5)
        safe_near = map_x(near_x + safe_edge_margin)
        safe_far = map_x(far_x - safe_edge_margin)
        if safe_far > safe_near:
            cv2.line(canvas, (safe_near, height_px - 8), (safe_far, height_px - 8), (70, 255, 70), 3)
        label = f"T{tread_index + 1}: {near_x:.2f}-{far_x:.2f}m  d={tread_depth:.2f}  c={tread_confidence:.2f}"
        cv2.putText(canvas, label, (max(left, near_px), max(top + 18, height_px - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    if feet_x is not None and feet_z is not None:
        foot_x_values = np.asarray(feet_x)
        foot_z_values = np.asarray(feet_z)
        contact_values = np.asarray(foot_contact) if foot_contact is not None else np.zeros(2, dtype=bool)
        confirmed_values = np.asarray(foot_confirmed) if foot_confirmed is not None else np.zeros(2, dtype=bool)
        for foot_index, (foot_x, foot_z) in enumerate(zip(foot_x_values, foot_z_values)):
            point = (
                int(np.clip(map_x(foot_x), left + 7, right - 7)),
                int(np.clip(map_z(foot_z), top + 7, bottom - 7)),
            )
            color = (70, 255, 70) if confirmed_values[foot_index] else (
                (0, 190, 255) if contact_values[foot_index] else (190, 190, 190)
            )
            cv2.circle(canvas, point, 7, color, -1)
            label_y = point[1] - 9 if foot_index == 0 else point[1] + 17
            cv2.putText(canvas, "LR"[foot_index], (point[0] + 9, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    terrain_class = "STAIRS_UP" if direction > 0 else "STAIRS_DOWN" if direction < 0 else "ROUGH" if roughness > 0.015 else "FLAT/UNKNOWN"
    if stair_mode is None:
        policy_mode = "N/A"
    else:
        if torch.is_tensor(stair_mode):
            mode_value = int(stair_mode[env_index].item()) if stair_mode.ndim else int(stair_mode.item())
        else:
            mode_value = int(stair_mode)
        policy_mode = "STAIRS_UP" if mode_value > 0 else "STAIRS_DOWN" if mode_value < 0 else "BLIND"
    cv2.rectangle(canvas, (0, 0), (width, 68), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        f"detector={terrain_class}  policy={policy_mode}  conf={confidence:.2f}",
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    point_count = int(result.valid_point_count[env_index].item()) if result.valid_point_count is not None else 0
    cv2.putText(
        canvas,
        f"roughness={roughness:.3f}m  slope={slope:.2f}  ROI points={point_count}",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "robot-forward distance (m)", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
    if feet_x is not None:
        cv2.putText(canvas, "feet: gray=swing  amber=contact  green=stable tread", (left + 240, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1)
    return canvas
