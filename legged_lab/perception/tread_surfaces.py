"""Observed 2-D support surfaces; diagnostic CPU path, independent of policy inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


def depth_horizontal_mask(depth, intrinsic, camera_rotation, max_tilt_deg=45.0,
                          window_size=7, radius=4):
    """Estimate local normals from inverse-depth planes; classification never fills depth."""
    depth = np.asarray(depth, dtype=np.float32)
    if window_size < 3 or window_size % 2 != 1 or radius < 1:
        raise ValueError("Local normal smoothing must use an odd window >= 3 and a positive radius.")
    observed = np.isfinite(depth) & (depth > 0)
    inverse = np.divide(1.0, depth, out=np.zeros_like(depth), where=observed)
    # Inverse depth is affine in image coordinates on a perspective plane.
    counts = cv2.boxFilter(observed.astype(np.float32), -1, (window_size, window_size), borderType=cv2.BORDER_CONSTANT)
    mean = cv2.boxFilter(inverse, -1, (window_size, window_size), borderType=cv2.BORDER_CONSTANT) / np.maximum(counts, 1.0e-6)
    du, dv = np.zeros_like(mean), np.zeros_like(mean)
    du[:, radius:-radius] = (mean[:, 2*radius:] - mean[:, :-2*radius]) / (2*radius)
    dv[radius:-radius] = (mean[2*radius:] - mean[:-2*radius]) / (2*radius)
    v, u = np.indices(depth.shape)
    normal = np.stack((intrinsic[0, 0] * du, intrinsic[1, 1] * dv,
                       mean - (u-intrinsic[0, 2]) * du - (v-intrinsic[1, 2]) * dv), axis=-1)
    normal_world = normal @ camera_rotation.T
    length = np.linalg.norm(normal_world, axis=-1)
    horizontal = np.abs(normal_world[..., 2]) >= math.cos(math.radians(max_tilt_deg)) * length
    horizontal &= observed & (counts >= 0.9) & (length > 1.0e-6)
    horizontal[:radius] = horizontal[-radius:] = False
    horizontal[:, :radius] = horizontal[:, -radius:] = False
    return horizontal


def footprint_cells(offset_xy, yaw, cfg, margins):
    """Rectangle/cell intersection using all four separating axes, excluding zero-area touch."""
    rear, front, side = margins
    half_length = 0.5 * (cfg.foot_rear_extent + cfg.foot_front_extent)
    center_offset = 0.5 * (front - rear)
    length = half_length + 0.5 * (rear + front)
    width = cfg.foot_half_width + side
    cosine, sine = math.cos(yaw), math.sin(yaw)
    shift = np.array([cosine, sine]) * center_offset
    delta = np.asarray(offset_xy) - shift
    half_cell = 0.5 * cfg.grid_size
    cell_projection = half_cell * (abs(cosine) + abs(sine))
    forward = delta[..., 0] * cosine + delta[..., 1] * sine
    lateral = -delta[..., 0] * sine + delta[..., 1] * cosine
    return ((np.abs(forward) < length + cell_projection - 1.0e-8)
            & (np.abs(lateral) < width + cell_projection - 1.0e-8)
            & (np.abs(delta[..., 0]) < abs(cosine)*length + abs(sine)*width + half_cell - 1.0e-8)
            & (np.abs(delta[..., 1]) < abs(sine)*length + abs(cosine)*width + half_cell - 1.0e-8))


@dataclass
class SurfaceValidationCfg:
    min_forward: float = 0.15
    max_forward: float = 2.0
    lateral_half_width: float = 0.35
    grid_size: float = 0.02
    min_cell_points: int = 2
    max_cell_std: float = 0.025
    height_band: float = 0.045
    min_surface_cells: int = 16
    max_surfaces: int = 8
    max_tilt_deg: float = 5.0
    max_plane_rms: float = 0.015
    min_riser_height: float = 0.08
    max_riser_height: float = 0.21
    max_edge_gap: float = 0.16
    edge_margin: float = 0.02
    other_end_margin: float = 0.01
    lateral_margin: float = 0.01
    uncertainty_margin: float = 0.01
    target_tracking_margin: float = 0.005
    # Conservative bounds of the ELF3 URDF ankle collision mesh, in metres.
    foot_rear_extent: float = 0.09
    foot_front_extent: float = 0.15
    foot_half_width: float = 0.042


@dataclass
class SurfaceEdge:
    x_at_y_zero: float
    dx_dy: float
    uncertainty: float
    observed_y_range: tuple[float, float]

    def x_at(self, y):
        return self.x_at_y_zero + self.dx_dy * y


@dataclass
class TreadSurface:
    surface_id: int
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    observed_bounds: np.ndarray
    tilt_deg: float
    rms: float
    observed_mask: np.ndarray
    valid: bool
    rejection: str
    kind: str = "partial"
    near_edge: SurfaceEdge | None = None
    far_edge: SurfaceEdge | None = None
    safe_center_mask: np.ndarray | None = None
    safe_center_offsets: np.ndarray | None = None
    track_id: int | None = None
    last_observed_time: float | None = None

    def height_at(self, xy):
        xy = np.asarray(xy)
        return -(xy @ self.normal[:2] + self.offset) / max(float(self.normal[2]), 1.0e-8)


@dataclass
class SurfaceGeometryResult:
    cfg: SurfaceValidationCfg
    grid_xy: np.ndarray
    grid_height: np.ndarray
    grid_observed: np.ndarray
    grid_labels: np.ndarray
    point_labels: np.ndarray
    surfaces: list[TreadSurface] = field(default_factory=list)
    heading_rad: float | None = None
    direction: int = 0
    memory_frame_count: int = 0
    timestamp_s: float | None = None

    def candidate_centers(self, surface_id):
        surface = self.surfaces[surface_id]
        offsets = 0 if surface.safe_center_offsets is None else surface.safe_center_offsets[surface.safe_center_mask]
        return self.grid_xy[surface.safe_center_mask] + offsets

    def support_margins(self):
        rear = self.cfg.edge_margin if self.direction >= 0 else self.cfg.other_end_margin
        front = self.cfg.edge_margin if self.direction <= 0 else self.cfg.other_end_margin
        uncertainty = self.cfg.uncertainty_margin
        return rear + uncertainty, front + uncertainty, self.cfg.lateral_margin + uncertainty

    def footprint_supported(self, surface_id: int, sole_center_xy, yaw: float = 0.0,
                            include_margin: bool = True) -> bool:
        surface = next((s for s in self.surfaces if s.surface_id == surface_id), None)
        if surface is None or not surface.valid:
            return False
        rear, front, side = self.support_margins() if include_margin else (0.0, 0.0, 0.0)
        half_length = 0.5 * (self.cfg.foot_rear_extent + self.cfg.foot_front_extent)
        width = self.cfg.foot_half_width + side
        polygon = np.array([[-half_length-rear, -width], [half_length+front, -width],
                            [half_length+front, width], [-half_length-rear, width]])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                             [math.sin(yaw), math.cos(yaw)]])
        polygon = polygon @ rotation.T + np.asarray(sole_center_xy)
        if (polygon[:, 0].min() < self.cfg.min_forward or polygon[:, 0].max() > self.cfg.max_forward
                or np.abs(polygon[:, 1]).max() > self.cfg.lateral_half_width):
            return False
        footprint = footprint_cells(self.grid_xy - np.asarray(sole_center_xy), yaw, self.cfg, (rear, front, side))
        return bool(footprint.any() and surface.observed_mask[footprint].all())


class TreadSurfaceExtractor:
    """Validate connected observed patches without filling unknown grid cells."""

    def __init__(self, cfg: SurfaceValidationCfg):
        self.cfg = cfg
        nx = math.ceil((cfg.max_forward - cfg.min_forward) / cfg.grid_size)
        ny = math.ceil(2.0 * cfg.lateral_half_width / cfg.grid_size)
        x = cfg.min_forward + (np.arange(nx) + 0.5) * cfg.grid_size
        y = -cfg.lateral_half_width + (np.arange(ny) + 0.5) * cfg.grid_size
        self.grid_xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)

    @staticmethod
    def _fit_plane(points):
        active = np.ones(len(points), dtype=bool)
        for _ in range(3):
            subset = points[active]
            if len(subset) < 6:
                return None
            centroid = subset.mean(axis=0)
            delta = subset - centroid
            eigenvalues, vectors = np.linalg.eigh(delta.T @ delta / len(subset))
            if eigenvalues[1] < 1.0e-6:
                return None
            normal = vectors[:, 0]
            if normal[2] < 0:
                normal = -normal
            offset = -float(normal @ centroid)
            distance = np.abs(points @ normal + offset)
            threshold = max(0.008, 3.0 * float(np.median(distance[active])))
            next_active = distance <= min(threshold, 0.03)
            if np.array_equal(active, next_active):
                break
            active = next_active
        # Refit after the final trim so all returned plane fields agree.
        subset = points[active]
        if len(subset) < 6:
            return None
        centroid = subset.mean(axis=0)
        delta = subset - centroid
        eigenvalues, vectors = np.linalg.eigh(delta.T @ delta / len(subset))
        if eigenvalues[1] < 1.0e-6:
            return None
        normal = vectors[:, 0]
        normal *= 1 if normal[2] >= 0 else -1
        offset = -float(normal @ centroid)
        rms = float(np.sqrt(np.mean((subset @ normal + offset) ** 2)))
        return normal, offset, centroid, rms, active

    def extract(self, points, point_valid=None) -> SurfaceGeometryResult:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Surface points must have shape [P,3].")
        valid = np.isfinite(points).all(axis=1)
        if point_valid is not None:
            valid &= np.asarray(point_valid, dtype=bool)
        valid &= (points[:, 0] >= self.cfg.min_forward) & (points[:, 0] < self.cfg.max_forward)
        valid &= np.abs(points[:, 1]) < self.cfg.lateral_half_width
        nx, ny = self.grid_xy.shape[:2]
        cell_ids = np.full(len(points), -1, dtype=np.int64)
        indices = np.floor((points[valid, :2] - [self.cfg.min_forward, -self.cfg.lateral_half_width])
                           / self.cfg.grid_size).astype(np.int64)
        cell_ids[valid] = indices[:, 0] * ny + indices[:, 1]
        ids = cell_ids[valid]
        count = np.bincount(ids, minlength=nx * ny)
        sum_xyz = np.stack([np.bincount(ids, weights=points[valid, axis], minlength=nx * ny)
                            for axis in range(3)], axis=-1)
        means = sum_xyz / np.maximum(count[:, None], 1)
        sum_z2 = np.bincount(ids, weights=points[valid, 2] ** 2, minlength=nx * ny)
        variance = np.maximum(0, sum_z2 / np.maximum(count, 1) - means[:, 2] ** 2)
        observed = ((count >= self.cfg.min_cell_points)
                    & (variance <= self.cfg.max_cell_std ** 2)).reshape(nx, ny)
        observed &= self.grid_xy[..., 0] + 0.5*self.cfg.grid_size <= self.cfg.max_forward + 1.0e-8
        mean_grid = means.reshape(nx, ny, 3)
        result = SurfaceGeometryResult(
            cfg=self.cfg, grid_xy=self.grid_xy.copy(), grid_height=mean_grid[..., 2],
            grid_observed=observed, grid_labels=np.full((nx, ny), -1, dtype=np.int32),
            point_labels=np.full(len(points), -1, dtype=np.int32),
        )
        heights = mean_grid[..., 2][observed]
        if not len(heights):
            return result
        bin_size = 0.02
        lower = math.floor(float(heights.min()) / bin_size) * bin_size
        bins = np.floor((heights - lower) / bin_size).astype(np.int64)
        hist = np.bincount(bins)
        scores = np.convolve(hist, [1, 2, 3, 2, 1], mode="full")[2:2 + len(hist)].astype(float)
        claimed = np.zeros((nx, ny), dtype=bool)
        for _ in range(self.cfg.max_surfaces):
            peak = int(scores.argmax())
            if scores[peak] == 0:
                break
            seed_height = lower + (peak + 0.5) * bin_size
            suppress = max(1, round(self.cfg.min_riser_height / bin_size))
            scores[max(0, peak - suppress):peak + suppress + 1] = 0
            candidate = observed & ~claimed & (np.abs(mean_grid[..., 2] - seed_height) <= self.cfg.height_band)
            _, components, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=4)
            for component in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1:
                if len(result.surfaces) >= self.cfg.max_surfaces:
                    break
                mask = components == component
                if int(mask.sum()) < self.cfg.min_surface_cells:
                    continue
                fit = self._fit_plane(mean_grid[mask])
                if fit is None:
                    continue
                normal, offset, centroid, rms, inliers = fit
                fitted_mask = np.zeros_like(mask)
                coordinates = np.argwhere(mask)[inliers]
                fitted_mask[coordinates[:, 0], coordinates[:, 1]] = True
                if int(fitted_mask.sum()) < self.cfg.min_surface_cells:
                    continue
                tilt = math.degrees(math.acos(float(np.clip(normal[2], -1, 1))))
                rejection = "tilt" if tilt > self.cfg.max_tilt_deg else "rms" if rms > self.cfg.max_plane_rms else ""
                point_members = valid.copy()
                point_members[valid] &= fitted_mask.ravel()[ids]
                point_members &= np.abs(points @ normal + offset) <= max(0.02, 3 * rms)
                observed_points = points[point_members]
                if not len(observed_points):
                    continue
                bounds = np.stack((observed_points[:, :2].min(axis=0), observed_points[:, :2].max(axis=0)))
                surface = TreadSurface(len(result.surfaces), normal, offset, centroid, bounds,
                                       tilt, rms, fitted_mask, not rejection, rejection)
                if bounds[1, 0] - bounds[0, 0] > 0.38:
                    surface.kind = "platform_candidate"
                result.surfaces.append(surface)
                claimed |= fitted_mask
                if surface.valid:
                    result.grid_labels[fitted_mask] = surface.surface_id
                    result.point_labels[point_members] = surface.surface_id
        self._find_edges(result, mean_grid)
        for surface in result.surfaces:
            if surface.near_edge is not None and surface.far_edge is not None:
                depth = surface.far_edge.x_at_y_zero - surface.near_edge.x_at_y_zero
                if 0.16 <= depth <= 0.36:
                    surface.kind = "tread"
            surface.safe_center_mask, surface.safe_center_offsets = self._safe_centers(
                surface, result.heading_rad or 0.0, result.support_margins())
        return result

    def _find_edges(self, result, means):
        pairs = {}
        labels = result.grid_labels
        for column in range(labels.shape[1]):
            rows = np.flatnonzero(labels[:, column] >= 0)
            for first, second in zip(rows[:-1], rows[1:]):
                a, b = int(labels[first, column]), int(labels[second, column])
                if a == b:
                    continue
                p, q = means[first, column], means[second, column]
                height_delta = float(q[2] - p[2])
                if (q[0] - p[0] > self.cfg.max_edge_gap
                        or not self.cfg.min_riser_height <= abs(height_delta) <= self.cfg.max_riser_height):
                    continue
                # A descending lower tread's near boundary may be hidden by the upper edge.
                edge_x = p[0] + 0.5 * self.cfg.grid_size if height_delta < 0 else 0.5 * (p[0] + q[0])
                pairs.setdefault((a, b), []).append((edge_x,
                                                    0.5 * (p[1] + q[1]),
                                                    q[0] - p[0], np.sign(height_delta)))
        headings, weights, directions = [], [], []
        for (a, b), samples in pairs.items():
            samples = np.asarray(samples)
            if len(samples) < 4 or np.ptp(samples[:, 1]) < 0.08:
                continue
            design = np.column_stack((samples[:, 1], np.ones(len(samples))))
            slope, intercept = np.linalg.lstsq(design, samples[:, 0], rcond=None)[0]
            residual = samples[:, 0] - design @ [slope, intercept]
            uncertainty = float(np.max(samples[:, 2]) + np.max(np.abs(residual)))
            edge = SurfaceEdge(float(intercept), float(slope), uncertainty,
                               (float(samples[:, 1].min()), float(samples[:, 1].max())))
            result.surfaces[a].far_edge = edge
            result.surfaces[b].near_edge = edge
            headings.append(math.atan2(-float(slope), 1.0))
            weights.append(len(samples))
            directions.append(float(np.median(samples[:, 3])))
        if headings:
            result.heading_rad = float(np.average(headings, weights=weights))
            result.direction = int(np.sign(np.average(directions, weights=weights)))

    def _safe_centers(self, surface, yaw, margins):
        offsets = np.zeros(surface.observed_mask.shape + (2,))
        if not surface.valid:
            return np.zeros_like(surface.observed_mask), offsets
        # Plan inside the accepted region, not exactly on a cell boundary.
        # Runtime whole-sole validation retains the original physical margins.
        preferred_margins = tuple(margin+self.cfg.target_tracking_margin for margin in margins)
        rear, front, side = preferred_margins
        half_length = 0.5 * (self.cfg.foot_rear_extent + self.cfg.foot_front_extent)
        half_width = self.cfg.foot_half_width + side
        radius = math.ceil(math.hypot(half_length + max(rear, front), half_width) / self.cfg.grid_size) + 1
        x, y = np.meshgrid(np.arange(-radius, radius + 1) * self.cfg.grid_size,
                           np.arange(-radius, radius + 1) * self.cfg.grid_size, indexing="ij")
        cell_offsets = np.stack((x, y), axis=-1)
        safe = np.zeros_like(surface.observed_mask)
        # Narrow treads can fit a sole between grid centers without any gap filling.
        # Prefer tracking reserve where observed coverage permits it; do not
        # fabricate coverage or remove previously valid narrow support patches.
        for candidate_margins in (preferred_margins, margins):
            for dx in (0.0, 0.25, -0.25, 0.5):
                for dy in (0.0, 0.25, -0.25, 0.5):
                    shift = np.array([dx, dy]) * self.cfg.grid_size
                    kernel = footprint_cells(cell_offsets - shift, yaw, self.cfg, candidate_margins).astype(np.uint8)
                    possible = cv2.erode(surface.observed_mask.astype(np.uint8), kernel,
                                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                    selected = possible & ~safe
                    offsets[selected] = shift
                    safe |= possible
        return safe, offsets


def draw_surface_geometry_debug(result: SurfaceGeometryResult, rgb, depth, image_shape,
                                feet_xy=None, feet_yaw=None, camera_result=None):
    """Show camera labels, observed cells, full-foot safe centers, and real soles."""
    import torch

    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    depth = np.nan_to_num(np.asarray(depth).squeeze(), nan=5.0, posinf=5.0, neginf=0.3)
    depth_color = cv2.applyColorMap((255 * (1 - np.clip(depth, 0.3, 5.0) / 5.0)).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
    if rgb is None:
        rgb = depth_color.copy()
    elif torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    rgb = cv2.cvtColor(np.asarray(rgb)[..., :3], cv2.COLOR_RGB2BGR)
    palette = np.array([(255, 180, 60), (80, 220, 170), (200, 120, 255),
                        (60, 200, 255), (220, 180, 110), (130, 230, 230),
                        (230, 130, 160), (170, 240, 110)], dtype=np.uint8)
    camera_result = result if camera_result is None else camera_result
    def surface_color(surface):
        identity = surface.surface_id if surface.track_id is None else surface.track_id
        return palette[identity % len(palette)]

    pixel_labels = camera_result.point_labels.reshape(image_shape)
    labels_rgb = cv2.resize(pixel_labels.astype(np.float32), (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
    labels_depth = cv2.resize(pixel_labels.astype(np.float32), (depth.shape[1], depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)
    for surface in camera_result.surfaces:
        for panel, labels in ((rgb, labels_rgb), (depth_color, labels_depth)):
            selected = labels == surface.surface_id
            panel[selected] = (0.65 * panel[selected] + 0.35 * surface_color(surface)).astype(np.uint8)
    width, camera_height = 1280, 360
    canvas = np.full((950, width, 3), 22, dtype=np.uint8)
    canvas[60:60 + camera_height, :640] = cv2.resize(rgb, (640, camera_height))
    canvas[60:60 + camera_height, 640:] = cv2.resize(depth_color, (640, camera_height))
    heading = "unknown" if result.heading_rad is None else f"{math.degrees(result.heading_rad):+.1f}deg"
    history = f" | observed history={result.memory_frame_count} frames" if result.memory_frame_count else ""
    cv2.putText(canvas, f"2-D depth diagnostics | direction={result.direction:+d} heading={heading}{history}",
                (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)
    cv2.putText(canvas, "RGB + accepted plane labels", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, "Depth + accepted plane labels", (658, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    left, top, map_width, map_height = 30, 455, 760, 425
    min_x, max_x = min(-0.30, result.cfg.min_forward), result.cfg.max_forward
    min_y, max_y = -0.45, 0.45

    def project(xy):
        xy = np.asarray(xy)
        return np.stack((left + (max_y - xy[..., 1]) / (max_y - min_y) * map_width,
                         top + (max_x - xy[..., 0]) / (max_x - min_x) * map_height), axis=-1).astype(np.int32)

    cv2.rectangle(canvas, (left, top), (left + map_width, top + map_height), (100, 100, 100), 1)
    cell_w = max(1, round(result.cfg.grid_size / (max_y - min_y) * map_width))
    cell_h = max(1, round(result.cfg.grid_size / (max_x - min_x) * map_height))
    for row, column in np.argwhere(result.grid_observed):
        px, py = project(result.grid_xy[row, column])
        label = result.grid_labels[row, column]
        color = tuple(int(v) for v in surface_color(result.surfaces[label])) if label >= 0 else (90, 90, 90)
        cv2.rectangle(canvas, (px - cell_w // 2, py - cell_h // 2),
                       (px + cell_w // 2, py + cell_h // 2), color, -1)
    for surface in result.surfaces:
        for xy in result.candidate_centers(surface.surface_id):
            px, py = project(xy)
            cv2.circle(canvas, (px, py), 2, (60, 255, 60), -1)
        for edge in (surface.near_edge, surface.far_edge):
            if edge is not None:
                ys = np.array(edge.observed_y_range)
                endpoints = project(np.column_stack((edge.x_at(ys), ys)))
                cv2.line(canvas, tuple(endpoints[0]), tuple(endpoints[1]), (255, 255, 255), 2)
        center = project(surface.centroid[:2])
        identity = f"S{surface.surface_id}" if surface.track_id is None else f"T{surface.track_id}"
        cv2.putText(canvas, identity, tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if feet_xy is not None:
        for index, center in enumerate(feet_xy):
            yaw = 0.0 if feet_yaw is None else float(feet_yaw[index])
            half_length = 0.5 * (result.cfg.foot_rear_extent + result.cfg.foot_front_extent)
            half_width = result.cfg.foot_half_width
            sole = np.array([[-half_length, -half_width], [half_length, -half_width],
                             [half_length, half_width], [-half_length, half_width]])
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            outline = project(sole @ rotation.T + center)
            cv2.polylines(canvas, [outline], True, (0, 190, 255), 2)
            cv2.putText(canvas, "LR"[index], tuple(project(center)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
    cv2.putText(canvas, "Top view: forward up, robot-left left", (left, top - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(canvas, "color=observed plane  black=unknown  green=full-foot candidate (unverified)", (30, 913),
                cv2.FONT_HERSHEY_SIMPLEX, 0.49, (230, 230, 230), 1)
    cv2.putText(canvas, f"sole={100*(result.cfg.foot_rear_extent+result.cfg.foot_front_extent):.0f}cm"
                f"  critical edge={100*result.cfg.edge_margin:.0f}cm other end={100*result.cfg.other_end_margin:.0f}cm"
                f"  uncertainty reserve={100*result.cfg.uncertainty_margin:.0f}cm", (30, 940),
                cv2.FONT_HERSHEY_SIMPLEX, 0.49, (230, 230, 230), 1)
    for index, surface in enumerate(result.surfaces):
        y = top + index * 51
        color = (200, 230, 220) if surface.valid else (80, 150, 255)
        status = surface.kind if surface.valid else f"REJECT {surface.rejection}"
        identity = f"S{surface.surface_id}" if surface.track_id is None else f"T{surface.track_id}"
        cv2.putText(canvas, f"{identity} {status}", (815, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        cv2.putText(canvas, f"tilt={surface.tilt_deg:.1f}deg rms={surface.rms*1000:.1f}mm"
                    f" centers={int(surface.safe_center_mask.sum())}", (815, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if result.timestamp_s is not None and surface.last_observed_time is not None:
            cv2.putText(canvas, f"last seen {result.timestamp_s-surface.last_observed_time:.2f}s ago",
                        (815, y + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
    return canvas
