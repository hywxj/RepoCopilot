"""Short-lived, pose-compensated observations; no inferred support in unknown cells."""

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from .tread_surfaces import TreadSurfaceExtractor


@dataclass
class SurfaceMemoryCfg:
    max_frames: int = 25
    max_age_s: float = 1.0
    max_points_per_frame: int = 20000
    rear_limit: float = -0.35
    max_pose_jump_m: float = 0.5
    height_tolerance_m: float = 0.035
    normal_tolerance_deg: float = 5.0
    association_gap_m: float = 0.06


@dataclass
class _SurfaceTrack:
    track_id: int
    normal: np.ndarray
    offset: float
    bounds: np.ndarray
    last_observed: float


class SurfaceMemory:
    def __init__(self, surface_cfg, cfg=None):
        self.cfg = SurfaceMemoryCfg() if cfg is None else cfg
        if self.cfg.max_frames < 1 or self.cfg.max_age_s <= 0 or self.cfg.max_points_per_frame < 1:
            raise ValueError("Surface memory limits must be positive.")
        self.extractor = TreadSurfaceExtractor(replace(surface_cfg, min_forward=self.cfg.rear_limit))
        self.reset()

    def reset(self):
        self.generation = getattr(self, "generation", -1) + 1
        self.frames = deque(maxlen=self.cfg.max_frames)
        self.tracks = []
        self.next_track_id = 0
        self.last_time = None
        self.last_position = None
        self.last_rotation = None
        self.result = None

    def _expire(self, timestamp):
        cutoff = timestamp - self.cfg.max_age_s
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()
        self.tracks = [track for track in self.tracks if track.last_observed >= cutoff]

    def _compose(self, root_position, root_rotation, timestamp):
        world = np.concatenate([points for _, points in self.frames]) if self.frames else np.empty((0, 3))
        current = (world-root_position) @ root_rotation
        self.result = self.extractor.extract(current)
        self.result.memory_frame_count = len(self.frames)
        self.result.timestamp_s = timestamp
        for surface in self.result.surfaces:
            if not surface.valid:
                continue
            normal = root_rotation @ surface.normal
            offset = surface.offset - float(normal @ root_position)
            members = world[self.result.point_labels == surface.surface_id]
            bounds = np.stack((members[:, :2].min(axis=0), members[:, :2].max(axis=0)))
            track = self._associate(normal, offset, bounds)
            if track is not None:
                surface.track_id, surface.last_observed_time = track.track_id, track.last_observed
        self.last_position, self.last_rotation = root_position.copy(), root_rotation.copy()
        return self.result

    def refresh(self, root_position, root_rotation, timestamp):
        """Reproject/expire history on the current clock even when the camera is stalled."""
        root_position, root_rotation = np.asarray(root_position), np.asarray(root_rotation)
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or not np.isfinite(root_position).all() or not np.isfinite(root_rotation).all():
            raise ValueError("Surface memory requires finite synchronized poses and timestamps.")
        if self.last_time is not None:
            if timestamp < self.last_time or np.linalg.norm(root_position-self.last_position) > self.cfg.max_pose_jump_m:
                self.reset()
        self._expire(timestamp)
        return self._compose(root_position, root_rotation, timestamp)

    def _associate(self, normal, offset, bounds):
        matches = []
        for track in self.tracks:
            if normal @ track.normal < math.cos(math.radians(self.cfg.normal_tolerance_deg)):
                continue
            xy = np.mean(bounds, axis=0)
            height = -(normal[:2] @ xy + offset) / normal[2]
            old_height = -(track.normal[:2] @ xy + track.offset) / track.normal[2]
            difference = abs(height-old_height)
            if difference > self.cfg.height_tolerance_m:
                continue
            separation = np.maximum(bounds[0] - track.bounds[1], track.bounds[0] - bounds[1])
            if (separation > self.cfg.association_gap_m).any():
                continue
            matches.append((difference, track))
        return min(matches, key=lambda pair: pair[0])[1] if matches else None

    def _invalidate_changed_heights(self, fresh_world):
        if not len(fresh_world):
            return
        size = self.extractor.cfg.grid_size
        cells, inverse = np.unique(np.floor(fresh_world[:, :2] / size).astype(np.int64), axis=0,
                                    return_inverse=True)
        means = np.bincount(inverse, weights=fresh_world[:, 2]) / np.bincount(inverse)
        minimum = cells.min(axis=0)
        shape = cells.max(axis=0)-minimum+1
        cell_keys = np.ravel_multi_index((cells-minimum).T, shape)
        for index, (time, old) in enumerate(self.frames):
            keys = np.floor(old[:, :2] / size).astype(np.int64)
            inside = ((keys >= minimum) & (keys < minimum+shape)).all(axis=1)
            old_keys = np.ravel_multi_index((keys[inside]-minimum).T, shape)
            matches = np.searchsorted(cell_keys, old_keys)
            in_range = matches < len(cell_keys)
            found = np.zeros(len(matches), dtype=bool)
            found[in_range] = cell_keys[matches[in_range]] == old_keys[in_range]
            expected = np.full(len(old), np.nan)
            expected[np.flatnonzero(inside)[found]] = means[matches[found]]
            keep = ~np.isfinite(expected) | (np.abs(old[:, 2] - expected) <= self.cfg.height_tolerance_m)
            self.frames[index] = (time, old[keep])

    def update(self, points_body, snapshot, root_position, root_rotation, timestamp, point_valid=None):
        timestamp = float(timestamp)
        root_position = np.asarray(root_position, dtype=np.float64)
        root_rotation = np.asarray(root_rotation, dtype=np.float64)
        if not np.isfinite(timestamp) or not np.isfinite(root_position).all() or not np.isfinite(root_rotation).all():
            raise ValueError("Surface memory requires finite synchronized poses and timestamps.")
        if self.last_time is not None:
            jump = np.linalg.norm(root_position - self.last_position) > self.cfg.max_pose_jump_m
            if timestamp < self.last_time or jump:
                self.reset()
            elif timestamp == self.last_time:
                if np.array_equal(root_position, self.last_position) and np.array_equal(root_rotation, self.last_rotation):
                    return self.result
                return self.refresh(root_position, root_rotation, timestamp)
        self._expire(timestamp)
        points_body = np.asarray(points_body)
        # Validate directly observed near-foot points too, before they leave the forward ROI.
        observations = self.extractor.extract(points_body, point_valid)
        fresh_world = points_body[observations.point_labels >= 0] @ root_rotation.T + root_position
        if len(fresh_world) > self.cfg.max_points_per_frame:
            selected = np.linspace(0, len(fresh_world)-1, self.cfg.max_points_per_frame, dtype=np.int64)
            fresh_world = fresh_world[selected]
        for surface in observations.surfaces:
            if not surface.valid:
                continue
            members = points_body[observations.point_labels == surface.surface_id] @ root_rotation.T + root_position
            normal = root_rotation @ surface.normal
            offset = surface.offset - float(normal @ root_position)
            bounds = np.stack((members[:, :2].min(axis=0), members[:, :2].max(axis=0)))
            track = self._associate(normal, offset, bounds)
            if track is None:
                track = _SurfaceTrack(self.next_track_id, normal, offset, bounds, timestamp)
                self.next_track_id += 1
                self.tracks.append(track)
            else:
                track.normal, track.offset = normal, offset
                track.bounds = np.stack((np.minimum(track.bounds[0], bounds[0]),
                                         np.maximum(track.bounds[1], bounds[1])))
                track.last_observed = timestamp
            surface.track_id = track.track_id
            surface.last_observed_time = timestamp
        for surface in snapshot.surfaces:
            if not surface.valid:
                continue
            normal = root_rotation @ surface.normal
            offset = surface.offset - float(normal @ root_position)
            members = points_body[snapshot.point_labels == surface.surface_id] @ root_rotation.T + root_position
            bounds = np.stack((members[:, :2].min(axis=0), members[:, :2].max(axis=0)))
            track = self._associate(normal, offset, bounds)
            if track is not None:
                surface.track_id, surface.last_observed_time = track.track_id, timestamp
        self._invalidate_changed_heights(fresh_world)
        self.frames.append((timestamp, fresh_world))
        self.last_time = timestamp
        return self._compose(root_position, root_rotation, timestamp)
