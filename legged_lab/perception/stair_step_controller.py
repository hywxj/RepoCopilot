"""Depth-only single-step supervisor. References guide a policy, not a joint servo."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np

from .tread_surfaces import SurfaceGeometryResult


class StepPhase(IntEnum):
    BLIND = 0
    OBSERVE = 1
    SHIFT_LEAD = 2
    LIFT_LEAD = 3
    LOWER_LEAD = 4
    TRANSFER = 5
    SHIFT_TRAIL = 6
    LIFT_TRAIL = 7
    LOWER_TRAIL = 8
    SETTLE = 9
    COMPLETE = 10
    RECOVER = 11


@dataclass
class StepControlCfg:
    confirmation_s: float = 0.12
    settle_s: float = 0.20
    phase_timeout_s: float = 6.0
    observe_timeout_s: float = 12.0
    target_max_age_s: float = 0.25
    flight_occlusion_s: float = 3.0
    max_heading_error_rad: float = math.radians(5.0)
    alignment_yaw_gain: float = 1.5
    alignment_max_yaw_rate: float = 0.25
    action_transition_s: float = 0.30
    min_step_height: float = 0.06
    max_step_height: float = 0.23
    min_forward_step: float = 0.03
    max_forward_step: float = 0.55
    max_lateral_adjustment: float = 0.08
    min_foot_separation: float = 0.16
    xy_tolerance: float = 0.025
    height_tolerance: float = 0.02
    foot_tilt_tolerance_rad: float = math.radians(7.0)
    max_sole_speed: float = 0.08
    max_slip: float = 0.015
    lift_clearance: float = 0.06
    crossing_clearance: float = 0.02
    overlap_swing_lift: bool = False
    lift_duration_s: float = 0.55
    max_lower_speed: float = 0.20
    enter_frames: int = 3
    leg_reach_m: float = 0.64
    leg_reach_reserve_m: float = 0.01
    transfer_body_fraction: float = 0.65
    # Optional interior CoM-reference region, configured from contact capability.
    # Zero retains point support for callers without a validated contact model.
    support_com_half_length_m: float = 0.0
    support_com_half_width_m: float = 0.0
    touchdown_load_fraction: float = 0.15
    touchdown_contact_fraction: float = 0.003
    touchdown_probe_depth: float = 0.003


@dataclass
class StepMeasurement:
    timestamp: float
    generation: int
    root_position: np.ndarray
    yaw_rotation: np.ndarray
    sole_positions: np.ndarray
    foot_rotations: np.ndarray
    sole_velocities: np.ndarray
    contact_forces: np.ndarray | None
    root_velocity: np.ndarray
    root_angular_velocity: np.ndarray
    body_weight: float
    camera_timestamp: float | None = None
    hip_offsets: np.ndarray | None = None
    com_offset: np.ndarray | None = None


def body_height_limit(root_xy, hip_offsets, ankle_positions, reach):
    """Necessary hip-to-ankle reach envelope, not a balance or IK solution."""
    horizontal = ankle_positions[:, :2] - (root_xy+hip_offsets[:, :2])
    remaining = reach**2 - np.sum(horizontal**2, axis=1)
    if np.any(remaining <= 0):
        return None
    return float(np.min(ankle_positions[:, 2]-hip_offsets[:, 2]+np.sqrt(remaining)))


@dataclass
class LockedTread:
    generation: int
    track_id: int
    surface_id: int
    geometry: SurfaceGeometryResult
    root_position: np.ndarray
    rotation: np.ndarray
    targets: np.ndarray
    heading: float
    last_seen: np.ndarray


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _smooth(t):
    t = float(np.clip(t, 0, 1))
    return t**3 * (10 - 15*t + 6*t*t)


class StairStepController:
    """One locked plane, leading/trailing feet, and physical support confirmation.

    No mesh heights, terrain row, ray-caster truth, or camera images of feet are
    consumed here. The copied observed mask permits bounded swing occlusion;
    it does not extrapolate into cells that were never observed.
    """

    num_features = len(StepPhase) + 6 + 6 + 2 + 3 + 1 + 1 + 1 + 2 + 2 + 2 + 1

    def __init__(self, cfg=None):
        self.cfg = cfg or StepControlCfg()
        self.reset()

    def reset(self):
        self.phase = StepPhase.BLIND
        self.direction = 0
        self.lock = None
        self.preview = None
        self.confirmation_plan = None
        self.lead = 0
        self.elapsed = self.confirmed = 0.0
        self.detection_count = 0
        self.last_detection_time = None
        self.failure_reason = ""
        self.progress_event = self.success_event = self.unsafe_contact_event = False
        self.reference_feet = np.zeros((2, 3))
        self.reference_root = np.zeros(3)
        self.planted_positions = [None, None]
        self.source_feet = None
        self.descent_targets = [None, None]
        self.confirmed_plants = np.zeros(2, dtype=bool)
        self.swing_start = None
        self.last_measurement = None
        self.support_valid = np.zeros(2, dtype=bool)
        self.observed_heading = 0.0
        self.alignment_heading_world = None
        self.alignment_observation_time = None
        self.alignment_block_reason = ""
        self.lift_clearance_confirmed = False
        self.observe_requirements = {}
        self.shift_requirements = {}
        self.support_block_reason = ""
        self.observe_support_height = None
        self.replan_reason = ""
        self.best_lift_height = 0.
        self.lift_progress = 0.

    @property
    def active(self):
        return self.phase != StepPhase.BLIND

    @property
    def swing_foot(self):
        if self.phase in (StepPhase.LIFT_LEAD, StepPhase.LOWER_LEAD):
            return self.lead
        if self.phase in (StepPhase.LIFT_TRAIL, StepPhase.LOWER_TRAIL):
            return 1-self.lead
        return None

    def alignment_heading_error(self, m):
        if self.alignment_heading_world is None or self.alignment_observation_time is None:
            return None
        age = m.timestamp-self.alignment_observation_time
        if not 0 <= age <= self.cfg.target_max_age_s:
            return None
        yaw = math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0])
        return _wrap(self.alignment_heading_world-yaw)

    def action_gate_target(self, m):
        if not self.active:
            return 0.
        return float(self.direction)

    def walking_command(self, m):
        """Request alignment before locking a stair target; never advance x/y."""
        command = np.zeros(3)
        self.alignment_block_reason = ""
        if self.phase != StepPhase.OBSERVE or self.lock is not None:
            return command
        heading = self.alignment_heading_error(m)
        if heading is None:
            self.alignment_block_reason = "heading_observation_unavailable"
            return command
        if abs(heading) <= self.cfg.max_heading_error_rad:
            return command
        if m.contact_forces is None:
            self.alignment_block_reason = "load_feedback_unavailable"
            return command
        load = np.maximum(m.contact_forces[:, 2], 0)/m.body_weight
        if load.sum() < 0.8 or load.min() < 0.12:
            self.alignment_block_reason = "insufficient_double_support"
            return command
        if abs(m.sole_positions[0, 2]-m.sole_positions[1, 2]) > self.cfg.height_tolerance:
            self.alignment_block_reason = "feet_not_on_same_level"
            return command
        if (np.linalg.norm(m.root_velocity) > 0.15
                or np.linalg.norm(m.root_angular_velocity) > 0.35):
            self.alignment_block_reason = "body_motion_too_fast"
            return command
        command[2] = np.clip(self.cfg.alignment_yaw_gain*heading,
                             -self.cfg.alignment_max_yaw_rate, self.cfg.alignment_max_yaw_rate)
        return command

    def plan(self, geometry, m):
        if geometry is None or geometry.direction == 0 or geometry.heading_rad is None:
            return None
        if abs(geometry.heading_rad) > self.cfg.max_heading_error_rad:
            return None
        feet = (m.sole_positions-m.root_position) @ m.yaw_rotation
        candidates = []
        # Without load feedback this is a geometry PREVIEW only. It cannot
        # authorize a support transfer, even when both soles look stationary.
        loaded = (np.ones(2, dtype=bool) if m.contact_forces is None
                  else m.contact_forces[:, 2] > 0.15*m.body_weight)
        if not loaded.any():
            return None
        support_z = float(feet[loaded, 2].min())
        for surface in geometry.surfaces:
            if not surface.valid or surface.track_id is None:
                continue
            dz = geometry.direction*(float(surface.height_at(surface.centroid[:2]))-support_z)
            if self.cfg.min_step_height <= dz <= self.cfg.max_step_height:
                candidates.append((dz, surface))
        if not candidates:
            return None
        # Never skip an unobserved/unreachable nearest level to choose a later one.
        surface = min(candidates, key=lambda item: item[0])[1]
        if (surface.last_observed_time is None
                or not 0 <= m.timestamp-surface.last_observed_time <= self.cfg.target_max_age_s):
            return None
        centers = geometry.candidate_centers(surface.surface_id)
        rear, front, _ = geometry.support_margins()
        pairs = []
        for foot in range(2):
            delta = centers-feet[foot, :2]
            valid = ((delta[:, 0] >= self.cfg.min_forward_step)
                     & (delta[:, 0] <= self.cfg.max_forward_step)
                     & (np.abs(delta[:, 1]) <= self.cfg.max_lateral_adjustment))
            pool = centers[valid]
            if not len(pool):
                return None
            # The task is to occupy this tread, not two grid-aligned footholds.
            # Preserve each foot's lateral position whenever the whole sole fits.
            forward = np.unique(pool[:, 0])
            aligned = np.column_stack((forward, np.full(len(forward), feet[foot, 1])))
            aligned = np.array([point for point in aligned
                                if geometry.footprint_supported(surface.surface_id, point, geometry.heading_rad)])
            if len(aligned):
                pool = np.vstack((aligned, pool))
            middle = float(surface.observed_bounds[:, 0].mean())
            if surface.near_edge is not None and surface.far_edge is not None:
                middle = .5*(surface.near_edge.x_at(feet[foot, 1])+surface.far_edge.x_at(feet[foot, 1]))
            middle += .5*(rear-front)*math.cos(geometry.heading_rad)
            order = np.lexsort(((pool[:, 0]-middle)**2, np.abs(pool[:, 1]-feet[foot, 1])))
            pairs.append(pool[order])
        chosen = None
        for left in pairs[0]:
            for right in pairs[1]:
                if left[1]-right[1] < self.cfg.min_foot_separation or abs(left[0]-right[0]) > 0.08:
                    continue
                if all(geometry.footprint_supported(surface.surface_id, p, geometry.heading_rad)
                       for p in (left, right)):
                    chosen = np.stack((left, right))
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None
        targets = np.column_stack((chosen, surface.height_at(chosen))) @ m.yaw_rotation.T + m.root_position
        heading = math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0])+geometry.heading_rad
        return LockedTread(m.generation, surface.track_id, surface.surface_id, copy.deepcopy(geometry),
                           m.root_position.copy(), m.yaw_rotation.copy(), targets, heading,
                           np.full(2, surface.last_observed_time))

    def _lowest_sole_height(self, m):
        corners = np.array([[-0.12, -0.042, 0], [0.12, -0.042, 0],
                            [0.12, 0.042, 0], [-0.12, 0.042, 0]])
        return (np.einsum('fij,cj->fci', m.foot_rotations, corners)+m.sole_positions[:, None])[:, :, 2].min(axis=1)

    def _same_plan(self, previous, current):
        return (previous is not None and current is not None
                and previous.generation == current.generation
                and previous.track_id == current.track_id
                and np.max(np.linalg.norm(previous.targets-current.targets, axis=1)) <= self.cfg.xy_tolerance
                and abs(_wrap(previous.heading-current.heading)) <= self.cfg.max_heading_error_rad)

    def _reachable_targets(self, m):
        delta = (self.execution_targets()-m.sole_positions) @ m.yaw_rotation
        return ((delta[:, 0] >= self.cfg.min_forward_step)
                & (delta[:, 0] <= self.cfg.max_forward_step)
                & (np.abs(delta[:, 1]) <= self.cfg.max_lateral_adjustment))

    def execution_targets(self):
        """Motion references within the tread; confirmed plants stay fixed."""
        targets = self.lock.targets.copy()
        for foot in range(2):
            if self.descent_targets[foot] is not None:
                targets[foot] = self.descent_targets[foot]
            if self.confirmed_plants[foot] and self.planted_positions[foot] is not None:
                targets[foot] = self.planted_positions[foot]
        return targets

    def _footprint_in_tread(self, m, foot):
        lock = self.lock
        position = (m.sole_positions[foot]-lock.root_position) @ lock.rotation
        rotation = lock.rotation.T @ m.foot_rotations[foot]
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        separation = ((m.sole_positions[0]-m.sole_positions[1]) @ lock.rotation)[1]
        return (separation >= self.cfg.min_foot_separation
                and lock.geometry.footprint_supported(lock.surface_id, position[:2], yaw))

    def physical_support_metrics(self, m, foot, min_load=0.12):
        """Read-only measurements and exact existing support gates for diagnosis."""
        if m.contact_forces is None or self.lock is None:
            return dict(available=False, valid=False, conditions={}, measurements={}, limits={},
                        failed_conditions=[name for name, missing in
                            (("contact_forces_unavailable", m.contact_forces is None),
                             ("target_lock_unavailable", self.lock is None)) if missing])
        lock = self.lock
        surface = lock.geometry.surfaces[lock.surface_id]
        position = (m.sole_positions[foot]-lock.root_position) @ lock.rotation
        rotation = lock.rotation.T @ m.foot_rotations[foot]
        footprint = self._footprint_in_tread(m, foot)
        corners = np.array([[-0.12, -0.042, 0], [0.12, -0.042, 0],
                            [0.12, 0.042, 0], [-0.12, 0.042, 0]]) @ rotation.T + position
        plane_error = np.abs(corners @ surface.normal + surface.offset).max()
        tilt = math.acos(float(np.clip(rotation[:, 2] @ surface.normal, -1, 1)))
        force = m.contact_forces[foot]
        force_norm = np.linalg.norm(force)
        speed = np.linalg.norm(m.sole_velocities[foot])
        planted = self.planted_positions[foot]
        slip_distance = None if planted is None else np.linalg.norm(m.sole_positions[foot, :2]-planted[:2])
        slip = planted is not None and slip_distance > self.cfg.max_slip
        conditions = dict(region=bool(footprint), plane=bool(plane_error <= self.cfg.height_tolerance),
                          load=bool(force[2] >= min_load*m.body_weight),
                          force_direction=bool(force[2] >= .75*force_norm),
                          speed=bool(speed <= self.cfg.max_sole_speed),
                          tilt=bool(tilt <= self.cfg.foot_tilt_tolerance_rad), slip=not bool(slip))
        return dict(available=True, valid=all(conditions.values()), conditions=conditions,
                    failed_conditions=[name for name, valid in conditions.items() if not valid],
                    measurements=dict(plane_error_m=float(plane_error), tilt_rad=float(tilt),
                        vertical_force_n=float(force[2]), force_norm_n=float(force_norm),
                        load_fraction=float(force[2]/m.body_weight) if m.body_weight > 0 else None,
                        vertical_force_fraction=float(force[2]/force_norm) if force_norm > 0 else None,
                        sole_speed_m_s=float(speed),
                        slip_distance_m=None if slip_distance is None else float(slip_distance)),
                    limits=dict(plane_error_m=float(self.cfg.height_tolerance),
                        tilt_rad=float(self.cfg.foot_tilt_tolerance_rad), load_fraction=float(min_load),
                        minimum_vertical_force_n=float(min_load*m.body_weight), vertical_force_fraction=.75,
                        sole_speed_m_s=float(self.cfg.max_sole_speed), slip_distance_m=float(self.cfg.max_slip)))

    def _physical_support(self, m, foot, min_load=0.12, record_plant=True):
        valid = self.physical_support_metrics(m, foot, min_load)["valid"]
        if valid and self.planted_positions[foot] is None and record_plant:
            self.planted_positions[foot] = m.sole_positions[foot].copy()
        return valid

    def _transition(self, phase, m):
        self.phase = phase
        self.elapsed = self.confirmed = 0.0
        self.progress_event = phase not in (StepPhase.OBSERVE, StepPhase.RECOVER)
        self.reference_feet = m.sole_positions.copy()
        if self.lock is not None:
            for foot in range(2):
                if self.planted_positions[foot] is not None:
                    self.reference_feet[foot] = self.planted_positions[foot]
        self.reference_root = m.root_position.copy()
        if phase == StepPhase.OBSERVE:
            # A gait may enter OBSERVE with one foot still airborne. Keep x/y
            # fixed but request landing on the measured source support level.
            if m.contact_forces is not None:
                loaded = m.contact_forces[:, 2] >= 0.15*m.body_weight
                if loaded.any():
                    ground = float(m.sole_positions[loaded, 2].min())
                    landing = ~loaded & (np.abs(m.sole_positions[:, 2]-ground) <= self.cfg.lift_clearance)
                    self.reference_feet[landing, 2] = ground
        if self.swing_foot is not None:
            self.swing_start = m.sole_positions[self.swing_foot].copy()
        if phase in (StepPhase.LIFT_LEAD, StepPhase.LIFT_TRAIL):
            self.lift_clearance_confirmed = False
            self.crossing_elapsed = 0.0
            self.best_lift_height = float(self.swing_start[2])

    def _fail(self, reason, m):
        self.failure_reason = reason
        self._transition(StepPhase.RECOVER, m)

    def _hold(self, condition, dt, duration=None):
        self.confirmed = self.confirmed + dt if condition else 0.0
        return self.confirmed+1.e-8 >= (self.cfg.confirmation_s if duration is None else duration)

    def update(self, geometry, m, dt):
        if dt <= 0 or m.body_weight <= 0:
            raise ValueError("Positive control dt and body weight are required.")
        self.last_measurement = m
        if (geometry is not None and geometry.direction != 0
                and geometry.heading_rad is not None and math.isfinite(geometry.heading_rad)):
            self.observed_heading = geometry.heading_rad
            frame_time = m.camera_timestamp if m.camera_timestamp is not None else geometry.timestamp_s
            if frame_time is not None and 0 <= m.timestamp-frame_time <= self.cfg.target_max_age_s:
                yaw = math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0])
                self.alignment_heading_world = _wrap(yaw+geometry.heading_rad)
                self.alignment_observation_time = frame_time
        self.progress_event = self.success_event = self.unsafe_contact_event = False
        self.lift_progress = 0.
        self.replan_reason = ""
        if self.phase in (StepPhase.COMPLETE, StepPhase.RECOVER):
            return
        self.elapsed += dt
        if self.phase == StepPhase.BLIND:
            self.reference_feet = m.sole_positions.copy()
            self.reference_root = m.root_position.copy()
            frame_time = m.camera_timestamp if m.camera_timestamp is not None else geometry.timestamp_s if geometry is not None else None
            if geometry is not None and frame_time != self.last_detection_time:
                self.last_detection_time = frame_time
                detected = geometry.direction != 0 and geometry.heading_rad is not None
                if detected and self.direction not in (0, geometry.direction):
                    self.detection_count = 0
                self.direction = geometry.direction if detected else 0
                self.detection_count = self.detection_count+1 if detected else 0
                if self.detection_count >= self.cfg.enter_frames:
                    self._transition(StepPhase.OBSERVE, m)
            return
        timeout = self.cfg.observe_timeout_s if self.phase == StepPhase.OBSERVE else self.cfg.phase_timeout_s
        if self.elapsed > timeout:
            self._fail("phase_timeout", m)
            return
        if m.contact_forces is None:
            self.support_valid[:] = False
            self.support_block_reason = "load_feedback_unavailable"
            self.observe_requirements = {"load_feedback": False}
            self.confirmed = 0.0
            if self.phase == StepPhase.OBSERVE:
                self.preview = self.plan(geometry, m)
            else:
                self._fail("load_feedback_unavailable", m)
            return
        self.support_block_reason = ""
        load = np.maximum(m.contact_forces[:, 2], 0)/m.body_weight
        stable = (np.linalg.norm(m.root_velocity) <= 0.12 and np.linalg.norm(m.root_angular_velocity) <= 0.30
                  and np.max(np.linalg.norm(m.sole_velocities, axis=1)) <= self.cfg.max_sole_speed)
        if self.phase == StepPhase.OBSERVE:
            contact_heights = self._lowest_sole_height(m)
            loaded = load >= 0.12
            if self.observe_support_height is None and loaded.any():
                self.observe_support_height = float(contact_heights[loaded].min())
            if (self.observe_support_height is not None
                    and (loaded & (np.abs(contact_heights-self.observe_support_height) > self.cfg.min_step_height)).any()):
                self.unsafe_contact_event = True
                self._fail("unexpected_support_height_before_lock", m)
                return
            # Entry references remain fixed: moving the robot must not move
            # the inspection goal or erase stance-position errors.
            planned = self.plan(geometry, m)
            self.preview = planned
            consistent = planned is not None and (self.confirmation_plan is None
                                                  or self._same_plan(self.confirmation_plan, planned))
            if not consistent:
                self.confirmed = 0.0
            if self.confirmation_plan is None or not consistent:
                self.confirmation_plan = planned
            heading_error = self.alignment_heading_error(m)
            self.observe_requirements = {
                "target": planned is not None,
                "heading_aligned": heading_error is not None and abs(heading_error) <= self.cfg.max_heading_error_rad,
                "root_speed": np.linalg.norm(m.root_velocity) <= 0.12,
                "root_rotation": np.linalg.norm(m.root_angular_velocity) <= 0.30,
                "feet_speed": np.max(np.linalg.norm(m.sole_velocities, axis=1)) <= self.cfg.max_sole_speed,
                "both_loaded": load.min() >= 0.20,
                "total_load": load.sum() >= 0.8,
            }
            self.observe_requirements["target_consistent"] = consistent
            if self._hold(all(self.observe_requirements.values()), dt):
                self.lock = planned
                self.direction = geometry.direction
                self.lead = int(np.argmin(load))
                self.initial_root_height = float(m.root_position[2])
                self.source_height = float(m.sole_positions[:, 2].mean())
                self.source_feet = m.sole_positions.copy()
                self._transition(StepPhase.SHIFT_LEAD, m)
            elif not all(self.observe_requirements.values()):
                self.confirmation_plan = planned
            return
        if m.generation != self.lock.generation:
            self._fail("map_epoch_changed", m)
            return
        if geometry is not None:
            for surface in geometry.surfaces:
                if not surface.valid:
                    continue
                for foot, target in enumerate(self.execution_targets()):
                    body_target = (target-m.root_position) @ m.yaw_rotation
                    heading = self.lock.heading
                    if self.descent_targets[foot] is not None or self.planted_positions[foot] is not None:
                        heading = math.atan2(m.foot_rotations[foot, 1, 0], m.foot_rotations[foot, 0, 0])
                    if geometry.footprint_supported(surface.surface_id, body_target[:2],
                                                    _wrap(heading-math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0]))):
                        if abs(float(surface.height_at(body_target[:2]))-body_target[2]) > self.cfg.height_tolerance:
                            self._fail("observed_target_height_changed", m)
                            return
                        if surface.last_observed_time is not None:
                            self.lock.last_seen[foot] = surface.last_observed_time
        self.support_valid = np.array([self._physical_support(m, i) for i in range(2)])
        for foot, planted in enumerate(self.planted_positions):
            if planted is not None and self.confirmed_plants[foot] and not self.support_valid[foot]:
                self._fail("planted_target_support_lost", m)
                return
        heading_error = _wrap(self.lock.heading-math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0]))
        lead, trail = self.lead, 1-self.lead
        if self.phase in (StepPhase.SHIFT_LEAD, StepPhase.SHIFT_TRAIL):
            swing = lead if self.phase == StepPhase.SHIFT_LEAD else trail
            stance = 1-swing
            self.reference_root[:2] = self.reference_feet[stance, :2]
            fresh = m.timestamp-self.lock.last_seen[swing] <= self.cfg.target_max_age_s
            reachable = self._reachable_targets(m)
            target_reachable = bool(reachable.all() if self.phase == StepPhase.SHIFT_LEAD else reachable[swing])
            # Replan only before the first lift, with measured stable support.
            # Once a foot has swung, the same tread remains the committed goal.
            if (self.phase == StepPhase.SHIFT_LEAD and (not target_reachable or not fresh)
                    and stable and load.min() >= 0.20 and load.sum() >= 0.8
                    and abs(m.sole_positions[0, 2]-m.sole_positions[1, 2]) <= self.cfg.height_tolerance):
                self.replan_reason = ("target_unreachable_before_lift" if not target_reachable
                                      else "target_stale_before_lift")
                self.lock = self.preview = self.confirmation_plan = None
                self.shift_requirements = {}
                self.observe_support_height = None
                self._transition(StepPhase.OBSERVE, m)
                return
            self.shift_requirements = {
                "target_recent": fresh,
                "target_reachable": target_reachable,
                "swing_unloaded": load[swing] <= 0.08,
                "stance_loaded": load[stance] >= 0.60,
                "stance_still": np.linalg.norm(m.sole_velocities[stance]) <= self.cfg.max_sole_speed,
                "body_stable": np.linalg.norm(m.root_velocity) <= .12 and np.linalg.norm(m.root_angular_velocity) <= .30,
                "heading_aligned": abs(heading_error) <= self.cfg.max_heading_error_rad,
            }
            if self._hold(all(self.shift_requirements.values()), dt):
                self._transition(StepPhase.LIFT_LEAD if swing == lead else StepPhase.LIFT_TRAIL, m)
        elif self.swing_foot is not None:
            swing = self.swing_foot
            stance = 1-swing
            target = self.execution_targets()[swing]
            if m.timestamp-self.lock.last_seen[swing] > self.cfg.flight_occlusion_s:
                self._fail("swing_target_expired", m)
                return
            if load[stance] < 0.45:
                self._fail("stance_load_lost", m)
                return
            lifting = self.phase in (StepPhase.LIFT_LEAD, StepPhase.LIFT_TRAIL)
            apex = max(float(self.swing_start[2]), float(target[2]))+self.cfg.lift_clearance
            if lifting:
                # Approach the riser while lifting; cross it only after measured
                # whole-sole clearance. This avoids a vertical-then-horizontal
                # box trajectory while retaining the physical clearance gate.
                corners = np.array([[-0.12, -0.042, 0], [0.12, -0.042, 0],
                                    [0.12, 0.042, 0], [-0.12, 0.042, 0]]) @ m.foot_rotations[swing].T + m.sole_positions[swing]
                reached = min(apex, max(self.best_lift_height, float(corners[:, 2].min())))
                if load[swing] <= 0.08:
                    self.lift_progress = (reached-self.best_lift_height)/(apex-self.swing_start[2])
                    self.best_lift_height = reached
                if not self.lift_clearance_confirmed:
                    start = (self.swing_start-self.lock.root_position) @ self.lock.rotation
                    surface = self.lock.geometry.surfaces[self.lock.surface_id]
                    geometry_cfg = self.lock.geometry.cfg
                    radius = math.hypot(.5*(geometry_cfg.foot_front_extent+geometry_cfg.foot_rear_extent),
                                        geometry_cfg.foot_half_width)
                    near = (min(surface.near_edge.x_at(start[1]-radius), surface.near_edge.x_at(start[1]+radius))
                            if surface.near_edge is not None else surface.observed_bounds[0, 0])
                    approach = start.copy()
                    approach[0] += max(0., near-radius-geometry_cfg.uncertainty_margin-.01-start[0])
                    approach = approach @ self.lock.rotation.T+self.lock.root_position
                    if not self.cfg.overlap_swing_lift:
                        approach = self.swing_start.copy()
                    approach[2] = apex
                    self.reference_feet[swing] = self.swing_start+_smooth(
                        self.elapsed/self.cfg.lift_duration_s)*(approach-self.swing_start)
                    clearance = max(float(self.swing_start[2]), float(target[2]))+self.cfg.crossing_clearance
                    if not self.cfg.overlap_swing_lift:
                        clearance = apex-.015
                    if corners[:, 2].min() >= clearance and load[swing] <= 0.08:
                        self.lift_clearance_confirmed = True
                        self.crossing_start = m.sole_positions[swing].copy()
                else:
                    self.crossing_elapsed += dt
                    endpoint = target.copy()
                    endpoint[2] = apex
                    self.reference_feet[swing] = self.crossing_start + _smooth(
                        self.crossing_elapsed/self.cfg.lift_duration_s)*(endpoint-self.crossing_start)
                arrived = (m.sole_positions[swing, 2] >= apex-0.015
                           and self.lift_clearance_confirmed
                           and np.linalg.norm(m.sole_velocities[swing, :2]) <= self.cfg.max_sole_speed
                           and self._footprint_in_tread(m, swing))
                if arrived:
                    # Slow down above the tread before descending: braking on
                    # first entry can pull the sole back outside the safe region.
                    # Keep this valid point fixed instead of chasing the mark.
                    position = (m.sole_positions[swing]-self.lock.root_position) @ self.lock.rotation
                    surface = self.lock.geometry.surfaces[self.lock.surface_id]
                    position[2] = float(surface.height_at(position[:2]))
                    self.descent_targets[swing] = position @ self.lock.rotation.T+self.lock.root_position
                    self._transition(StepPhase.LOWER_LEAD if swing == lead else StepPhase.LOWER_TRAIL, m)
            else:
                # A bounded compliance reference establishes contact; the accepted
                # sole must still match the original observed plane and footprint.
                contact_height = target[2]-self.cfg.touchdown_probe_depth
                duration = max(0.4, 1.875*abs(self.swing_start[2]-contact_height)/self.cfg.max_lower_speed)
                self.reference_feet[swing] = target.copy()
                if self.planted_positions[swing] is not None:
                    self.reference_feet[swing, :2] = self.planted_positions[swing][:2]
                self.reference_feet[swing, 2] = self.swing_start[2]+_smooth(self.elapsed/duration)*(contact_height-self.swing_start[2])
                touched = load[swing] >= 0.12
                self.unsafe_contact_event = bool(touched and not self.support_valid[swing])
                if self._hold(self.support_valid[swing], dt):
                    self.confirmed_plants[swing] = True
                    self._transition(StepPhase.TRANSFER if swing == lead else StepPhase.SETTLE, m)
        elif self.phase == StepPhase.TRANSFER:
            targets = self.execution_targets()
            self.reference_root[:2] = (self.reference_feet[trail, :2] + self.cfg.transfer_body_fraction
                                       * (targets[lead, :2]-self.reference_feet[trail, :2]))
            self.reference_root[2] = self.initial_root_height + float(targets[lead, 2])-self.source_height
            if self._hold(self.support_valid[lead] and load[lead] >= 0.60 and load[trail] >= 0.08
                          and abs(m.root_velocity[2]) <= 0.10, dt):
                self._transition(StepPhase.SHIFT_TRAIL, m)
        elif self.phase == StepPhase.SETTLE:
            targets = self.execution_targets()
            self.reference_feet = targets.copy()
            self.reference_root[:2] = targets[:, :2].mean(axis=0)
            self.reference_root[2] = self.initial_root_height + float(targets[:, 2].mean())-self.source_height
            done = (self.support_valid.all() and load.min() >= 0.20 and load.sum() >= 0.8
                    and stable and abs(heading_error) <= self.cfg.max_heading_error_rad)
            if self._hold(done, dt, self.cfg.settle_s):
                self._transition(StepPhase.COMPLETE, m)
                self.success_event = True

    def _stance_com_anchor(self, stance):
        """Use an interior support point toward the fixed pre-step stance.

        A planted foot supports an area. Forcing CoM to its exact center before
        releasing the rear foot unnecessarily extends that rear leg. Keep this
        anchor fixed during swing; it must not track CoM drift or foot slip.
        """
        center = (self.execution_targets()[stance, :2]
                  if stance == self.lead and self.phase >= StepPhase.SHIFT_TRAIL
                  else self.reference_feet[stance, :2])
        if self.source_feet is None:
            return center.copy()
        cosine, sine = math.cos(self.lock.heading), math.sin(self.lock.heading)
        axes = np.array([[cosine, -sine], [sine, cosine]])
        toward = .5*(self.source_feet[1-stance, :2]-center) @ axes
        extent = np.array([self.cfg.support_com_half_length_m, self.cfg.support_com_half_width_m])
        return center+np.clip(toward, -extent, extent) @ axes.T

    def constrain_body_reference(self, m):
        if (m.hip_offsets is None or self.lock is None
                or not StepPhase.SHIFT_LEAD <= self.phase <= StepPhase.SETTLE):
            return
        targets = self.execution_targets()
        if self.phase <= StepPhase.LOWER_LEAD:
            anchor = self._stance_com_anchor(1-self.lead)
        elif self.phase in (StepPhase.SHIFT_TRAIL, StepPhase.LIFT_TRAIL, StepPhase.LOWER_TRAIL):
            anchor = self._stance_com_anchor(self.lead)
        elif self.phase == StepPhase.TRANSFER:
            source = self.reference_feet[1-self.lead, :2]
            anchor = source+self.cfg.transfer_body_fraction*(targets[self.lead, :2]-source)
        else:
            anchor = targets[:, :2].mean(axis=0)
        if self.phase in (StepPhase.LOWER_LEAD, StepPhase.LOWER_TRAIL):
            swing = self.lead if self.phase == StepPhase.LOWER_LEAD else 1-self.lead
            # Do not move CoM outside the stance foot while the other foot is airborne.
            # Light contact permits loading, but never counts as a confirmed plant.
            if self._physical_support(m, swing, min_load=self.cfg.touchdown_contact_fraction, record_plant=False):
                landing_xy = (self.planted_positions[swing][:2] if self.planted_positions[swing] is not None
                              else m.sole_positions[swing, :2])
                anchor = anchor+self.cfg.touchdown_load_fraction*(landing_xy-anchor)
        self.reference_root[:2] = anchor-(m.com_offset[:2] if m.com_offset is not None else 0.)
        nominal = self.initial_root_height
        if self.phase >= StepPhase.TRANSFER:
            nominal += float(targets[:, 2].mean())-self.source_height
        ankle_offsets = np.einsum("fij,j->fi", m.foot_rotations, [-.03, 0., .04])
        bounds = [body_height_limit(self.reference_root[:2], m.hip_offsets, feet+ankle_offsets,
                                   self.cfg.leg_reach_m-self.cfg.leg_reach_reserve_m)
                  for feet in (m.sole_positions, self.reference_feet)]
        if any(bound is None for bound in bounds):
            self._fail("body_reference_unreachable", m)
            return
        self.reference_root[2] = min(nominal, *bounds)

    def features(self, m, privileged=False):
        self.constrain_body_reference(m)
        phase = np.eye(len(StepPhase))[int(self.phase)]
        chosen = self.lock if self.lock is not None else self.preview
        targets = self.execution_targets() if self.lock is not None else None if chosen is None else chosen.targets
        target_error = np.zeros((2, 3)) if targets is None else (targets-m.sole_positions) @ m.yaw_rotation
        reference_error = (self.reference_feet-m.sole_positions) @ m.yaw_rotation
        root_error = (self.reference_root-m.root_position) @ m.yaw_rotation
        yaw = math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0])
        heading = (self.alignment_heading_error(m) or 0.) if self.lock is None else _wrap(self.lock.heading-yaw)
        freshness = np.zeros(2) if chosen is None else np.clip(
            1-(m.timestamp-chosen.last_seen)/self.cfg.flight_occlusion_s, 0, 1)
        # No physical force or force-qualified support labels in the actor.
        # The phase itself is still generated by the simulation teacher.
        load = (np.clip(m.contact_forces[:, 2]/m.body_weight, 0, 2)
                if privileged and m.contact_forces is not None else np.zeros(2))
        support = self.support_valid if privileged else np.zeros(2)
        return np.concatenate((phase, target_error.ravel(), reference_error.ravel(),
                               load, root_error,
                               [heading, float(chosen is not None), self.direction],
                               np.eye(2)[self.lead], support, freshness,
                               [float(self.lift_clearance_confirmed)])).astype(np.float32)

    def reward_metrics(self, m, dt):
        self.constrain_body_reference(m)
        active = float(self.active)
        feet_error = np.sum((self.reference_feet-m.sole_positions)**2, axis=1)
        root_error = np.sum((self.reference_root-m.root_position)**2)
        load = None if m.contact_forces is None else np.maximum(m.contact_forces[:, 2], 0)/m.body_weight
        desired = np.array([0.5, 0.5])
        if self.phase in (StepPhase.SHIFT_LEAD, StepPhase.LIFT_LEAD, StepPhase.LOWER_LEAD):
            desired[self.lead], desired[1-self.lead] = 0, 1
        elif self.phase in (StepPhase.TRANSFER, StepPhase.SHIFT_TRAIL, StepPhase.LIFT_TRAIL, StepPhase.LOWER_TRAIL):
            desired[self.lead], desired[1-self.lead] = 1, 0
        if self.phase in (StepPhase.LOWER_LEAD, StepPhase.LOWER_TRAIL):
            swing = self.lead if self.phase == StepPhase.LOWER_LEAD else 1-self.lead
            desired[swing] = self.cfg.touchdown_load_fraction
            desired[1-swing] = 1-self.cfg.touchdown_load_fraction
        yaw = math.atan2(m.yaw_rotation[1, 0], m.yaw_rotation[0, 0])
        heading_error = self.alignment_heading_error(m) if self.lock is None else _wrap(self.lock.heading-yaw)
        heading = heading_error or 0.
        aligning = self.phase == StepPhase.OBSERVE and abs(self.walking_command(m)[2]) > 0
        stopping = self.phase == StepPhase.SETTLE or (self.phase == StepPhase.OBSERVE and not aligning)
        shift_near = (self.phase in (StepPhase.SHIFT_LEAD, StepPhase.SHIFT_TRAIL)
                      and np.linalg.norm(self.reference_root[:2]-m.root_position[:2]) <= .06)
        body_stopping = stopping or shift_near or self.swing_foot is not None
        return {
            # Smooth tails give credit for approaching a distant reference;
            # score feet separately so swing error cannot erase stance credit.
            "feet_reference": active*float(np.mean(1/(1+feet_error/0.08**2))),
            "body_reference": active/(1+root_error/0.12**2),
            "load_reference": 0. if load is None else active/(1+float(np.sum((load-desired)**2))/0.25**2),
            "heading": 0. if heading_error is None else active*math.exp(-heading**2/math.radians(5)**2),
            "sole_tilt": active*float(np.mean(np.sum(m.foot_rotations[:, :2, 2]**2, axis=1))),
            "unsafe_contact": float(self.unsafe_contact_event),
            "downward_speed": active*max(0., -m.root_velocity[2]-0.10)**2,
            "progress": float(self.progress_event)/dt,
            "lift_progress": self.lift_progress/dt,
            "success": float(self.success_event)/dt,
            "stop_root_motion": float(m.root_velocity @ m.root_velocity) if body_stopping else 0.,
            "stop_feet_motion": float(np.mean(np.sum(m.sole_velocities**2, axis=1))) if stopping else 0.,
            "stop_angular_motion": float(m.root_angular_velocity @ m.root_angular_velocity) if body_stopping else 0.,
        }
