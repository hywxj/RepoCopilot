"""Experimental constant-height, planar center-of-mass (COM) preview.

This is a constant-height, zero centroidal angular-momentum LIPM approximation.
Feasible preview inequalities do not prove 3-D stair, contact, or joint feasibility.
The collision geometry and controller's contact-wrench limits are not changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from qpsolvers import solve_qp
from scipy.spatial import ConvexHull


def _finite_array(value, shape, name, *, nonnegative=False):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _finite_scalar(value, name, *, positive=False, nonnegative=False):
    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(array)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass
class COMPreview:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    zmp: np.ndarray
    zmp_end: np.ndarray
    phase: np.ndarray
    contacts: np.ndarray
    zmp_support_phase: np.ndarray
    phase_edges: np.ndarray
    metrics: dict

    def sample(self, time: float):
        """Return exact piecewise-constant-acceleration position, velocity, accel."""
        time = _finite_scalar(time, "time")
        if time >= self.time[-1]:
            return self.position[-1].copy(), np.zeros(2), np.zeros(2)
        index = max(0, min(len(self.acceleration)-1,
                           int(np.searchsorted(self.time, time, side="right"))-1))
        elapsed = max(0., float(time-self.time[index]))
        acceleration = self.acceleration[index]
        return (self.position[index]+elapsed*self.velocity[index]+.5*elapsed**2*acceleration,
                self.velocity[index]+elapsed*acceleration,
                acceleration.copy())

    def save(self, path):
        np.savez_compressed(path, time=self.time, position=self.position,
                            velocity=self.velocity, acceleration=self.acceleration,
                            zmp=self.zmp, zmp_end=self.zmp_end, phase=self.phase,
                            contacts=self.contacts, zmp_support_phase=self.zmp_support_phase,
                            phase_edges=self.phase_edges,
                            metrics=json.dumps(self.metrics))


def _polygon(centers, half_extents):
    offsets = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])*half_extents
    points = (np.asarray(centers)[:, None, :]+offsets).reshape(-1, 2)
    # Normalized half spaces: normal @ point <= offset.
    equations = ConvexHull(points).equations
    return equations[:, :2], -equations[:, 2]


def plan_com_preview(source_feet, target_feet, *, initial_com=None,
                     initial_velocity=(0., 0.), final_com=None, dt=.02,
                     phase_durations=(.6, .9, .3, .9, .6),
                     com_height=.65, gravity=9.81, half_extents=(.06, .012),
                     max_acceleration=3., max_velocity=.7,
                     max_jerk=None,
                     unload_lead_time=.06,
                     position_weights=(5., 100.), velocity_weights=(.2, .2),
                     acceleration_weight=.015, jerk_weight=.0005,
                     zero_endpoint_acceleration=True):
    """Plan prep / right swing / transfer / left swing / settle in world x/y.

    Feet are [left, right] rows. This prototype assumes feet face world +x,
    rectangular support patches, and constant height above a common plane.
    Durations are rounded to whole dt intervals. Positions/velocities have N+1
    rows; acceleration, support labels, and interval-endpoint ZMP have N rows.
    During the final unload_lead_time before either liftoff, ZMP is already
    constrained to the upcoming stance sole while physical contacts remain
    double. This prepares unloading before the measured-load transition gate.
    """
    source = _finite_array(source_feet, (2, 2), "source_feet")
    target = _finite_array(target_feet, (2, 2), "target_feet")
    dt = _finite_scalar(dt, "dt", positive=True)
    com_height = _finite_scalar(com_height, "com_height", positive=True)
    gravity = _finite_scalar(gravity, "gravity", positive=True)
    max_acceleration = _finite_scalar(max_acceleration, "max_acceleration", positive=True)
    max_velocity = _finite_scalar(max_velocity, "max_velocity", positive=True)
    if max_jerk is not None:
        max_jerk = _finite_scalar(max_jerk, "max_jerk", positive=True)
    position_weights = _finite_array(position_weights, (2,), "position_weights", nonnegative=True)
    velocity_weights = _finite_array(velocity_weights, (2,), "velocity_weights", nonnegative=True)
    acceleration_weight = _finite_scalar(acceleration_weight, "acceleration_weight", nonnegative=True)
    jerk_weight = _finite_scalar(jerk_weight, "jerk_weight", nonnegative=True)
    durations = _finite_array(phase_durations, (5,), "phase_durations", nonnegative=True)
    counts = np.rint(durations/dt).astype(int)
    if np.any(counts < 1):
        raise ValueError("five phase durations of at least one dt are required")
    unload_lead_time = _finite_scalar(unload_lead_time, "unload_lead_time", nonnegative=True)
    unload_intervals = int(np.ceil(unload_lead_time/dt-1.e-10))
    if unload_intervals >= min(counts[0], counts[2]):
        raise ValueError("unload lead time must be shorter than prep and transfer")
    half_extents = _finite_array(half_extents, (2,), "half_extents")
    if np.any(half_extents <= 0):
        raise ValueError("positive rectangular half extents are required")
    initial = source.mean(axis=0) if initial_com is None else _finite_array(initial_com, (2,), "initial_com")
    terminal = target.mean(axis=0) if final_com is None else _finite_array(final_com, (2,), "final_com")
    initial_velocity = _finite_array(initial_velocity, (2,), "initial_velocity")
    total = int(counts.sum())
    time = np.arange(total+1)*dt
    phase = np.repeat(np.arange(5), counts)
    zmp_support_phase = phase.copy()
    if unload_intervals:
        for double_phase in (0, 2):
            end = int(counts[:double_phase+1].sum())
            zmp_support_phase[end-unload_intervals:end] = double_phase+1
    phase_contacts = np.array([[1, 1], [1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)
    phase_support = [source, source[[0]], np.stack((source[0], target[1])),
                     target[[1]], target]
    halfspaces = [_polygon(centers, half_extents) for centers in phase_support]

    # p[k] = p0 + k*dt*v0 + dt^2 sum_{j<k}(k-j-.5)*a[j].
    node, interval = np.arange(total+1)[:, None], np.arange(total)[None, :]
    p_scalar = np.where(interval < node, (node-interval-.5)*dt**2, 0.)
    v_scalar = (interval < node)*dt
    p_map, v_map = np.kron(p_scalar, np.eye(2)), np.kron(v_scalar, np.eye(2))
    free_position = initial+time[:, None]*initial_velocity
    free_velocity = np.tile(initial_velocity, (total+1, 1))

    u = time/time[-1]
    blend = 10*u**3-15*u**4+6*u**5
    blend_velocity = (30*u**2-60*u**3+30*u**4)/time[-1]
    desired_position = initial+blend[:, None]*(terminal-initial)
    desired_velocity = blend_velocity[:, None]*(terminal-initial)
    p_weights = np.tile(np.asarray(position_weights), total+1)
    v_weights = np.tile(np.asarray(velocity_weights), total+1)
    hessian = p_map.T @ (p_weights[:, None]*p_map)+v_map.T @ (v_weights[:, None]*v_map)
    linear = (p_map.T @ (p_weights*(free_position-desired_position).ravel())
              +v_map.T @ (v_weights*(free_velocity-desired_velocity).ravel()))
    hessian += (acceleration_weight+1.e-8)*np.eye(2*total)
    difference = np.diff(np.eye(total), axis=0)/dt
    jerk_map = np.kron(difference, np.eye(2))
    hessian += jerk_weight*(jerk_map.T @ jerk_map)

    # Also constrain interval-end ZMP under that interval's contact polygon.
    # A bounded quadratic deviates from its endpoint chord by at most
    # |n @ a|*dt^2/8. Reserving the acceleration-box worst case guarantees that
    # between-node ZMP stays inside the original polygon, not just its samples.
    identity = np.eye(2*total)
    zmp_maps = (p_map[:-2]-com_height/gravity*identity,
                p_map[2:]-com_height/gravity*identity)
    rows, upper = [], []
    for k in range(total):
        normal, bound = halfspaces[zmp_support_phase[k]]
        chord_reserve = max_acceleration*np.sum(np.abs(normal), axis=1)*dt**2/8
        for offset, zmap in enumerate(zmp_maps):
            rows.append(normal @ zmap[2*k:2*k+2])
            upper.append(bound-chord_reserve-normal @ free_position[k+offset])
    rows.extend((v_map, -v_map))
    upper.extend((np.full(2*(total+1), max_velocity)-free_velocity.ravel(),
                  np.full(2*(total+1), max_velocity)+free_velocity.ravel()))
    if max_jerk is not None:
        rows.extend((jerk_map, -jerk_map))
        upper.extend((np.full(len(jerk_map), max_jerk), np.full(len(jerk_map), max_jerk)))
    equality = [p_map[-2:], v_map[-2:]]
    equality_rhs = [terminal-free_position[-1], -initial_velocity]
    if zero_endpoint_acceleration:
        equality.extend((identity[:2], identity[-2:]))
        equality_rhs.extend((np.zeros(2), np.zeros(2)))

    solution = solve_qp(hessian, linear, np.vstack(rows), np.concatenate(upper),
                        np.vstack(equality), np.concatenate(equality_rhs),
                        lb=np.full(2*total, -max_acceleration),
                        ub=np.full(2*total, max_acceleration), solver="quadprog")
    if solution is None or not np.isfinite(solution).all():
        raise RuntimeError("No feasible planar preview for these timings/bounds")
    acceleration = solution.reshape(total, 2)
    position = free_position+(p_map @ solution).reshape(total+1, 2)
    velocity = free_velocity+(v_map @ solution).reshape(total+1, 2)
    zmp = position[:-1]-com_height/gravity*acceleration
    zmp_end = position[1:]-com_height/gravity*acceleration

    # Exactly check potential extrema of each half-space polynomial, including
    # zero velocity along oblique double-support polygon edges.
    maximum_violation = -np.inf
    for k in range(total):
        normal, bound = halfspaces[zmp_support_phase[k]]
        for n, b in zip(normal, bound):
            samples = [0., dt]
            curvature = n @ acceleration[k]
            if abs(curvature) > 1.e-12:
                critical = -(n @ velocity[k])/curvature
                if 0 < critical < dt:
                    samples.append(critical)
            for t in samples:
                z = zmp[k]+t*velocity[k]+.5*t*t*acceleration[k]
                maximum_violation = max(maximum_violation, float(n @ z-b))
    if maximum_violation > 1.e-6:
        raise RuntimeError(f"Between-node ZMP violation: {maximum_violation:.6g} m")

    metrics = dict(duration_s=float(time[-1]), phase_durations_s=(counts*dt).tolist(),
                   unload_lead_time_s=float(unload_intervals*dt),
                   com_height_m=float(com_height), support_half_extents_m=half_extents.tolist(),
                   peak_abs_com_y_m=float(np.max(np.abs(position[:, 1]))),
                   peak_lateral_deviation_from_midline_m=float(np.max(
                       np.abs(position[:, 1]-desired_position[:, 1]))),
                   com_y_range_m=[float(position[:, 1].min()), float(position[:, 1].max())],
                   max_abs_velocity_m_s=np.abs(velocity).max(axis=0).tolist(),
                   max_abs_acceleration_m_s2=np.abs(acceleration).max(axis=0).tolist(),
                   max_abs_jerk_m_s3=(np.abs(np.diff(acceleration, axis=0))/dt).max(axis=0).tolist(),
                   max_support_inequality_violation_m=maximum_violation,
                   final_position_error_m=float(np.linalg.norm(position[-1]-terminal)),
                   final_speed_m_s=float(np.linalg.norm(velocity[-1])),
                   dynamics_position_residual_m=float(np.max(np.abs(
                       position[1:]-position[:-1]-dt*velocity[:-1]-.5*dt*dt*acceleration))),
                   model="constant-height LIPM preview; not physical validation")
    return COMPreview(time, position, velocity, acceleration, zmp, zmp_end,
                      phase, phase_contacts[phase], zmp_support_phase,
                      np.r_[0, np.cumsum(counts)]*dt, metrics)
