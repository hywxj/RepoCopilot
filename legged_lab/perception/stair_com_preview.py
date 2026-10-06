"""Experimental approximate 3-D centroidal preview with prescribed COM height.

Prescribed COM height and proportional horizontal/vertical force sharing reduce
the centroidal moment equations to an iterated planar QP. Every returned sample
is checked against the 3-D force/moment equations at a finite set of times.
This remains a centroidal approximation, not a proof of continuous-time wrench,
articulated-body, contact tracking, joint, balance, or stair-clearance feasibility.
"""
from __future__ import annotations

import json
import numpy as np
from qpsolvers import solve_qp
from .com_preview import COMPreview, _finite_array, _finite_scalar, _polygon, plan_com_preview


def quintic_height(start_z, end_z, start_time, end_time):
    """Return a smooth prescribed height, held constant outside its time span."""
    start_z = _finite_scalar(start_z, "start_z")
    end_z = _finite_scalar(end_z, "end_z")
    start_time = _finite_scalar(start_time, "start_time")
    end_time = _finite_scalar(end_time, "end_time")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    def sample(t):
        t = _finite_scalar(t, "time")
        duration = end_time-start_time
        if t <= start_time:
            return float(start_z), 0., 0.
        if t >= end_time:
            return float(end_z), 0., 0.
        u = (t-start_time)/duration
        return (start_z+(end_z-start_z)*(10*u**3-15*u**4+6*u**5),
                (end_z-start_z)*(30*u*u-60*u**3+30*u**4)/duration,
                (end_z-start_z)*(60*u-180*u*u+120*u**3)/duration**2)
    return sample


def _force_split(com, acceleration, feet, bounds, half_extents, preference, gravity):
    """Exact feasible force share with a common local COP offset on both feet.

    Forces on the two soles are parallel, scaled by lambda and 1-lambda.
    This permits COP displacement inside a sole without inventing second-foot
    loading, unlike projection onto the line between sole centers.
    """
    effective_gravity = gravity+acceleration[2]
    if effective_gravity <= 0:
        raise RuntimeError("COM vertical acceleration requires nonpositive contact load")
    beta = (feet[0, 2]-feet[1, 2])*acceleration[:2]/effective_gravity
    span = feet[0, :2]-feet[1, :2]-beta
    offset = com[:2]-(com[2]-feet[1, 2])*acceleration[:2]/effective_gravity-feet[1, :2]
    low, high = map(float, bounds)
    for axis in range(2):
        if abs(span[axis]) < 1.e-12:
            if abs(offset[axis]) > half_extents[axis]+1.e-7:
                return None
        else:
            a, b = sorted(((offset[axis]-half_extents[axis])/span[axis],
                           (offset[axis]+half_extents[axis])/span[axis]))
            low, high = max(low, a), min(high, b)
    if low > high+1.e-7:
        return None
    share = float(np.clip(preference, low, max(low, high)))
    local_cop = offset-share*span
    forces = np.array([share, 1-share])[:, None]*np.r_[acceleration[:2]/gravity,
                                                                   effective_gravity/gravity]
    moments = np.cross(np.r_[local_cop, 0.], forces)
    centroidal_moment = np.cross(feet-com, forces).sum(axis=0)+moments.sum(axis=0)
    return share, local_cop, np.column_stack((forces, moments)), centroidal_moment


class HeightAwarePreview:
    """COM and normalized wrench references under the centroidal approximation."""
    def __init__(self, planar, source, target, height_profile, bounds, preference, half_extents, gravity):
        self.planar, self.source, self.target = planar, source, target
        self.height_profile, self.load_bounds = height_profile, bounds
        self.preference, self.half_extents, self.gravity = preference, half_extents, gravity
        self.metrics = planar.metrics
        self.phase_edges, self.time = planar.phase_edges, planar.time

    def _index(self, t):
        return max(0, min(len(self.planar.phase)-1, int(np.searchsorted(self.time, t, side="right"))-1))

    def feet_for_phase(self, phase):
        if not isinstance(phase, (int, np.integer)) or not 0 <= phase < 5:
            raise ValueError("phase must be an integer from 0 to 4")
        if phase < 2:
            return self.source
        if phase == 2:
            return np.stack((self.source[0], self.target[1]))
        return self.target

    def sample(self, t):
        """COM position, velocity and acceleration in 3-D, not torso references."""
        p, v, a = self.planar.sample(t)
        z, vz, az = self.height_profile(t)
        return np.r_[p, z], np.r_[v, vz], np.r_[a, az]

    def sample_wrenches(self, t):
        """Per-foot world-frame wrench divided by body weight, about sole center."""
        k = self._index(t)
        p, _, a = self.sample(t)
        result = _force_split(p, a, self.feet_for_phase(self.planar.phase[k]),
                              self.load_bounds[k], self.half_extents, self.preference[k], self.gravity)
        if result is None:
            raise RuntimeError(f"No 3-D wrench split at t={t:.6f}")
        return result[2]

    def sample_loads(self, t):
        """Desired normal forces divided by body weight; sum includes vertical accel."""
        return self.sample_wrenches(t)[:, 2]

    def save(self, path):
        pva = [self.sample(t) for t in self.time]
        np.savez_compressed(path, time=self.time, position=np.stack([x[0] for x in pva]),
                            velocity=np.stack([x[1] for x in pva]), acceleration=np.stack([x[2] for x in pva]),
                            phase=self.planar.phase, contacts=self.planar.contacts,
                            phase_edges=self.phase_edges, left_normal_force_share_bounds=self.load_bounds,
                            force_preference=self.preference,
                            wrenches=np.stack([self.sample_wrenches(t) for t in self.time]),
                            source_feet=self.source, target_feet=self.target,
                            metrics=json.dumps(self.metrics))


def plan_height_aware(source_feet, target_feet, *, initial_com, initial_velocity=(0., 0., 0.),
                      dt=.02, phase_durations=(.6, .8, .3, .8, .6), height_profile=None,
                      gravity=9.81, half_extents=(.06, .012), unload_lead_time=.06,
                      unload_max_load_bw=.04, contact_min_load_bw=.05,
                      max_acceleration=3., max_velocity=.7, max_jerk=20.,
                      position_weights=(5., 1000.), velocity_weights=(.2, .2),
                      acceleration_weight=.015, jerk_weight=.0005, friction=.6,
                      position_waypoints=(), position_bounds=(),
                      max_iterations=15):
    """Prescribe COM z and solve planar motion with mixed-height support forces.

    source/target are [left,right] XYZ sole centers. Optional height_profile(t)
    returns world z, vz, az. Default rises/descends from first landing to the
    second landing; callers may supply another physically assessed profile.
    position_waypoints entries are (time, target_xy, weight_xy); position_bounds
    entries are (time, axis, lower, upper). Both snap to the nearest dt node.
    """
    source = _finite_array(source_feet, (2, 3), "source_feet")
    target = _finite_array(target_feet, (2, 3), "target_feet")
    initial = _finite_array(initial_com, (3,), "initial_com")
    initial_v = _finite_array(initial_velocity, (3,), "initial_velocity")
    dt = _finite_scalar(dt, "dt", positive=True)
    gravity = _finite_scalar(gravity, "gravity", positive=True)
    max_jerk = _finite_scalar(max_jerk, "max_jerk", positive=True)
    friction = _finite_scalar(friction, "friction", positive=True)
    unload_max_load_bw = _finite_scalar(unload_max_load_bw, "unload_max_load_bw", nonnegative=True)
    contact_min_load_bw = _finite_scalar(contact_min_load_bw, "contact_min_load_bw", nonnegative=True)
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0):
        raise ValueError("max_iterations must be a positive integer")
    half = _finite_array(half_extents, (2,), "half_extents")
    durations = _finite_array(phase_durations, (5,), "phase_durations", nonnegative=True)
    counts = np.rint(durations/dt).astype(int)
    if np.any(counts < 1):
        raise ValueError("five phase durations of at least one dt are required")
    edges = np.r_[0, np.cumsum(counts)]*dt
    if height_profile is None:
        height_profile = quintic_height(initial[2], initial[2]+np.mean(target[:, 2]-source[:, 2]),
                                         edges[2], edges[4])
    if not callable(height_profile):
        raise ValueError("height_profile must be callable")
    supplied_profile = height_profile
    def height_profile(t):
        return _finite_array(supplied_profile(t), (3,), "height_profile result")
    waypoints = []
    for at, target_xy, weights_xy in position_waypoints:
        waypoints.append((_finite_scalar(at, "waypoint time"),
                          _finite_array(target_xy, (2,), "waypoint target"),
                          _finite_array(weights_xy, (2,), "waypoint weights", nonnegative=True)))
    position_waypoints = waypoints
    bounded_positions = []
    for at, axis, low, high in position_bounds:
        if not isinstance(axis, (int, np.integer)) or axis not in (0, 1):
            raise ValueError("position bound axis must be 0 or 1")
        low = _finite_scalar(low, "position lower bound")
        high = _finite_scalar(high, "position upper bound")
        if low > high:
            raise ValueError("position lower bound must not exceed upper bound")
        bounded_positions.append((_finite_scalar(at, "position bound time"), axis, low, high))
    position_bounds = bounded_positions
    seed = plan_com_preview(source[:, :2], target[:, :2], initial_com=initial[:2],
                            initial_velocity=initial_v[:2], dt=dt, phase_durations=phase_durations,
                            com_height=initial[2]-source[:, 2].mean(), gravity=gravity,
                            half_extents=half, unload_lead_time=unload_lead_time,
                            max_acceleration=max_acceleration, max_velocity=max_velocity,
                            max_jerk=max_jerk, position_weights=position_weights,
                            velocity_weights=velocity_weights,
                            acceleration_weight=acceleration_weight, jerk_weight=jerk_weight)
    phase, time = seed.phase, seed.time
    total = len(phase)
    heights = np.array([height_profile(t) for t in time])
    gamma = 1+heights[:, 2]/gravity
    if gamma.min() <= .2:
        raise ValueError("Vertical trajectory requires less than 0.2 BW total normal force")
    phase_feet = [source, source, np.stack((source[0], target[1])), target, target]
    bounds, preference, polygons = [], [], []
    for k, ph in enumerate(phase):
        low, high, pref = 0., 1., .5
        if ph == 1:
            low = high = pref = 1.
        elif ph == 3:
            low = high = pref = 0.
        elif ph == 2:
            high = 1-contact_min_load_bw/min(gamma[k:k+2])
            progress = np.clip((time[k]-edges[2])/max(dt, edges[3]-edges[2]-unload_lead_time), 0, 1)
            pref = 1-(10*progress**3-15*progress**4+6*progress**5)
        elif ph == 4:
            low = contact_min_load_bw/min(gamma[k:k+2])
            high = 1-low
        if ph == 0 and seed.zmp_support_phase[k] == 1:
            low, pref = 1-unload_max_load_bw/max(gamma[k:k+2]), 1.
        if ph == 2 and seed.zmp_support_phase[k] == 3:
            high, pref = unload_max_load_bw/max(gamma[k:k+2]), 0.
        if not 0 <= low <= high <= 1:
            raise ValueError("load bounds are incompatible with the prescribed vertical acceleration")
        bounds.append((low, high)); preference.append(pref)
        feet = phase_feet[ph]
        centers = np.array([lo*feet[0, :2]+(1-lo)*feet[1, :2] for lo in (low, high)])
        polygons.append(_polygon(centers, half))
    bounds, preference = np.asarray(bounds), np.asarray(preference)

    node, interval = np.arange(total+1)[:, None], np.arange(total)[None, :]
    ps = np.where(interval < node, (node-interval-.5)*dt**2, 0.)
    vs = (interval < node)*dt
    pm, vm = np.kron(ps, np.eye(2)), np.kron(vs, np.eye(2))
    pf = initial[:2]+time[:, None]*initial_v[:2]
    vf = np.tile(initial_v[:2], (total+1, 1))
    terminal = target[:, :2].mean(axis=0)
    u = time/time[-1]
    desired = initial[:2]+(10*u**3-15*u**4+6*u**5)[:, None]*(terminal-initial[:2])
    desired_v = ((30*u*u-60*u**3+30*u**4)/time[-1])[:, None]*(terminal-initial[:2])
    pw, vw = np.tile(position_weights, total+1), np.tile(velocity_weights, total+1)
    hessian = pm.T @ (pw[:, None]*pm)+vm.T @ (vw[:, None]*vm)
    linear = pm.T @ (pw*(pf-desired).ravel())+vm.T @ (vw*(vf-desired_v).ravel())
    ident = np.eye(2*total)
    jm = np.kron(np.diff(np.eye(total), axis=0)/dt, np.eye(2))
    hessian += (acceleration_weight+1.e-8)*ident+jerk_weight*(jm.T @ jm)
    for at, target_xy, weights_xy in position_waypoints:
        k = int(np.clip(np.rint(at/dt), 0, total))
        mapping = pm[2*k:2*k+2]
        weights_xy = np.asarray(weights_xy)
        hessian += mapping.T @ (weights_xy[:, None]*mapping)
        linear += mapping.T @ (weights_xy*(pf[k]-target_xy))
    equality = np.vstack((pm[-2:], vm[-2:], ident[:2], ident[-2:]))
    rhs = np.r_[terminal-pf[-1], -initial_v[:2], 0., 0., 0., 0.]
    shares = preference.copy()
    last_infeasible = None
    for iteration in range(max_iterations):
        ground = np.array([share*phase_feet[ph][0, 2]+(1-share)*phase_feet[ph][1, 2]
                           for share, ph in zip(shares, phase)])
        alpha = ((heights[:-1, 0]-ground)/(gravity*gamma[:-1]),
                 (heights[1:, 0]-ground)/(gravity*gamma[1:]))
        maps = (pm[:-2]-np.repeat(alpha[0], 2)[:, None]*ident,
                pm[2:]-np.repeat(alpha[1], 2)[:, None]*ident)
        rows, upper = [vm, -vm, jm, -jm], [np.full(2*(total+1), max_velocity)-vf.ravel(),
                     np.full(2*(total+1), max_velocity)+vf.ravel(),
                     np.full(len(jm), max_jerk), np.full(len(jm), max_jerk)]
        for at, axis, low, high in position_bounds:
            k = int(np.clip(np.rint(at/dt), 0, total))
            mapping = pm[2*k+axis]
            rows.extend((mapping[None], -mapping[None]))
            upper.extend((np.array([high-pf[k, axis]]), np.array([pf[k, axis]-low])))
        for k in range(total):
            n, b = polygons[k]
            reserve = max_acceleration*np.abs(n).sum(axis=1)*dt**2/8+.0004
            for side, zm in enumerate(maps):
                rows.append(n @ zm[2*k:2*k+2]); upper.append(b-reserve-n @ pf[k+side])
            for sx in (-1, 1):
                for sy in (-1, 1):
                    row = np.zeros(2*total); row[2*k:2*k+2] = [sx, sy]
                    rows.append(row[None]); upper.append(np.array([friction*gravity*min(gamma[k:k+2])]))
        solution = solve_qp(hessian, linear, np.vstack(rows), np.concatenate(upper), equality, rhs,
                            lb=np.full(2*total, -max_acceleration), ub=np.full(2*total, max_acceleration),
                            solver="quadprog")
        if solution is None or not np.isfinite(solution).all():
            raise RuntimeError(f"Height-aware QP infeasible at iteration {iteration}")
        acceleration = solution.reshape(total, 2)
        position, velocity = pf+(pm @ solution).reshape(total+1, 2), vf+(vm @ solution).reshape(total+1, 2)
        new_shares, infeasible = [], []
        for k in range(total):
            t = time[k]+.5*dt; z, vz, az = height_profile(t)
            p = np.r_[position[k]+.5*dt*velocity[k]+.125*dt*dt*acceleration[k], z]
            result = _force_split(p, np.r_[acceleration[k], az], phase_feet[phase[k]],
                                  bounds[k], half, preference[k], gravity)
            if result is None:
                infeasible.append(k); new_shares.append(shares[k])
            else:
                new_shares.append(result[0])
        new_shares = np.asarray(new_shares)
        difference = np.max(np.abs(shares-new_shares))
        shares = .25*shares+.75*new_shares
        last_infeasible = infeasible
        if not infeasible and difference < .001:
            break
    if last_infeasible:
        raise RuntimeError(f"Height iteration left force split infeasible at indices {last_infeasible}")
    metrics = dict(model="height-aware iterated centroidal preview, not physical validation",
                   duration_s=float(time[-1]), phase_durations_s=(counts*dt).tolist(),
                   unload_lead_time_s=float(unload_lead_time), unload_max_load_bw=unload_max_load_bw,
                   iterations=iteration+1, peak_abs_com_y_m=float(np.abs(position[:, 1]).max()),
                   max_abs_velocity_m_s=np.abs(velocity).max(axis=0).tolist(),
                   max_abs_acceleration_m_s2=np.abs(acceleration).max(axis=0).tolist(),
                   support_half_extents_m=half.tolist())
    planar = COMPreview(time, position, velocity, acceleration,
                        position[:-1]-alpha[0][:, None]*acceleration,
                        position[1:]-alpha[1][:, None]*acceleration,
                        phase, seed.contacts, seed.zmp_support_phase, edges, metrics)
    result = HeightAwarePreview(planar, source, target, height_profile, bounds, preference, half, gravity)
    max_moment, max_cop, max_friction, max_torsion, max_unloading = 0., 0., 0., 0., 0.
    for k in range(total):
        # Stay strictly within this interval at its right boundary, where both
        # acceleration and contact set may change. Also validate exact nodes via
        # the next interval's own first sample.
        for f in (0., .25, .5, .75, 1.-1.e-9):
            t = time[k]+f*dt
            p, _, a = result.sample(t)
            split = _force_split(p, a, phase_feet[phase[k]], bounds[k], half, preference[k], gravity)
            if split is None:
                raise RuntimeError(f"3-D centroidal validation failed at t={t:.6f}")
            share, local_cop, w, moment = split
            max_moment = max(max_moment, float(np.abs(moment).max()))
            max_cop = max(max_cop, float(np.max(np.abs(local_cop)-half)))
            max_friction = max(max_friction, float(np.max(np.abs(w[:, 0])+np.abs(w[:, 1])-friction*w[:, 2])))
            max_torsion = max(max_torsion, float(np.max(np.abs(w[:, 5])-.015*w[:, 2])))
            if phase[k] in (0, 2) and seed.zmp_support_phase[k] in (1, 3):
                swing = 1 if phase[k] == 0 else 0
                max_unloading = max(max_unloading, float(w[swing, 2]))
    metrics.update(max_centroidal_moment_residual_bw_m=max_moment,
                   max_local_cop_violation_m=max_cop,
                   max_friction_violation_bw=max_friction,
                   max_torsion_violation_bw_m=max_torsion,
                   max_pre_liftoff_swing_load_bw=max_unloading,
                   final_position_error_m=float(np.linalg.norm(position[-1]-terminal)),
                   final_speed_m_s=float(np.linalg.norm(velocity[-1])))
    if max_moment > 1.e-6 or max_cop > 1.e-6 or max_friction > 1.e-5 or max_torsion > 1.e-5:
        raise RuntimeError("3-D centroidal preview violates wrench constraints: "+str(metrics))
    return result
