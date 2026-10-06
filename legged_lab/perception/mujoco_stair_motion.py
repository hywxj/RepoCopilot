"""Single-step dynamic MuJoCo motion candidates with measured contact gates.

The scene supplies known, static tread geometry. These are joint-motor
controllers, not learned policies or a general approach/continuous-stair skill.
Up and down share support validation and inverse dynamics, with separately
validated motion profiles. Preview feasibility alone never counts as success.
"""

import contextlib
import io

import mujoco
import numpy as np

from .com_preview import plan_com_preview
from .stair_com_preview import plan_height_aware, quintic_height
from .mujoco_step_teacher import MujocoStepTeacher
from .stair_step_controller import StepPhase


def _smooth(t, duration):
    if t <= 0:
        return 0., 0., 0.
    if t >= duration:
        return 1., 0., 0.
    s = t/duration
    return (10*s**3-15*s**4+6*s**5,
            (30*s**2-60*s**3+30*s**4)/duration,
            (60*s-180*s**2+120*s**3)/duration**2)


def _interpolate(start, end, t, duration):
    f, v, a = _smooth(t, duration)
    delta = end-start
    return start+f*delta, v*delta, a*delta


class DynamicTeachingEpisode:
    """Physically execute one up/down cycle from a declared starting distance."""

    def __init__(self, height=.11, width=.32, direction=1, initial_forward_offset=.05):
        # Local import keeps the original staged teacher independently usable.
        from legged_lab.scripts.mujoco_stair_teacher import TeachingEpisode
        if direction not in (-1, 1) or not np.isfinite(initial_forward_offset):
            raise ValueError("Direction must be +1/-1 and the initial offset must be finite.")
        self.height, self.width, self.direction = height, width, direction
        self.initial_forward_offset = float(initial_forward_offset)
        self._context = TeachingEpisode(height, width, direction)
        self.model, self.data = self._context.model, self._context.data
        # Set the initial scene pose once, before any physics. Never teleport
        # during an action to conceal an approach or contact failure.
        self.data.qpos[0] += self.initial_forward_offset
        mujoco.mj_forward(self.model, self.data)
        self._context.initial_root = self._context.measurement().root_position.copy()
        self._context.initial_feet = self._context.measurement().sole_positions.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            while self._context.controller.lock is None:
                self._context.tick()
        self.controller = self._context.controller
        self.teacher = self._make_teacher()
        self._context.teacher = self.teacher
        self.dt = .01
        m = self.measurement()
        self.source = m.sole_positions.copy()
        self.target = self.controller.lock.targets.copy()
        if direction < 0:
            self.target[:, 0] -= .005
        lock = self.controller.lock
        for foot in range(2):
            local = (self.target[foot]-lock.root_position) @ lock.rotation
            if not lock.geometry.footprint_supported(lock.surface_id, local[:2], lock.geometry.heading_rad):
                raise RuntimeError("Motion reference is outside the full-sole support region.")
        lock.targets[:] = self.target
        self.target_region = self._context.target_region
        self.target_region["reference_soles_world"] = self.target.tolist()
        self.target_region_quads = self._context.target_region_quads
        self.root_start = m.root_position.copy()
        self.com_start = self.data.subtree_com[self.teacher.reader.root_id].copy()
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac, self.teacher.reader.root_id)
        velocity = jac @ self.data.qvel
        if direction > 0:
            self.plan = plan_com_preview(self.source[:, :2], self.target[:, :2],
                                        initial_com=self.com_start[:2], initial_velocity=velocity[:2],
                                        com_height=.75, phase_durations=(.6, .8, .3, .8, .6),
                                        position_weights=(5., 1000.), max_jerk=20., unload_lead_time=.06)
        else:
            height_profile = quintic_height(self.com_start[2], self.com_start[2]+
                                            self.target[:, 2].mean()-self.source[:, 2].mean(), .6, 1.40)
            self.plan = plan_height_aware(self.source, self.target, initial_com=self.com_start,
                                         initial_velocity=velocity, phase_durations=(.6, .92, .4, .92, .6),
                                         height_profile=height_profile, position_weights=(5., 1000.),
                                         max_jerk=20., unload_lead_time=.06, contact_min_load_bw=.15,
                                         position_bounds=((1.52, 0, .20, .23),))
        self.edges = self.plan.phase_edges
        self.start_time = float(self.data.time)
        self.plants = self.source.copy()
        self.touching = np.zeros(2, dtype=bool)
        self.confirmed = np.zeros(2, dtype=bool)
        self.support_dwell = np.zeros(2)
        self.previous_phase = 0
        self.stable_for = 0.
        self.samples, self.trace = [], []
        self.profile = "dynamic_ascent" if direction > 0 else "dynamic_descent"

    def _make_teacher(self):
        weights = (120., 120., 100.) if self.direction > 0 else (40., 40., 15.)
        return MujocoStepTeacher(self.model, self.data, posture_control=True, com_tracking_weights=weights)

    def measurement(self):
        return self._context.measurement()

    def reset(self):
        """New episode while preserving the viewer's model/data identities."""
        replacement = type(self)(self.height, self.width, self.direction, self.initial_forward_offset)
        mujoco.mj_copyData(self.data, self.model, replacement.data)
        replacement.model, replacement.data = self.model, self.data
        replacement._context.model, replacement._context.data = self.model, self.data
        replacement.teacher = replacement._make_teacher()
        replacement._context.teacher = replacement.teacher
        return replacement

    def _swing(self, foot, elapsed, duration):
        start, end = self.source[foot], self.target[foot]
        if self.direction > 0:
            p, v, a = _interpolate(start, end, elapsed-.35*duration, .57*duration)
            peak_time, lower_time, end_time = .45*duration, .65*duration, .95*duration
            apex = max(start[2], end[2])+.055
        else:
            duration -= .12  # Leave time for real touchdown and speed convergence.
            p, v, a = _interpolate(start, end, elapsed, .65*duration)
            peak_time, lower_time, end_time = .30*duration, .58*duration, duration
            apex = max(start[2], end[2])+.025
        if elapsed < peak_time:
            p[2], v[2], a[2] = _interpolate(start[2], apex, elapsed, peak_time)
        elif elapsed < lower_time:
            p[2], v[2], a[2] = apex, 0., 0.
        else:
            p[2], v[2], a[2] = _interpolate(apex, end[2]-.002, elapsed-lower_time, end_time-lower_time)
        return p, v, a

    def _touch(self, m, foot):
        if not self.touching[foot]:
            self.plants[foot] = m.sole_positions[foot]
            self.plants[foot, 2] -= self.controller.cfg.touchdown_probe_depth
            self.touching[foot] = True

    def _check_clearance(self, m, foot):
        corners = np.array([[-.12, -.042, 0], [-.12, .042, 0], [.12, -.042, 0], [.12, .042, 0]])
        corners = corners @ m.foot_rotations[foot].T+m.sole_positions[foot]
        # This scene's first riser is x=.22. The goal region itself remains the
        # observed surface mask and full-foot validator, not this edge test.
        if self.direction > 0 and corners[:, 0].max() >= .22 and corners[:, 0].min() < .22:
            if corners[:, 2].min() < self.target[foot, 2]+.02:
                raise RuntimeError("Measured whole sole lacks stair-edge clearance.")
        if self.direction < 0 and corners[:, 0].min() < .22:
            if corners[:, 2].min() < self.source[foot, 2]-.002:
                raise RuntimeError("Descending heel has not cleared the source edge.")

    def _advance(self):
        mujoco.mj_forward(self.model, self.data)
        m, c = self.measurement(), self.controller
        t = float(self.data.time-self.start_time)
        phase = min(4, int(np.searchsorted(self.edges[1:], t+1.e-9, side="right")))
        measured_load = m.contact_forces[:, 2]/m.body_weight
        if phase != self.previous_phase:
            if phase in (1, 3):
                foot = 1 if phase == 1 else 0
                if phase == 3 and not self.confirmed[1-foot]:
                    raise RuntimeError("Leading support is not confirmed before trailing liftoff.")
                if measured_load[foot] > .08 or measured_load[1-foot] < .60:
                    raise RuntimeError(f"Planned liftoff lacks measured unloading: {measured_load}.")
            if phase in (2, 4):
                foot = 1 if phase == 2 else 0
                if not c._physical_support(m, foot, min_load=c.cfg.touchdown_contact_fraction, record_plant=False):
                    raise RuntimeError("Planned landing did not establish regional light contact.")
                self._touch(m, foot)
            self.previous_phase = phase
        feet, fv, fa = self.plants.copy(), np.zeros((2, 3)), np.zeros((2, 3))
        contacts = np.ones(2, dtype=bool)
        if phase in (1, 3):
            foot = 1 if phase == 1 else 0
            contacts[foot] = False
            feet[foot], fv[foot], fa[foot] = self._swing(foot, t-self.edges[phase], self.edges[phase+1]-self.edges[phase])
            self._check_clearance(m, foot)
            early_window = .10 if self.direction > 0 else .20
            if self.touching[foot] or (t > self.edges[phase+1]-early_window and
                                      c._physical_support(m, foot, min_load=c.cfg.touchdown_contact_fraction, record_plant=False)):
                self._touch(m, foot)
                contacts[foot], feet[foot], fv[foot], fa[foot] = True, self.plants[foot], 0., 0.
        maximum_loads = np.full(2, 1.5)
        if self.direction > 0:
            cp, cv, ca = self.plan.sample(t)
            z, vz, az = _interpolate(self.com_start[2], self.com_start[2]+
                                     self.target[:, 2].mean()-self.source[:, 2].mean(),
                                     t-self.edges[3], self.edges[4]-self.edges[3])
            com, com_v, com_a = np.r_[cp, z], np.r_[cv, vz], np.r_[ca, az]
            if contacts.all():
                zmp = cp-.75/9.81*ca
                span = feet[0, :2]-feet[1, :2]
                left = float(np.clip(np.dot(zmp-feet[1, :2], span)/np.dot(span, span), 0, 1))
                loads = np.array([left, 1-left])
            else:
                loads = contacts.astype(float)
            if phase in (0, 2) and self.edges[phase+1]-t < .12:
                maximum_loads[1 if phase == 0 else 0] = .02+1.48*_smooth(self.edges[phase+1]-t, .12)[0]
        else:
            com, com_v, com_a = self.plan.sample(t)
            loads = self.plan.sample_loads(t)
            if phase in (0, 2) and t >= self.edges[phase+1]-.06:
                maximum_loads[1 if phase == 0 else 0] = .04
        for foot in range(2):
            if self.touching[foot]:
                support = c._physical_support(m, foot, min_load=.12, record_plant=False)
                if self.confirmed[foot] and not support:
                    raise RuntimeError(f"Confirmed support lost on foot {foot}.")
                self.support_dwell[foot] = self.support_dwell[foot]+self.model.opt.timestep if support else 0.
                if self.support_dwell[foot] >= c.cfg.confirmation_s:
                    self.confirmed[foot] = True
                    c._physical_support(m, foot, min_load=.12, record_plant=True)
        root = com-(self.com_start-self.root_start)
        stable = (phase == 4 and self.confirmed.all() and measured_load.sum() >= .8 and
                  all(c._physical_support(m, f, min_load=.2, record_plant=False) for f in range(2)) and
                  np.linalg.norm(m.root_velocity) < .05 and np.linalg.norm(m.root_angular_velocity) <= .30)
        self.stable_for = self.stable_for+self.model.opt.timestep if stable else 0.
        finished = bool(t >= self.edges[-1]+1.-1.e-8)
        if finished and self.stable_for < .25:
            raise RuntimeError("No final continuous double-support stability.")
        tag = (StepPhase.SHIFT_LEAD, StepPhase.LIFT_LEAD, StepPhase.TRANSFER,
               StepPhase.LIFT_TRAIL, StepPhase.SETTLE)[phase]
        if finished:
            tag = StepPhase.COMPLETE
        sample = {"time": float(self.data.time), "motion_time": t, "phase": tag.name,
                  "root": m.root_position.tolist(), "feet": m.sole_positions.tolist(),
                  "root_velocity": m.root_velocity.tolist(), "root_angular_velocity": m.root_angular_velocity.tolist(),
                  "load_fraction": measured_load.tolist(), "confirmed_plants": self.confirmed.tolist(),
                  "com": self.data.subtree_com[self.teacher.reader.root_id].tolist(), "com_reference": com.tolist(),
                  "foot_reference": feet.tolist(), "stable_for_s": self.stable_for, "failure": "", "success": finished}
        self.data.ctrl[:] = self.teacher.control(root, feet, contacts, self.model.opt.timestep, loads,
                                                minimum_loads=np.where(self.confirmed, .18, np.where(self.touching, .05, 0.)),
                                                maximum_loads=maximum_loads, com_reference=com,
                                                com_velocity=com_v, com_acceleration=com_a,
                                                sole_velocity_reference=fv, sole_acceleration_reference=fa,
                                                com_acceleration_limit=3.)
        torque = self.data.ctrl.copy()
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or self.data.qpos[2] < self.root_start[2]-.25:
            raise RuntimeError("Dynamic teacher lost physical balance.")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError("External applied force is not permitted in this motion candidate.")
        return sample, torque

    def tick(self):
        state = (float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy())
        results = [self._advance() for _ in range(4)]
        sample = results[0][0]
        sample["next_time"] = float(self.data.time)
        phase = StepPhase[sample["phase"]]
        self.trace.append((*state, np.eye(len(StepPhase))[int(phase)], np.stack([row[1] for row in results])))
        self.samples.append(sample)
        return sample

    def save_trace(self, path):
        if not self.samples or not self.samples[-1]["success"]:
            raise ValueError("Only physically successful episodes can be saved as motion candidates.")
        np.savez_compressed(path, state_time=np.array([s[0] for s in self.trace]),
                            qpos=np.stack([s[1] for s in self.trace]), qvel=np.stack([s[2] for s in self.trace]),
                            step_features=np.stack([s[3] for s in self.trace]), motor_torques=np.stack([s[4] for s in self.trace]),
                            step_features_semantics="phase_onehot_only_not_policy_observation",
                            actuator_joint_names=np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                                                           for j in self.teacher.joints]),
                            physics_dt=self.model.opt.timestep, supervisor_dt=self.dt,
                            task_goal="both_full_soles_supported_on_same_observed_tread",
                            target_region_quads_world=self.target_region_quads, reference_soles_world=self.target,
                            initial_forward_offset_m=self.initial_forward_offset,
                            trace_role="physical_motion_candidate", motion_quality_validated=False,
                            controller_profile=self.profile, known_geometry=True, force_oracle=True, learned_policy=False)
