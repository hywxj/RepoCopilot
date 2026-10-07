"""Event-driven single-step teacher supervision with unchanged physical gates.

The dynamic teacher is used only to initialize a declared standing state and
provide optional reference curves. During execution this class requires an
explicit motor-control callback; it cannot fall back to teacher motor control.
Foot placement is accepted anywhere inside the complete-foot safe region.
Contact/pose and the known scene edge remain explicit simulation oracles.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .mujoco_stair_motion import DynamicTeachingEpisode, _interpolate
from .stair_step_controller import StepPhase, body_height_limit


@dataclass(frozen=True)
class EventStairCfg:
    clock_ramp_s: float = .10
    phase_timeout_s: float = 8.
    episode_timeout_s: float = 32.
    stable_confirmation_s: float = .25
    finish_hold_s: float = 1.

    def __post_init__(self):
        values = np.array([self.clock_ramp_s, self.phase_timeout_s, self.episode_timeout_s,
                           self.stable_confirmation_s, self.finish_hold_s])
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Event supervisor durations must be finite and positive.")
        if self.stable_confirmation_s < .25 or self.finish_hold_s < 1.:
            raise ValueError("Keep the original 0.25 s stability test and at least 1 s autonomous hold.")


def phase_clock(elapsed, duration, ramp=.10):
    """C2 virtual time with unit-speed interior and zero-speed saturated ends.

    Unlike clamping a moving trajectory, position/velocity/acceleration remain
    continuous when a physical event requires an arbitrarily longer wait.
    """
    if duration <= 0 or ramp <= 0 or not np.isfinite([elapsed, duration, ramp]).all():
        raise ValueError("Finite time and positive clock durations are required.")
    ramp = min(float(ramp), float(duration))
    if elapsed <= 0:
        return 0., 0., 0., False
    if elapsed >= duration+ramp:
        return float(duration), 0., 0., True
    if elapsed < ramp:
        u = elapsed/ramp
        integral = 2.5*u**4-3*u**5+u**6
        rate = 10*u**3-15*u**4+6*u**5
        acceleration = (30*u**2-60*u**3+30*u**4)/ramp
        return ramp*integral, rate, acceleration, False
    if elapsed <= duration:
        return elapsed-ramp/2, 1., 0., False
    u = (elapsed-duration)/ramp
    integral = 2.5*u**4-3*u**5+u**6
    rate = 10*u**3-15*u**4+6*u**5
    acceleration = (30*u**2-60*u**3+30*u**4)/ramp
    return duration-ramp/2+ramp*(u-integral), 1-rate, -acceleration, False


class EventDrivenStairEpisode(DynamicTeachingEpisode):
    reference_supervisor = "event"

    def __init__(self, *args, event_cfg=None, **kwargs):
        self.event_cfg = EventStairCfg() if event_cfg is None else event_cfg
        super().__init__(*args, **kwargs)
        self.event_phase = 0
        self.event_phase_start = float(self.data.time)
        self.event_last_time = float(self.data.time)
        self.event_gate_dwell = 0.
        self.event_count = 0
        self.last_event = "initialized"
        self.clearance_confirmed = np.zeros(2, dtype=bool)
        self.step_completed = False
        self._event_success = False
        self.failure_diagnostics = None
        self.profile = "event_teacher_ascent" if self.direction > 0 else "event_teacher_descent"
        # A dynamic preview's zero-speed point need not be statically balanced.
        # Every event wait instead ends above the current supporting sole, with
        # zero reference velocity/acceleration. The old dynamic teacher is intact.
        self.reference_profile = "pause_feasible_position_v2"
        # With fixed 20 ms position targets, the .8 s ascent swing crossed the
        # riser 1.1 mm below the unchanged whole-sole clearance gate. Allow the
        # physical foot to lift and level before forward travel; descent keeps
        # its independently verified timing.
        durations = (1.6, 1. if self.direction > 0 else .92, 1.8,
                     1. if self.direction > 0 else .92, 1.6)
        self.edges = np.r_[0., np.cumsum(durations)]
        self.body_offset = self.com_start-self.root_start
        self.final_com_height = (self.com_start[2]+self.target[:, 2].mean()
                                 -self.source[:, 2].mean())
        self.support_com = np.tile(self.com_start, (2, 1))
        rotation = self.controller.lock.rotation[:2, :2]
        self.support_com[0, :2] = self.source[0, :2]+rotation @ np.array([.045, -.012])
        self.support_com[1, :2] = self.target[1, :2]+rotation @ np.array([-.045, .012])
        m = self.measurement()
        self.reference_hip_offsets = m.hip_offsets.copy()
        mixed_feet = np.stack((self.source[0], self.target[1]))
        ankle_offset = np.r_[rotation @ np.array([-.03, 0.]), .04]
        for goal in self.support_com:
            height = body_height_limit(goal[:2]-self.body_offset[:2], m.hip_offsets,
                                       mixed_feet+ankle_offset,
                                       self.controller.cfg.leg_reach_m-self.controller.cfg.leg_reach_reserve_m)
            if height is None:
                self.close()
                raise RuntimeError("Event mixed-support reference is outside leg reach.")
            goal[2] = min(self.com_start[2], height+self.body_offset[2]-.005)
        if self.direction < 0:
            self.support_com[1, 2] = min(self.final_com_height, self.support_com[1, 2])
        self.event_com_start = self.com_start.copy()
        self.event_com_goal = self.support_com[0].copy()

    def reset(self):
        # Same constructor/reset identity contract as the base viewer episode.
        cfg = self.event_cfg
        replacement = super().reset()
        replacement.event_cfg = cfg
        return replacement

    def _advance(self, torque_callback=None):
        if torque_callback is None:
            raise RuntimeError("Event execution requires an explicit motor-control callback; no teacher fallback.")
        return super()._advance(torque_callback)

    def _heading_aligned(self, measured):
        yaw = np.arctan2(measured.yaw_rotation[1, 0], measured.yaw_rotation[0, 0])
        error = (self.controller.lock.heading-yaw+np.pi) % (2*np.pi)-np.pi
        return abs(error) <= self.controller.cfg.max_heading_error_rad

    def _reference_state(self, elapsed):
        phase = self.event_phase
        duration = self.edges[phase+1]-self.edges[phase]
        com, velocity, acceleration = _interpolate(self.event_com_start, self.event_com_goal,
                                                  elapsed, duration)
        feet, fv, fa = self.plants.copy(), np.zeros((2, 3)), np.zeros((2, 3))
        contacts = np.ones(2, dtype=bool)
        if phase in (1, 3):
            foot = 1 if phase == 1 else 0
            if not self.touching[foot]:
                contacts[foot] = False
                feet[foot], fv[foot], fa[foot] = self._swing(foot, elapsed, duration)
        if phase in (0, 1):
            loads = np.array([1., 0.])
        else:
            span = feet[0, :2]-feet[1, :2]
            cop = com[:2]-(com[2]-feet[1, 2])/(9.81+acceleration[2])*acceleration[:2]
            left = float(np.clip(np.dot(cop-feet[1, :2], span)/np.dot(span, span), 0., 1.))
            loads = np.array([left, 1-left])
        return com, velocity, acceleration, feet, fv, fa, contacts, loads

    def _begin_next_phase(self, measured):
        # The finished quintic has zero derivatives: all later phases join C2,
        # including a physical event which arrives after an arbitrary wait.
        self.event_com_start = self.event_com_goal.copy()
        self.event_phase += 1
        if self.event_phase == 1:
            self.event_com_goal = self.support_com[0].copy()
        elif self.event_phase == 2:
            # Region acceptance permits an actual plant away from the nominal
            # target. The next single-support endpoint must follow that plant;
            # retaining the old target would put the waiting CoM off its sole.
            rotation = self.controller.lock.rotation[:2, :2]
            anchor = self.support_com[1].copy()
            anchor[:2] = self.plants[1, :2]+rotation @ np.array([-.045, .012])
            ankle_offset = np.r_[rotation @ np.array([-.03, 0.]), .04]
            mixed_feet = np.stack((self.source[0], self.plants[1]))
            height = body_height_limit(anchor[:2]-self.body_offset[:2], self.reference_hip_offsets,
                                       mixed_feet+ankle_offset,
                                       self.controller.cfg.leg_reach_m-self.controller.cfg.leg_reach_reserve_m)
            if height is None:
                raise RuntimeError("Accepted leading plant leaves no reachable mixed-support body reference.")
            anchor[2] = min(self.com_start[2], height+self.body_offset[2]-.005)
            if self.direction < 0:
                anchor[2] = min(self.final_com_height, anchor[2])
            self.support_com[1] = anchor
            self.event_com_goal = self.support_com[1].copy()
        elif self.event_phase == 3:
            self.event_com_goal = np.r_[self.support_com[1, :2], self.final_com_height]
        else:
            self.event_com_goal = np.r_[self.plants[:, :2].mean(axis=0)+
                self.com_start[:2]-self.source[:, :2].mean(axis=0), self.final_com_height]
        self.previous_phase = self.event_phase
        self.event_phase_start = measured.timestamp
        self.event_gate_dwell = 0.
        self.event_count += 1

    def _physical_events(self, measured, dt, clock_finished, reference_loads=None):
        """Advance only physical events; no teacher-point tracking tolerance."""
        c, phase = self.controller, self.event_phase
        load = measured.contact_forces[:, 2]/measured.body_weight
        if self.geometry_source == "depth":
            if measured.generation != c.lock.generation:
                raise RuntimeError("Depth map pose continuity changed after target lock.")
        for foot in range(2):
            if self.confirmed[foot]:
                support = c.physical_support_metrics(measured, foot, min_load=.12)
                if not support["valid"]:
                    self.failure_diagnostics = dict(kind="confirmed_support_lost", foot=foot,
                        motion_time=float(measured.timestamp-self.start_time), phase_index=int(phase),
                        support=support, sole_position_world=measured.sole_positions[foot].tolist(),
                        sole_velocity_world=measured.sole_velocities[foot].tolist(),
                        root_velocity_world=measured.root_velocity.tolist(),
                        root_angular_velocity_world=measured.root_angular_velocity.tolist())
                    raise RuntimeError(f"Confirmed support lost on foot {foot}.")
            # A touchdown impulse is not a weight-transfer confirmation. First
            # permit light regional contact to start TRANSFER/SETTLE. Confirm
            # the new support only once that stage actually intends to load it.
            loading = (foot == 1 and phase >= 2) or (foot == 0 and phase >= 4)
            intended = reference_loads is None or reference_loads[foot] >= .12
            if self.touching[foot] and loading and intended:
                supported = c._physical_support(measured, foot, min_load=.12, record_plant=False)
                self.support_dwell[foot] = self.support_dwell[foot]+dt if supported else 0.
                if self.support_dwell[foot]+1.e-9 >= c.cfg.confirmation_s:
                    self.confirmed[foot] = True
                    c._physical_support(measured, foot, min_load=.12, record_plant=True)
            elif not self.confirmed[foot]:
                self.support_dwell[foot] = 0.
        condition, event = False, ""
        if phase in (0, 2):
            swing = 1 if phase == 0 else 0
            fresh = (self.geometry_source != "depth" or
                     measured.timestamp-c.lock.last_seen[swing] <= c.cfg.target_max_age_s)
            condition = (load[swing] <= .08 and load[1-swing] >= .60
                         and np.linalg.norm(measured.sole_velocities[1-swing]) <= c.cfg.max_sole_speed
                         and self._heading_aligned(measured) and fresh
                         and (phase == 0 or self.confirmed[1]))
            self.event_gate_dwell = self.event_gate_dwell+dt if condition else 0.
            condition = self.event_gate_dwell+1.e-9 >= c.cfg.confirmation_s
            event = "leading_unloaded" if phase == 0 else "trailing_unloaded"
        elif phase in (1, 3):
            foot = 1 if phase == 1 else 0
            if load[1-foot] < .45:
                raise RuntimeError("Stance load lost during swing.")
            if (self.geometry_source == "depth" and
                    measured.timestamp-c.lock.last_seen[foot] > c.cfg.flight_occlusion_s):
                raise RuntimeError("Locked depth target expired during swing.")
            self._check_clearance(measured, foot)
            corners = np.array([[-.12, -.042, 0], [-.12, .042, 0], [.12, -.042, 0], [.12, .042, 0]])
            corners = corners @ measured.foot_rotations[foot].T+measured.sole_positions[foot]
            # Same known-riser oracle as the original teacher, not a camera claim.
            if self.direction > 0:
                cleared = corners[:, 2].min() >= self.target[foot, 2]+.02 and corners[:, 0].max() >= .22
            else:
                cleared = corners[:, 0].min() >= .22
            self.clearance_confirmed[foot] |= cleared
            at_target_height = abs(measured.sole_positions[foot, 2]-self.target[foot, 2]) <= c.cfg.height_tolerance
            if load[foot] >= .12 and at_target_height and not c._footprint_in_tread(measured, foot):
                raise RuntimeError("Unsafe loaded contact outside the complete-foot target region.")
            light_contact = c._physical_support(measured, foot, min_load=c.cfg.touchdown_contact_fraction,
                                                record_plant=False)
            if light_contact and self.clearance_confirmed[foot]:
                self._touch(measured, foot)
            condition = bool(self.touching[foot] and light_contact)
            event = "leading_region_contact" if phase == 1 else "trailing_region_contact"
        else:
            stable = (self.confirmed.all() and load.sum() >= .8
                      and all(c._physical_support(measured, f, min_load=.2, record_plant=False) for f in range(2))
                      and np.linalg.norm(measured.root_velocity) < .05
                      and np.linalg.norm(measured.root_angular_velocity) <= .30
                      and self._heading_aligned(measured))
            self.stable_for = self.stable_for+dt if stable else 0.
            self.step_completed |= self.stable_for+1.e-9 >= self.event_cfg.stable_confirmation_s
            self._event_success = bool(self.stable_for+1.e-9 >=
                                       self.event_cfg.stable_confirmation_s+self.event_cfg.finish_hold_s)
        if condition and clock_finished:
            self._begin_next_phase(measured)
            self.last_event = event

    def prepare_step(self):
        if self._prepared is not None:
            return self._prepared
        mujoco.mj_forward(self.model, self.data)
        if self.geometry_source == "depth":
            geometry = self._context.refresh_geometry(tracking_only=True)
            for surface in geometry.surfaces:
                if surface.track_id == self.controller.lock.track_id:
                    self.controller.lock.last_seen[:] = np.maximum(self.controller.lock.last_seen,
                                                                  surface.last_observed_time)
        m, cfg = self.measurement(), self.event_cfg
        if m.contact_forces is None:
            raise RuntimeError("Event phase supervision requires declared simulation contact truth.")
        dt = max(0., float(m.timestamp-self.event_last_time))
        self.event_last_time = float(m.timestamp)
        elapsed = float(m.timestamp-self.event_phase_start)
        duration = self.edges[self.event_phase+1]-self.edges[self.event_phase]
        reference_loads = self._reference_state(elapsed)[-1]
        self._physical_events(m, dt, elapsed >= duration, reference_loads)
        elapsed = float(m.timestamp-self.event_phase_start)
        if elapsed > cfg.phase_timeout_s:
            raise RuntimeError(f"stalled_phase_timeout: phase={self.event_phase}, elapsed={elapsed:.3f}s")
        motion_time = float(m.timestamp-self.start_time)
        if motion_time > cfg.episode_timeout_s:
            raise RuntimeError("event_episode_timeout")
        phase = self.event_phase
        duration = self.edges[phase+1]-self.edges[phase]
        local_time, rate = min(elapsed, duration), float(elapsed < duration)
        virtual_time = float(self.edges[phase]+local_time)
        com, com_v, com_a, feet, fv, fa, contacts, loads = self._reference_state(elapsed)
        root = com-(self.com_start-self.root_start)
        tag = (StepPhase.SHIFT_LEAD, StepPhase.LIFT_LEAD, StepPhase.TRANSFER,
               StepPhase.LIFT_TRAIL, StepPhase.SETTLE)[phase]
        if self._event_success:
            tag = StepPhase.COMPLETE
        sample = dict(time=float(m.timestamp), motion_time=motion_time, virtual_motion_time=virtual_time,
            phase=tag.name, phase_index=phase, phase_elapsed_s=elapsed, phase_clock_rate=rate,
            phase_event=self.last_event, phase_event_count=self.event_count,
            task_progress=1. if self.step_completed else phase/5., step_completed=bool(self.step_completed),
            finish_hold_s=max(0., self.stable_for-cfg.stable_confirmation_s), reference_supervisor="event",
            reference_profile=self.reference_profile,
            root=m.root_position.tolist(), feet=m.sole_positions.tolist(),
            root_velocity=m.root_velocity.tolist(), root_angular_velocity=m.root_angular_velocity.tolist(),
            load_fraction=(m.contact_forces[:, 2]/m.body_weight).tolist(), confirmed_plants=self.confirmed.tolist(),
            clearance_confirmed=self.clearance_confirmed.tolist(),
            com=self.data.subtree_com[self.teacher.reader.root_id].tolist(), com_reference=com.tolist(),
            root_reference=root.tolist(), foot_reference=feet.tolist(), stable_for_s=self.stable_for,
            failure="", success=self._event_success)
        maximum_loads = np.full(2, 1.5)
        if phase in (0, 2) and elapsed >= duration:
            maximum_loads[1 if phase == 0 else 0] = .04
        control = dict(root_reference=root, feet_reference=feet, contacts=contacts,
            dt=self.model.opt.timestep, load_reference=loads,
            minimum_loads=np.where(self.confirmed, .12, np.where(self.touching,
                                  self.controller.cfg.touchdown_contact_fraction, 0.)),
            maximum_loads=maximum_loads, com_reference=com, com_velocity=com_v, com_acceleration=com_a,
            sole_velocity_reference=fv, sole_acceleration_reference=fa, com_acceleration_limit=3.)
        self._prepared = sample, control
        return self._prepared
