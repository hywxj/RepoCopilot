"""Batched simulation-only motor curriculum; never a replacement for depth planning."""

import math

import torch
import torch.nn.functional as F

from .stair_step_controller import StepPhase, StepControlCfg


class StairSkillTeacher:
    """Train the same 39-feature stair actor on physical, known simulation treads."""

    def __init__(self, origins, near_edge, width, height, direction, body_weight):
        self.cfg = StepControlCfg()
        self.origins = origins
        self.rows = torch.arange(len(origins), device=origins.device)
        self.direction = direction
        self.body_weight = body_weight
        self.near = origins[:, 0] + near_edge
        self.far = self.near + width
        self.height = origins[:, 2] + direction * height
        self.phase = torch.full_like(self.rows, int(StepPhase.OBSERVE))
        self.lead = self.rows % 2
        self.elapsed = torch.zeros_like(self.near)
        self.confirmed = torch.zeros_like(self.near)
        self.pending = torch.ones_like(self.rows, dtype=torch.bool)
        self.clearance = torch.zeros_like(self.pending)
        self.crossing_elapsed = torch.zeros_like(self.near)
        self.lift_limit = torch.full_like(self.near, .02)
        self.lift_hold = torch.zeros_like(self.near)
        self.best_lift_height = torch.zeros_like(self.near)
        self.source_feet = torch.zeros(len(origins), 2, 3, device=origins.device)
        self.targets = torch.zeros_like(self.source_feet)
        self.reference_feet = torch.zeros_like(self.source_feet)
        self.swing_start = torch.zeros(len(origins), 3, device=origins.device)
        self.crossing_start = torch.zeros_like(self.swing_start)
        self.reference_root = torch.zeros_like(self.swing_start)
        self.initial_root = torch.zeros_like(self.swing_start)
        self.planted = torch.zeros(len(origins), 2, dtype=torch.bool, device=origins.device)
        self.plant_positions = torch.zeros_like(self.source_feet)
        self.success = torch.zeros_like(self.pending)
        self.failure = torch.zeros_like(self.pending)
        self.progress = torch.zeros_like(self.pending)

    def reset(self, ids):
        self.pending[ids] = True
        self.phase[ids] = int(StepPhase.OBSERVE)
        self.elapsed[ids] = self.confirmed[ids] = 0
        self.lead[ids] = torch.randint(2, (len(ids),), device=ids.device)
        self.planted[ids] = False
        self.clearance[ids] = False

    @staticmethod
    def smooth(value):
        value = value.clamp(0, 1)
        return value**3 * (10 - 15*value + 6*value**2)

    def update(self, root, rotation, soles, velocities, corners, normals, forces, root_vel, angular_vel, dt,
               hip_offsets=None, ankle_offsets=None, com_offset=None):
        cfg, rows = self.cfg, self.rows
        load = forces[:, :, 2].clamp_min(0)/self.body_weight[:, None]
        speed = velocities.norm(dim=-1)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        heading = -yaw
        aligned = heading.abs() <= cfg.max_heading_error_rad
        stable = (root_vel.norm(dim=-1) <= .12) & (angular_vel.norm(dim=-1) <= .30) & (speed.amax(dim=1) <= .08)
        both = (load.amin(dim=1) >= .20) & (load.sum(dim=1) >= .8)
        init = self.pending
        self.source_feet[init] = soles[init]
        self.source_feet[init, :, 2] = self.origins[init, None, 2]
        self.reference_feet[init] = self.source_feet[init]
        self.reference_root[init] = root[init]
        self.reference_root[init, 2] -= (soles[init, :, 2]-self.origins[init, None, 2]).mean(dim=1)
        self.initial_root[init] = self.reference_root[init]
        self.targets[init] = self.source_feet[init]
        self.targets[init, :, 0] = ((self.near+self.far)/2)[init, None]
        self.targets[init, :, 2] = self.height[init, None]
        # Half the resets start from an already aligned double-support setup.
        # This exposes weight-transfer data without counting OBSERVE as solved.
        self.phase[init & (self.rows % 2 == 0)] = int(StepPhase.SHIFT_LEAD)
        self.pending[:] = False
        self.elapsed += dt
        self.success[:] = self.failure[:] = self.progress[:] = False
        old = self.phase.clone()
        shift = (old == int(StepPhase.SHIFT_LEAD)) | (old == int(StepPhase.SHIFT_TRAIL))
        lift = (old == int(StepPhase.LIFT_LEAD)) | (old == int(StepPhase.LIFT_TRAIL))
        lower = (old == int(StepPhase.LOWER_LEAD)) | (old == int(StepPhase.LOWER_TRAIL))
        second = (old >= int(StepPhase.SHIFT_TRAIL)) & (old <= int(StepPhase.LOWER_TRAIL))
        swing = torch.where(second, 1-self.lead, self.lead)
        stance = 1-swing
        condition = torch.zeros_like(self.pending)
        next_phase = old.clone()

        # The entire sole must remain inside a bounded tread, not just its center.
        near_margin = .03 if self.direction > 0 else .02
        far_margin = .02 if self.direction > 0 else .03
        contained = ((corners[..., 0] >= self.near[:, None, None]+near_margin)
                     & (corners[..., 0] <= self.far[:, None, None]-far_margin)
                     & ((corners[..., 1]-self.origins[:, None, None, 1]).abs() <= 1.4)).all(dim=2)
        on_plane = (corners[..., 2]-self.height[:, None, None]).abs().amax(dim=2) <= cfg.height_tolerance
        centered = (soles[..., :2]-self.targets[..., :2]).norm(dim=-1) <= cfg.xy_tolerance
        upright = normals[..., 2] >= math.cos(cfg.foot_tilt_tolerance_rad)
        upward = forces[..., 2] >= .75*forces.norm(dim=-1)
        slip = (soles[..., :2]-self.plant_positions[..., :2]).norm(dim=-1) > cfg.max_slip
        supported = contained & on_plane & centered & upright & upward & (load >= .12) & (speed <= cfg.max_sole_speed)
        supported &= ~(self.planted & slip)

        observing = old == int(StepPhase.OBSERVE)
        condition[observing] = (stable & both & aligned)[observing]
        next_phase[observing] = int(StepPhase.SHIFT_LEAD)
        self.reference_root[shift, :2] = self.reference_feet[rows[shift], stance[shift], :2]
        condition[shift] = ((load[rows, swing] <= .08) & (load[rows, stance] >= .60)
                            & (speed[rows, stance] <= .08) & aligned
                            & (root_vel.norm(dim=-1) <= .12) & (angular_vel.norm(dim=-1) <= .30))[shift]
        self.shift_requirements = {
            "swing_unloaded": load[rows, swing] <= .08,
            "stance_loaded": load[rows, stance] >= .60,
            "stance_still": speed[rows, stance] <= .08,
            "heading_aligned": aligned,
            "body_stable": (root_vel.norm(dim=-1) <= .12) & (angular_vel.norm(dim=-1) <= .30),
        }
        self.last_soles, self.last_load = soles, load
        self.last_sole_speed = speed
        next_phase[shift] = old[shift]+1

        flight = lift | lower
        start = self.swing_start
        target = self.targets[rows, swing]
        apex = torch.maximum(start[:, 2], target[:, 2])+cfg.lift_clearance
        minimum_height = corners[rows, swing, :, 2].amin(dim=1)
        reached = torch.minimum(apex, torch.maximum(self.best_lift_height, minimum_height))
        lifting_unloaded = lift & (load[rows, swing] <= .08)
        lift_progress = torch.where(lifting_unloaded, (reached-self.best_lift_height)/(apex-start[:, 2]).clamp_min(.01), 0.)
        self.best_lift_height[lifting_unloaded] = reached[lifting_unloaded]
        vertical = lift & ~self.clearance
        tracked = (vertical & (minimum_height >= start[:, 2]+self.lift_limit-.005)
                   & (load[rows, swing] <= .08) & (speed[rows, swing] <= .08))
        self.lift_hold = torch.where(tracked, self.lift_hold+dt, 0.)
        increment = self.lift_hold+1.e-6 >= cfg.confirmation_s
        self.lift_limit[increment] += .02
        self.lift_hold[increment] = 0
        reference_apex = torch.minimum(apex, start[:, 2]+self.lift_limit)
        self.reference_feet[rows[vertical], swing[vertical]] = start[vertical]
        self.reference_feet[rows[vertical], swing[vertical], 2] += (
            self.smooth(self.elapsed/cfg.lift_duration_s)*(reference_apex-start[:, 2]))[vertical]
        cleared = vertical & (corners[rows, swing, :, 2].amin(dim=1) >= apex-.015) & (load[rows, swing] <= .08)
        self.clearance[cleared] = True
        self.crossing_start[cleared] = soles[rows[cleared], swing[cleared]]
        crossing = lift & self.clearance & ~cleared
        self.crossing_elapsed[crossing] += dt
        endpoint = target.clone()
        endpoint[:, 2] = apex
        horizontal = self.crossing_start + self.smooth(self.crossing_elapsed/cfg.lift_duration_s)[:, None]*(endpoint-self.crossing_start)
        self.reference_feet[rows[crossing], swing[crossing]] = horizontal[crossing]
        condition[lift] = (self.clearance & (soles[rows, swing, 2] >= apex-.015)
                           & ((soles[rows, swing, :2]-target[:, :2]).norm(dim=-1) <= cfg.xy_tolerance))[lift]
        next_phase[lift] = old[lift]+1
        contact_height = target[:, 2]-cfg.touchdown_probe_depth
        duration = (1.875*(start[:, 2]-contact_height).abs()/cfg.max_lower_speed).clamp_min(.4)
        descending = target.clone()
        descending[:, 2] = start[:, 2]+self.smooth(self.elapsed/duration)*(contact_height-start[:, 2])
        self.reference_feet[rows[lower], swing[lower]] = descending[lower]
        condition[lower] = supported[rows[lower], swing[lower]]
        next_phase[old == int(StepPhase.LOWER_LEAD)] = int(StepPhase.TRANSFER)
        next_phase[old == int(StepPhase.LOWER_TRAIL)] = int(StepPhase.SETTLE)

        transfer = old == int(StepPhase.TRANSFER)
        source = self.reference_feet[rows, 1-self.lead, :2]
        self.reference_root[transfer, :2] = (source+cfg.transfer_body_fraction*(
            self.targets[rows, self.lead, :2]-source))[transfer]
        self.reference_root[transfer, 2] = (self.initial_root[:, 2]+self.height-self.origins[:, 2])[transfer]
        condition[transfer] = (supported[rows, self.lead] & (load[rows, self.lead] >= .60)
                               & (load[rows, 1-self.lead] >= .08) & (root_vel[:, 2].abs() <= .10))[transfer]
        next_phase[transfer] = int(StepPhase.SHIFT_TRAIL)
        settling = old == int(StepPhase.SETTLE)
        self.reference_feet[settling] = self.targets[settling]
        self.reference_root[settling, :2] = self.targets[settling, :, :2].mean(dim=1)
        self.reference_root[settling, 2] = (self.initial_root[:, 2]+self.height-self.origins[:, 2])[settling]
        condition[settling] = (supported.all(dim=1) & both & stable & aligned)[settling]
        next_phase[settling] = int(StepPhase.COMPLETE)
        self.confirmed = torch.where(condition, self.confirmed+dt, 0.)
        hold = torch.where(settling, cfg.settle_s, cfg.confirmation_s)
        advance = (self.confirmed+1.e-6 >= hold) | (lift & condition)
        planted_now = advance & lower
        self.planted[rows[planted_now], swing[planted_now]] = True
        self.plant_positions[rows[planted_now], swing[planted_now]] = soles[rows[planted_now], swing[planted_now]]
        self.phase[advance] = next_phase[advance]
        self.elapsed[advance] = self.confirmed[advance] = 0
        self.reference_feet[advance] = torch.where(self.planted[advance, :, None], self.targets[advance], soles[advance])
        self.reference_root[advance] = root[advance]
        next_swing = torch.where(self.phase >= int(StepPhase.SHIFT_TRAIL), 1-self.lead, self.lead)
        self.swing_start[advance] = soles[rows[advance], next_swing[advance]]
        entering_lift = advance & ((self.phase == int(StepPhase.LIFT_LEAD)) |
                                   (self.phase == int(StepPhase.LIFT_TRAIL)))
        self.clearance[entering_lift] = False
        self.best_lift_height[entering_lift] = self.swing_start[entering_lift, 2]
        self.crossing_elapsed[advance] = 0
        self.progress = advance
        self.success = self.phase == int(StepPhase.COMPLETE)
        unplanned_height = observing & (load >= .12).any(dim=1) & (
            ((soles[..., 2]-self.origins[:, None, 2] > .06) & (load >= .12)).any(dim=1))
        self.failure = ((self.elapsed > torch.where(observing, cfg.observe_timeout_s, cfg.phase_timeout_s))
                        | (flight & (load[rows, stance] < .45))
                        | (self.planted & ~supported).any(dim=1) | unplanned_height)
        self.phase[self.failure] = int(StepPhase.RECOVER)
        if com_offset is not None:
            anchor = self.reference_feet[rows, 1-self.lead, :2].clone()
            trailing = ((self.phase >= int(StepPhase.SHIFT_TRAIL)) & (self.phase <= int(StepPhase.LOWER_TRAIL)))
            anchor[trailing] = self.targets[rows[trailing], self.lead[trailing], :2]
            transferring = self.phase == int(StepPhase.TRANSFER)
            source = self.reference_feet[rows, 1-self.lead, :2]
            anchor[transferring] = (source+cfg.transfer_body_fraction*(self.targets[rows, self.lead, :2]-source))[transferring]
            settling = self.phase == int(StepPhase.SETTLE)
            anchor[settling] = self.targets[settling, :, :2].mean(dim=1)
            lowering = ((self.phase == int(StepPhase.LOWER_LEAD)) | (self.phase == int(StepPhase.LOWER_TRAIL)))
            landing_foot = torch.where(self.phase == int(StepPhase.LOWER_TRAIL), 1-self.lead, self.lead)
            touching = (contained & on_plane & centered & upright & upward & (speed <= cfg.max_sole_speed)
                        & (load >= cfg.touchdown_contact_fraction))
            loading = lowering & touching[rows, landing_foot]
            anchor[loading] += cfg.touchdown_load_fraction*(self.targets[rows, landing_foot, :2]-anchor)[loading]
            moving = ((self.phase >= int(StepPhase.SHIFT_LEAD)) & (self.phase <= int(StepPhase.SETTLE)))
            self.reference_root[moving, :2] = (anchor-com_offset[:, :2])[moving]
        if hip_offsets is not None:
            nominal_height = self.initial_root[:, 2]+torch.where(
                self.phase >= int(StepPhase.TRANSFER), self.height-self.origins[:, 2], 0.)
            reachable = torch.ones_like(self.pending)
            limit = nominal_height.clone()
            for feet in (soles, self.reference_feet):
                ankles = feet+ankle_offsets
                horizontal = ankles[..., :2]-(self.reference_root[:, None, :2]+hip_offsets[..., :2])
                remaining = (cfg.leg_reach_m-cfg.leg_reach_reserve_m)**2-horizontal.square().sum(dim=-1)
                reachable &= (remaining > 0).all(dim=1)
                ceiling = ankles[..., 2]-hip_offsets[..., 2]+remaining.clamp_min(0).sqrt()
                limit = torch.minimum(limit, ceiling.amin(dim=1))
            moving = ((self.phase >= int(StepPhase.SHIFT_LEAD)) & (self.phase <= int(StepPhase.SETTLE)))
            self.reference_root[moving, 2] = limit[moving]
            self.failure |= moving & ~reachable
            self.phase[self.failure] = int(StepPhase.RECOVER)
        self.last_phases = self.phase.clone()
        self.success &= ~self.failure
        self.progress &= ~self.failure
        lift_progress = torch.where(self.failure, 0., lift_progress)

        features = torch.zeros(len(root), 39, device=root.device)
        features[:, :12] = F.one_hot(self.phase, 12)
        features[:, 12:18] = ((self.targets-soles) @ rotation).flatten(1)
        features[:, 18:24] = ((self.reference_feet-soles) @ rotation).flatten(1)
        features[:, 26:29] = ((self.reference_root-root)[:, None] @ rotation).squeeze(1)
        features[:, 29] = heading
        features[:, 30:32] = torch.tensor([1., self.direction], device=root.device)
        features[:, 32:34] = F.one_hot(self.lead, 2)
        features[:, 36:38] = 1
        features[:, 38] = self.clearance
        critic = features.clone()
        critic[:, 24:26] = load.clamp(0, 2)
        critic[:, 34:36] = supported
        desired = torch.full_like(load, .5)
        asymmetric = shift | flight | transfer
        desired[rows[asymmetric], swing[asymmetric]] = 0
        desired[rows[asymmetric], stance[asymmetric]] = 1
        desired[rows[transfer], self.lead[transfer]] = 1
        desired[rows[transfer], 1-self.lead[transfer]] = 0
        desired[rows[lower], swing[lower]] = cfg.touchdown_load_fraction
        desired[rows[lower], stance[lower]] = 1-cfg.touchdown_load_fraction
        stopping = observing | settling
        body_stopping = stopping | flight | (shift & ((self.reference_root[:, :2]-root[:, :2]).norm(dim=-1) <= .06))
        metrics = {
            "feet_reference": (1/(1+(self.reference_feet-soles).square().sum(dim=-1)/.08**2)).mean(dim=1),
            "body_reference": 1/(1+(self.reference_root-root).square().sum(dim=-1)/.12**2),
            "load_reference": 1/(1+(load-desired).square().sum(dim=-1)/.25**2),
            "heading": torch.exp(-heading.square()/math.radians(5)**2),
            "sole_tilt": normals[..., :2].square().sum(dim=-1).mean(dim=1),
            "unsafe_contact": (lower & (load[rows, swing] >= .12) & ~supported[rows, swing]).float(),
            "downward_speed": (-root_vel[:, 2]-.10).clamp_min(0).square(),
            "progress": self.progress.float()/dt,
            "lift_progress": lift_progress/dt,
            "success": self.success.float()/dt,
            "stop_root_motion": body_stopping*root_vel.square().sum(dim=-1),
            "stop_feet_motion": stopping*velocities.square().sum(dim=-1).mean(dim=1),
            "stop_angular_motion": body_stopping*angular_vel.square().sum(dim=-1),
        }
        return features, critic, metrics
