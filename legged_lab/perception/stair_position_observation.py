"""Position-teacher recording contract, without direct contact-force inputs."""

import numpy as np


class StairPositionObservation:
    def __init__(self, interface):
        self.interface = interface
        self.history = None
        self.last_action = np.zeros(29)

    def observe(self, episode):
        interface, data = self.interface, episode.data
        features = episode.actor_features()
        measured = episode.measurement()
        rotation = data.xmat[episode.teacher.reader.root_id].reshape(3, 3)
        s = interface.obs_scales
        # A deliberate single step has no forward velocity command. Dynamic
        # motion references are carried in the 39 supervisor feature slots.
        frame = np.concatenate((measured.root_angular_velocity @ rotation*s["ang_vel"],
                                np.array([0., 0., -1.]) @ rotation*s["projected_gravity"],
                                np.zeros(3)*s["commands"],
                                (data.qpos[interface.qpos]-interface.default)*s["joint_pos"],
                                data.qvel[interface.dofs]*s["joint_vel"],
                                self.last_action*s["actions"]))
        if self.history is None:
            self.history = np.tile(frame, (interface.history_length, 1))
        else:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = frame
        observation = np.r_[self.history.ravel(), features, episode.direction]
        if (observation.shape != (1000,) or not np.isfinite(observation).all()
                or observation[984:986].any() or observation[994:996].any()):
            raise ValueError("Expected 1000 finite features with no direct force/support truth.")
        return np.clip(observation, -interface.clip_obs, interface.clip_obs).astype(np.float32)

    def metadata(self, episode=None):
        i = self.interface
        source = getattr(episode, "geometry_source", "known")
        supervisor = getattr(episode, "reference_supervisor", "dynamic")
        return dict(schema_version="elf3_stair_position_v2",
                    observation_schema="elf3_960_step39_gate_v1",
                    action_schema="isaac_default_plus_scaled_position_v1",
                    joint_names=list(i.joint_names), control_dt=i.dt,
                    physics_dt=float(i.model.opt.timestep), history_length=i.history_length,
                    history_order="oldest_to_newest", action_scales=i.scales.tolist(),
                    default_joint_positions=i.default.tolist(), stiffness=i.kp.tolist(),
                    damping=i.kd.tolist(), clip_actions=i.clip_actions,
                    clip_observations=i.clip_obs,
                    observation_scales={k: float(i.obs_scales[k]) for k in
                        ("ang_vel", "projected_gravity", "commands", "joint_pos", "joint_vel", "actions")},
                    torque_limits=i.model.actuator_ctrlrange[i.motors].tolist(),
                    reference_source=f"{supervisor}_{source}_geometry_contact_oracle",
                    force_oracle=True, known_geometry=source == "known",
                    initialization_teacher=True, depth_transfer_verified=False)
