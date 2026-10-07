"""Fixed-period ELF3 position actions from an isolated torque-teacher preview.

The predictive teacher only advances an isolated simulator copy. The real robot
holds one position target for the entire policy period; its motor torques are
ordinary feedback PD torques, recomputed at the physics frequency.
"""

import copy
from pathlib import Path
import re

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares


# Isaac articulation order (not the motor declaration order in the MuJoCo XML).
POLICY_JOINT_NAMES = (
    "l_shoulder_y_joint", "r_shoulder_y_joint", "waist_y_joint",
    "l_shoulder_x_joint", "r_shoulder_x_joint", "waist_x_joint",
    "l_shoulder_z_joint", "r_shoulder_z_joint", "waist_z_joint",
    "l_elbow_y_joint", "r_elbow_y_joint", "l_hip_y_joint", "r_hip_y_joint",
    "l_wrist_x_joint", "r_wrist_x_joint", "l_hip_x_joint", "r_hip_x_joint",
    "l_wrist_y_joint", "r_wrist_y_joint", "l_hip_z_joint", "r_hip_z_joint",
    "l_wrist_z_joint", "r_wrist_z_joint", "l_knee_y_joint", "r_knee_y_joint",
    "l_ankle_y_joint", "r_ankle_y_joint", "l_ankle_x_joint", "r_ankle_x_joint",
)


class _ConfigLoader(yaml.SafeLoader):
    """Only the two harmless Python tags used in exported Isaac configs."""


_ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                              lambda loader, node: tuple(loader.construct_sequence(node)))
_ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                              lambda loader, node: slice(*loader.construct_sequence(node)))


def _joint_values(mapping, names):
    result = []
    for name in names:
        matches = [value for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one configuration value for {name}; found {len(matches)}.")
        result.append(matches[0])
    return np.asarray(result, dtype=float)


class MujocoPositionInterface:
    """Named-joint PD contract loaded from the actual Isaac environment config."""

    def __init__(self, model, environment_config):
        cfg = (environment_config if isinstance(environment_config, dict) else
               yaml.load(Path(environment_config).read_text(), Loader=_ConfigLoader))
        self.model, self.joint_names = model, POLICY_JOINT_NAMES
        joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                           for name in self.joint_names])
        if (joints < 0).any() or model.nu != len(joints):
            raise ValueError("Position policy requires all 29 ELF3 joints.")
        motor_joints = list(model.actuator_trnid[:, 0])
        if len(set(motor_joints)) != model.nu or not np.allclose(model.actuator_gear[:, 0], 1.):
            raise ValueError("Position policy requires one unit-gear motor per joint.")
        self.motors = np.array([motor_joints.index(joint) for joint in joints])
        self.qpos, self.dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        robot = cfg["scene"]["robot"]
        self.default = _joint_values(robot["init_state"]["joint_pos"], self.joint_names)
        self.kp = _joint_values({key: value for group in robot["actuators"].values()
                                 for key, value in group["stiffness"].items()}, self.joint_names)
        self.kd = _joint_values({key: value for group in robot["actuators"].values()
                                 for key, value in group["damping"].items()}, self.joint_names)
        self.scales = np.broadcast_to(np.asarray(cfg["robot"]["action_scale"], dtype=float), (model.nu,)).copy()
        self.dt = float(cfg["sim"]["dt"]*cfg["sim"]["decimation"])
        self.clip_actions = float(cfg["normalization"]["clip_actions"])
        self.history_length = int(cfg["robot"]["actor_obs_history_length"])
        self.obs_scales = dict(cfg["normalization"]["obs_scales"])
        self.clip_obs = float(cfg["normalization"]["clip_observations"])
        values = np.r_[self.default, self.kp, self.kd, self.scales, self.dt,
                       self.clip_actions, self.clip_obs]
        if (not np.isfinite(values).all() or (self.kp <= 0).any() or (self.kd < 0).any()
                or (self.scales <= 0).any() or self.dt <= 0 or self.clip_actions <= 0
                or self.clip_obs <= 0 or self.history_length != 10):
            raise ValueError("Invalid ELF3 position-control contract.")
        self.physics_steps = int(round(self.dt/model.opt.timestep))
        if self.physics_steps < 1 or not np.isclose(self.dt, self.physics_steps*model.opt.timestep):
            raise ValueError("Policy period must be an integer multiple of the physics period.")

    def action_from_torque(self, data, torque):
        torque = np.asarray(torque)
        if torque.shape != (self.model.nu,) or not np.isfinite(torque).all():
            raise ValueError("Expected 29 finite motor torques.")
        target = data.qpos[self.qpos]+(torque[self.motors]+self.kd*data.qvel[self.dofs])/self.kp
        return self.action_from_target(target)

    def action_from_target(self, target):
        action = (np.asarray(target)-self.default)/self.scales
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError("Expected 29 finite joint position targets.")
        if (np.abs(action) > self.clip_actions).any():
            raise ValueError("Teacher position targets exceed the position-action range.")
        return action

    def target_from_action(self, action):
        action = np.asarray(action)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError("Expected 29 finite policy actions.")
        return self.default+self.scales*np.clip(action, -self.clip_actions, self.clip_actions)

    def torque_from_action(self, data, action):
        target = self.target_from_action(action)
        torque = self.kp*(target-data.qpos[self.qpos])-self.kd*data.qvel[self.dofs]
        result = np.empty(self.model.nu)
        result[self.motors] = torque
        return np.clip(result, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    def contract(self):
        return {"policy_joint_names": list(self.joint_names), "action_scales": self.scales.copy(),
                "default_joint_positions": self.default.copy(), "stiffness": self.kp.copy(),
                "damping": self.kd.copy(), "control_dt": self.dt,
                "physics_dt": float(self.model.opt.timestep), "clip_actions": self.clip_actions}


class DynamicPositionBridge:
    """Predict one policy period and fit a single held position target.

    The clone preserves touchdown locks, support dwell, and dynamic references.
    COM drift starts from current analytic kinematics; the teacher's smooth torso
    reference persists across intervals. No live episode state is advanced or
    corrected by prediction. Hold each returned action for ``physics_steps``.
    """

    def __init__(self, episode, interface, *, refine=True):
        if episode.model is not interface.model:
            raise ValueError("Episode and position interface must share their model.")
        self.episode, self.interface = episode, interface
        self._teacher_memory = None
        self.refine = refine

    def action(self):
        episode, interface = self.episode, self.interface
        # Share the immutable model and geometric point cloud; all mutable
        # controller, measurement, and simulation state remains isolated.
        memo = {id(episode.model): episode.model}
        if hasattr(episode._context, "world_points"):
            memo[id(episode._context.world_points)] = episode._context.world_points
        preview = copy.deepcopy(episode, memo)
        preview.samples, preview.trace = [], []
        if self._teacher_memory is not None:
            preview.teacher.torso_rpy_reference = self._teacher_memory.copy()
        # The last *predicted* Jacobian is not the previous real Jacobian. Its
        # state error divided by 2.5 ms would create artificial COM drift. Seed
        # the first derivative with the mass-weighted analytic body COM J-dot.
        model, data, reader = preview.model, preview.data, preview.teacher.reader
        mujoco.mj_forward(model, data)
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jac, reader.root_id)
        derivative = np.zeros_like(jac)
        translational, rotational = np.zeros_like(jac), np.zeros_like(jac)
        mass = model.body_mass[reader.robot_mask].sum()
        for body in np.flatnonzero(reader.robot_mask):
            mujoco.mj_jacDot(model, data, translational, rotational, data.xipos[body], int(body))
            derivative += model.body_mass[body]/mass*translational
        preview.teacher.previous_com_jacobian = jac-model.opt.timestep*derivative

        def prediction_torque(predicted_episode, sample):
            # Event supervision deliberately has no implicit motor fallback.
            # Only this isolated label query explicitly supplies its teacher.
            # Respect the copied instance method: never recover a disabled
            # disabled teacher through its class or the live context alias.
            if predicted_episode is not preview:
                raise RuntimeError("Teacher prediction must remain in the isolated episode copy.")
            prepared_sample, control = predicted_episode.prepare_step()
            if prepared_sample is not sample:
                raise RuntimeError("Teacher prediction must reuse the current prepared physics step.")
            return predicted_episode.teacher.control(**control)

        actions = []
        for _ in range(interface.physics_steps):
            before = copy.copy(preview.data)
            _, torque = preview._advance(prediction_torque)
            actions.append(interface.action_from_torque(before, torque))
        self._teacher_memory = preview.teacher.torso_rpy_reference.copy()
        action = np.mean(actions, axis=0)
        if self.refine:
            # Shooting uses only cloned free-base physics. Fit the terminal
            # generalized state, retaining the exact held-target PD execution.
            rollout = mujoco.MjData(episode.model)
            weights = np.ones(episode.model.nv)
            weights[:6] = 3.
            position_error = np.empty(episode.model.nv)

            def residual(candidate):
                mujoco.mj_copyData(rollout, episode.model, episode.data)
                for _ in range(interface.physics_steps):
                    rollout.ctrl[:] = interface.torque_from_action(rollout, candidate)
                    mujoco.mj_step(episode.model, rollout)
                mujoco.mj_differentiatePos(episode.model, position_error, interface.dt,
                                           preview.data.qpos, rollout.qpos)
                return np.r_[weights*position_error,
                             weights*(rollout.qvel-preview.data.qvel), .001*(candidate-action)]

            def derivative(candidate):
                baseline = residual(candidate)
                result = np.empty((len(baseline), len(candidate)))
                for column in range(len(candidate)):
                    shifted = candidate.copy()
                    shifted[column] += 1.e-4
                    result[:, column] = (residual(shifted)-baseline)/1.e-4
                return result

            fitted = least_squares(residual, action, jac=derivative, max_nfev=8, x_scale="jac",
                                   bounds=(-interface.clip_actions, interface.clip_actions),
                                   ftol=1.e-6, xtol=1.e-6, gtol=1.e-6)
            self.fit_diagnostics = {"cost": float(fitted.cost), "optimality": float(fitted.optimality),
                                    "evaluations": int(fitted.nfev)}
            action = fitted.x
        return action
