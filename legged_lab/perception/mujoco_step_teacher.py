"""Simulation-only inverse-dynamics teacher, not a learned or hardware controller."""

import mujoco
import numpy as np
from qpsolvers import solve_qp
from scipy.spatial.transform import Rotation

from .mujoco_support import MujocoSupportReader


class MujocoStepTeacher:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.reader = MujocoSupportReader(model)
        self.nv = model.nv
        self.joints = model.actuator_trnid[:, 0]
        self.dofs = model.jnt_dofadr[self.joints]
        self.qpos = model.jnt_qposadr[self.joints]
        if (len(set(self.dofs)) != model.nu or model.nv != model.nu+6
                or not np.allclose(model.actuator_gear[:, 0], 1.)):
            raise ValueError("Teacher requires a floating base and one unit-gear motor per joint.")
        self.nominal = data.qpos[self.qpos].copy()
        self.mass_matrix = np.zeros((self.nv, self.nv))
        self.previous_com_jacobian = None
        self.max_dynamics_residual = 0.

    def point_jacobians(self, body, point):
        jac, derivative = np.zeros((6, self.nv)), np.zeros((6, self.nv))
        mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], point, body)
        mujoco.mj_jacDot(self.model, self.data, derivative[:3], derivative[3:], point, body)
        return jac, derivative @ self.data.qvel

    def control(self, root_reference, feet_reference, contacts, dt, load_reference, minimum_loads=None):
        model, data, nv = self.model, self.data, self.nv
        mujoco.mj_forward(model, data)
        m = self.reader.measurement(data, include_contact_truth=True)
        active = np.flatnonzero(contacts)
        if not len(active):
            raise RuntimeError("No support contact is available to the action teacher.")
        count, weight = len(active), self.reader.body_weight
        dim = nv+6*count
        rows, values, costs = [], [], []

        def task(jacobian, desired, cost):
            rows.append(np.pad(jacobian, ((0, 0), (0, 6*count))))
            values.append(np.atleast_1d(desired))
            costs.extend(np.broadcast_to(cost, len(np.atleast_1d(desired))))

        com_jac = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, com_jac, self.reader.root_id)
        com_drift = (np.zeros(3) if self.previous_com_jacobian is None else
                     (com_jac-self.previous_com_jacobian) @ data.qvel/dt)
        self.previous_com_jacobian = com_jac.copy()
        com_offset = data.subtree_com[self.reader.root_id]-m.root_position
        com_acceleration = np.clip(35*(root_reference+com_offset-data.subtree_com[self.reader.root_id])
                                   -12*(com_jac @ data.qvel), -.6, .6)-com_drift
        task(com_jac, com_acceleration, [40., 40., 15.])
        root_jac, root_drift = self.point_jacobians(self.reader.root_id, m.root_position)
        root_acceleration = np.clip(45*(root_reference-m.root_position)-14*m.root_velocity, -1., 1.)
        root_rotation = data.xmat[self.reader.root_id].reshape(3, 3)
        rotation_error = -Rotation.from_matrix(root_rotation).as_rotvec()
        task(root_jac[:3], root_acceleration-root_drift[:3], [2., 2., 10.])
        task(root_jac[3:], 100*rotation_error-20*m.root_angular_velocity-root_drift[3:], 40.)
        posture = np.zeros((model.nu, nv))
        posture[np.arange(model.nu), self.dofs] = 1.
        task(posture, 8*(self.nominal-data.qpos[self.qpos])-6*data.qvel[self.dofs], .05)

        contact_jacs, contact_accelerations = [], []
        contact_feedback = np.zeros(model.nu)
        for foot, body in enumerate(self.reader.foot_ids):
            jac, drift = self.point_jacobians(body, m.sole_positions[foot])
            angular_velocity = jac[3:] @ data.qvel
            orientation_error = -Rotation.from_matrix(m.foot_rotations[foot]).as_rotvec()
            acceleration = np.r_[200*(feet_reference[foot]-m.sole_positions[foot])-28*m.sole_velocities[foot],
                                 150*orientation_error-24*angular_velocity]-drift
            if contacts[foot]:
                contact_jacs.append(jac)
                contact_accelerations.append(acceleration)
                # Compliant contacts do not realize the ideal QP wrench instantly.
                # Joint motors correct measured sole rotation, especially yaw slip.
                attitude_torque = np.array([20., 20., 80.])*orientation_error-np.array([2., 2., 8.])*angular_velocity
                contact_feedback += jac[3:, self.dofs].T @ attitude_torque
            else:
                task(jac, acceleration, [20., 20., 30., 12., 12., 12.])

        objectives = np.vstack(rows)
        targets = np.concatenate(values)
        costs = np.asarray(costs)
        hessian = objectives.T @ (costs[:, None]*objectives)+np.eye(dim)*1.e-6
        linear = -objectives.T @ (costs*targets)
        for slot, foot in enumerate(active):
            indices = nv+6*slot+np.arange(6)
            hessian[indices, indices] += [1., 1., .2, 10., 10., 10.]
            linear[nv+6*slot+2] -= .2*load_reference[foot]

        mujoco.mj_fullM(model, data, self.mass_matrix)
        contact_jac = np.vstack(contact_jacs)
        bias = data.qfrc_bias-data.qfrc_passive
        # The first six equations are unactuated; only actual joint motors may act.
        equality = np.vstack((np.c_[self.mass_matrix[:6]/weight, -contact_jac[:, :6].T],
                              np.c_[contact_jac, np.zeros((6*count, 6*count))]))
        equality_value = np.r_[-bias[:6]/weight, np.concatenate(contact_accelerations)]
        torque_map = np.c_[self.mass_matrix[self.dofs], -weight*contact_jac[:, self.dofs].T]
        inequalities = [torque_map, -torque_map]
        bounds = [model.actuator_ctrlrange[:, 1]-bias[self.dofs],
                  -model.actuator_ctrlrange[:, 0]+bias[self.dofs]]
        # Rounded foot capsules cannot support pressure at the visual sole edge.
        # Keep CoP inside their flat contact region, with a stability margin.
        for slot in range(count):
            # A conservative inner friction pyramid, not independent x/y boxes.
            for sign_x in (-1., 1.):
                for sign_y in (-1., 1.):
                    row = np.zeros(dim)
                    row[nv+6*slot], row[nv+6*slot+1], row[nv+6*slot+2] = sign_x, sign_y, -.7
                    inequalities.append(row[None])
                    bounds.append(np.zeros(1))
            for axis, limit in ((3, .018), (4, .080), (5, .015)):
                for sign in (-1., 1.):
                    row = np.zeros(dim)
                    row[nv+6*slot+axis], row[nv+6*slot+2] = sign, -limit
                    inequalities.append(row[None])
                    bounds.append(np.zeros(1))
        low, high = np.full(dim, -np.inf), np.full(dim, np.inf)
        low[:nv], high[:nv] = -100., 100.
        joint_positions = data.qpos[self.qpos]
        horizon = .15
        low[self.dofs] = np.maximum(low[self.dofs], 2*(model.jnt_range[self.joints, 0]+.01-joint_positions
                                                      -horizon*data.qvel[self.dofs])/horizon**2)
        high[self.dofs] = np.minimum(high[self.dofs], 2*(model.jnt_range[self.joints, 1]-.01-joint_positions
                                                       -horizon*data.qvel[self.dofs])/horizon**2)
        for slot in range(count):
            low[nv+6*slot+2] = 0. if minimum_loads is None else minimum_loads[active[slot]]
            high[nv+6*slot+2] = 1.5
        self.last_problem = (hessian, linear, np.vstack(inequalities), np.concatenate(bounds),
                             equality, equality_value, low, high)
        solution = solve_qp(hessian, linear, np.vstack(inequalities), np.concatenate(bounds),
                            equality, equality_value, lb=low, ub=high, solver="quadprog")
        if solution is None or not np.isfinite(solution).all():
            raise RuntimeError("Action teacher inverse-dynamics QP is infeasible.")
        self.last_solution = solution.copy()
        torque = torque_map @ solution+bias[self.dofs]
        self.max_dynamics_residual = max(self.max_dynamics_residual,
                                        float(np.max(np.abs(equality[:6] @ solution-equality_value[:6]))))
        return np.clip(torque+contact_feedback, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
