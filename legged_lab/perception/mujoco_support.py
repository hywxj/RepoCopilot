"""MuJoCo kinematics and contact ORACLE for sim2sim tests, not a hardware sensor."""

import math

import mujoco
import numpy as np

from .stair_step_controller import StepMeasurement


class MujocoSupportReader:
    def __init__(self, model, root_body="torso_link",
                 foot_bodies=("l_ankle_x_link", "r_ankle_x_link")):
        self.model = model
        self.root_id = self._body(root_body)
        self.foot_ids = [self._body(name) for name in foot_bodies]
        if self.root_id == 0 or len(self.foot_ids) != 2 or len(set(self.foot_ids)) != 2:
            raise ValueError("A robot root and distinct left/right foot bodies are required.")
        self.robot_mask = np.array([self._descendant(i, self.root_id) for i in range(model.nbody)])
        if not all(self.robot_mask[foot] for foot in self.foot_ids):
            raise ValueError("Feet must belong to the selected robot root subtree.")
        self.geom_foot = np.full(model.ngeom, -1, dtype=int)
        for geom, body in enumerate(model.geom_bodyid):
            for foot, parent in enumerate(self.foot_ids):
                if self._descendant(body, parent):
                    self.geom_foot[geom] = foot
        self.body_weight = float(model.body_mass[self.robot_mask].sum()*np.linalg.norm(model.opt.gravity))
        if self.body_weight <= 0:
            raise ValueError("A nonzero robot mass and gravity are required.")

    def _body(self, name):
        result = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if result < 0:
            raise ValueError(f"Unknown MuJoCo body: {name}")
        return result

    def _descendant(self, body, ancestor):
        while body != 0:
            if body == ancestor:
                return True
            body = self.model.body_parentid[body]
        return ancestor == 0

    def contact_forces_world(self, data):
        """Sum external contacts on each foot; no mjData.sensordata is read."""
        forces = np.zeros((2, 3))
        wrench = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 < 0 or geom2 < 0:
                continue
            body1, body2 = self.model.geom_bodyid[[geom1, geom2]]
            # Self-collision cannot prove ground support or load transfer.
            if self.robot_mask[body1] and self.robot_mask[body2]:
                continue
            left, right = self.geom_foot[[geom1, geom2]]
            if left < 0 and right < 0:
                continue
            mujoco.mj_contactForce(self.model, data, contact_id, wrench)
            force_on_geom2 = contact.frame.reshape(3, 3).T @ wrench[:3]
            if left >= 0:
                forces[left] -= force_on_geom2
            if right >= 0:
                forces[right] += force_on_geom2
        return forces

    def _point_velocity(self, data, body_id, point):
        jacobian, angular = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jacobian, angular, point, body_id)
        return jacobian @ data.qvel, angular @ data.qvel

    def measurement(self, data, generation=0, include_contact_truth=False, camera_timestamp=None):
        """Without the opt-in contact oracle, load stays UNKNOWN, never zero."""
        root = data.xpos[self.root_id].copy()
        root_rotation = data.xmat[self.root_id].reshape(3, 3)
        yaw = math.atan2(root_rotation[1, 0], root_rotation[0, 0])
        yaw_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                 [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
        rotations = np.stack([data.xmat[foot].reshape(3, 3) for foot in self.foot_ids])
        soles = data.xpos[self.foot_ids]+np.einsum("fij,j->fi", rotations, [0.03, 0., -0.04])
        sole_velocity = np.stack([self._point_velocity(data, foot, point)[0]
                                  for foot, point in zip(self.foot_ids, soles)])
        root_velocity, root_angular_velocity = self._point_velocity(data, self.root_id, root)
        forces = self.contact_forces_world(data) if include_contact_truth else None
        return StepMeasurement(float(data.time), generation, root, yaw_rotation, soles, rotations,
                               sole_velocity, forces, root_velocity, root_angular_velocity,
                               self.body_weight, camera_timestamp)
