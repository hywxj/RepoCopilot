"""D435i-shaped rendered depth observations for the MuJoCo stair supervisor.

Only visible depth pixels enter the shared surface extractor and world-memory.
Simulation poses and contact qualification remain explicit simulation inputs;
this module does not turn their availability into a hardware deployment claim.
"""

import copy
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from .surface_memory import SurfaceMemory, SurfaceMemoryCfg
from .tread_surfaces import TreadSurfaceExtractor, depth_horizontal_mask


@dataclass(frozen=True)
class DepthCameraSpec:
    # Same optics and torso extrinsics as camera_cfgs/d435i_depth_config.py.
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 87.
    vertical_fov_deg: float = 58.
    position: tuple = (.1152, .0175, -.1358)
    quaternion_ros_wxyz: tuple = (.122796911, -.696361316, .696363874, -.122797362)
    near: float = .3
    far: float = 5.
    update_period: float = .04
    # Full input resolution keeps enough observed edge pixels for a 32 cm
    # descending tread after the local-normal mask. Do not shrink foot margins.
    processing_width: int = 1280
    processing_height: int = 720

    def intrinsic(self):
        fx = .5*self.width/math.tan(math.radians(self.horizontal_fov_deg)/2)
        fy = .5*self.height/math.tan(math.radians(self.vertical_fov_deg)/2)
        return np.array([[fx, 0., .5*self.width], [0., fy, .5*self.height], [0., 0., 1.]])


CAMERA_NAME = "elf3_d435i"


def add_depth_camera(xml, spec=None):
    """Install a named camera without changing dynamics or contact geometry."""
    spec = DepthCameraSpec() if spec is None else spec
    body = xml.find("worldbody/body[@name='torso_link']")
    if body is None:
        raise ValueError("The depth camera requires torso_link.")
    # MuJoCo looks along -Z with +Y up; ROS optical uses +Z and +Y down.
    # q_ros multiplied on the right by a pi rotation about optical X.
    w, x, y, z = np.asarray(spec.quaternion_ros_wxyz, dtype=float)
    gl_quaternion = np.array([-x, w, z, -y])
    gl_quaternion /= np.linalg.norm(gl_quaternion)
    intrinsic = spec.intrinsic()
    ET.SubElement(body, "camera", name=CAMERA_NAME,
                  pos=" ".join(map(str, spec.position)), quat=" ".join(map(str, gl_quaternion)),
                  resolution=f"{spec.width} {spec.height}", sensorsize=".00384 .00216",
                  focalpixel=f"{intrinsic[0, 0]} {intrinsic[1, 1]}")
    visual = xml.find("visual")
    if visual is None:
        visual = ET.SubElement(xml, "visual")
    global_ = visual.find("global")
    if global_ is None:
        global_ = ET.SubElement(visual, "global")
    global_.set("offwidth", str(spec.width))
    global_.set("offheight", str(spec.height))


class MujocoDepthGeometry:
    """Rendered depth -> measured point cloud -> common 2-D surfaces/memory."""

    def __init__(self, model, surface_cfg, spec=None):
        self.model = model
        self.spec = DepthCameraSpec() if spec is None else spec
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self.root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                         for name in ("l_ankle_x_link", "r_ankle_x_link")]
        if min(self.camera_id, self.root_id, *self.foot_ids) < 0:
            raise ValueError("Named depth camera, torso and feet are required.")
        self.extractor = TreadSurfaceExtractor(surface_cfg)
        # Preserve two actual pixels per observed map cell. A linear stride over
        # the raster can leave periodic holes that erase an otherwise visible
        # full-foot region. This budget bounds storage without inventing points.
        point_budget = 2*self.extractor.grid_xy.shape[0]*self.extractor.grid_xy.shape[1]
        self.memory = SurfaceMemory(surface_cfg, SurfaceMemoryCfg(max_points_per_frame=point_budget))
        self.intrinsic = self.spec.intrinsic()
        self.renderer = None
        self.frozen = False
        self.last_frame_time = -math.inf
        self.last_depth = self.last_rgb = None
        self.frame_count = 0
        self.last_valid_points = 0
        self.last_camera_position = self.last_camera_rotation = None
        self.last_snapshot = None
        self.last_update_time = None

    def __deepcopy__(self, memo):
        # Predictive teacher clones must never duplicate OpenGL contexts or
        # render future observations. Map points are immutable between captures.
        result = copy.copy(self)
        memo[id(self)] = result
        for _, points in self.memory.frames:
            memo[id(points)] = points
        result.memory = copy.deepcopy(self.memory, memo)
        result.renderer = None
        result.frozen = True
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _root_pose(self, data):
        root = data.xpos[self.root_id].copy()
        rotation = data.xmat[self.root_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        return root, np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def render(self, data):
        """Return metric optical-axis depth and RGB of this exact simulator pose."""
        if self.frozen:
            raise RuntimeError("A predictive geometry snapshot cannot acquire camera frames.")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.spec.height, width=self.spec.width)
        self.renderer.update_scene(data, camera=self.camera_id)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()
        # MuJoCo Renderer already converts the depth buffer to metres and flips
        # image rows. Missing/out-of-range pixels stay invalid, never get filled.
        depth[(depth < self.spec.near) | (depth > self.spec.far)] = np.nan
        return depth, rgb

    def ingest(self, data, depth, *, timestamp=None, rgb=None):
        """Fuse one acquisition with its matching camera/root/foot pose."""
        if self.frozen:
            raise RuntimeError("A predictive geometry snapshot cannot ingest new frames.")
        timestamp = float(data.time if timestamp is None else timestamp)
        if not np.isclose(timestamp, data.time, rtol=0., atol=1.e-8):
            raise ValueError("Depth ingestion requires its matching simulation pose timestamp.")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (self.spec.height, self.spec.width):
            raise ValueError("Depth dimensions differ from the camera calibration.")
        h, w = self.spec.processing_height, self.spec.processing_width
        reduced = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        intrinsic = self.intrinsic.copy()
        intrinsic[0] *= w/self.spec.width
        intrinsic[1] *= h/self.spec.height
        camera_position = data.cam_xpos[self.camera_id].copy()
        camera_rotation = data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        valid = np.isfinite(reduced) & (reduced >= self.spec.near) & (reduced <= self.spec.far)
        valid &= depth_horizontal_mask(reduced, intrinsic, camera_rotation)
        v, u = np.indices((h, w))
        z = np.nan_to_num(reduced, nan=0., posinf=0., neginf=0.)
        camera = np.stack(((u-intrinsic[0, 2])*z/intrinsic[0, 0],
                           (v-intrinsic[1, 2])*z/intrinsic[1, 1], z), axis=-1).reshape(-1, 3)
        world = camera @ camera_rotation.T+camera_position
        valid = valid.ravel()
        # Kinematic self-mask: identical foot boxes to project_surface_points.
        # Do not use renderer segmentation labels to invent visible ground.
        for foot in self.foot_ids:
            local = (world-data.xpos[foot]) @ data.xmat[foot].reshape(3, 3)
            valid &= ~((local >= [-.10, -.052, -.05]) & (local <= [.16, .052, .025])).all(axis=1)
        root, yaw_rotation = self._root_pose(data)
        body = (world-root) @ yaw_rotation
        valid &= (body[:, 2] >= -1.5) & (body[:, 2] <= .35)
        cfg = self.extractor.cfg
        valid &= ((body[:, 0] >= cfg.min_forward) & (body[:, 0] < cfg.max_forward)
                  & (np.abs(body[:, 1]) < cfg.lateral_half_width))
        visible = body[valid]
        if len(visible):
            cells = np.floor((visible[:, :2]-[cfg.min_forward, -cfg.lateral_half_width])/cfg.grid_size).astype(int)
            keys = cells[:, 0]*self.extractor.grid_xy.shape[1]+cells[:, 1]
            # Retain height extrema, so thinning cannot hide a cell that mixed
            # two levels or exceeded the original within-cell height variation.
            order = np.lexsort((visible[:, 2], keys))
            boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order]))+1, len(order)]
            spans = boundaries[1:]-boundaries[:-1]
            selected = np.r_[order[boundaries[:-1]], order[boundaries[1:][spans > 1]-1]]
            visible = visible[selected]
        snapshot = self.extractor.extract(visible)
        geometry = self.memory.update(visible, snapshot, root, yaw_rotation, timestamp)
        self.last_frame_time = timestamp
        self.last_depth = depth.copy()
        self.last_rgb = None if rgb is None else rgb.copy()
        self.last_camera_position, self.last_camera_rotation = camera_position, camera_rotation.copy()
        self.frame_count += 1
        self.last_valid_points = int(valid.sum())
        self.last_snapshot = snapshot
        return geometry

    def update(self, data, *, acquire=True, tracking_only=False):
        """Acquire at 25 Hz and expire prior observations on every control tick.

        Tracking-only callers may read track identity and last-observed time,
        never use the cached body-coordinate raster for planning. Locked world
        targets need no new raster between camera frames. Normal callers retain
        current-pose reprojection. Neither path refreshes an observation's age.
        """
        mujoco.mj_forward(self.model, data)
        timestamp = float(data.time)
        root, rotation = self._root_pose(data)
        if not np.isfinite(timestamp) or not np.isfinite(root).all() or not np.isfinite(rotation).all():
            raise ValueError("Depth tracking requires finite synchronized poses and timestamps.")
        previous_time = self.last_update_time
        if previous_time is None:
            previous_time = self.memory.last_time
        discontinuity = (previous_time is not None and timestamp < previous_time) or (
            self.memory.last_position is not None
            and np.linalg.norm(root-self.memory.last_position) > self.memory.cfg.max_pose_jump_m)
        if discontinuity:
            self.memory.reset()
        self.last_update_time = timestamp
        if (acquire and not self.frozen
                and (discontinuity or timestamp-self.last_frame_time >= self.spec.update_period-1.e-9)):
            depth, rgb = self.render(data)
            return self.ingest(data, depth, rgb=rgb)
        if tracking_only and self.memory.result is not None:
            self.memory._expire(timestamp)
            live_ids = {track.track_id for track in self.memory.tracks}
            geometry = copy.copy(self.memory.result)
            geometry.surfaces = [surface for surface in geometry.surfaces if surface.track_id in live_ids]
            geometry.memory_frame_count = len(self.memory.frames)
            self.memory.last_position = root.copy()
            self.memory.last_rotation = rotation.copy()
            return geometry
        return self.memory.refresh(root, rotation, timestamp)
