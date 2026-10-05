# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

import math

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from legged_lab.sensors.camera import CameraCfg, SensorNoiseCfg, TiledCameraCfg

D435I_DEPTH_WIDTH = 1280
D435I_DEPTH_HEIGHT = 720
D435I_DEPTH_HFOV_DEG = 87.0
D435I_DEPTH_VFOV_DEG = 58.0
D435I_HORIZONTAL_APERTURE_CM = 2.4
D435I_FOCAL_LENGTH_CM = (D435I_HORIZONTAL_APERTURE_CM / 2.0) / math.tan(
    math.radians(D435I_DEPTH_HFOV_DEG) / 2.0
)
D435I_VERTICAL_APERTURE_CM = 2.0 * D435I_FOCAL_LENGTH_CM * math.tan(
    math.radians(D435I_DEPTH_VFOV_DEG) / 2.0
)


@configclass
class D435iCamera:
    """RealSense D435i depth intrinsics used by simulation and deployment."""

    enable_depth_camera = True
    debug_vis = False
    update_latest_camera_pose = True

    width: int = D435I_DEPTH_WIDTH
    height: int = D435I_DEPTH_HEIGHT
    max_range: float = 5.0
    min_range: float = 0.3
    # RGB remains available for diagnostics and future visual perception, but
    # only depth-derived geometry is consumed by the locomotion policy.
    data_types: list[str] = ["rgb", "distance_to_image_plane"]

    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(0.1152, 0.0175, -0.1358),
        rot=(0.122796911, -0.696361316, 0.696363874, -0.122797362),
        convention="ros",
    )
    spawn: PinholeCameraCfg = PinholeCameraCfg(
        focal_length=D435I_FOCAL_LENGTH_CM,
        horizontal_aperture=D435I_HORIZONTAL_APERTURE_CM,
        vertical_aperture=D435I_VERTICAL_APERTURE_CM,
        clipping_range=(min_range, max_range),
    )
    sensor_noise: SensorNoiseCfg = SensorNoiseCfg(
        enable=True,
        mode="combined",
        depth_std=0.005,
        depth_std_multiplier=0.01,
        dropout_prob=0.005,
        dropout_value=-1.0,
    )


@configclass
class D435iCameraCfg(D435iCamera, CameraCfg):
    pass


@configclass
class TiledD435iCameraCfg(D435iCamera, TiledCameraCfg):
    pass
