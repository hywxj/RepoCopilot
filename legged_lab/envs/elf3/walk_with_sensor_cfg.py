# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import copy

from isaaclab.utils import configclass

from legged_lab.envs.base.base_config import HeightScannerCfg
from legged_lab.envs.elf3.walk_cfg import Elf3WalkAgentCfg, Elf3WalkFlatEnvCfg
from legged_lab.sensors.camera.camera_cfgs import TiledD455CameraCfg


@configclass
class Elf3WalkWithSensorFlatEnvCfg(Elf3WalkFlatEnvCfg):
    scene = copy.deepcopy(Elf3WalkFlatEnvCfg().scene)
    scene.height_scanner = HeightScannerCfg(
        enable_height_scan=False,
        prim_body_name="torso_link",
        resolution=0.1,
        size=(1.6, 1.0),
        debug_vis=False,
        drift_range=(0.0, 0.0),
    )
    scene.depth_camera = TiledD455CameraCfg(
        prim_body_name="torso_link/depth_camera",
        offset=TiledD455CameraCfg.OffsetCfg(
            pos=(0.1152, 0.0175, -0.1358),
            rot=(0.122796911, -0.696361316, 0.696363874, -0.122797362),
            convention="ros",
        ),
        width=64,
        height=48,
        min_range=0.3,
        max_range=5.0,
        feature_width=8,
        feature_height=12,
        depth_history_length=3,
        frame_hold_prob=0.05,
        depth_quantization=0.001,
        update_period=0.04,
        debug_vis=False,
    )
    scene.depth_camera.sensor_noise.dropout_prob = 0.005


@configclass
class Elf3WalkWithSensorAgentCfg(Elf3WalkAgentCfg):
    experiment_name = "walk"
    run_name = "elf3_depth_sensor"
    neptune_project = "walk_elf3_depth_sensor"
    wandb_project = "walk_elf3_depth_sensor"
