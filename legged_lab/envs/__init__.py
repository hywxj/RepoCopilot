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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg


from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)

from legged_lab.envs.elf3.elf3_env import Elf3Env
from legged_lab.envs.elf3.walk_cfg import (
    Elf3WalkAgentCfg,
    Elf3WalkFlatEnvCfg,
)
from legged_lab.envs.elf3.walk_with_sensor_cfg import (
    Elf3WalkWithSensorAgentCfg,
    Elf3WalkWithSensorFlatEnvCfg,
)
from legged_lab.envs.elf3.walk_terrain_teacher_cfg import (
    Elf3WalkGeometryCourseAgentCfg,
    Elf3WalkGeometryCourseEnvCfg,
    Elf3WalkGeometryDebugAgentCfg,
    Elf3WalkGeometryDebugEnvCfg,
    Elf3WalkGeometryFusionAgentCfg,
    Elf3WalkGeometryFusionEnvCfg,
    Elf3WalkGeometryStairsBootstrapAgentCfg,
    Elf3WalkGeometryStairsBootstrapEnvCfg,
    Elf3WalkGeometryStairsDownBootstrapAgentCfg,
    Elf3WalkGeometryStairsDownBootstrapEnvCfg,
    Elf3WalkGeometryStairsDownFullControlAgentCfg,
    Elf3WalkStairsCurriculumAgentCfg,
    Elf3WalkStairsCurriculumEnvCfg,
    Elf3WalkTerrainTeacherAgentCfg,
    Elf3WalkTerrainTeacherEnvCfg,
    Elf3WalkTerrainTeacherSensorAgentCfg,
    Elf3WalkTerrainTeacherSensorEnvCfg,
)

from legged_lab.utils.task_registry import task_registry
from legged_lab.envs.elf3.stair_step_cfg import Elf3SingleStepUpEnvCfg, Elf3SingleStepDownEnvCfg, Elf3SingleStepAgentCfg

task_registry.register("walk_elf3_geometry_step_up", Elf3Env, Elf3SingleStepUpEnvCfg(), Elf3SingleStepAgentCfg())
task_registry.register("walk_elf3_geometry_step_down", Elf3Env, Elf3SingleStepDownEnvCfg(), Elf3SingleStepAgentCfg())

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)

task_registry.register("walk_elf3", Elf3Env, Elf3WalkFlatEnvCfg(), Elf3WalkAgentCfg())
task_registry.register(
    "walk_elf3_with_sensor", Elf3Env, Elf3WalkWithSensorFlatEnvCfg(), Elf3WalkWithSensorAgentCfg()
)
task_registry.register(
    "walk_elf3_depth_sensor", Elf3Env, Elf3WalkWithSensorFlatEnvCfg(), Elf3WalkWithSensorAgentCfg()
)
task_registry.register(
    "walk_elf3_terrain_teacher", Elf3Env, Elf3WalkTerrainTeacherEnvCfg(), Elf3WalkTerrainTeacherAgentCfg()
)
task_registry.register(
    "walk_elf3_stairs_curriculum",
    Elf3Env,
    Elf3WalkStairsCurriculumEnvCfg(),
    Elf3WalkStairsCurriculumAgentCfg(),
)
task_registry.register(
    "walk_elf3_geometry_fusion",
    Elf3Env,
    Elf3WalkGeometryFusionEnvCfg(),
    Elf3WalkGeometryFusionAgentCfg(),
)
task_registry.register(
    "walk_elf3_geometry_course",
    Elf3Env,
    Elf3WalkGeometryCourseEnvCfg(),
    Elf3WalkGeometryCourseAgentCfg(),
)
task_registry.register(
    "walk_elf3_geometry_stairs_bootstrap",
    Elf3Env,
    Elf3WalkGeometryStairsBootstrapEnvCfg(),
    Elf3WalkGeometryStairsBootstrapAgentCfg(),
)
task_registry.register(
    "walk_elf3_geometry_stairs_down_bootstrap",
    Elf3Env,
    Elf3WalkGeometryStairsDownBootstrapEnvCfg(),
    Elf3WalkGeometryStairsDownBootstrapAgentCfg(),
)
task_registry.register(
    "walk_elf3_geometry_stairs_down_full_control",
    Elf3Env,
    Elf3WalkGeometryStairsDownBootstrapEnvCfg(),
    Elf3WalkGeometryStairsDownFullControlAgentCfg(),
)
task_registry.register(
    "walk_elf3_geometry_debug",
    Elf3Env,
    Elf3WalkGeometryDebugEnvCfg(),
    Elf3WalkGeometryDebugAgentCfg(),
)
task_registry.register(
    "walk_elf3_terrain_teacher_sensor",
    Elf3Env,
    Elf3WalkTerrainTeacherSensorEnvCfg(),
    Elf3WalkTerrainTeacherSensorAgentCfg(),
)
