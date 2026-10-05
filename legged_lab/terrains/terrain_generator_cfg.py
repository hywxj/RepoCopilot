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

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import copy

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh, height_field_to_mesh
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


@height_field_to_mesh
def rounded_pebbles_terrain(difficulty: float, cfg) -> np.ndarray:
    """Generate rounded cobblestone-like bumps as a height field."""

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    heights = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    x = np.arange(width_pixels, dtype=np.float32)[:, None] * cfg.horizontal_scale
    y = np.arange(length_pixels, dtype=np.float32)[None, :] * cfg.horizontal_scale

    num_stones = int(cfg.num_stones_range[0] + difficulty * (cfg.num_stones_range[1] - cfg.num_stones_range[0]))
    height_max = cfg.stone_height_range[0] + difficulty * (cfg.stone_height_range[1] - cfg.stone_height_range[0])
    margin = max(cfg.stone_radius_range[1], cfg.horizontal_scale)

    for _ in range(num_stones):
        cx = np.random.uniform(margin, max(margin, cfg.size[0] - margin))
        cy = np.random.uniform(margin, max(margin, cfg.size[1] - margin))

        if cfg.center_clearance_width > 0.0:
            center_x = 0.5 * cfg.size[0]
            center_y = 0.5 * cfg.size[1]
            if abs(cx - center_x) < 0.5 * cfg.center_clearance_width and abs(cy - center_y) < 0.5:
                continue

        rx = np.random.uniform(*cfg.stone_radius_range)
        ry = np.random.uniform(*cfg.stone_radius_range)
        stone_height = np.random.uniform(cfg.stone_height_range[0], height_max)

        normalized_dist = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        mask = normalized_dist <= 1.0
        cap = 0.5 * (1.0 + np.cos(np.pi * normalized_dist))
        heights[mask] = np.maximum(heights[mask], stone_height * cap[mask])

    return np.rint(heights / cfg.vertical_scale).astype(np.int16)


def straight_slope_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a ramp with exact endpoint heights and no zero-height border."""

    width_pixels = round(cfg.size[0] / cfg.horizontal_scale) + 1
    length_pixels = round(cfg.size[1] / cfg.horizontal_scale) + 1
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    x = np.linspace(0.0, cfg.size[0], width_pixels, dtype=np.float32)
    if cfg.direction == "down":
        x = cfg.size[0] - x
    heights = np.rint((x * slope)[:, None] / cfg.vertical_scale).astype(np.int16)
    heights = np.broadcast_to(heights, (width_pixels, length_pixels)).copy()
    vertices, faces = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, slope_threshold=None
    )
    center_height = heights[width_pixels // 2, length_pixels // 2] * cfg.vertical_scale
    return [trimesh.Trimesh(vertices=vertices, faces=faces)], np.array(
        [0.5 * cfg.size[0], 0.5 * cfg.size[1], center_height]
    )


def straight_slope_plateau_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Match the uphill endpoint and downhill starting height."""

    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    height = round(cfg.ramp_length * slope / cfg.vertical_scale) * cfg.vertical_scale
    mesh = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], height + 0.02),
        transform=trimesh.transformations.translation_matrix(
            (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * (height - 0.02))
        ),
    )
    return [mesh], np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], height])


def straight_stairs_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate full-width straight stairs along +x."""

    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.step_width_range is None:
        step_width = cfg.step_width
    else:
        step_width = cfg.step_width_range[0] + difficulty * (
            cfg.step_width_range[1] - cfg.step_width_range[0]
        )
    num_steps = max(1, cfg.num_steps)
    final_height = num_steps * step_height
    approach_length = min(cfg.approach_length, max(0.0, cfg.size[0] - num_steps * step_width))
    meshes = []

    base_dims = (cfg.size[0], cfg.size[1], 0.02)
    base_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.01)
    meshes.append(trimesh.creation.box(base_dims, trimesh.transformations.translation_matrix(base_pos)))

    if cfg.direction == "down" and approach_length > 0.0:
        box_dims = (approach_length, cfg.size[1], final_height)
        box_pos = (0.5 * approach_length, 0.5 * cfg.size[1], 0.5 * final_height)
        meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    for step_idx in range(num_steps):
        x_start = approach_length + step_idx * step_width
        x_end = min(cfg.size[0], approach_length + (step_idx + 1) * step_width)
        if x_end <= x_start:
            continue

        if cfg.direction == "up":
            height = (step_idx + 1) * step_height
        else:
            height = (num_steps - step_idx) * step_height

        box_dims = (x_end - x_start, cfg.size[1], max(height, 0.002))
        box_pos = ((x_start + x_end) * 0.5, cfg.size[1] * 0.5, height * 0.5)
        meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    platform_start = approach_length + num_steps * step_width
    if platform_start < cfg.size[0] and cfg.direction == "up":
        box_dims = (cfg.size[0] - platform_start, cfg.size[1], final_height)
        box_pos = ((platform_start + cfg.size[0]) * 0.5, 0.5 * cfg.size[1], 0.5 * final_height)
        meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    x_center = 0.5 * cfg.size[0]
    if x_center < approach_length:
        origin_z = final_height if cfg.direction == "down" else 0.0
    elif x_center < approach_length + num_steps * step_width:
        center_step = int((x_center - approach_length) / step_width)
        origin_z = (center_step + 1) * step_height if cfg.direction == "up" else (num_steps - center_step) * step_height
    else:
        origin_z = final_height if cfg.direction == "up" else 0.0

    return meshes, np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])


@configclass
class HfRoundedPebblesTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    function = rounded_pebbles_terrain

    stone_radius_range: tuple[float, float] = (0.06, 0.16)
    stone_height_range: tuple[float, float] = (0.01, 0.055)
    num_stones_range: tuple[int, int] = (60, 130)
    center_clearance_width: float = 0.0


@configclass
class HfStraightSlopeTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    function = straight_slope_terrain

    slope_range: tuple[float, float] = (0.04, 0.18)
    direction: str = "up"


@configclass
class MeshStraightSlopePlateauTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    function = straight_slope_plateau_terrain

    slope_range: tuple[float, float] = (0.04, 0.18)
    ramp_length: float = 5.0
    vertical_scale: float = 0.005


@configclass
class MeshStraightStairsTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    function = straight_stairs_terrain

    step_height_range: tuple[float, float] = (0.046, 0.161)
    step_width: float = 0.368
    step_width_range: tuple[float, float] | None = None
    num_steps: int = 6
    approach_length: float = 1.0
    direction: str = "up"

class AtecTerrainGenerator(TerrainGenerator):
    """Terrain generator with fixed colors per ATEC-style sub-terrain type."""

    COLORS = {
        "plane": (145, 145, 145, 255),
        "grass_mild": (76, 153, 82, 255),
        "gravel_rough": (178, 150, 93, 255),
        "slope_up": (73, 143, 210, 255),
        "slope_down": (92, 184, 212, 255),
        "stairs_up": (151, 112, 204, 255),
        "stairs_down": (197, 122, 188, 255),
        "pebbles": (206, 101, 72, 255),
        "border": (50, 50, 50, 255),
    }

    def _color_for_cfg(self, cfg):
        cls_name = type(cfg).__name__
        if cls_name == "MeshPlaneTerrainCfg":
            return self.COLORS["plane"]
        if cls_name == "HfRandomUniformTerrainCfg":
            return self.COLORS["grass_mild"] if cfg.noise_range[1] <= 0.03 else self.COLORS["gravel_rough"]
        if cls_name == "HfInvertedPyramidSlopedTerrainCfg":
            return self.COLORS["slope_up"]
        if cls_name == "HfPyramidSlopedTerrainCfg":
            return self.COLORS["slope_down"]
        if cls_name == "HfStraightSlopeTerrainCfg":
            return self.COLORS["slope_down"] if cfg.direction == "down" else self.COLORS["slope_up"]
        if cls_name == "MeshStraightSlopePlateauTerrainCfg":
            return self.COLORS["slope_up"]
        if cls_name == "MeshInvertedPyramidStairsTerrainCfg":
            return self.COLORS["stairs_up"]
        if cls_name == "MeshPyramidStairsTerrainCfg":
            return self.COLORS["stairs_down"]
        if cls_name == "MeshStraightStairsTerrainCfg":
            return self.COLORS["stairs_down"] if cfg.direction == "down" else self.COLORS["stairs_up"]
        if cls_name == "HfDiscreteObstaclesTerrainCfg":
            return self.COLORS["pebbles"]
        if cls_name == "HfRoundedPebblesTerrainCfg":
            return self.COLORS["pebbles"]
        return (128, 128, 128, 255)

    def _add_terrain_border(self):
        super()._add_terrain_border()
        border = self.terrain_meshes[-1]
        border.visual.vertex_colors = np.tile(np.array(self.COLORS["border"], dtype=np.uint8), (len(border.vertices), 1))

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        color = self._color_for_cfg(sub_terrain_cfg)
        mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


class AtecObstacleCourseTerrainGenerator(AtecTerrainGenerator):
    """ATEC-style obstacle course with terrain types arranged along x.

    IsaacLab's default curriculum varies difficulty along rows (x) and terrain
    type along columns (y). For route-style playback we want the opposite: each
    row is the next course segment, while columns are parallel lanes.
    """

    def _generate_random_terrains(self):
        self._generate_course_terrains()

    def _generate_curriculum_terrains(self):
        self._generate_course_terrains()

    def _generate_course_terrains(self):
        sub_terrain_items = list(self.cfg.sub_terrains.items())
        lower, upper = self.cfg.difficulty_range

        for sub_row in range(self.cfg.num_rows):
            _, sub_terrain_cfg = sub_terrain_items[sub_row % len(sub_terrain_items)]
            for sub_col in range(self.cfg.num_cols):
                # Rows are consecutive course sections, so course difficulty
                # must vary across parallel lanes rather than along the route.
                col_ratio = 0.0 if self.cfg.num_cols <= 1 else sub_col / (self.cfg.num_cols - 1)
                difficulty = lower + (upper - lower) * col_ratio
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrain_cfg)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_cfg)


GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="none",
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Keep enough flat environments to retain the locomotion policy while
        # stairs are introduced. Row zero starts with real, low steps instead
        # of a zero-height terrain that can be solved by standing still.
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.6),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.16),
            step_width=0.30,
            platform_width=1.5,
            border_width=0.5,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.16),
            step_width=0.30,
            platform_width=1.5,
            border_width=0.5,
            holes=False,
        ),
    },
)


# Dedicated ELF3 stair curriculum. Rows increase riser height while both
# travel directions retain 32 cm treads for step-to support training.
ELF3_STAIRS_CURRICULUM_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=AtecTerrainGenerator,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Two columns retain flat-ground behavior throughout fine-tuning.
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        # The 11-16 cm range makes the final rows concentrate around the
        # nominal 15 cm riser instead of exposing it at the start of training.
        "stairs_up_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.45,
            step_height_range=(0.11, 0.16),
            step_width=0.32,
            platform_width=1.5,
            border_width=0.5,
            holes=False,
        ),
        "stairs_down_32": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.45,
            step_height_range=(0.11, 0.16),
            step_width=0.32,
            platform_width=1.5,
            border_width=0.5,
            holes=False,
        ),
    },
)


# Continuous training route for the geometry-fusion policy. Each curriculum
# lane varies riser height from 11 to 16 cm with fixed 32 cm treads.
# Insertion order is the travel order used by the course generator.
ELF3_GEOMETRY_COURSE_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=AtecObstacleCourseTerrainGenerator,
    curriculum=True,
    size=(5.0, 3.0),
    border_width=12.0,
    num_rows=8,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "plane_start": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "stairs_up": MeshStraightStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.11, 0.16),
            step_width=0.32,
            step_width_range=None,
            num_steps=8,
            approach_length=1.0,
            direction="up",
        ),
        "stairs_down": MeshStraightStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.11, 0.16),
            step_width=0.32,
            step_width_range=None,
            num_steps=8,
            approach_length=1.0,
            direction="down",
        ),
        "slope_up": HfStraightSlopeTerrainCfg(
            proportion=1.0,
            slope_range=(0.06, 0.18),
            direction="up",
            border_width=0.20,
        ),
        "slope_top": MeshStraightSlopePlateauTerrainCfg(
            proportion=1.0,
            slope_range=(0.06, 0.18),
            ramp_length=5.0,
        ),
        "slope_down": HfStraightSlopeTerrainCfg(
            proportion=1.0,
            slope_range=(0.06, 0.18),
            direction="down",
            border_width=0.20,
        ),
        "pebbles": HfRoundedPebblesTerrainCfg(
            proportion=1.0,
            stone_radius_range=(0.06, 0.16),
            stone_height_range=(0.01, 0.055),
            num_stones_range=(70, 150),
            center_clearance_width=0.0,
            border_width=0.20,
        ),
        "plane_finish": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


# Fixed 32 cm / 15 cm single-lane course for repeatable checkpoint evaluation.
ELF3_GEOMETRY_EVAL_COURSE_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=AtecObstacleCourseTerrainGenerator,
    curriculum=False,
    size=(5.0, 3.0),
    border_width=12.0,
    num_rows=8,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.8, 0.8),
    use_cache=False,
    sub_terrains=copy.deepcopy(ELF3_GEOMETRY_COURSE_TERRAINS_CFG.sub_terrains),
)
ELF3_GEOMETRY_EVAL_COURSE_TERRAINS_CFG.sub_terrains["stairs_up"].step_width_range = None
ELF3_GEOMETRY_EVAL_COURSE_TERRAINS_CFG.sub_terrains["stairs_down"].step_width_range = None


ATEC_OFFLINE_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=AtecTerrainGenerator,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Keep flat routes in the mix so the policy does not forget the v12 gait.
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.14),
        # Grass-like low unevenness: frequent but small height changes.
        "grass_mild": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.18,
            noise_range=(-0.015, 0.025),
            noise_step=0.01,
            border_width=0.25,
        ),
        # Gravel-like roughness: larger random height changes.
        "gravel_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(-0.025, 0.040),
            noise_step=0.015,
            border_width=0.25,
        ),
        # Gentle ramps for the route-guided cross-country task.
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.12,
            slope_range=(0.0, 0.18),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.12,
            slope_range=(0.0, 0.18),
            platform_width=2.0,
            border_width=0.25,
        ),
        # Low to medium stairs. Height and tread width are 15% above the previous stage.
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.046, 0.161),
            step_width=0.368,
            platform_width=2.0,
            border_width=0.5,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.046, 0.161),
            step_width=0.368,
            platform_width=2.0,
            border_width=0.5,
            holes=False,
        ),
        # Pebble/cobble-like discrete obstacles. This stays modest for the first terrain stage.
        "pebbles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.06,
            obstacle_width_range=(0.10, 0.22),
            obstacle_height_range=(0.01, 0.05),
            num_obstacles=50,
            platform_width=0.9,
            border_width=0.20
        ),
    },
)


ATEC_OBSTACLE_COURSE_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=AtecObstacleCourseTerrainGenerator,
    curriculum=True,
    size=(8.0, 3.0),
    border_width=12.0,
    num_rows=9,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plane_start": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "grass_mild": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.015, 0.025),
            noise_step=0.01,
            border_width=0.25,
        ),
        "gravel_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.025, 0.040),
            noise_step=0.015,
            border_width=0.25,
        ),
        "pebbles": HfRoundedPebblesTerrainCfg(
            proportion=1.0,
            stone_radius_range=(0.06, 0.16),
            stone_height_range=(0.01, 0.055),
            num_stones_range=(70, 150),
            center_clearance_width=0.0,
            border_width=0.20,
        ),
        "stairs_up": MeshStraightStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.046, 0.161),
            step_width=0.368,
            direction="up",
        ),
        "stairs_down": MeshStraightStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.046, 0.161),
            step_width=0.368,
            direction="down",
        ),
        "slope_up": HfStraightSlopeTerrainCfg(
            proportion=1.0,
            slope_range=(0.06, 0.18),
            direction="up",
            border_width=0.20,
        ),
        "slope_down": HfStraightSlopeTerrainCfg(
            proportion=1.0,
            slope_range=(0.06, 0.18),
            direction="down",
            border_width=0.20,
        ),
        "plane_finish": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)
