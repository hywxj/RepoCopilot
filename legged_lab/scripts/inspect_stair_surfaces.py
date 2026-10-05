"""Capture and audit D435i 2-D surfaces without training a locomotion policy."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", default="logs/stair_surfaces_32cm")
parser.add_argument("--heights", type=float, nargs="+", default=[0.15])
parser.add_argument("--processing_width", type=int, default=640)
parser.add_argument("--walking_steps", type=int, default=240)
parser.add_argument("--capture_every", type=int, default=20)
parser.add_argument("--surface_memory", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--normal_window_size", type=int, default=7)
parser.add_argument("--normal_radius", type=int, default=4)
parser.add_argument("--checkpoint", default="logs/walk/2026-09-19_12-54-59_elf3_atec_blind_final_v18_1024env_3k/model_43997.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import cv2
import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from legged_lab.envs.elf3.elf3_env import Elf3Env
from legged_lab.envs.elf3.walk_terrain_teacher_cfg import Elf3WalkGeometryStairsBootstrapEnvCfg
from legged_lab.perception.stair_geometry import _yaw_matrix
from legged_lab.perception.surface_audit import audit_surfaces, observed_next_tread_centers, observed_next_tread_coverage
from legged_lab.perception.tread_surfaces import draw_surface_geometry_debug


def staircase_truth(env):
    """Build upper-envelope intervals from generated mesh bounds, including both stairs."""
    terrain_cfg = env.cfg.scene.terrain_generator
    rows = []
    for row, name in ((1, "stairs_up"), (2, "stairs_down")):
        cfg = terrain_cfg.sub_terrains[name].copy()
        cfg.size = terrain_cfg.size
        meshes, _ = cfg.function(0.0, cfg)
        translation = env.scene.terrain.terrain_origins[row, 0, :2].detach().cpu().numpy() - 0.5 * np.array(cfg.size)
        for mesh in meshes:
            bounds = mesh.bounds.copy()
            bounds[:, :2] += translation
            rows.append(bounds)
    bounds = np.stack(rows)
    edges = np.unique(np.round(bounds[:, :, 0].ravel(), 8))
    intervals = []
    for near, far in zip(edges[:-1], edges[1:]):
        x = 0.5 * (near + far)
        active = (bounds[:, 0, 0] < x) & (bounds[:, 1, 0] > x)
        if not active.any():
            continue
        height = float(bounds[active, 1, 2].max())
        if intervals and abs(intervals[-1][2] - height) < 1.0e-6:
            intervals[-1][1] = float(far)
        else:
            intervals.append([float(near), float(far), height])
    return np.asarray(intervals)


def capture(env, output, name, mode, height, truth, save_image=True):
    # Explicitly read the current RTX frame before composing its matching body poses.
    env._update_terrain_geometry()
    geometry = env.terrain_geometry
    snapshot = geometry.surface_geometry[0]
    result = snapshot if geometry.surface_memory is None else geometry.surface_memory[0]
    root_position = env.robot.data.root_pos_w[0].detach().cpu().numpy()
    root_rotation = _yaw_matrix(env.robot.data.root_quat_w)[0].detach().cpu().numpy()
    records, heading_error = audit_surfaces(result, root_position, root_rotation, truth)
    snapshot_records, snapshot_heading = audit_surfaces(snapshot, root_position, root_rotation, truth)
    rgb = env.depth_camera.data.output["rgb"][0]
    depth = env.depth_camera.data.output["distance_to_image_plane"][0]
    feet = env.robot.data.body_pos_w[0, env.feet_body_ids]
    quat = env.robot.data.body_quat_w[0, env.feet_body_ids]
    forward = quat_apply(quat, torch.tensor([[1., 0., 0.]], device=env.device).expand(2, -1))
    sole_xy = ((feet[:, :2] + 0.03 * forward[:, :2]).detach().cpu().numpy() - root_position[:2]) @ root_rotation[:2, :2]
    foot_direction = forward[:, :2].detach().cpu().numpy() @ root_rotation[:2, :2]
    foot_yaw = np.arctan2(foot_direction[:, 1], foot_direction[:, 0])
    sole_positions_w = feet + quat_apply(quat, torch.tensor([[0.03, 0., -0.04]], device=env.device).expand(2, -1))
    timestamp = float(env.depth_camera.frame_timestamp[0])
    if save_image:
        image = draw_surface_geometry_debug(result, rgb, depth, geometry.processed_image_shape,
                                            feet_xy=sole_xy, feet_yaw=foot_yaw, camera_result=snapshot)
        if not cv2.imwrite(str(output / f"{name}.png"), image):
            raise OSError(f"Failed to save image {name}.")
    np.savez_compressed(output / "raw" / f"{name}.npz",
                            depth=depth.detach().cpu().numpy(), rgb=rgb.detach().cpu().numpy(),
                            intrinsic=env.depth_camera.data.intrinsic_matrices[0].detach().cpu().numpy(),
                            camera_pos=env._current_depth_camera_pose()[0][0].detach().cpu().numpy(),
                            camera_quat=env._current_depth_camera_pose()[1][0].detach().cpu().numpy(),
                            root_pos=root_position, root_quat=env.robot.data.root_quat_w[0].detach().cpu().numpy(),
                            foot_pos=feet.detach().cpu().numpy(), foot_quat=quat.detach().cpu().numpy(),
                            truth=truth, timestamp_s=timestamp,
                            memory_generation=(env.terrain_geometry_extractor.surface_memories[0].generation
                                               if geometry.surface_memory is not None else 0),
                            sequence_id=name.split("_walking_")[0] + "_walking" if "_walking_" in name
                                        else "_".join(name.split("_")[:2]))
    report = {
        "case": name, "mode": mode, "step_height_m": height, "direction": result.direction,
        "heading_error_deg": heading_error, "surfaces": records,
        "timestamp_s": timestamp, "memory_frame_count": result.memory_frame_count,
        "snapshot_surfaces": snapshot_records, "snapshot_direction": snapshot.direction,
        "snapshot_heading_error_deg": snapshot_heading,
        "observed_next_tread_centers": observed_next_tread_centers(
            result, root_position, root_rotation, sole_positions_w.detach().cpu().numpy()),
        "snapshot_next_tread_centers": observed_next_tread_centers(
            snapshot, root_position, root_rotation, sole_positions_w.detach().cpu().numpy()),
        "next_tread_lateral_coverage": observed_next_tread_coverage(
            result, root_position, root_rotation, sole_positions_w.detach().cpu().numpy()),
        "accepted_surfaces": sum(s.valid for s in result.surfaces),
        "full_foot_safe_centers": sum(int(s.safe_center_mask.sum()) for s in result.surfaces),
        "terminal_platform_detected": any(s.valid and s.kind == "platform_candidate" and
                                          abs((root_rotation @ s.centroid + root_position)[2]
                                              - (8 * height if mode == "up" else 0)) < 0.02
                                          for s in result.surfaces),
        "legacy_valid_treads": int((geometry.treads[0, :, 5] > 0.5).sum()),
    }
    print(f"[SURFACE] {name}: accepted={report['accepted_surfaces']}"
          f" safe_centers={report['full_foot_safe_centers']} next={report['observed_next_tread_centers']}"
          f" snapshot_next={report['snapshot_next_tread_centers']} heading_error={heading_error}", flush=True)
    return report


def render_pose(env, root_state, joint_position):
    ids = torch.tensor([0], device=env.device)
    env.robot.write_root_state_to_sim(root_state, env_ids=ids)
    env.robot.write_joint_state_to_sim(joint_position, torch.zeros_like(joint_position), env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    # Sensors run at 25 Hz; force a fresh render after each controlled pose change.
    for _ in range(4):
        env.sim.render()
        env.depth_camera.update(0.04, force_recompute=True)
    env._last_geometry_camera_frame = None


def load_blind_actor(path, device):
    state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    weights = sorted(((int(key.split(".")[1]), value) for key, value in state.items()
                      if key.startswith("actor.") and key.endswith(".weight")), key=lambda item: item[0])
    if not weights or weights[0][1].shape[1] != 960:
        raise ValueError("The diagnostic walking checkpoint must contain the original 960-D blind actor.")
    layers = []
    for index, (_, weight) in enumerate(weights):
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if index + 1 < len(weights):
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device)
    actor.load_state_dict({key[len("actor."):]: value for key, value in state.items() if key.startswith("actor.")})
    actor.eval()
    return actor


@torch.inference_mode()
def main():
    if args.capture_every <= 0 or args.walking_steps < 0:
        raise ValueError("capture_every must be positive and walking_steps nonnegative.")
    if args.processing_width <= 0:
        raise ValueError("processing_width must be positive.")
    if len(args.heights) != 1 or not 0.11 <= args.heights[0] <= 0.16:
        raise ValueError("Pass one height between 0.11 and 0.16 per invocation.")
    actual_height = args.heights[0]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    cfg = copy.deepcopy(Elf3WalkGeometryStairsBootstrapEnvCfg())
    cfg.amp_motion_files_display = cfg.amp_motion_files_display[:1]
    cfg.scene.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.scene.terrain_generator.num_cols = 1
    cfg.scene.depth_camera.width = 1280
    cfg.scene.depth_camera.height = 720
    cfg.scene.depth_camera.geometry.surface_validation_enabled = True
    cfg.scene.depth_camera.geometry.surface_memory_enabled = args.surface_memory
    cfg.scene.depth_camera.geometry.surface_normal_window_size = args.normal_window_size
    cfg.scene.depth_camera.geometry.surface_normal_radius = args.normal_radius
    cfg.scene.depth_camera.geometry.surface_processing_width = args.processing_width
    cfg.scene.depth_camera.geometry.surface_processing_height = round(args.processing_width * 9 / 16)
    cfg.scene.depth_camera.geometry.stair_step_settle_enabled = False
    cfg.scene.depth_camera.geometry.stair_heading_feedback_enabled = False
    cfg.noise.add_noise = False
    cfg.domain_rand.events.push_robot = None
    cfg.commands.ranges.lin_vel_x = (0.3, 0.3)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.rel_standing_envs = 0.0
    cfg.commands.heading_command = False
    cfg.scene.terrain_generator.difficulty_range = (0.0, 0.0)
    for name in ("stairs_up", "stairs_down"):
        cfg.scene.terrain_generator.sub_terrains[name].step_height_range = (actual_height, actual_height)
    env = Elf3Env(cfg, headless=args.headless)
    original_root = env.robot.data.default_root_state.clone()
    joints = env.robot.data.default_joint_pos.clone()
    reports = []
    truth = staircase_truth(env)
    width = 0.32
    for mode, row in (("up", 1), ("down", 2)):
        tile_center = env.scene.terrain.terrain_origins[row, 0].detach().cpu().numpy()
        tile_start = tile_center[:2] - 0.5 * np.asarray(cfg.scene.terrain_generator.size)
        positions = {"approach": 0.62, "middle": 1.0 + 3.5 * width, "platform": 3.8}
        for location, local_x in positions.items():
            env.terrain_geometry_extractor.reset_surface_memory([0])
            world_x = tile_start[0] + local_x
            interval = truth[(truth[:, 0] <= world_x) & (truth[:, 1] > world_x)][0]
            views = [("nominal", 0.0, 0.0, 0.0), ("yaw8", 8.0, 0.0, 0.0),
                     ("pitch8", 0.0, 8.0, 0.0)]
            if location == "middle":
                # A controlled extra viewpoint, not a demonstrated balanced stepping motion.
                views.append(("prestep_view", 0.0, 8.0, 0.12))
            for variant, yaw, pitch, shift in views:
                root = original_root.clone()
                root[0, :3] = torch.tensor([world_x+shift, tile_center[1], interval[2] + original_root[0, 2].item()], device=env.device)
                root[0, 3:7] = quat_from_euler_xyz(
                    torch.zeros(1, device=env.device), torch.tensor([math.radians(pitch)], device=env.device),
                    torch.tensor([math.radians(yaw)], device=env.device),
                )[0]
                root[0, 7:] = 0.0
                render_pose(env, root, joints)
                name = f"{mode}_{location}_{variant}_{actual_height*100:.0f}cm"
                reports.append(capture(env, output, name, mode, actual_height, truth))
    if args.walking_steps:
        actor = load_blind_actor(args.checkpoint, env.device)
        for mode, row in (("up", 1), ("down", 2)):
            env.cfg.scene.depth_camera.geometry.bootstrap_stair_row = row
            tile_center = env.scene.terrain.terrain_origins[row, 0]
            env.scene.terrain.env_origins[0] = tile_center
            env.scene.terrain.env_origins[0, 0] -= 2.0
            env.scene.terrain.env_origins[0, 2] = 0.0 if mode == "up" else 8 * actual_height
            env.reset(torch.tensor([0], device=env.device))
            obs, _ = env.compute_observations()
            for step in range(args.walking_steps):
                env.command_generator.command[0] = torch.tensor([0.3, 0., 0.], device=env.device)
                with torch.inference_mode():
                    obs, _, dones, _ = env.step(actor(obs[:, :960]))
                # A reset can pair a pre-reset camera frame with a new root pose.
                if not bool(dones[0]) and (step + 1) % args.capture_every == 0:
                    reports.append(capture(env, output, f"{mode}_walking_{step+1:04d}", mode,
                                           actual_height, truth, save_image=(step + 1) % 80 == 0))
    result = {
        "source": "Isaac Sim RTX D435i; not a physical RealSense recording",
        "seed": args.seed,
        "native_resolution": [1280, 720],
        "processing_resolution": [cfg.scene.depth_camera.geometry.surface_processing_width,
                                   cfg.scene.depth_camera.geometry.surface_processing_height],
        "legacy_processing_resolution": [cfg.scene.depth_camera.geometry.processing_width,
                                          cfg.scene.depth_camera.geometry.processing_height],
        "sensor_noise": {
            "enabled": cfg.scene.depth_camera.sensor_noise.enable,
            "mode": cfg.scene.depth_camera.sensor_noise.mode,
            "depth_std_m": cfg.scene.depth_camera.sensor_noise.depth_std,
            "depth_std_multiplier": cfg.scene.depth_camera.sensor_noise.depth_std_multiplier,
            "dropout_prob": cfg.scene.depth_camera.sensor_noise.dropout_prob,
        },
        "tread_depth_m": width, "step_height_m": actual_height,
        "foot_length_m": 0.24, "critical_edge_margin_m": 0.02, "other_end_margin_m": 0.01,
        "uncertainty_reserve_m": 0.01,
        "surface_memory_enabled": args.surface_memory,
        "surface_normal_filter": {"window_size": args.normal_window_size, "radius": args.normal_radius},
        "timestamp_source": "sensor acquisition clock, not frame count",
        "prestep_view_is_controlled_pose_not_stable_locomotion": True,
        "cases": reports,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (output / "surfaces.csv").open("w", newline="") as handle:
        rows = [dict(case=case["case"], mode=case["mode"], **surface)
                for case in reports for surface in case["surfaces"]]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[SURFACE] Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(skip_cleanup=True)
