"""Replay saved RTX depth frames without launching Isaac Sim or loading a policy."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from legged_lab.perception.stair_geometry import project_surface_points, _matrix_from_quat, _yaw_matrix
from legged_lab.perception.surface_audit import audit_surfaces, observed_next_tread_centers, observed_next_tread_coverage
from legged_lab.perception.surface_memory import SurfaceMemory
from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor, draw_surface_geometry_debug


def load_surface_frame(path, extractor, processing_width=640, normals_enabled=True, local_tilt_deg=45.0,
                       normal_window_size=7, normal_radius=4):
    with np.load(path) as archive:
        raw = {key: archive[key] for key in archive.files}
    tensor = lambda key: torch.from_numpy(raw[key]).unsqueeze(0)
    cfg = SimpleNamespace(min_forward=extractor.cfg.min_forward, max_forward=extractor.cfg.max_forward,
                          min_height_from_root=-1.5, max_height_from_root=0.35,
                          surface_processing_width=processing_width,
                          surface_processing_height=round(processing_width * 9 / 16),
                          surface_normals_enabled=normals_enabled, surface_local_tilt_deg=local_tilt_deg,
                          surface_normal_window_size=normal_window_size, surface_normal_radius=normal_radius)
    points, valid, shape = project_surface_points(
        tensor("depth"), tensor("intrinsic"), tensor("camera_pos"), tensor("camera_quat"),
        tensor("root_pos"), tensor("root_quat"), cfg, tensor("foot_pos"), tensor("foot_quat"),
    )
    result = extractor.extract(points[0].numpy(), valid[0].numpy())
    rotation = _yaw_matrix(tensor("root_quat"))[0].numpy()
    records, heading_error = audit_surfaces(result, raw["root_pos"], rotation, raw["truth"])
    return raw, result, shape, rotation, records, heading_error, points[0].numpy(), valid[0].numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--processing_width", type=int, default=640)
    parser.add_argument("--grid_size", type=float, default=0.02)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--disable_surface_normals", action="store_true")
    parser.add_argument("--local_tilt_deg", type=float, default=45.0)
    parser.add_argument("--surface_memory", action="store_true")
    parser.add_argument("--normal_window_size", type=int, default=7)
    parser.add_argument("--normal_radius", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extractor = TreadSurfaceExtractor(SurfaceValidationCfg(grid_size=args.grid_size))
    cases = []
    for directory in args.input_dirs:
        paths = sorted((Path(directory) / "raw").glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No captured depth frames in {directory}/raw.")
        if args.surface_memory:
            def frame_order(path):
                with np.load(path) as raw:
                    if "timestamp_s" not in raw or "sequence_id" not in raw:
                        raise ValueError("Memory replay requires recorded timestamps and reset sequences; "
                                         "do not invent timestamps for older captures.")
                    return (str(raw["sequence_id"]), int(raw.get("memory_generation", 0)),
                            float(raw["timestamp_s"]))
            paths.sort(key=frame_order)
        memory, sequence = SurfaceMemory(extractor.cfg), None
        for path in paths:
            raw, snapshot, shape, rotation, records, heading, points, valid = load_surface_frame(
                path, extractor, args.processing_width, not args.disable_surface_normals, args.local_tilt_deg,
                args.normal_window_size, args.normal_radius)
            result = snapshot
            if args.surface_memory:
                current_sequence = (str(raw["sequence_id"]), int(raw.get("memory_generation", 0)))
                if current_sequence != sequence:
                    memory.reset()
                    sequence = current_sequence
                result = memory.update(points, snapshot, raw["root_pos"], rotation,
                                       float(raw["timestamp_s"]), valid)
                records, heading = audit_surfaces(result, raw["root_pos"], rotation, raw["truth"])
            foot_rotation = _matrix_from_quat(torch.from_numpy(raw["foot_quat"])).numpy()
            sole_positions = raw["foot_pos"] + foot_rotation @ np.array([0.03, 0., -0.04])
            case = {"case": path.stem, "source": str(path), "heading_error_deg": heading,
                    "direction": result.direction, "surfaces": records,
                    "memory_frame_count": result.memory_frame_count,
                    "observed_next_tread_centers": observed_next_tread_centers(
                        result, raw["root_pos"], rotation, sole_positions),
                    "next_tread_lateral_coverage": observed_next_tread_coverage(
                        result, raw["root_pos"], rotation, sole_positions)}
            cases.append(case)
            if args.save_images:
                image_output = output / Path(directory).name
                image_output.mkdir(exist_ok=True)
                forward = _matrix_from_quat(torch.from_numpy(raw["foot_quat"]))[:, :, 0].numpy()[:, :2]
                feet_xy = (raw["foot_pos"][:, :2] + 0.03 * forward - raw["root_pos"][:2]) @ rotation[:2, :2]
                direction = forward @ rotation[:2, :2]
                feet_yaw = np.arctan2(direction[:, 1], direction[:, 0])
                image = draw_surface_geometry_debug(result, raw["rgb"], raw["depth"], shape, feet_xy, feet_yaw,
                                                    camera_result=snapshot)
                image_path = image_output / f"{path.stem}.png"
                case["image_path"] = str(image_path)
                if not cv2.imwrite(str(image_path), image):
                    raise OSError(f"Failed to write {path.stem}.png.")
            violations = sum(s["safe_center_truth_violations"] or 0 for s in records)
            if violations:
                print(f"[REPLAY] {path.stem}: violations={violations}", flush=True)
    summary = {"frames": len(cases), "surface_memory_enabled": args.surface_memory,
               "surface_normal_filter": {"window_size": args.normal_window_size, "radius": args.normal_radius},
               "memory_replay_uses_only_saved_frames_not_every_live_camera_frame": args.surface_memory,
               "safe_centers": sum(s["safe_centers"] for c in cases for s in c["surfaces"]),
               "truth_violations": sum(s["safe_center_truth_violations"] or 0 for c in cases for s in c["surfaces"]),
               "height_violations": sum(s["safe_center_height_violations"] or 0 for c in cases for s in c["surfaces"]),
               "cases": cases}
    (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"[REPLAY] {len(cases)} frames, {summary['safe_centers']} candidates, "
          f"{summary['truth_violations']} truth clearance violations: {output / 'report.json'}")


if __name__ == "__main__":
    main()
