"""Collect pure position-teacher demonstrations with a fixed physical scene split.

Only successful teacher episodes enter the demonstration paths. Failed attempts
remain in the manifest and retain their diagnostics. This command does not load
or train a learned policy.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


SCENE_DEFAULTS = dict(height=.11, width=.32, geometry_source="known", initial_forward_offset=.05,
                      initial_lateral_offset=0., initial_yaw=0.,
                      initial_linear_velocity=[0., 0., 0.],
                      initial_angular_velocity=[0., 0., 0.],
                      initial_velocity_noise=0., seed=42)


def canonical_scenario(scene):
    """Canonical physical identity; renaming/seeding a deterministic replay is not diversity."""
    allowed = set(SCENE_DEFAULTS) | {"direction", "name", "dataset_split"}
    if not isinstance(scene, dict) or set(scene)-allowed:
        raise ValueError(f"Unsupported scenario fields: {set(scene)-allowed if isinstance(scene, dict) else scene}")
    result = dict(SCENE_DEFAULTS, **scene)
    result["direction"] = {"up": 1, "down": -1}.get(result.get("direction"), result.get("direction"))
    if result["direction"] not in (-1, 1):
        raise ValueError("Every scenario needs direction up/down or +1/-1.")
    if result["geometry_source"] not in ("known", "depth"):
        raise ValueError("geometry_source must be known or depth.")
    for key in ("initial_linear_velocity", "initial_angular_velocity"):
        values = np.asarray(result[key], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain three finite values.")
        result[key] = values.tolist()
    for key in ("height", "width", "initial_forward_offset", "initial_lateral_offset", "initial_yaw", "initial_velocity_noise"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError(f"{key} must be finite.")
    if result["height"] <= 0 or result["width"] <= 0 or result["initial_velocity_noise"] < 0:
        raise ValueError("Invalid height, width or joint velocity noise.")
    if not isinstance(result["seed"], int) or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    identity = {key: value for key, value in result.items() if key not in ("name", "dataset_split")}
    if not result["initial_velocity_noise"]:
        identity.pop("seed")
    result["scenario_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return result


def assign_splits(scenes, validation_fraction=.25, split_seed=20261006):
    """Stratified physical scene holdout, fixed before teacher collection."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    canonical = [canonical_scenario(scene) for scene in scenes]
    ids = [scene["scenario_id"] for scene in canonical]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate physical scenarios do not provide independent validation.")
    explicit = ["dataset_split" in scene for scene in canonical]
    if any(explicit):
        if not all(explicit) or any(scene["dataset_split"] not in ("train", "validation") for scene in canonical):
            raise ValueError("Provide valid dataset_split for all scenarios or none.")
        return canonical
    for direction in (1, -1):
        group = sorted((scene for scene in canonical if scene["direction"] == direction),
                       key=lambda scene: hashlib.sha256(f"{split_seed}:{scene['scenario_id']}".encode()).hexdigest())
        count = min(len(group)-1, max(1, round(len(group)*validation_fraction))) if len(group) >= 2 and validation_fraction else 0
        for index, scene in enumerate(group):
            scene["dataset_split"] = "validation" if index < count else "train"
    return canonical


def summarize(manifest):
    entries = manifest["episodes"]
    paths = {split: [entry["trajectory"] for entry in entries
                     if entry["dataset_split"] == split and entry["training_eligible"]]
             for split in ("train", "validation")}
    return dict(attempts=len(entries), physical_successes=sum(entry["physical_success"] for entry in entries),
                complete_teacher_demonstrations=sum(entry["complete_teacher_demonstration"] for entry in entries),
                complete_perturbed_teacher_demonstrations=sum(entry["complete_perturbed_teacher_demonstration"]
                                                              for entry in entries),
                training_eligible_frames=sum(entry["frames"] for entry in entries if entry["training_eligible"]),
                failure_counts=dict(Counter(entry["failure"] for entry in entries if entry["failure"])),
                paths=paths)


def collect(args):
    from legged_lab.scripts.mujoco_stair_position_teacher import (
        build_parser, run_episode, validate_args, teacher_noise_settings,
    )

    noise_std, noise_seed = teacher_noise_settings(args)
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("duration must be finite and positive.")
    supplied = json.loads(Path(args.scenarios).read_text())
    supplied_scenes = supplied["scenarios"] if isinstance(supplied, dict) else supplied
    if args.geometry_source:
        supplied_scenes = [dict(scene, geometry_source=args.geometry_source) for scene in supplied_scenes]
    scenes = assign_splits(supplied_scenes, args.validation_fraction, args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Dataset manifest already exists; use a new directory to retain provenance.")
    manifest = dict(schema_version="elf3_stair_dataset_manifest_v1", mode="collect", status="running",
                    supervisor=args.supervisor, duration=args.duration,
                    scenario_file=str(Path(args.scenarios).resolve()), split_seed=args.seed,
                    validation_fraction=args.validation_fraction, scenarios=scenes,
                    position_config=str(Path(args.position_config).resolve()),
                    teacher_action_noise_std=noise_std, teacher_action_noise_seed=noise_seed,
                    teacher_noise_changes_physical_scenario_identity=False,
                    geometry_sources=sorted({scene["geometry_source"] for scene in scenes}),
                    force_oracle=True, depth_transfer_verified=False, episodes=[])

    def save():
        manifest["summary"] = summarize(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")

    save()
    try:
        for scene in scenes:
            episode_dir = output/scene["dataset_split"]/scene["scenario_id"]
            parser = build_parser()
            episode_args = parser.parse_args([
                "--position_config", args.position_config, "--output_dir", str(episode_dir),
                "--headless", "--duration", str(args.duration), "--supervisor", args.supervisor,
                "--teacher_action_noise_std", str(noise_std), "--teacher_action_noise_seed", str(noise_seed),
            ])
            for key, value in scene.items():
                if key not in ("direction", "name"):
                    setattr(episode_args, key, value)
            validate_args(episode_args, parser)
            report = run_episode(episode_args, scene["direction"])
            path = report.get("trajectory")
            frames = 0
            phase_counts, source_counts = {}, {}
            if path:
                with np.load(path, allow_pickle=False) as trace:
                    frames = len(trace["observations"])
                    phase_counts = dict(Counter(map(str, np.argmax(trace["observations"][:, 960:972], axis=1))))
                    source_counts = dict(Counter(map(str, trace["action_sources"])))
            manifest["episodes"].append(dict(scenario_id=scene["scenario_id"], supervisor=args.supervisor,
                data_kind="teacher_perturbed" if noise_std > 0 else "teacher",
                teacher_action_noise_std=noise_std, teacher_action_noise_seed=noise_seed,
                name=scene.get("name"), dataset_split=scene["dataset_split"], direction=scene["direction"],
                report=str(episode_dir/f"{'up' if scene['direction'] > 0 else 'down'}_000.json"),
                trajectory=path, frames=frames, phase_counts=phase_counts, action_source_counts=source_counts,
                physical_success=report["success"], failure=report["failure"],
                failure_stage=report["failure_stage"], training_eligible=bool(frames >= 2 and report["success"]),
                complete_teacher_demonstration=bool(not noise_std and report["success"]),
                complete_perturbed_teacher_demonstration=bool(noise_std > 0 and report["success"])))
            save()
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        save()
    print("[DATASET] "+json.dumps(manifest["summary"]), flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True, help="JSON list of explicit physical scenarios.")
    parser.add_argument("--supervisor", choices=("dynamic", "event"), default="event")
    parser.add_argument("--position_config", default=str(Path(__file__).resolve().parents[1]/"configs/elf3_stair_position.yaml"))
    parser.add_argument("--geometry_source", choices=("known", "depth"),
                        help="Override every scenario's geometry source; otherwise use JSON/default known.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--duration", type=float, default=40.)
    parser.add_argument("--teacher_action_noise_std", type=float, default=0.,
                        help="Gaussian noise on execution while preserving clean teacher labels.")
    parser.add_argument("--teacher_action_noise_seed", type=int, default=0,
                        help="Execution-noise seed, independent of scene initialization and split seeds.")
    parser.add_argument("--validation_fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=20261006, help="Split seed; physical seeds are in each scenario.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    manifest = collect(args)
    if any(not episode["physical_success"] for episode in manifest["episodes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
