"""Export audited ELF3 clips to InstinctLab's existing retargeted-motion format.

This changes container field names only: no retargeting, pose offsets, quaternion
reordering, velocity fitting, or resampling takes place during export.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import yaml

from legged_lab.motion.elf3_contract import AMP_JOINT_NAMES, JOINT_NAMES


ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCHEMA = "elf3_amp70_absolute_joint_pos_vel_root_local_end_effectors_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def export_hiking_motion(manifest_path, output_dir):
    """Validate the manifest, export eligible train clips, and retain provenance."""
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; existing exports are never overwritten.")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    _require(manifest.get("schema_version") == "elf3_gmr_dataset_v1", "Unsupported manifest schema_version")
    _require(np.isclose(manifest.get("sample_dt", 0), .02, rtol=0., atol=1.e-12), "Expected a 20 ms manifest")
    contract = manifest.get("contract", {})
    for key, expected in (("feature_schema", FEATURE_SCHEMA), ("quaternion_order", "wxyz"),
                          ("root_frame", "torso_link"), ("world_axes", "right_handed_z_up")):
        _require(contract.get(key) == expected, f"Incompatible contract {key}")
    _require(tuple(contract.get("joint_names", ())) == JOINT_NAMES, "Incompatible contract joint_names")
    _require(tuple(contract.get("amp_joint_names", ())) == AMP_JOINT_NAMES, "Incompatible contract amp_joint_names")
    clips = manifest.get("clips", [])
    _require(isinstance(clips, list) and clips, "Manifest must contain clips")
    groups, ids, seen_outputs, seen_hashes, selected = {}, set(), set(), set(), []
    validation = []
    for row in clips:
        for name in ("id", "group", "category"):
            _require(isinstance(row.get(name), str) and bool(row[name]), f"Missing clip {name}")
        _require(row["id"] not in ids, "Duplicate clip id")
        ids.add(row["id"])
        _require(type(row.get("eligible")) is bool and row.get("split") in ("train", "validation"),
                 "Every clip needs an explicit eligible flag and split")
        previous = groups.setdefault(row["group"], row["split"])
        _require(previous == row["split"], "Train and validation groups overlap")
        if row["split"] == "validation":
            validation.append({key: row[key] for key in ("id", "group", "category", "eligible", "source", "source_sha256",
                                                        "output", "output_sha256") if key in row})
        if not row["eligible"] or row["split"] != "train":
            continue
        relative = Path(row.get("output", ""))
        path = (manifest_path.parent / relative).resolve()
        _require(not relative.is_absolute() and path.is_relative_to(manifest_path.parent) and path.suffix == ".npz",
                 "Clip path must be an NPZ inside the manifest directory")
        content = path.read_bytes()
        digest = _sha256(content)
        _require(digest == row.get("output_sha256"), f"Clip sha256 mismatch: {row['id']}")
        _require(path not in seen_outputs and digest not in seen_hashes, "Duplicate training clip output or content")
        seen_outputs.add(path)
        seen_hashes.add(digest)
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            _require(data["schema_version"].shape == () and data["schema_version"].item() == "elf3_gmr_motion_v1",
                     "Unsupported clip schema_version")
            _require(data["feature_schema"].shape == () and data["feature_schema"].item() == FEATURE_SCHEMA,
                     "Incompatible clip feature_schema")
            _require(tuple(data["joint_names"].tolist()) == JOINT_NAMES, "Incompatible clip joint_names")
            _require(tuple(data["amp_joint_names"].tolist()) == AMP_JOINT_NAMES, "Incompatible clip amp_joint_names")
            _require(data["sample_dt"].shape == () and np.isclose(data["sample_dt"].item(), .02, rtol=0., atol=1.e-12),
                     "Clip sample_dt must be 20 ms")
            time = data["time"]
            _require(time.ndim == 1 and len(time) >= 11 and np.isfinite(time).all()
                     and np.allclose(time, np.arange(len(time)) * .02, rtol=0., atol=1.e-7),
                     "Clip time must contain at least 11 consecutive 20 ms frames starting at zero")
            arrays = {"joint_pos": data["joint_positions"], "base_pos_w": data["root_position_world"],
                      "base_quat_w": data["root_quaternion_wxyz"]}
            for key, size in (("joint_pos", 29), ("base_pos_w", 3), ("base_quat_w", 4)):
                _require(arrays[key].shape == (len(time), size) and arrays[key].dtype == np.float32
                         and np.isfinite(arrays[key]).all(), f"Invalid float32 {key}")
            _require(np.allclose(np.linalg.norm(arrays["base_quat_w"], axis=1), 1., atol=2.e-5),
                     "Root quaternions must be unit WXYZ")
            # Keep in memory until all selected source files pass validation.
            selected.append((row, {key: value.copy() for key, value in arrays.items()}, len(time)))
    _require(selected, "No eligible train clips")
    counts = Counter(row["category"] for row, _, _ in selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clips").mkdir()
    records, files, weights = [], [], []
    for index, (row, arrays, frames) in enumerate(selected):
        # Numeric output names avoid interpreting source identifiers as paths.
        relative = f"clips/{index:04d}_retargeted.npz"
        np.savez_compressed(output_dir / relative, framerate=np.array(50.), joint_names=np.asarray(JOINT_NAMES), **arrays)
        weight = 1. / counts[row["category"]]
        files.append(relative)
        weights.append(weight)
        records.append({"id": row["id"], "group": row["group"], "category": row["category"], "split": "train",
                        "source_output": row["output"], "source_output_sha256": row["output_sha256"],
                        "source": row.get("source"), "source_sha256": row.get("source_sha256"),
                        "output": relative, "output_sha256": _sha256((output_dir / relative).read_bytes()),
                        "frames": frames, "motion_weight": weight})
    selection = {"selected_files": files, "motion_weights": weights}
    (output_dir / "selection_train.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    provenance = {
        "schema_version": "elf3_hiking_motion_export_v1", "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(payload), "sample_dt": .02, "quaternion_order": "wxyz",
        "root_frame": "torso_link", "joint_names": list(JOINT_NAMES), "split": "train",
        "conversion": "field mapping only; source poses and frame timing unchanged",
        "velocity_estimation": "official AmassMotion frontward differences at load time, not source central velocities",
        "motion_sampling": "inverse category clip count weights at motion resampling; not duration-balanced frame sampling",
        "selection": "selection_train.yaml", "selection_sha256": _sha256((output_dir / "selection_train.yaml").read_bytes()),
        "clips": records, "validation_sources_not_exported": validation,
        "summary": {"train_clips": len(records), "train_frames": sum(row["frames"] for row in records),
                    "categories": dict(sorted(counts.items())), "validation_clips_not_exported": len(validation),
                    "eligible_validation_clips_not_exported": sum(row["eligible"] for row in validation)},
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "logs/data_preparation/elf3_gmr_v1/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/data_preparation/elf3_hiking_v1")
    args = parser.parse_args()
    result = export_hiking_motion(args.manifest, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
