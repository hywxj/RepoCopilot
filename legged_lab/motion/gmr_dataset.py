"""Auditable GMR preparation for ELF3; independent of Isaac Sim and training.

Input pickle files must be trusted local GMR exports. Output NPZ files never
require pickle. Eligibility is an explicit kinematic screen, not a contact,
naturalness, or dynamics certificate. Original motions are never modified.
"""

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .elf3_contract import AMP_INDICES, AMP_JOINT_NAMES, JOINT_NAMES, PROJECT_XML, Elf3Kinematics


FEATURE_SCHEMA = "elf3_amp70_absolute_joint_pos_vel_root_local_end_effectors_v1"
MOTION_SCHEMA = "elf3_gmr_motion_v1"
DATASET_SCHEMA = "elf3_gmr_dataset_v1"
DEFAULT_FILTERS = {
    "min_duration_s": 0.5,
    "max_joint_speed_rad_s": 16.0,
    "max_joint_limit_excess_rad": 0.0001,
    "max_root_tilt_deg": 60.0,
    "min_mean_forward_speed_m_s": 0.1,
    "max_backward_frame_fraction": 0.15,
    "backward_speed_threshold_m_s": -0.1,
    "max_source_fk_error_m": 0.0001,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_motion(relative_path):
    """Names classify content; measured forward motion is checked separately."""
    name = Path(relative_path).stem.lower()
    if re.search(r"back(?:ward|wards)|downstairs_b\d", name):
        return "backward"
    if re.search(r"(?:^|_)(run|running|jog|jump|scamper)(?:_|\d|$)", name):
        return "other"
    for token, category in (
        ("upstairs_downstairs", "stairs_mixed"), ("upstairs", "stairs_up"),
        ("downstairs", "stairs_down"), ("normal_walk", "walk_stationary_root"),
        ("treadmill", "walk_treadmill"), ("circle_walk", "walk_turn"),
        ("turn_left", "walk_turn"), ("turn_right", "walk_turn"),
        ("slope_up", "walk_slope"), ("slope_down", "walk_slope"),
        ("step_stones", "walk_stones"),
    ):
        if token in name:
            return category
    return "other"


def subject_group(relative_path):
    """Keep KIT subjects together even across stairs and terrain directories."""
    parts = Path(relative_path).parts
    if "bmlrub" in parts:
        index = parts.index("bmlrub")
        return "bmlrub/" + parts[index + 1]
    if parts[0] == "elf3_kit_stairs_gmr":
        return "kit/" + parts[1]
    if "kit_terrain_mix" in parts:
        index = parts.index("kit_terrain_mix")
        return "kit/" + parts[index + 1]
    raise ValueError(f"Unknown source family; cannot safely split {relative_path}")


def subject_splits(groups, validation_fraction=0.2, seed=20261007, categories_by_group=None):
    """Deterministic subject holdout within each source family, never by frame."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    families = defaultdict(set)
    for group in groups:
        families[group.split("/")[0]].add(group)
    result = {}
    for family, subjects in sorted(families.items()):
        if len(subjects) < 2:
            raise ValueError(f"At least two subjects needed for holdout in {family}")
        ordered = sorted(subjects, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
        count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
        validation = set(ordered[:count])
        if categories_by_group:
            all_categories = set().union(*(categories_by_group[s] for s in ordered))
            counts = {s: Counter(categories_by_group[s]) for s in ordered}
            totals = sum(counts.values(), Counter())
            candidates = combinations(ordered, count)
            if math.comb(len(ordered), count) > 10000:
                rng = np.random.default_rng(seed)
                candidates = [tuple(ordered[:count])] + [tuple(rng.choice(ordered, count, replace=False)) for _ in range(1000)]

            def score(candidate):
                held = set(candidate)
                valid_categories = set().union(*(categories_by_group[s] for s in held))
                train_categories = set().union(*(categories_by_group[s] for s in subjects - held))
                missing = len(all_categories - valid_categories) + len(all_categories - train_categories)
                valid_counts = sum((counts[s] for s in held), Counter())
                imbalance = sum(abs(valid_counts[c] / totals[c] - validation_fraction) for c in all_categories)
                return missing, imbalance, hashlib.sha256(f"{seed}:{','.join(sorted(held))}".encode()).hexdigest()

            validation = set(min(candidates, key=score))
        result.update({s: "validation" if s in validation else "train" for s in ordered})
    return result


def validate_raw(raw):
    fps = float(raw["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be a finite positive scalar")
    result = {k: np.asarray(raw[k], dtype=np.float64) for k in ("root_pos", "root_rot", "dof_pos")}
    n = len(result["root_pos"])
    for key, size in (("root_pos", 3), ("root_rot", 4), ("dof_pos", 29)):
        if result[key].shape != (n, size) or n < 2 or not np.isfinite(result[key]).all():
            raise ValueError(f"{key} must have at least two aligned finite frames of width {size}")
    norms = np.linalg.norm(result["root_rot"], axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError("Source root_rot must contain unit XYZW quaternions")
    result["root_rot"] /= norms[:, None]
    result["fps"] = fps
    return result


def resample_motion(raw, target_fps=50.0):
    """Resample actual source seconds; no looping, stretching, or extrapolation."""
    raw = validate_raw(raw)
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    source_time = np.arange(len(raw["root_pos"]), dtype=np.float64) / raw["fps"]
    time = np.arange(int(np.floor(source_time[-1] * target_fps + 1e-9)) + 1) / target_fps
    # Numerical near-integer durations must still stay within Slerp's domain.
    time = time[time <= source_time[-1] + 1e-12]
    if len(time) < 3:
        raise ValueError("Motion is too short for target-rate velocity estimation")
    query_time = np.minimum(time, source_time[-1])
    interpolate = lambda a: np.column_stack([np.interp(query_time, source_time, a[:, j]) for j in range(a.shape[1])])
    positions = interpolate(raw["root_pos"])
    joints = interpolate(raw["dof_pos"])
    rotation = Slerp(source_time, Rotation.from_quat(raw["root_rot"]))(query_time)
    quaternions = rotation.as_quat()[:, [3, 0, 1, 2]]  # serialized output is WXYZ
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i - 1], quaternions[i]) < 0:
            quaternions[i] *= -1
    dt = 1.0 / target_fps
    # First-order endpoints avoid extrapolated velocity spikes at clip edges.
    linear_world = np.gradient(positions, dt, axis=0, edge_order=1)
    # Relative rotations in world coordinates, averaged at interior samples.
    interval_angular = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
    angular_world = np.vstack((interval_angular[0], (interval_angular[:-1] + interval_angular[1:]) / 2,
                               interval_angular[-1]))
    return {
        "time": time, "root_position_world": positions, "root_quaternion_wxyz": quaternions,
        "joint_positions": joints, "joint_velocities": np.gradient(joints, dt, axis=0, edge_order=1),
        "root_linear_velocity_world": linear_world,
        "root_linear_velocity_local": rotation.inv().apply(linear_world),
        "root_angular_velocity_local": rotation.inv().apply(angular_world),
        "sample_dt": np.array(dt),
    }


def verify_source_model(source, target):
    """Export and execution models may differ in dynamics, but FK must agree."""
    a, b = source.model, target.model
    names_a = [a.joint(i).name for i in range(1, a.njnt)]
    names_b = [b.joint(i).name for i in range(1, b.njnt)]
    if names_a != names_b or tuple(names_a) != tuple(JOINT_NAMES):
        raise ValueError("GMR and project joint orders disagree")
    for key in ("jnt_axis", "jnt_pos", "jnt_range", "jnt_type", "jnt_qposadr", "body_parentid"):
        if not np.allclose(getattr(a, key), getattr(b, key), atol=1e-10, rtol=0):
            raise ValueError(f"GMR and project models disagree on {key}")
    poses = np.random.default_rng(0).uniform(target.joint_limits[:, 0], target.joint_limits[:, 1], size=(8, 29))
    fk_a, fk_b = source.forward(poses), target.forward(poses)
    if list(fk_a["body_names"]) != list(fk_b["body_names"]):
        raise ValueError("GMR and project body names disagree")
    if not np.allclose(fk_a["body_positions_local"], fk_b["body_positions_local"], atol=1e-8, rtol=0):
        raise ValueError("GMR and project forward kinematics disagree")


def quality_metrics(raw, motion, kinematics):
    source = validate_raw(raw)
    q = source["dof_pos"]
    limits = kinematics.joint_limits
    source_joint_speed = np.abs(np.diff(q, axis=0) * source["fps"])
    speed_index = np.unravel_index(np.argmax(source_joint_speed), source_joint_speed.shape)
    source_rotation = Rotation.from_quat(source["root_rot"])
    forward = source_rotation[:-1].inv().apply(np.diff(source["root_pos"], axis=0) * source["fps"])[:, 0]
    up = source_rotation.apply(np.tile([0., 0., 1.], (len(q), 1)))
    # Compare the export's redundant FK against the explicit named model.
    sample_ids = np.unique(np.linspace(0, len(q) - 1, min(5, len(q)), dtype=int))
    fk = kinematics.forward(q[sample_ids])
    names = list(raw["link_body_list"])
    if len(set(names)) != len(names) or set(names) != set(fk["body_names"]):
        raise ValueError("Export link_body_list must match the ELF3 body contract")
    local = np.asarray(raw["local_body_pos"], dtype=float)
    if local.shape != (len(q), len(names), 3) or not np.isfinite(local).all():
        raise ValueError("Export local_body_pos must be finite aligned body positions")
    gather = [names.index(name) for name in fk["body_names"]]
    fk_error = np.linalg.norm(local[sample_ids][:, gather] - fk["body_positions_local"], axis=-1).max()
    return {
        "source_frames": len(q), "source_fps": source["fps"],
        "source_duration_s": (len(q) - 1) / source["fps"],
        "target_frames": len(motion["time"]), "target_duration_s": float(motion["time"][-1]),
        "source_max_joint_speed_rad_s": float(source_joint_speed.max()),
        "source_peak_speed_joint": JOINT_NAMES[speed_index[1]],
        "source_peak_speed_time_s": float(speed_index[0] / source["fps"]),
        "target_max_joint_speed_rad_s": float(np.abs(motion["joint_velocities"]).max()),
        "max_joint_limit_excess_rad": float(max(0, (limits[:, 0] - q).max(), (q - limits[:, 1]).max())),
        "max_root_tilt_deg": float(np.rad2deg(np.arccos(np.clip(up[:, 2], -1, 1))).max()),
        "mean_forward_speed_m_s": float(forward.mean()),
        "backward_frame_fraction": float(np.mean(forward < DEFAULT_FILTERS["backward_speed_threshold_m_s"])),
        "source_fk_error_m": float(fk_error),
        "source_fk_checked_frames": sample_ids.tolist(),
    }


def exclusion_reasons(category, metrics, filters):
    reasons = []
    if category in ("backward", "other"):
        reasons.append("motion_category")
    if category in ("walk_treadmill", "walk_stationary_root"):
        reasons.append("stationary_root_style_only")
    checks = (
        (metrics["source_duration_s"] < filters["min_duration_s"], "too_short"),
        (max(metrics["source_max_joint_speed_rad_s"], metrics["target_max_joint_speed_rad_s"])
         > filters["max_joint_speed_rad_s"], "joint_speed_spike"),
        (metrics["max_joint_limit_excess_rad"] > filters["max_joint_limit_excess_rad"], "joint_limit"),
        (metrics["max_root_tilt_deg"] > filters["max_root_tilt_deg"], "root_tilt"),
        (metrics["mean_forward_speed_m_s"] < filters["min_mean_forward_speed_m_s"] or
         metrics["backward_frame_fraction"] > filters["max_backward_frame_fraction"], "not_forward_motion"),
        (metrics["source_fk_error_m"] > filters["max_source_fk_error_m"], "source_fk_mismatch"),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return reasons


def build_dataset(source_root, index_root, source_xml, output_dir, *, target_fps=50.0,
                  validation_fraction=0.2, seed=20261007, filters=None):
    """Use the existing clean TXT tree only as a filename allowlist, not data."""
    source_root, index_root, source_xml, output_dir = map(Path, (source_root, index_root, source_xml, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; use a new dataset version")
    if not np.isclose(target_fps, 50., atol=0, rtol=0):
        raise ValueError("The current continuous-control contract requires 50 Hz")
    filters = dict(DEFAULT_FILTERS, **(filters or {}))
    if set(filters) != set(DEFAULT_FILTERS) or not all(np.isfinite(v) for v in filters.values()):
        raise ValueError("Unknown or nonfinite quality thresholds")
    if filters["backward_speed_threshold_m_s"] != DEFAULT_FILTERS["backward_speed_threshold_m_s"]:
        raise ValueError("Changing the velocity sign threshold requires a new metric contract")
    references = sorted(p.relative_to(index_root) for p in index_root.rglob("*.txt"))
    if not references:
        raise ValueError("Clean index contains no TXT files")
    paths = [p.with_suffix(".pkl") for p in references]
    missing = [str(p) for p in paths if not (source_root / p).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} source motions: {missing[:5]}")
    groups = {p: subject_group(p) for p in paths}
    categories_by_group = defaultdict(Counter)
    for p, group in groups.items():
        category = classify_motion(p)
        if category not in ("backward", "other", "walk_stationary_root", "walk_treadmill"):
            categories_by_group[group][category] += 1
        else:
            categories_by_group[group]  # Include subjects with only excluded categories.
    splits = subject_splits(groups.values(), validation_fraction, seed, categories_by_group)
    kinematics, source_kinematics = Elf3Kinematics(), Elf3Kinematics(source_xml)
    verify_source_model(source_kinematics, kinematics)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clips").mkdir()
    contract = {
        "feature_schema": FEATURE_SCHEMA, "joint_names": list(JOINT_NAMES),
        "amp_joint_names": list(AMP_JOINT_NAMES), "quaternion_order": "wxyz",
        "world_axes": "right_handed_z_up", "root_frame": "torso_link",
        "velocity_units": "m/s and rad/s", "joint_positions": "absolute radians, no default-pose subtraction",
        "end_effectors": "LH,RH: elbow_y_link + rotated(0,0,-0.3); LF,RF: ankle_x_link origins; root local",
        "contact_labels": "unavailable; no contact or terrain inferred from world height",
        "root_height": "original GMR global height offset retained; not terrain calibrated",
        "mirror": "verified ELF3 sagittal FK transform; not augmented in this dataset",
        "source_fk_check": "up to five equally spaced source frames per clip; not every frame",
        "stored_precision": "float64 time/sample_dt; float32 poses, velocities, endpoints and AMP features",
        "resampling": "source fps timestamps; linear translation/joints, quaternion SLERP; central velocities, one-sided endpoints",
    }
    clips, hashes = [], {}
    for relative, reference in zip(paths, references):
        source_path = source_root / relative
        digest = sha256(source_path)
        record = {"id": hashlib.sha256(str(relative).encode()).hexdigest()[:20],
                  "source": str(relative), "source_resolved": str(source_path.resolve()), "source_sha256": digest,
                  "index": str(reference), "index_sha256": sha256(index_root / reference),
                  "group": groups[relative], "split": splits[groups[relative]],
                  "category": classify_motion(relative), "eligible": False,
                  "physical_execution_validated": False, "motion_quality_validated": False}
        if digest in hashes:
            raise ValueError(f"Duplicate source contents require grouped resolution: {relative} and {hashes[digest]}")
        hashes[digest] = str(relative)
        try:
            with source_path.open("rb") as stream:
                raw = pickle.load(stream)
            motion = resample_motion(raw, target_fps)
            metrics = quality_metrics(raw, motion, kinematics)
            reasons = exclusion_reasons(record["category"], metrics, filters)
            record.update(metrics=metrics, exclusion_reasons=reasons, eligible=not reasons)
            if record["eligible"]:
                fk = kinematics.forward(motion["joint_positions"])
                end_effectors = fk["end_effectors_local"]
                amp = np.concatenate((motion["joint_positions"][:, AMP_INDICES],
                                      motion["joint_velocities"][:, AMP_INDICES],
                                      end_effectors.reshape(len(motion["time"]), 12)), axis=1)
                relative_output = Path("clips") / (record["id"] + ".npz")
                stored = {k: v if k in ("time", "sample_dt") else v.astype(np.float32) for k, v in motion.items()}
                np.savez_compressed(output_dir / relative_output, **stored,
                                    schema_version=np.array(MOTION_SCHEMA), feature_schema=np.array(FEATURE_SCHEMA),
                                    joint_names=np.asarray(JOINT_NAMES), amp_joint_names=np.asarray(AMP_JOINT_NAMES),
                                    end_effectors_local=end_effectors.astype(np.float32), amp_observations=amp.astype(np.float32),
                                    metadata_json=np.array(json.dumps(record, sort_keys=True)))
                record.update(output=str(relative_output), output_sha256=sha256(output_dir / relative_output))
        except (ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as error:
            record.update(eligible=False, exclusion_reasons=["invalid_source"], error=str(error))
        clips.append(record)
    eligible = [c for c in clips if c["eligible"]]
    by_category = {}
    for category in sorted({c["category"] for c in clips}):
        selected = [c for c in clips if c["category"] == category]
        by_category[category] = {"total": len(selected),
                                 **{split: sum(c["eligible"] and c["split"] == split for c in selected)
                                    for split in ("train", "validation")}}
    summary = {"source_clips": len(clips), "eligible_clips": len(eligible),
               "excluded_clips": len(clips) - len(eligible),
               "source_frames": sum(c.get("metrics", {}).get("source_frames", 0) for c in clips),
               "eligible_frames": sum(c["metrics"]["target_frames"] for c in eligible),
               "by_category": by_category,
               "exclusion_reasons": dict(Counter(r for c in clips for r in c["exclusion_reasons"])),
               "split_clips": dict(Counter(c["split"] for c in eligible)),
               "split_groups": {split: sorted(g for g, s in splits.items() if s == split)
                                for split in ("train", "validation")}}
    eligible_categories = {c["category"] for c in eligible}
    summary["missing_split_categories"] = {
        split: sorted(eligible_categories - {c["category"] for c in eligible if c["split"] == split})
        for split in ("train", "validation")}
    manifest = {"schema_version": DATASET_SCHEMA, "sample_dt": 1. / target_fps,
                "contract": contract, "source_root": str(source_root.resolve()),
                "index_root": str(index_root.resolve()), "source_xml": str(source_xml.resolve()),
                "source_xml_sha256": sha256(source_xml), "project_xml_sha256": sha256(PROJECT_XML),
                "preparation_code_sha256": sha256(__file__),
                "kinematics_code_sha256": sha256(Path(__file__).with_name("elf3_contract.py")),
                "selection_thresholds": filters, "selection_scope": "kinematic_screen_only",
                "split_seed": seed, "validation_fraction": validation_fraction,
                "split_policy": "subject holdout shared across KIT stairs and terrain, source families and category coverage",
                "summary": summary, "clips": clips}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not all(summary["split_clips"].get(s) for s in ("train", "validation")):
        raise ValueError("No eligible train or validation data; inspect the written manifest")
    return manifest
