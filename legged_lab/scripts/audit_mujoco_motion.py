"""Audit saved MuJoCo teacher states with forward kinematics, without stepping physics.

The measurements describe the recorded motion. They do not certify natural gait,
disturbance recovery, robustness across initial states, or hardware readiness.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.perception.stair_step_controller import StepPhase
from legged_lab.scripts.mujoco_stair_teacher import build_model


def _load_trace(path, model):
    required = ("state_time", "qpos", "qvel", "step_features", "actuator_joint_names")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required)-set(archive.files)
        if missing:
            raise ValueError("Trace is missing fields: "+", ".join(sorted(missing)))
        trace = {name: archive[name] for name in required}
        if "supervisor_dt" in archive.files:
            trace["supervisor_dt"] = archive["supervisor_dt"]
    times = trace["state_time"]
    if times.ndim != 1 or not len(times):
        raise ValueError("state_time must be a nonempty vector.")
    count = len(times)
    for name, shape in (("qpos", (count, model.nq)), ("qvel", (count, model.nv))):
        if trace[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {trace[name].shape}.")
    features = trace["step_features"]
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] < len(StepPhase):
        raise ValueError("step_features must contain one phase vector per recorded state.")
    for name in ("state_time", "qpos", "qvel", "step_features"):
        if trace[name].dtype.kind not in "fiu" or not np.isfinite(trace[name]).all():
            raise ValueError(f"{name} must contain finite numeric values.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("state_time must increase strictly.")
    if "supervisor_dt" in trace:
        dt = trace["supervisor_dt"]
        if dt.shape != () or dt.dtype.kind not in "fiu" or not np.isfinite(dt) or dt <= 0:
            raise ValueError("supervisor_dt must be a finite positive scalar.")
        if not np.allclose(np.diff(times), float(dt), rtol=1.e-4, atol=1.e-6):
            raise ValueError("Recorded state times do not match supervisor_dt.")
    names = trace["actuator_joint_names"]
    expected = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
                for joint in model.actuator_trnid[:, 0]]
    if (names.shape != (model.nu,) or names.dtype.kind not in "US"
            or names.astype(str).tolist() != expected):
        raise ValueError("actuator_joint_names must match the current model's actuator order.")
    phase_ids = np.argmax(features[:, :len(StepPhase)], axis=1)
    if not np.allclose(features[:, :len(StepPhase)], np.eye(len(StepPhase))[phase_ids], atol=1.e-6):
        raise ValueError("Recorded phase vectors must be one-hot StepPhase values.")
    trace["phase_ids"] = phase_ids
    return trace


def _load_report(trace_path, report_path, trace):
    """A paired report supplies measured loads; absence does not block FK audit."""
    if report_path is None:
        stem = trace_path.stem.removesuffix("_expert").removesuffix("_candidate")
        report_path = trace_path.with_name(stem+".json")
        if not report_path.is_file():
            return None, None
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    samples = report.get("samples") if isinstance(report, dict) else None
    if not isinstance(samples, list) or len(samples) != len(trace["state_time"]):
        raise ValueError("Paired report must have one sample per recorded trace state.")
    if not all(isinstance(row, dict) for row in samples):
        raise ValueError("Paired report samples must be objects.")
    try:
        times = np.asarray([row["time"] for row in samples], dtype=float)
        phases = [row["phase"] for row in samples]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Paired report samples require numeric time and phase names.") from error
    expected_phases = [StepPhase(int(phase)).name for phase in trace["phase_ids"]]
    if (times.shape != trace["state_time"].shape or not np.isfinite(times).all()
            or not np.allclose(times, trace["state_time"], atol=1.e-5, rtol=0)
            or phases != expected_phases):
        raise ValueError("Paired report times and phases must match the trace.")
    if not any("load_fraction" in row for row in samples):
        return report_path, None
    try:
        loads = np.asarray([row["load_fraction"] for row in samples], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Paired report must supply left/right loads for every sample.") from error
    if loads.shape != (len(samples), 2) or not np.isfinite(loads).all():
        raise ValueError("Paired load_fraction must be finite left/right values per state.")
    return report_path, loads


def audit(trace_path, direction=1, report_path=None):
    """Return observed posture and joint margins; never a motion-quality pass flag."""
    if direction not in (-1, 1):
        raise ValueError("direction must be 1 (up) or -1 (down).")
    trace_path = Path(trace_path)
    model, data = build_model(direction=direction)
    trace = _load_trace(trace_path, model)
    paired_report, loads = _load_report(trace_path, report_path, trace)
    body_names = ("torso_link", "waist_z_link")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if min(body_ids) < 0:
        raise ValueError("Current model must contain torso_link and waist_z_link.")
    hip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hip_z_link")
               for side in ("l", "r")]
    ankle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_x_link")
                 for side in ("l", "r")]
    if min(hip_ids+ankle_ids) < 0:
        raise ValueError("Current model must contain left/right hip_z_link and ankle_x_link bodies.")
    angles, centers, roots, leg_vectors = [], [], [], []
    for qpos, qvel in zip(trace["qpos"], trace["qvel"]):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        angles.append([Rotation.from_matrix(data.xmat[body].reshape(3, 3)).as_euler("xyz", degrees=True)
                       for body in body_ids])
        centers.append(data.subtree_com[body_ids[0]].copy())
        roots.append(data.xpos[body_ids[0]].copy())
        leg_vectors.append(data.xpos[hip_ids]-data.xpos[ankle_ids])
    angles, centers, roots = np.asarray(angles), np.asarray(centers), np.asarray(roots)
    leg_vectors = np.asarray(leg_vectors)
    # The stair scenes face +x. Measure the leg line in the world y/z plane,
    # independently of pelvis and torso orientation; do not infer foot support.
    leg_inclination = np.rad2deg(np.arctan2(leg_vectors[:, :, 1], leg_vectors[:, :, 2]))
    times, phase_ids = trace["state_time"], trace["phase_ids"]

    def location(index):
        return {"time_s": float(times[index]), "phase": StepPhase(int(phase_ids[index])).name}

    def leg_metrics(mask, loaded=False):
        result = {}
        for foot, side in enumerate(("left", "right")):
            selected = mask & (loads[:, foot] >= .60) if loaded else mask
            indices = np.flatnonzero(selected)
            if not len(indices):
                result[side] = None
                continue
            index = int(indices[np.argmax(np.abs(leg_inclination[indices, foot]))])
            vector = leg_vectors[index, foot]
            result[side] = {
                "sample_count": len(indices),
                "peak_abs_deg": float(abs(leg_inclination[index, foot])),
                "signed_angle_at_peak_deg": float(leg_inclination[index, foot]),
                "peak_at": location(index), "hip_minus_ankle_world_m": vector.tolist(),
                "lateral_displacement_m": float(vector[1]), "vertical_separation_m": float(vector[2]),
            }
            if loaded:
                result[side]["load_fraction_at_peak"] = float(loads[index, foot])
        return result

    joints = {}
    for name, joint in zip(trace["actuator_joint_names"].astype(str), model.actuator_trnid[:, 0]):
        joint = int(joint)
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected a hinge joint for angular audit: {name}.")
        values = np.rad2deg(trace["qpos"][:, model.jnt_qposadr[joint]])
        metrics = {"min_deg": float(values.min()), "max_deg": float(values.max()),
                   "max_abs_change_from_initial_deg": float(np.abs(values-values[0]).max())}
        if model.jnt_limited[joint]:
            limits = np.rad2deg(model.jnt_range[joint])
            margin = np.minimum(values-limits[0], limits[1]-values)
            index = int(np.argmin(margin))
            metrics.update(limit_deg=limits.tolist(), minimum_limit_margin_deg=float(margin[index]),
                           minimum_limit_margin_at=location(index))
        else:
            metrics.update(limit_deg=None, minimum_limit_margin_deg=None)
        joints[name] = metrics

    phases = {}
    for phase in dict.fromkeys(phase_ids):
        mask = phase_ids == phase
        indices = np.flatnonzero(mask)
        breaks = np.flatnonzero(np.diff(indices) > 1)+1
        intervals = [times[run[[0, -1]]].tolist() for run in np.split(indices, breaks)]
        phases[StepPhase(int(phase)).name] = {
            "time_s": times[indices[[0, -1]]].tolist(), "sample_count": int(mask.sum()),
            "contiguous_sample_intervals_s": intervals,
            "torso_max_abs_rpy_deg": np.max(np.abs(angles[mask, 0]), axis=0).tolist(),
            "pelvis_max_abs_rpy_deg": np.max(np.abs(angles[mask, 1]), axis=0).tolist(),
            "com_min_xy_m": centers[mask, :2].min(axis=0).tolist(),
            "com_max_xy_m": centers[mask, :2].max(axis=0).tolist(),
            "leg_frontal_inclination": leg_metrics(mask),
        }
        if loads is not None:
            phases[StepPhase(int(phase)).name]["loaded_leg_frontal_inclination"] = leg_metrics(mask, loaded=True)

    loaded_ankles = {}
    if loads is not None:
        for foot, side in enumerate(("l", "r")):
            mask = loads[:, foot] >= .60
            indices = np.flatnonzero(mask)
            for axis in ("x", "y"):
                name = f"{side}_ankle_{axis}_joint"
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint < 0:
                    continue
                if not len(indices):
                    loaded_ankles[name] = None
                    continue
                values = np.rad2deg(trace["qpos"][:, model.jnt_qposadr[joint]])
                index = int(indices[np.argmax(np.abs(values[indices]))])
                limits = np.rad2deg(model.jnt_range[joint])
                margin = np.minimum(values-limits[0], limits[1]-values)
                loaded_ankles[name] = {
                    "loaded_sample_count": int(mask.sum()), "peak_abs_deg": float(abs(values[index])),
                    "signed_angle_at_peak_deg": float(values[index]), "peak_at": location(index),
                    "load_fraction_at_peak": float(loads[index, foot]),
                    "minimum_limit_margin_deg": float(margin[indices].min()),
                    "samples_within_one_degree_of_limit": int(np.sum(mask & (margin < 1.))),
                }

    lowest = int(np.argmin(roots[:, 2]))
    result = {
        "source": str(trace_path), "direction": "up" if direction > 0 else "down",
        "method": "forward_kinematics_of_recorded_states_without_physics_steps",
        "scope": "Observed motion metrics only; not certification of natural or robust gait.",
        "rpy_convention": "world-frame xyz Euler angles, degrees",
        "leg_frontal_inclination_convention": (
            "Signed atan2(hip_y - ankle_y, hip_z - ankle_z) in degrees; world y/z plane "
            "for +x-facing stair scenes; hip_z_link to ankle_x_link, independent of pelvis/torso attitude. "
            "Unfiltered metrics include swinging legs; loaded metrics require measured report loads >= 0.60 BW."
        ),
        "limit_margin_convention": "Positive inside XML limits; negative means recorded limit exceedance.",
        "sample_count": len(times), "start_time_s": float(times[0]), "end_time_s": float(times[-1]),
        "duration_s": float(times[-1]-times[0]),
        "torso_max_abs_rpy_deg": np.abs(angles[:, 0]).max(axis=0).tolist(),
        "pelvis_max_abs_rpy_deg": np.abs(angles[:, 1]).max(axis=0).tolist(),
        "leg_frontal_inclination": leg_metrics(np.ones(len(times), dtype=bool)),
        "root_initial_height_m": float(roots[0, 2]), "root_min_height_m": float(roots[lowest, 2]),
        "root_max_drop_from_initial_m": float(roots[0, 2]-roots[lowest, 2]),
        "root_min_height_at": location(lowest), "joint_angles": joints, "phases": phases,
        "paired_report": None if paired_report is None else str(paired_report),
        "load_feedback_available": loads is not None, "loaded_threshold_body_weight": .60,
        "loaded_ankles": loaded_ankles,
    }
    if loads is not None:
        result["loaded_leg_frontal_inclination"] = leg_metrics(np.ones(len(times), dtype=bool), loaded=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Teacher motion NPZ containing recorded qpos/qvel.")
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print JSON to stdout.")
    parser.add_argument("--report", type=Path, help="Optional paired teacher report; auto-detected beside the trace.")
    args = parser.parse_args()
    try:
        result = audit(args.trace, 1 if args.direction == "up" else -1, args.report)
        content = json.dumps(result, indent=2, allow_nan=False)+"\n"
        if args.output is None:
            print(content, end="")
        else:
            args.output.write_text(content)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
