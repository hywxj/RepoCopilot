"""Collect single-step teacher demonstrations with fixed 20 ms position PD.

The dynamic/event reference, rendered or known geometry, and simulation contact
oracle are retained. Teacher labels come from isolated predictive copies; live
physics executes only the held position target. No learned policy is loaded.
"""

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import time
import uuid

import mujoco
import numpy as np

from legged_lab.perception.mujoco_position_bridge import DynamicPositionBridge, MujocoPositionInterface
from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
from legged_lab.perception.stair_position_observation import StairPositionObservation
from legged_lab.perception.stair_event_supervisor import EventDrivenStairEpisode


# Capture at import, before a long batch can outlive an edit to the source files.
# A manifest must distinguish older and newer controller implementations.
IMPLEMENTATION_SHA256 = {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                         for cls in (DynamicTeachingEpisode, EventDrivenStairEpisode,
                                     DynamicPositionBridge, StairPositionObservation)}
IMPLEMENTATION_SHA256["position_recorder"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_episode(args, direction, index=0):
    teacher_noise_settings(args, index)  # Reject invalid noise before creating an episode.
    with ExitStack() as cleanup:
        try:
            return _run_episode(args, direction, index, cleanup)
        except (RuntimeError, ValueError) as exc:
            # A failed initialization is a dataset outcome too. In particular,
            # do not silently omit scenarios for which target locking failed.
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            path = output / f"{'up' if direction > 0 else 'down'}_{index:03d}.json"
            if path.exists():
                raise
            report = dict(mode="collect", direction=direction, success=False,
                          physical_success=False, independent_policy_success=False,
                          failure=str(exc), failure_stage="initialization", samples=[],
                          scenario_id=getattr(args, "scenario_id", None),
                          dataset_split=getattr(args, "dataset_split", None),
                          geometry_source=getattr(args, "geometry_source", "known"),
                          initial_conditions=initial_conditions(args, index),
                          data_kind="teacher", learned_policy=False)
            if getattr(args, "teacher_action_noise_std", 0.) > 0:
                report.update(data_kind="teacher_perturbed", learned_policy=False,
                              **teacher_noise_metadata(args, index))
            path.write_text(json.dumps(report, indent=2)+"\n")
            print("[RESULT] "+json.dumps(report), flush=True)
            return report


def initial_conditions(args, index=0):
    """The requested physical initial condition, applied before preparation."""
    return dict(height=getattr(args, "height", .11), width=getattr(args, "width", .32),
                initial_forward_offset=args.initial_forward_offset,
                initial_lateral_offset=getattr(args, "initial_lateral_offset", 0.),
                initial_yaw=getattr(args, "initial_yaw", 0.),
                initial_linear_velocity=list(getattr(args, "initial_linear_velocity", (0., 0., 0.))),
                initial_angular_velocity=list(getattr(args, "initial_angular_velocity", (0., 0., 0.))),
                initial_joint_velocity_noise=args.initial_velocity_noise, seed=args.seed+index)


def teacher_noise_settings(args, index=0):
    """A separate, explicit collection RNG; scene/reset seeds remain untouched."""
    std = float(getattr(args, "teacher_action_noise_std", 0.))
    seed = getattr(args, "teacher_action_noise_seed", 0)
    if (not np.isfinite(std) or std < 0 or type(seed) is not int or seed < 0):
        raise ValueError("Teacher action noise requires finite nonnegative std and a nonnegative integer seed.")
    return std, seed+index


def teacher_noise_metadata(args, index=0):
    std, seed = teacher_noise_settings(args, index)
    return dict(teacher_action_noise_std=std, teacher_action_noise_seed=seed,
                teacher_action_noise_process="independent_gaussian_per_control_step",
                teacher_action_noise_units="normalized_position_action",
                teacher_label_semantics="clean_teacher_action_at_actual_state")


def _run_episode(args, direction, index, cleanup):
    name = f"{'up' if direction > 0 else 'down'}_{index:03d}"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/f"{name}.json").exists() or list(output.glob(f"{name}_*.npz")):
        raise FileExistsError(f"Refusing to overwrite {name}; select a new output directory.")
    noise_std, noise_seed = teacher_noise_settings(args, index)
    # Do not even construct a generator on the historical zero-noise path.
    noise_rng = np.random.Generator(np.random.PCG64(noise_seed)) if noise_std > 0 else None
    requested_initial = initial_conditions(args, index)
    geometry_source = getattr(args, "geometry_source", "known")
    supervisor = getattr(args, "supervisor", "event")
    episode_class = DynamicTeachingEpisode
    if supervisor == "event":
        episode_class = EventDrivenStairEpisode
    episode = episode_class(direction=direction, geometry_source=geometry_source, **requested_initial)
    if hasattr(episode, "close"):
        cleanup.callback(episode.close)
    interface = MujocoPositionInterface(episode.model, args.position_config)
    observer = StairPositionObservation(interface)
    metadata = observer.metadata(episode)
    metadata.update(direction=direction, initial_forward_offset_m=args.initial_forward_offset,
                    initial_joint_velocity_noise=args.initial_velocity_noise, seed=args.seed+index,
                    teacher_labels=True, learned_policy=False, data_kind="teacher",
                    trajectory_id=uuid.uuid4().hex,
                    scenario_id=getattr(args, "scenario_id", None),
                    dataset_split=getattr(args, "dataset_split", None),
                    initial_conditions=requested_initial,
                    motion_start_qpos=episode.data.qpos.tolist(),
                    motion_start_qvel=episode.data.qvel.tolist(),
                    motion_start_time=float(episode.data.time),
                    implementation_sha256=IMPLEMENTATION_SHA256,
                    applied_actions_dtype="float64",
                    applied_actions_semantics="post_clip_position_actions_held_by_live_pd",
                    planned_source_soles_world=(episode.source.tolist() if hasattr(episode, "source") else None),
                    planned_target_soles_world=(episode.target.tolist() if hasattr(episode, "target") else None),
                    geometry_source=geometry_source,
                    supervisor=supervisor,
                    known_geometry=geometry_source == "known", depth_transfer_verified=False,
                    initialization_teacher=True)
    if hasattr(episode, "reference_profile"):
        metadata["reference_profile"] = episode.reference_profile
    if noise_rng is not None:
        metadata.update(data_kind="teacher_perturbed", **teacher_noise_metadata(args, index))
    bridge = DynamicPositionBridge(episode, interface)
    viewer = None
    if not args.headless:
        from mujoco import viewer as viewer_api
        from legged_lab.scripts.mujoco_stair_teacher import draw_target_region, draw_motion_status
        viewer = viewer_api.launch_passive(episode.model, episode.data)
        cleanup.callback(viewer.close)
        viewer.cam.lookat[:] = [.3, 0., .65]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 130., -18.
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          f"{'UP' if direction > 0 else 'DOWN'} | TEACHER | 20 ms position PD",
                          ("Rendered depth geometry; simulation contact supervisor" if geometry_source == "depth"
                           else "Known geometry/contact supervisor; no depth transfer")))
    records, samples, failure, failure_stage = [], [], "", ""
    last_phase, closed = "", False
    try:
        while episode.data.time-episode.start_time < args.duration:
            if viewer is not None and not viewer.is_running():
                closed = True
                break
            started = time.monotonic()
            failure_stage = "supervisor_observation"
            observation = observer.observe(episode)
            state_time = float(episode.data.time)
            qpos, qvel = episode.data.qpos.copy(), episode.data.qvel.copy()
            failure_stage = "teacher_label_query"
            label = bridge.action()
            elapsed = float(episode.data.time-episode.start_time)
            action, source = label.copy(), "teacher"
            if noise_rng is not None:
                # The label was queried at this actual perturbed rollout state.
                # Only execution is disturbed; never overwrite the clean label.
                action = action+noise_rng.normal(0., noise_std, size=action.shape)
                if not np.isfinite(action).all():
                    raise ValueError("Nonfinite perturbed teacher action; no PD command was applied.")
                source = "teacher_perturbed"
            action = np.clip(action, -interface.clip_actions, interface.clip_actions)
            records.append([state_time, observation, label, action.copy(), qpos, qvel, 0, source, elapsed])
            failure_stage = "position_physics"
            for _ in range(interface.physics_steps):
                sample, _ = episode._advance(lambda ep, _: interface.torque_from_action(ep.data, action))
                records[-1][6] += 1
            observer.last_action = action.copy()
            samples.append(sample)
            if sample["phase"] != last_phase:
                print(json.dumps({k: sample[k] for k in ("motion_time", "phase", "success")}), flush=True)
                last_phase = sample["phase"]
            if viewer is not None:
                with viewer.lock():
                    draw_target_region(viewer.user_scn, episode.target_region_quads, episode.controller.lock, False)
                    draw_motion_status(viewer.user_scn, sample)
                viewer.sync()
                time.sleep(max(0., interface.dt/args.playback_speed-(time.monotonic()-started)))
            if sample["success"]:
                break
        failure_stage = "" if samples and samples[-1]["success"] else "viewer" if closed else "timeout"
    except (RuntimeError, ValueError) as exc:
        failure = str(exc)
    success = bool(samples and samples[-1]["success"] and not failure and not closed)
    if not success and not failure:
        failure = "viewer_closed" if closed else "episode_timeout"
    metadata["physical_success"] = success
    sources = [record[7] for record in records]
    metadata["motor_teacher_applied"] = bool(records)
    metadata["independent_policy_success"] = False
    report = dict(metadata, mode="collect", success=success, failure=failure,
                  failure_diagnostics=getattr(episode, "failure_diagnostics", None),
                  failure_stage=failure_stage, terminal_time=float(episode.data.time),
                  terminal_qpos=episode.data.qpos.tolist(), terminal_qvel=episode.data.qvel.tolist(),
                  action_hold_steps=interface.physics_steps,
                  applied_motor_controller="fixed_target_position_pd",
                  inverse_dynamics_torque_applied=False,
                  teacher_label_query=True,
                  teacher_label_execution="isolated_predictive_clone",
                  completed_control_intervals=sum(r[6] == interface.physics_steps for r in records),
                  samples=samples, motion_time=float(episode.data.time-episode.start_time),
                  initial_state_is_locked_standing=True,
                  external_forces_applied=False, runtime_state_corrections=False)
    # Keep failures for diagnosis; the dataset manifest excludes them from
    # successful demonstration paths.
    if records:
        path = output/f"{name}_positions{'' if success else '_failed'}.npz"
        arrays = dict(state_time=np.array([r[0] for r in records]),
                      schema_version=metadata["schema_version"],
                      observations=np.stack([r[1] for r in records]),
                      # Preserve exactly the command held by live PD. Labels
                      # retain the historical float32 data format; quantizing
                      # executed commands would change replay contact dynamics.
                      applied_actions=np.stack([r[3] for r in records]).astype(np.float64),
                      action_sources=np.asarray(sources),
                      motion_time=np.asarray([r[8] for r in records]),
                      executed_physics_steps=np.array([r[6] for r in records]),
                      qpos=np.stack([r[4] for r in records]), qvel=np.stack([r[5] for r in records]),
                      metadata_json=json.dumps(metadata), direction=direction)
        arrays["actions"] = np.stack([r[2] for r in records]).astype(np.float32)
        np.savez_compressed(path, **arrays)
        report["trajectory"] = str(path)
    (output/f"{name}.json").write_text(json.dumps(report, indent=2)+"\n")
    print("[RESULT] "+json.dumps({k: report[k] for k in
          ("mode", "direction", "success", "failure", "motion_time", "motor_teacher_applied")}), flush=True)
    if viewer is not None and args.hold_viewer and viewer.is_running():
        status = "COMPLETE" if success else "STOPPED"
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          f"{'UP' if direction > 0 else 'DOWN'} | TEACHER | {status}",
                          (failure or "Physical acceptance passed")+"\nClose this window to continue."))
        print("[VIEWER] Result saved; simulation is stopped. Close the window to continue.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(.03)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor", choices=("dynamic", "event"), default="event")
    parser.add_argument("--direction", choices=("up", "down", "both"), default="both")
    parser.add_argument("--position_config", default=str(Path(__file__).resolve().parents[1]/"configs/elf3_stair_position.yaml"),
                        help="ELF3 joint PD/scales configuration.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold_viewer", action="store_true",
                        help="Keep the terminal pose and result visible until the window is closed.")
    parser.add_argument("--playback_speed", type=float, default=1.,
                        help="Visible playback rate only (0.25 is quarter speed); physics timing is unchanged.")
    parser.add_argument("--duration", type=float, default=40.)
    parser.add_argument("--initial_forward_offset", type=float, default=.05)
    parser.add_argument("--height", type=float, default=.11)
    parser.add_argument("--width", type=float, default=.32)
    parser.add_argument("--geometry_source", choices=("known", "depth"), default="known",
                        help="Source of the target tread; depth uses rendered depth, not the known terrain point cloud.")
    parser.add_argument("--initial_lateral_offset", type=float, default=0.)
    parser.add_argument("--initial_yaw", type=float, default=0., help="Initial yaw in radians.")
    parser.add_argument("--initial_linear_velocity", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--initial_angular_velocity", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--initial_velocity_noise", type=float, default=0.)
    parser.add_argument("--teacher_action_noise_std", type=float, default=0.,
                        help="Independent Gaussian execution noise in normalized position-action units; clean labels are unchanged.")
    parser.add_argument("--teacher_action_noise_seed", type=int, default=0,
                        help="Separate execution-noise seed (+ episode index); never changes the physical initialization seed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    return parser


def validate_args(args, parser):
    try:
        teacher_noise_settings(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.headless and args.hold_viewer:
        parser.error("--hold_viewer requires a visible viewer.")
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("--playback_speed must be finite and positive.")
    if args.episodes < 1 or args.duration <= 0 or args.initial_velocity_noise < 0 or args.seed < 0:
        parser.error("Invalid episode count, duration or initial velocity noise.")
    values = np.r_[args.height, args.width, args.initial_forward_offset, args.initial_lateral_offset,
                   args.initial_yaw, args.initial_linear_velocity, args.initial_angular_velocity,
                   args.initial_velocity_noise, args.duration]
    if not np.isfinite(values).all() or args.height <= 0 or args.width <= 0:
        parser.error("Initial conditions must be finite; height and width must be positive.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    directions = (1, -1) if args.direction == "both" else (1,) if args.direction == "up" else (-1,)
    reports = [run_episode(args, direction, index) for index in range(args.episodes) for direction in directions]
    if not all(report["success"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
