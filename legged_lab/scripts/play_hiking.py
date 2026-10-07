"""Replay an ELF3 Hiking checkpoint with the camera and policy depth side by side.

This is deterministic policy inference in the training task, including its random
resets, observation noise and falls. It is not a fixed-start traversal benchmark.
"""

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from legged_lab.hiking.bootstrap import ROOT, activate


def resolve_checkpoint(requested=None):
    if requested is not None:
        result = Path(requested).expanduser().resolve()
    else:
        active = json.loads((ROOT / "logs/elf3_hiking/active_run.json").read_text())
        candidates = list(Path(active["run_dir"]).glob("model_*.pt"))
        candidates = [p for p in candidates if p.stem.removeprefix("model_").isdigit()]
        # Prefer the latest saved model; resume_checkpoint may name the older
        # model from which the current run started.
        result = max(candidates, key=lambda p: int(p.stem.removeprefix("model_"))) if candidates else Path(
            active["resume_checkpoint"])
        result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def configure_terrain(cfg, terrain_name, num_envs):
    aliases = {"stairs_up": "pyramid_stairs", "stairs_down": "pyramid_stairs_inv", "flat": "perlin_rough"}
    selected = aliases.get(terrain_name, terrain_name)
    terrain = cfg.scene.terrain.terrain_generator
    if selected != "mixed":
        if selected not in terrain.sub_terrains:
            raise ValueError(f"Unknown terrain {terrain_name!r}; choose mixed, stairs_up, stairs_down, flat or "
                             + ", ".join(terrain.sub_terrains))
        sub_terrain = terrain.sub_terrains[selected]
        sub_terrain.proportion = 1.0
        if terrain_name == "flat":
            sub_terrain.noise_scale = [0.0, 0.0]
        terrain.sub_terrains = {selected: sub_terrain}
        terrain.num_cols = min(num_envs, 4)
        command = cfg.commands.base_velocity
        command.velocity_ranges = {selected: command.velocity_ranges[selected]}
        command.random_velocity_terrain = [name for name in command.random_velocity_terrain if name == selected]
    return list(terrain.sub_terrains)


def terrain_assignments(env, terrain_cfg):
    import numpy as np

    names = list(terrain_cfg.sub_terrains)
    proportions = np.asarray([v.proportion for v in terrain_cfg.sub_terrains.values()])
    thresholds = np.cumsum(proportions / proportions.sum())
    columns = [int(np.where(i / terrain_cfg.num_cols + .001 < thresholds)[0][0])
               for i in range(terrain_cfg.num_cols)]
    return [names[columns[col]] for col in env.scene.terrain.terrain_types.cpu().tolist()]


def depth_panel(raw_depth, policy_depth, lines):
    import cv2
    import numpy as np

    def color(values, maximum):
        values = np.nan_to_num(values, nan=maximum, posinf=maximum, neginf=0.)
        pixels = (np.clip(values / maximum, 0., 1.) * 255).astype(np.uint8)
        result = cv2.applyColorMap(pixels, cv2.COLORMAP_TURBO)
        return cv2.resize(result, (640, 360), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((490, 1280, 3), dtype=np.uint8)
    canvas[45:405, :640] = color(raw_depth, 2.5)
    canvas[45:405, 640:] = color(policy_depth, 1.)
    labels = [("Camera raw 36x64 | display 0-2.5 m", (12, 28)),
              ("Policy depth 18x32 | newest of 8 | 0-1", (650, 28))]
    labels.extend((line, (12, 428 + i * 25)) for i, line in enumerate(lines[:3]))
    for label, position in labels:
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, .57, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def main():
    activate()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=4)
    parser.add_argument("--terrain", default="stairs_up", help="stairs_up (default), stairs_down, flat, mixed, or exact terrain name; mixed renders the full training terrain grid")
    parser.add_argument("--env-index", type=int, help="Initially followed environment; default is the first up-stairs robot")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many policy steps; 0 runs until the viewer closes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-realtime", action="store_true", help="Run without the 50 Hz wall-clock limit")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(rendering_mode="performance")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0:
        parser.error("num_envs must be positive and steps cannot be negative")
    if args.env_index is not None and not 0 <= args.env_index < args.num_envs:
        parser.error("env-index must be between 0 and num_envs - 1")
    if args.headless and args.steps == 0:
        args.steps = 500
    checkpoint = resolve_checkpoint(args.checkpoint)
    output = (args.output_dir or ROOT / "logs/elf3_hiking/playbacks" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"HIKING_PLAY_CHECKPOINT={checkpoint}\nHIKING_PLAY_OUTPUT={output}", flush=True)
    args.kit_args = (args.kit_args or "") + " --/app/window/saveSizeOnExit=false"
    app = AppLauncher(args, width=1280, height=720, window_width=1280, window_height=800).app
    env = None
    failed = False
    cv2 = None
    try:
        import cv2
        import numpy as np
        import torch
        if not args.headless:
            import omni.appwindow
            from omni.kit.viewport.utility import get_active_viewport
            window = omni.appwindow.get_default_app_window()
            window.restore_window()
            window.resize(1280, 800)
            viewport = get_active_viewport()
            viewport.fill_frame = False
            viewport.set_texture_resolution((640, 360))
            viewport.resolution_scale = 1.0
            print(f"HIKING_PLAY_VIEWPORT={viewport.resolution}", flush=True)
        from instinctlab.envs import InstinctRlEnv
        from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg
        from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
        from legged_lab.hiking.elf3_env_cfg import Elf3HikingEnvCfg
        from legged_lab.hiking.runner import HikingRunner

        cfg = Elf3HikingEnvCfg()
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        # Replay uses only a few robots. Training-sized broad-phase buffers
        # reserve GPU memory even when most of their capacity is unused.
        if args.num_envs <= 20:
            cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
            cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
            cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
            cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
            cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
            cfg.sim.physx.gpu_collision_stack_size = 2**26
        if args.device:
            cfg.sim.device = args.device
        categories = configure_terrain(cfg, args.terrain, args.num_envs)
        cfg.viewer.eye = (3., -3., 2.)
        cfg.viewer.lookat = (0., 0., .6)
        env = InstinctRlVecEnvWrapper(InstinctRlEnv(cfg=cfg))
        base = env.unwrapped
        agent_cfg = G1ParkourPPORunnerCfg()
        agent_cfg.device = cfg.sim.device
        agent_cfg.seed = args.seed
        runner = HikingRunner(env, agent_cfg.to_dict(), log_dir=None, device=cfg.sim.device)
        # The pinned runner.load has no load_optimizer switch. Restore the
        # inference state explicitly and strictly, without allocating Adam state.
        state = torch.load(checkpoint, map_location=cfg.sim.device, weights_only=True)
        runner.alg.actor_critic.load_state_dict(state["model_state_dict"], strict=True)
        normalizer_groups = {key.removesuffix("_normalizer_state_dict") for key in state
                             if key.endswith("_normalizer_state_dict")}
        if normalizer_groups != set(runner.normalizers):
            raise RuntimeError(f"Checkpoint normalizers {normalizer_groups} differ from configuration {set(runner.normalizers)}")
        for name, normalizer in runner.normalizers.items():
            normalizer.load_state_dict(state[f"{name}_normalizer_state_dict"], strict=True)
        checkpoint_iteration = int(state["iter"])
        del state
        policy = runner.get_inference_policy(device=cfg.sim.device)
        model = runner.alg.actor_critic
        if type(model).__name__ != "EncoderMoEActorCritic" or model.num_moe_experts != 4:
            raise RuntimeError("Expected the trained four-expert Hiking policy")
        obs, _ = env.get_observations()
        fmt = env.get_obs_format()["policy"]
        depth_offset = 0
        for name, shape in fmt.items():
            if name == "depth_image":
                depth_shape = tuple(int(d) for d in shape)
                break
            depth_offset += math.prod(shape)
        else:
            raise RuntimeError("The actor has no depth image input")
        if depth_shape != (8, 18, 32):
            raise RuntimeError(f"Unexpected depth history shape {depth_shape}")
        depth_end = depth_offset + math.prod(depth_shape)
        assignments = terrain_assignments(base, cfg.scene.terrain.terrain_generator)
        selected = args.env_index if args.env_index is not None else next(
            (i for i, name in enumerate(assignments) if name == "pyramid_stairs"), 0)
        print("HIKING_PLAY_TERRAINS=" + json.dumps(dict(enumerate(assignments))), flush=True)
        print("Depth window keys: N/P next/previous robot, 1..9/0 terrain, Q/Esc quit. "
              "Falls reset normally. Training random starts and sensor noise remain enabled.", flush=True)
        window = "ELF3 Hiking | raw camera + policy depth"
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 490)
        resets = np.zeros(args.num_envs, dtype=np.int64)
        termination_counts = {name: 0 for name in base.termination_manager.active_terms}
        steps = 0
        all_finite = True
        frames_vary = False
        first_depth = None
        last_reset_causes = [[] for _ in range(args.num_envs)]
        last_reset_step = np.full(args.num_envs, -1, dtype=np.int64)
        started = time.monotonic()
        samples = []

        def capture():
            nonlocal first_depth, frames_vary
            # Read the exact observation passed to policy(). Do not recompute
            # observation groups: that can change history, delay or noise.
            history = obs[selected, depth_offset:depth_end].reshape(depth_shape)
            policy_depth = history[-1].detach().cpu().numpy().copy()
            raw = base.scene["camera"].data.output["distance_to_image_plane"][selected].squeeze(-1).detach().cpu().numpy().copy()
            root = base.scene["robot"].data.root_pos_w[selected].detach().cpu().numpy()
            origin = base.scene.env_origins[selected].detach().cpu().numpy()
            velocity = base.scene["robot"].data.root_lin_vel_b[selected].detach().cpu().tolist()
            command = base.command_manager.get_command("base_velocity")[selected].detach().cpu().tolist()
            action_term = base.action_manager.get_term("joint_pos")
            joint_ids = action_term._joint_ids
            robot = base.scene["robot"]
            def joint_values(value):
                return value[selected, joint_ids].detach().cpu().tolist()
            # Before the next action: targets are those last sent to the drive.
            # Preserve both policy-space actions and physical joint targets so
            # a small actor output can be distinguished from tracking failure.
            control = {
                "joint_names": [robot.joint_names[i] for i in joint_ids],
                "raw_action": action_term.raw_actions[selected].detach().cpu().tolist(),
                "processed_target_rad": action_term.processed_actions[selected].detach().cpu().tolist(),
                "last_drive_target_rad": joint_values(robot.data.joint_pos_target),
                "joint_position_rad": joint_values(robot.data.joint_pos),
                "joint_velocity_rad_s": joint_values(robot.data.joint_vel),
                "reported_applied_torque_nm": joint_values(robot.data.applied_torque),
            }
            if first_depth is None:
                first_depth = policy_depth.copy()
            else:
                frames_vary |= not np.array_equal(first_depth, policy_depth)
            details = {
                "step": steps, "env_index": selected, "terrain": assignments[selected],
                "terrain_level": int(base.scene.terrain.terrain_levels[selected]),
                "episode_seconds": float(base.episode_length_buf[selected]) * base.step_dt,
                "completed_episodes": int(resets[selected]),
                "root_height_above_env_origin_m": float(root[2] - origin[2]),
                "root_position_world_m": root.tolist(), "base_velocity_mps": velocity,
                "velocity_command": command, "latest_reset_causes": last_reset_causes[selected],
                "latest_reset_step": int(last_reset_step[selected]),
                "raw_depth_shape": list(raw.shape), "policy_depth_history_shape": list(depth_shape),
                "policy_depth_min": float(policy_depth.min()), "policy_depth_max": float(policy_depth.max()),
                "control_snapshot_before_next_action": control,
            }
            lines = [f"step {steps} | env {selected} | {assignments[selected]} | level {details['terrain_level']} | "
                     f"episode {details['episode_seconds']:.2f}s | resets {resets[selected]}",
                     f"vx {velocity[0]:+.2f} / command {command[0]:+.2f} m/s | root above tile origin {root[2]-origin[2]:.2f} m | "
                     f"last reset: {','.join(last_reset_causes[selected]) or '-'}",
                     "N/P robot | 1..9/0 terrain | Q/Esc quit (focus this window) | blue=near red=far | random-start replay"]
            return depth_panel(raw, policy_depth, lines), raw, history.detach().cpu().numpy().copy(), details

        while app.is_running() and (args.steps == 0 or steps < args.steps):
            loop_start = time.monotonic()
            if not torch.isfinite(obs).all():
                all_finite = False
                raise RuntimeError("Non-finite policy observation during replay")
            panel, raw, history, details = capture()
            if steps % 50 == 0:
                cv2.imwrite(str(output / "depth_latest.png"), panel)
                np.savez_compressed(output / "depth_latest.npz", raw_depth_m=raw, policy_depth_history=history)
                samples.append(details)
                print("HIKING_PLAY_STATE=" + json.dumps(details), flush=True)
            if not args.headless:
                root = base.scene["robot"].data.root_pos_w[selected].detach().cpu().numpy()
                base.sim.set_camera_view(tuple(root + np.array([2.5, -3.0, 1.4])), tuple(root + np.array([0., 0., -.2])))
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("n"), ord("p")):
                    selected = (selected + (1 if key == ord("n") else -1)) % args.num_envs
                elif chr(key) in "1234567890":
                    index = "1234567890".index(chr(key))
                    if index < len(categories):
                        candidates = [i for i, name in enumerate(assignments) if name == categories[index]]
                        if candidates:
                            selected = next((i for i in candidates if i > selected), candidates[0])
            with torch.inference_mode():
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    all_finite = False
                    raise RuntimeError("Non-finite policy action during replay")
                obs, _, dones, _ = env.step(actions)
                model.reset(dones)
            steps += 1
            done_ids = dones.nonzero(as_tuple=True)[0].cpu().tolist()
            resets[done_ids] += 1
            if done_ids:
                flags = {name: base.termination_manager.get_term(name).cpu().numpy()
                         for name in termination_counts}
                for name, values in flags.items():
                    termination_counts[name] += int(values.sum())
                for index in done_ids:
                    last_reset_causes[index] = [name for name, values in flags.items() if values[index]]
                    last_reset_step[index] = steps
            if not args.headless and not args.no_realtime:
                time.sleep(max(0., base.step_dt - (time.monotonic() - loop_start)))
        panel, raw, history, details = capture()
        cv2.imwrite(str(output / "depth_latest.png"), panel)
        np.savez_compressed(output / "depth_latest.npz", raw_depth_m=raw, policy_depth_history=history)
        report = {
            "checkpoint": str(checkpoint), "checkpoint_iteration": checkpoint_iteration,
            "steps": steps, "num_envs": args.num_envs, "simulated_seconds_per_environment": steps * base.step_dt,
            "wall_seconds": time.monotonic() - started, "terrain_requested": args.terrain,
            "terrain_environment_assignments": assignments, "completed_episodes_per_env": resets.tolist(),
            "termination_counts": termination_counts, "all_observations_and_actions_finite": all_finite,
            "policy_depth_changed_during_replay": frames_vary, "loaded_normalizers": sorted(normalizer_groups),
            "model_class": type(model).__name__, "deterministic_policy": True,
            "observation_noise_enabled": cfg.observations.policy.enable_corruption,
            "evaluation_protocol": "Training-task random starts and automatic fall/time-limit resets; not a traversal benchmark",
            "capability_validated": False, "latest_state": details, "samples": samples,
        }
        (output / "state.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"HIKING_PLAY_COMPLETED={output}", flush=True)
    except KeyboardInterrupt:
        print("Hiking playback interrupted; training was not started.", flush=True)
    except BaseException:
        failed = True
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if cv2 is not None and not args.headless:
            cv2.destroyAllWindows()
        if env is not None:
            env.close()
        if not failed:
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
