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

import argparse
import glob
import os
import time

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the txt file")
parser.add_argument("--motion_file", type=str, default=None, help="Motion visualization file to play or convert")
parser.add_argument("--motion_dir", type=str, default=None, help="Directory of motion visualization files to convert")
parser.add_argument("--save_dir", type=str, default=None, help="Output directory for batch converted AMP expert files")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
parser.add_argument("--fps", type=float, default=30.0, help="Target fps")
parser.add_argument("--play_loop", action="store_true", help="Loop the selected motion in the viewer")
parser.add_argument("--max_loops", type=int, default=0, help="Stop after this many playback loops; zero runs forever")
parser.add_argument("--playback_speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--root_z_offset", type=float, default=0.0, help="Extra root height offset for viewer playback")
parser.add_argument("--display_forward_speed", type=float, default=0.0, help="Viewer-only forward speed added to root x")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from rsl_rl.utils import AMPLoaderDisplay


def write_motion_file(save_path, frames, fps):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savetxt(save_path, frames, fmt='%f', delimiter=', ')

    with open(save_path, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(save_path, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')


def set_display_motion(env, motion_file):
    env.amp_loader_display = AMPLoaderDisplay(
        motion_files=[motion_file], device=env.device, time_between_frames=env.physics_dt
    )
    env.motion_len = int(env.amp_loader_display.trajectory_num_frames[0])


def collect_motion_frames(env, fps):
    all_frames = []
    for frame_cnt in range(max(int(env.motion_len) - 1, 1)):
        time = (frame_cnt % int(env.motion_len)) * (1.0 / fps)
        frame = env.visualize_motion(time)
        all_frames.append(frame.cpu().numpy().reshape(-1))
    return np.stack(all_frames, axis=0)


def play_motion_loop(env, fps, max_loops=0, playback_speed=1.0, root_z_offset=0.0, display_forward_speed=0.0):
    motion_len = max(int(env.motion_len), 1)
    frame_dt = 1.0 / fps
    sleep_dt = frame_dt / max(playback_speed, 1e-6)
    frame_cnt = 0
    print(
        f"[INFO] Playing AMP motion: frames={motion_len}, fps={fps}, speed={playback_speed}, "
        f"root_z_offset={root_z_offset}, display_forward_speed={display_forward_speed}"
    )
    while simulation_app.is_running():
        loop_idx = frame_cnt // motion_len
        if max_loops > 0 and loop_idx >= max_loops:
            break
        motion_time = (frame_cnt % motion_len) * frame_dt
        display_time = frame_cnt * frame_dt
        env.visualize_motion(
            motion_time,
            root_z_offset=root_z_offset,
            root_x_offset=display_forward_speed * display_time,
        )
        frame_cnt += 1
        time.sleep(sleep_dt)


def play_amp_animation():
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    motion_files = []
    if args_cli.motion_dir:
        motion_files = sorted(glob.glob(os.path.join(args_cli.motion_dir, "**", "*.txt"), recursive=True))
        if not motion_files:
            raise FileNotFoundError(f"No .txt motion files found under {args_cli.motion_dir}")
        if not args_cli.save_dir:
            raise ValueError("--save_dir is required when using --motion_dir")
        env_cfg.amp_motion_files_display = [motion_files[0]]
    elif args_cli.motion_file:
        motion_files = [args_cli.motion_file]
        env_cfg.amp_motion_files_display = [args_cli.motion_file]

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    if args_cli.motion_dir:
        converted = 0
        skipped = 0
        for idx, motion_file in enumerate(motion_files, start=1):
            rel_path = os.path.relpath(motion_file, args_cli.motion_dir)
            save_path = os.path.join(args_cli.save_dir, rel_path)
            if os.path.exists(save_path) and not args_cli.overwrite:
                skipped += 1
                print(f"Skipped {idx}/{len(motion_files)}: {save_path}")
                continue
            set_display_motion(env, motion_file)
            all_frames_np = collect_motion_frames(env, args_cli.fps)
            write_motion_file(save_path, all_frames_np, args_cli.fps)
            converted += 1
            print(f"Converted {idx}/{len(motion_files)}: {motion_file} -> {save_path}")
        print(f"Done. converted={converted}, skipped={skipped}, save_dir={args_cli.save_dir}")
    else:
        if args_cli.motion_file:
            set_display_motion(env, args_cli.motion_file)
        if args_cli.play_loop:
            play_motion_loop(env, args_cli.fps, args_cli.max_loops, args_cli.playback_speed, args_cli.root_z_offset, args_cli.display_forward_speed)
            return
        all_frames_np = collect_motion_frames(env, args_cli.fps)
        if args_cli.save_path:
            write_motion_file(args_cli.save_path, all_frames_np, args_cli.fps)
            print(f"✅ Successfully converted to {args_cli.save_path}")


if __name__ == "__main__":
    play_amp_animation()
    simulation_app.close()
