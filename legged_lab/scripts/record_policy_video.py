import argparse
import asyncio
import os
import shutil
import tempfile

import imageio.v2 as imageio
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

import legged_lab.utils.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Record a policy rollout video.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video_path", type=str, default="videos/policy_rollout.mp4", help="Path to save the mp4 file")
parser.add_argument("--video_duration", type=float, default=12.0, help="Video duration in seconds")
parser.add_argument("--video_fps", type=int, default=10, help="Output video fps")
parser.add_argument("--warmup_steps", type=int, default=30, help="Steps before recording starts")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path
from omni.kit.viewport.utility import capture_viewport_to_file, frame_viewport_prims, get_active_viewport

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


async def capture_frame(viewport, frame_path):
    capture_helper = capture_viewport_to_file(viewport, file_path=frame_path)
    await capture_helper.wait_for_result()


def record_policy_video():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = max(env_cfg.scene.max_episode_length_s, args_cli.video_duration + 5.0)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (-0.5, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if hasattr(agent_cfg, "amp_num_preload_transitions"):
        agent_cfg.amp_num_preload_transitions = 1
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    print(f"[INFO] Saving video to: {os.path.abspath(args_cli.video_path)}")

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.get_observations()
    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport found. Run without --headless to record video.")
    frame_viewport_prims(viewport, ["/World/envs/env_0/Robot"])

    tmp_dir = tempfile.mkdtemp(prefix="g1_policy_video_")
    frame_paths = []
    output_frames = max(int(args_cli.video_duration * args_cli.video_fps), 1)
    steps_per_frame = max(int(round((1.0 / args_cli.video_fps) / env.step_dt)), 1)

    try:
        for frame_idx in range(output_frames):
            for _ in range(steps_per_frame):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:05d}.png")
            asyncio.get_event_loop().run_until_complete(capture_frame(viewport, frame_path))
            frame_paths.append(frame_path)
            print(f"[INFO] Captured frame {frame_idx + 1}/{output_frames}")

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video_path)), exist_ok=True)
        with imageio.get_writer(args_cli.video_path, fps=args_cli.video_fps, codec="libx264", quality=8) as writer:
            for frame_path in frame_paths:
                frame = imageio.imread(frame_path)
                writer.append_data(frame[:, :, :3])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    record_policy_video()
    simulation_app.close()
