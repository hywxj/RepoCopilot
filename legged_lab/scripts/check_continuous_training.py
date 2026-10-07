"""Run a bounded Isaac/PPO integration check, not a stair capability evaluation."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("elf3_continuous_flat", "elf3_continuous_up", "elf3_continuous_down"), required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--stair-height", type=float, default=None)
    parser.add_argument("--episode-seconds", type=float, default=.4,
                        help="Short episodes deliberately exercise reset transitions")
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.iterations < 1 or args.episode_seconds <= 0:
        parser.error("Use at least two environments, a positive iteration count and episode duration")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    app = AppLauncher(args).app
    try:
        import torch
        import legged_lab.envs  # noqa: F401 -- register tasks after Kit starts
        from legged_lab.utils.task_registry import task_registry
        from rsl_rl.runners import AmpOnPolicyRunner

        env_cfg, agent_cfg = (deepcopy(c) for c in task_registry.get_cfgs(args.task))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.max_episode_length_s = args.episode_seconds
        if args.stair_height is not None:
            if not 0 < args.stair_height < float("inf"):
                raise ValueError("--stair-height must be finite and positive")
            env_cfg.continuous = replace(env_cfg.continuous, step_height=args.stair_height)
            env_cfg.scene.terrain_generator.sub_terrains["continuous"].course = env_cfg.continuous
        agent_cfg.logger = "tensorboard"
        agent_cfg.save_interval = args.iterations + 1
        env_cfg.scene.seed = agent_cfg.seed
        env = task_registry.get_task_class(args.task)(env_cfg, headless=args.headless)
        runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(args.output_dir), device=agent_cfg.device)
        policy_before = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
        discriminator_before = [p.detach().clone() for p in runner.alg.discriminator.parameters()]
        report = {"task": args.task, "purpose": "optimizer_and_physics_integration_only", "num_envs": args.num_envs,
                  "iterations": args.iterations, "control_dt": env.step_dt, "episode_seconds": args.episode_seconds,
                  "environment_steps": 0, "terminal_transitions": 0, "terminal_rows_different_from_reset": 0,
                  "task_reward_min": float("inf"), "task_reward_max": float("-inf"),
                  "capability_validated": False, "latest_metrics": {}}
        report["course"] = vars(env.cfg.continuous)
        report["contact_history_length"] = env.contact_sensor.cfg.history_length
        report["actor_observation_dim"] = runner.alg.policy.actor[0].in_features
        report["critic_observation_dim"] = runner.alg.policy.critic[0].in_features
        report["action_joint_names"] = list(env.robot.joint_names)
        report["action_scales"] = env.action_scale.tolist()
        project = Path(__file__).resolve().parents[2]
        report["source_sha256"] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (project/"legged_lab/envs/elf3/continuous_cfg.py",
                                                project/"legged_lab/envs/elf3/continuous_env.py",
                                                project/"legged_lab/utils/continuous_course.py",
                                                project/"rsl_rl/rsl_rl/algorithms/amp_ppo.py",
                                                project/"rsl_rl/rsl_rl/runners/amp_on_policy_runner.py")}
        step = env.step

        def recorded_step(actions):
            observations, rewards, dones, infos = step(actions)
            for name, value in (("observations", observations), ("rewards", rewards),
                                ("critic", infos["observations"]["critic"])):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite {name} during physics rollout")
            ids = infos["terminal_amp_env_ids"]
            terminal = infos["terminal_amp_observations"]
            if len(ids):
                reset_states = env.get_amp_obs_for_expert_trans()[ids]
                report["terminal_rows_different_from_reset"] += int((terminal != reset_states).any(dim=1).sum().item())
            report["environment_steps"] += env.num_envs
            report["terminal_transitions"] += len(ids)
            report["task_reward_min"] = min(report["task_reward_min"], rewards.min().item())
            report["task_reward_max"] = max(report["task_reward_max"], rewards.max().item())
            for key, value in infos.get("log", {}).items():
                if "continuous" in key.lower() and isinstance(value, torch.Tensor) and value.numel() == 1:
                    report["latest_metrics"][key] = value.item()
            return observations, rewards, dones, infos

        env.step = recorded_step
        start = time.perf_counter()
        runner.learn(args.iterations, init_at_random_ep_len=False)
        report["training_wall_seconds"] = time.perf_counter() - start
        for label, before, parameters in (("actor", policy_before, list(runner.alg.policy.actor.parameters())),
                                          ("discriminator", discriminator_before, list(runner.alg.discriminator.parameters()))):
            report[label + "_max_parameter_change"] = max((a - b).abs().max().item() for a, b in zip(before, parameters))
            report[label + "_finite"] = all(bool(torch.isfinite(p).all()) for p in parameters)
        report["amp_normalizer_count"] = float(runner.alg.amp_normalizer.count)
        report["motion_manifest_sha256"] = runner.motion_prior_provenance["manifest_sha256"]
        report["passed"] = (report["terminal_transitions"] > 0 and report["terminal_rows_different_from_reset"] > 0
                            and report["actor_finite"] and report["discriminator_finite"]
                            and report["actor_max_parameter_change"] > 0 and report["discriminator_max_parameter_change"] > 0)
        (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("CONTINUOUS_TRAINING_VALIDATION " + json.dumps(report), flush=True)
        if not report["passed"]:
            raise RuntimeError("Continuous training integration check failed")
    finally:
        app.close(skip_cleanup=bool(args.headless))


if __name__ == "__main__":
    main()
