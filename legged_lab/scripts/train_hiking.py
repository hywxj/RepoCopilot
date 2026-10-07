"""Train ELF3 with the official depth-conditioned Hiking PPO + AMP stack."""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from legged_lab.hiking.bootstrap import ROOT, activate
from legged_lab.hiking.checkpoint_config import load_checkpoint_action_scale


def main():
    upstream = activate()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--resume", type=Path, help="Resume with the checkpoint's saved action scales; omit for new defaults.")
    parser.add_argument("--check", action="store_true", help="Verify depth/AMP and parameter updates during a short run.")
    parser.add_argument("--log_dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 4 or args.max_iterations < 1:
        parser.error("Use at least 4 environments and one training iteration")
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            parser.error(f"Resume checkpoint does not exist: {args.resume}")
    checkpoint_scale = load_checkpoint_action_scale(args.resume) if args.resume else None
    app = AppLauncher(args).app
    env = None
    failed = False
    try:
        import torch
        from isaaclab.utils.io import dump_yaml
        from instinctlab.envs import InstinctRlEnv
        from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg
        from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
        from legged_lab.hiking.elf3_env_cfg import Elf3HikingEnvCfg
        from legged_lab.hiking.runner import HikingRunner

        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = Elf3HikingEnvCfg()
        if checkpoint_scale is not None:
            cfg.actions.joint_pos.scale = checkpoint_scale
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        if args.device:
            cfg.sim.device = args.device
        agent = G1ParkourPPORunnerCfg()
        agent.seed = args.seed
        agent.device = cfg.sim.device
        agent.experiment_name = "elf3_hiking"
        agent.max_iterations = args.max_iterations
        agent.save_interval = 100
        log_dir = args.log_dir or ROOT / "logs/elf3_hiking" / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ("_" + args.run_name if args.run_name else "")
        )
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent)
        provenance = {
            "upstream": upstream, "python": platform.python_version(), "torch": torch.__version__,
            "seed": args.seed, "num_envs": args.num_envs, "iterations": args.max_iterations,
            "resume": str(args.resume) if args.resume else None,
            "action_scale_source": str(args.resume.expanduser().resolve().parent / "params/env.yaml")
                                   if args.resume else "training_defaults",
            "action_scale": cfg.actions.joint_pos.scale,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT / "legged_lab/hiking").glob("*.py")},
            "observation_source": "simulated noisy depth history and proprioception",
        }
        selection = Path(cfg.scene.motion_reference.motion_buffers["elf3_gmr_train"].filtered_motion_selection_filepath)
        provenance["motion_selection_sha256"] = hashlib.sha256(selection.read_bytes()).hexdigest()
        provenance["motion_provenance"] = json.loads(selection.with_name("provenance.json").read_text())
        provenance["source_sha256"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (log_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"HIKING_LOG_DIR={log_dir}", flush=True)
        print(f"HIKING_ACTION_SCALE_SOURCE={provenance['action_scale_source']}\n"
              f"HIKING_ACTION_SCALE={json.dumps(cfg.actions.joint_pos.scale)}", flush=True)
        env = InstinctRlVecEnvWrapper(InstinctRlEnv(cfg=cfg))
        runner = HikingRunner(env, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
        runner.add_git_repo_to_log(__file__)
        if args.resume:
            runner.load(str(args.resume.resolve()))
        obs, extras = env.get_observations()
        fmt = {group: {name: tuple(int(d) for d in shape) for name, shape in terms.items()}
               for group, terms in env.get_obs_format().items()}
        print(f"HIKING_OBSERVATIONS={fmt}", flush=True)
        if not torch.isfinite(obs).all():
            raise RuntimeError("Non-finite initial policy observation")
        if "depth_image" not in fmt["policy"]:
            raise RuntimeError("Depth image missing from the policy input")
        for name in ("amp_policy", "amp_reference"):
            if name not in extras["observations"] or not torch.isfinite(extras["observations"][name]).all():
                raise RuntimeError(f"Missing or non-finite {name} observations")
        model = runner.alg.actor_critic
        if type(model).__name__ != "EncoderMoEActorCritic" or model.num_moe_experts != 4:
            raise RuntimeError("Expected the official depth encoder and four-expert Hiking policy")
        modules = {"actor_depth_encoder": model.encoders, "actor_moe": model.actor,
                   "critic_depth_encoder": model.critic_encoders, "critic_moe": model.critic,
                   "discriminator": runner.alg.discriminator}
        modules["actor_gate"] = model.actor.gate
        modules.update({f"actor_expert_{i}": expert for i, expert in enumerate(model.actor.experts)})
        before = {key: {n: p.detach().cpu().clone() for n, p in module.named_parameters()}
                  for key, module in modules.items()} if args.check else {}
        depth_offset = 0
        for name, shape in fmt["policy"].items():
            if name == "depth_image":
                break
            depth_offset += math.prod(shape)
        depth = obs[:, depth_offset:].reshape(args.num_envs, *fmt["policy"]["depth_image"])
        if args.check:
            import numpy as np
            np.savez_compressed(log_dir / "initial_depth.npz", depth=depth.detach().cpu().numpy())
            if float(depth.std()) <= 1.e-6:
                raise RuntimeError("Depth observations are constant")
        # Match the official terrain generator's column allocation and audit
        # actual environment assignments, including rare 5% terrain families.
        import numpy as np
        terrain_cfg = cfg.scene.terrain.terrain_generator
        names = list(terrain_cfg.sub_terrains)
        probabilities = np.asarray([v.proportion for v in terrain_cfg.sub_terrains.values()])
        thresholds = np.cumsum(probabilities / probabilities.sum())
        columns = [int(np.where(i / terrain_cfg.num_cols + .001 < thresholds)[0][0])
                   for i in range(terrain_cfg.num_cols)]
        terrain_counts = {name: 0 for name in names}
        for col in env.unwrapped.scene.terrain.terrain_types.cpu().tolist():
            terrain_counts[names[columns[col]]] += 1
        start = time.monotonic()
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        elapsed = time.monotonic() - start
        deltas = {key: max(float((p.detach().cpu() - before[key][n]).abs().max())
                          for n, p in module.named_parameters())
                  for key, module in modules.items()} if args.check else {}
        report = {
            "training_completed": True, "iterations": args.max_iterations,
            "transitions": args.num_envs * agent.num_steps_per_env * args.max_iterations,
            "training_seconds": elapsed, "observation_format": fmt,
            "all_policy_parameters_finite": all(bool(torch.isfinite(p).all()) for p in runner.alg.actor_critic.parameters()),
            "parameter_max_deltas": deltas,
            "all_module_parameters_finite": {key: all(bool(torch.isfinite(p).all()) for p in module.parameters())
                                             for key, module in modules.items()},
            "depth_shape": list(depth.shape), "depth_std": float(depth.std()),
            "terrain_categories": list(cfg.scene.terrain.terrain_generator.sub_terrains),
            "terrain_environment_counts": terrain_counts,
            "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
            "capability_validated": False,
        }
        (log_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        if not all(report["all_module_parameters_finite"].values()) or (args.check and not all(v > 0 for v in deltas.values())):
            raise RuntimeError("Training did not produce a finite policy update")
        if runner.writer:
            runner.writer.close()
        print(f"HIKING_COMPLETED={log_dir}", flush=True)
    except BaseException:
        failed = True
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        if not failed:
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
