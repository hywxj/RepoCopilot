"""Verify real sensor clip changes do not leak into another AMP window."""

import argparse
import json
import traceback
from pathlib import Path

from legged_lab.hiking.bootstrap import ROOT, activate


def main():
    activate()
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    failed = False
    env = None
    try:
        import torch
        from instinctlab.envs import InstinctRlEnv
        from legged_lab.hiking.elf3_env_cfg import Elf3HikingEnvCfg
        from legged_lab.hiking.amp_history import dataset_exhausted_with_reference_history_reset

        cfg = Elf3HikingEnvCfg()
        cfg.scene.num_envs = 16
        cfg.seed = 42
        env = InstinctRlEnv(cfg=cfg)
        env.reset()
        with torch.inference_mode():
            for _ in range(12):
                env.step(torch.zeros(env.num_envs, 29, device=env.device))
            manager = env.observation_manager
            reference = env.scene["motion_reference"]
            ids = torch.tensor([1, 6], device=env.device)
            unaffected = torch.tensor([i for i in range(env.num_envs) if i not in (1, 6)], device=env.device)
            history = manager._group_obs_term_history_buffer
            old = {group: {name: buffer.buffer.clone() for name, buffer in terms.items()}
                   for group, terms in history.items()}
            old_pushes = {group: {name: buffer._num_pushes.clone() for name, buffer in terms.items()}
                          for group, terms in history.items()}
            robot_state = env.scene["robot"].data.root_state_w.clone()
            episode_lengths = env.episode_length_buf.clone()
            # Ensure a deterministic boundary on selected rows. This alters only
            # the test's validity flags, not any stored source motion or model.
            data = reference.data
            data.validity[reference.ALL_INDICES, reference.aiming_frame_idx] = True
            data.validity[ids, reference.aiming_frame_idx[ids]] = False
            dones = dataset_exhausted_with_reference_history_reset(env, reset_without_notice=True)
            assert not dones.any()
            torch.testing.assert_close(env.episode_length_buf, episode_lengths)
            torch.testing.assert_close(env.scene["robot"].data.root_state_w, robot_state)
            for group, terms in history.items():
                for name, buffer in terms.items():
                    if group == "amp_reference":
                        assert not buffer._num_pushes[ids].any()
                        torch.testing.assert_close(buffer.buffer[unaffected], old[group][name][unaffected])
                    else:
                        torch.testing.assert_close(buffer.buffer, old[group][name])
                        torch.testing.assert_close(buffer._num_pushes, old_pushes[group][name])
            # Exactly the end-of-step update used by Isaac: each group advances
            # once and only restarted reference rows receive first-frame padding.
            manager.compute(update_history=True)
            actual_frames = {name: [buffer.buffer[ids, -1].clone()]
                             for name, buffer in history["amp_reference"].items()}
            for group, terms in history.items():
                for name, buffer in terms.items():
                    if group == "amp_reference":
                        first = actual_frames[name][0]
                        torch.testing.assert_close(buffer.buffer[ids], first[:, None].expand_as(buffer.buffer[ids]))
                        torch.testing.assert_close(buffer.buffer[unaffected, :-1], old[group][name][unaffected, 1:])
                    else:
                        torch.testing.assert_close(buffer.buffer[:, :-1], old[group][name][:, 1:])
            changed = {name: bool((buffer.buffer[ids] - old["amp_reference"][name][ids]).abs().max() > 1.e-6)
                       for name, buffer in history["amp_reference"].items()}
            assert changed["joint_pos_rel"], "Selected test rows did not change motion state"
            # Keep the same newly assigned clip while filling nine distinct
            # reference times, even if a short sampled suffix would normally end.
            for _ in range(9):
                reference.update(env.step_dt, force_recompute=True)
                manager.compute(update_history=True)
                for name, buffer in history["amp_reference"].items():
                    actual_frames[name].append(buffer.buffer[ids, -1].clone())
            for name, buffer in history["amp_reference"].items():
                torch.testing.assert_close(buffer.buffer[ids], torch.stack(actual_frames[name], dim=1))
            frozen = {name: buffer._num_pushes.clone() for name, buffer in history["amp_reference"].items()}
            manager.compute(update_history=False)
            manager.compute(update_history=False)
            for name, buffer in history["amp_reference"].items():
                torch.testing.assert_close(buffer._num_pushes, frozen[name])
            report = {
                "passed": True, "num_envs": env.num_envs, "switched_env_ids": ids.cpu().tolist(),
                "reference_terms": list(history["amp_reference"]), "changed_terms": changed,
                "no_old_clip_history": True, "unaffected_histories_continuous": True,
                "robot_episode_not_reset": True, "observation_read_does_not_advance_history": True,
                "warmup": "First new frame pads the window; ten subsequent recorded frames stay within the new assignment.",
                "capability_validated": False,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("AMP_HISTORY_VALIDATION=" + json.dumps(report), flush=True)
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if not failed:
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
