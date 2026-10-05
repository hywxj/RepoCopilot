"""Small RTX/physics integration check, not an evaluation of a trained stair skill."""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--direction", choices=("up", "down"), default="up")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--motor_only", action="store_true", help="Evaluate physical motor curriculum without cameras; this is NOT a depth-transfer acceptance test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require_successes", type=int, default=0,
                    help="Fail after writing the report if fewer complete physical episodes succeed. Zero runs diagnostics only; passing one run is not a reliability certification.")
parser.add_argument("--initial_yaw_deg", type=float, default=0., help="Initial yaw error for an alignment check.")
parser.add_argument("--disable_alignment", action="store_true", help="Keep OBSERVE commands at zero for comparison.")
parser.add_argument("--output_dir", default="logs/stair_step_phase2")
parser.add_argument("--checkpoint", default="logs/walk/2026-09-19_12-54-59_elf3_atec_blind_final_v18_1024env_3k/model_43997.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.motor_only
launcher = AppLauncher(args)
app = launcher.app

import cv2
import numpy as np
import torch

from legged_lab.envs.elf3.elf3_env import Elf3Env
from legged_lab.envs.elf3.stair_step_cfg import Elf3SingleStepUpEnvCfg, Elf3SingleStepDownEnvCfg, Elf3SingleStepAgentCfg
from legged_lab.perception.tread_surfaces import draw_surface_geometry_debug
from legged_lab.perception.stair_step_controller import StepPhase
from rsl_rl.modules import GatedResidualActorCritic


def check_motor(env, policy, obs, report, output):
    report.update({"perception_source": "simulation_known_treads", "depth_transfer_verified": False,
                   "completed_episodes": 0, "failed_episodes": 0, "phase_observations": 0, "num_envs": env.num_envs,
                   "shift_condition_failures": {}, "action_guidance_at_inference": False,
                   "motor_reset_x_range": [.04, .10]})
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy.act_inference(obs)
            closed = obs[:, -1] == 0
            if closed.any():
                delta = (actions-policy.actor.base_actor(obs[:, :960]))[closed].abs().max()
                report["off_stair_blind_max_error"] = max(report["off_stair_blind_max_error"], float(delta))
                report["off_stair_blind_checked_frames"] += int(closed.sum())
            obs, rewards, dones, extras = env.step(actions)
            teacher = env.step_skill_teacher
            report["completed_episodes"] += int(dones.sum())
            report["successes"] += int((env.step_success_event & dones).sum())
            report["failed_episodes"] += int((~env.step_success_event & dones).sum())
            report["finite_rewards"] &= bool(torch.isfinite(obs).all() and torch.isfinite(rewards).all())
            for phase in StepPhase:
                count = int((teacher.last_phases == int(phase)).sum())
                report["phase_counts"][phase.name] = report["phase_counts"].get(phase.name, 0)+count
                report["phase_observations"] += count
            shifting = ((teacher.phase == int(StepPhase.SHIFT_LEAD)) |
                        (teacher.phase == int(StepPhase.SHIFT_TRAIL))) & ~dones
            for name, satisfied in teacher.shift_requirements.items():
                counts = report["shift_condition_failures"]
                counts[name] = counts.get(name, 0)+int((shifting & ~satisfied).sum())
            if step % 80 == 0:
                record = {"step": step, "completed_episodes": report["completed_episodes"],
                          "successes": report["successes"], "phase_counts": dict(report["phase_counts"]),
                          "phase": teacher.phase[:2].cpu().tolist(),
                          "reset_occurred": dones[:2].cpu().tolist(),
                          "lead": teacher.lead[:2].cpu().tolist(),
                          "sole_positions": teacher.last_soles[:2].cpu().tolist(),
                          "sole_speed": teacher.last_sole_speed[:2].cpu().tolist(),
                          "load_fraction": teacher.last_load[:2].cpu().tolist(),
                          "reference_feet": teacher.reference_feet[:2].cpu().tolist(),
                          "reference_root": teacher.reference_root[:2].cpu().tolist(),
                          "root_position": env.robot.data.root_pos_w[:2].cpu().tolist(),
                          "com_position": (env.robot.data.root_pos_w+env._step_com_offset())[:2].cpu().tolist(),
                          "root_velocity": env.robot.data.root_lin_vel_w[:2].cpu().tolist(),
                          "shift_requirements": {name: satisfied[:2].cpu().tolist()
                                                 for name, satisfied in teacher.shift_requirements.items()}}
                report["frames"].append(record)
                print("[MOTOR] "+json.dumps(record), flush=True)
    episodes = report["completed_episodes"]
    report["completion_rate"] = report["successes"]/episodes if episodes else None
    report["required_successes"] = args.require_successes
    report["acceptance_passed"] = report["successes"] >= args.require_successes if args.require_successes else None
    (output/f"{args.direction}_motor_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("[RESULT] "+json.dumps({key: value for key, value in report.items() if key != "frames"}), flush=True)
    if not report["finite_rewards"] or report["off_stair_blind_max_error"] > 1.e-6:
        raise AssertionError("Nonfinite motor state or changed blind actor.")
    if report["acceptance_passed"] is False:
        raise AssertionError(f"Motor acceptance failed: {report['successes']} complete episodes; {args.require_successes} required. This was not a depth-transfer test.")


def main():
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.require_successes < 0:
        raise ValueError("require_successes must be nonnegative.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = (Elf3SingleStepUpEnvCfg if args.direction == "up" else Elf3SingleStepDownEnvCfg)()
    if args.num_envs <= 0 or (not args.motor_only and args.num_envs != 1):
        raise ValueError("Only --motor_only supports batch environments.")
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = args.seed
    if args.motor_only:
        cfg.scene.depth_camera.enable_depth_camera = False
        cfg.scene.depth_camera.geometry.step_skill_pretrain = True
        cfg.scene.terrain_generator.num_cols = args.num_envs
        cfg.domain_rand.events.reset_base.params["pose_range"]["x"] = (.04, .10)
    if not np.isfinite(args.initial_yaw_deg) or abs(args.initial_yaw_deg) > 30:
        raise ValueError("initial_yaw_deg must be finite and within +/-30 degrees.")
    yaw = float(np.deg2rad(args.initial_yaw_deg))
    cfg.domain_rand.events.reset_base.params["pose_range"]["yaw"] = (yaw, yaw)
    cfg.amp_motion_files_display = cfg.amp_motion_files_display[:1]
    env = Elf3Env(cfg, headless=args.headless)
    if args.disable_alignment:
        for controller in env.step_controllers:
            controller.cfg.alignment_yaw_gain = 0.
    obs, extras = env.get_observations()
    policy_cfg = Elf3SingleStepAgentCfg().policy.to_dict()
    policy_cfg.pop("class_name")
    policy = GatedResidualActorCritic(obs.shape[1], extras["observations"]["critic"].shape[1],
                                      env.num_actions, **policy_cfg).to(env.device)
    checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    state = checkpoint["model_state_dict"]
    has_stair_branch = any(key.startswith('actor.residual_actor.') for key in state)
    policy.load_state_dict(state)
    policy.eval()
    report = {"direction": args.direction, "trained_stair_policy": None if has_stair_branch else False,
              "actor_observation_dim": obs.shape[1], "phase_counts": {}, "failures": [],
              "successes": 0, "finite_rewards": True, "off_stair_blind_max_error": 0., "frames": []}
    report.update({"perception_source": "rtx_depth_surface_map", "depth_transfer_verified": False,
                   "action_guidance_at_inference": False, "completed_episodes": 0, "failed_episodes": 0})
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["checkpoint_iteration"] = checkpoint.get('iter')
    report["checkpoint_has_stair_branch"] = has_stair_branch
    report["actor_mode"] = 'independent_stair_actor' if policy.initialize_stair_actor_from_blind else 'residual'
    report["planned_target_frames"] = 0
    report["target_feature_action_max_delta"] = 0.0
    report["target_feature_ablation_scope"] = "Six target-error inputs only; same history, phase and gate. Not a success-rate comparison."
    report["observe_condition_failures"] = {}
    report["shift_condition_failures"] = {}
    report["replans"] = []
    report["actor_has_direct_load_inputs"] = False
    report["supervisor_uses_simulation_load_truth"] = True
    report["initial_yaw_deg"] = args.initial_yaw_deg
    report["alignment_enabled"] = not args.disable_alignment
    report["alignment_command_frames"] = 0
    report["alignment_blocked_frames"] = {}
    report["off_stair_blind_checked_frames"] = 0
    if args.motor_only:
        check_motor(env, policy, obs, report, output)
        return
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy.act_inference(obs)
            without_target_error = obs.clone()
            target_start = 960 + len(StepPhase)
            without_target_error[:, target_start:target_start + 6] = 0.0
            target_delta = float((actions - policy.act_inference(without_target_error)).abs().max())
            report["target_feature_action_max_delta"] = max(report["target_feature_action_max_delta"], target_delta)
            blind = policy.actor.base_actor(obs[:, :960])
            if obs[0, -1] == 0:
                report["off_stair_blind_checked_frames"] += 1
                error = float((actions-blind).abs().max())
                report["off_stair_blind_max_error"] = max(report["off_stair_blind_max_error"], error)
            obs, reward, dones, extras = env.step(actions)
            controller = env.step_controllers[0]
            m = controller.last_measurement
            geometry = env.terrain_geometry.surface_memory[0]
            if controller.replan_reason:
                report["replans"].append({"step": step, "reason": controller.replan_reason})
            report["alignment_command_frames"] += int(env.command_generator.command[0, 2] != 0)
            if controller.alignment_block_reason:
                reason = controller.alignment_block_reason
                report["alignment_blocked_frames"][reason] = report["alignment_blocked_frames"].get(reason, 0)+1
            proposed = None if m is None else controller.plan(geometry, m)
            report["planned_target_frames"] += int(proposed is not None)
            if controller.phase.name == "OBSERVE":
                for name, satisfied in controller.observe_requirements.items():
                    if not satisfied:
                        report["observe_condition_failures"][name] = report["observe_condition_failures"].get(name, 0)+1
            if controller.phase in (StepPhase.SHIFT_LEAD, StepPhase.SHIFT_TRAIL):
                for name, satisfied in controller.shift_requirements.items():
                    if not satisfied:
                        report["shift_condition_failures"][name] = report["shift_condition_failures"].get(name, 0)+1
            # A reset clears controller state, but event tensors remain the pre-reset outcome.
            if dones[0]:
                report["completed_episodes"] += 1
                report["successes"] += int(env.step_success_event[0])
                report["failed_episodes"] += int(not env.step_success_event[0])
                if not env.step_success_event[0] and not env.step_last_failures[0]:
                    report["failures"].append({"step": step, "reason": "physical_or_episode_termination"})
            phase = env.step_last_phases[0].name
            report["phase_counts"][phase] = report["phase_counts"].get(phase, 0)+1
            report["finite_rewards"] &= bool(torch.isfinite(reward).all() and torch.isfinite(obs).all())
            if env.step_last_failures[0]:
                report["failures"].append({"step": step, "reason": env.step_last_failures[0]})
            if step % 40 == 0:
                record = {"step": step, "phase": phase, "gate": float(env.step_gate[0]),
                          "reward": float(reward[0]), "target_locked": controller.lock is not None,
                          "geometric_plan_available": proposed is not None}
                record["target_feature_action_max_delta"] = target_delta
                record["observe_conditions"] = {name: bool(value) for name, value in controller.observe_requirements.items()}
                record["shift_conditions"] = {name: bool(value) for name, value in controller.shift_requirements.items()}
                record["support_block_reason"] = controller.support_block_reason
                record["alignment_block_reason"] = controller.alignment_block_reason
                record["command"] = env.command_generator.command[0].cpu().tolist()
                record["reference_feet"] = controller.reference_feet.tolist()
                record["reference_root"] = controller.reference_root.tolist()
                record["phase_elapsed_s"] = controller.elapsed
                record["lead_foot"] = controller.lead
                record["replan_reason"] = controller.replan_reason
                if controller.lock is not None:
                    record["locked_targets"] = controller.lock.targets.tolist()
                    record["locked_surface_id"] = int(controller.lock.surface_id)
                heading = None if m is None else controller.alignment_heading_error(m)
                record["heading_error_deg"] = None if heading is None else float(np.rad2deg(heading))
                if m is not None:
                    record["sole_positions"] = m.sole_positions.tolist()
                    record["vertical_load_fraction"] = (m.contact_forces[:, 2]/m.body_weight).tolist()
                    record["root_velocity"] = m.root_velocity.tolist()
                    record["root_angular_velocity"] = m.root_angular_velocity.tolist()
                report["frames"].append(record)
                print("[STEP] "+json.dumps(record), flush=True)
                feet_xy = feet_yaw = None
                if m is not None:
                    feet_xy = ((m.sole_positions-m.root_position) @ m.yaw_rotation)[:, :2]
                    rotations = np.einsum("ij,fjk->fik", m.yaw_rotation.T, m.foot_rotations)
                    feet_yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
                canvas = draw_surface_geometry_debug(geometry, env.depth_camera.data.output["rgb"][0],
                    env.depth_camera.data.output["distance_to_image_plane"][0], env.terrain_geometry.processed_image_shape,
                    feet_xy=feet_xy, feet_yaw=feet_yaw, camera_result=env.terrain_geometry.surface_geometry[0])
                canvas = cv2.copyMakeBorder(canvas, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                cv2.putText(canvas, f"CHECK: {phase}", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not cv2.imwrite(str(output/f"{args.direction}_{step:04d}.png"), canvas):
                    raise OSError("Cannot save diagnostic frame.")
    episodes = report["completed_episodes"]
    report["completion_rate"] = report["successes"]/episodes if episodes else None
    report["required_successes"] = args.require_successes
    report["acceptance_passed"] = report["successes"] >= args.require_successes if args.require_successes else None
    (output/f"{args.direction}_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("[RESULT] "+json.dumps(report), flush=True)
    if not report["finite_rewards"] or report["off_stair_blind_max_error"] > 1.e-6:
        raise AssertionError("Nonfinite state/reward or changed off-stair blind actor.")
    if report["acceptance_passed"] is False:
        raise AssertionError(f"Depth acceptance failed: {report['successes']} complete episodes; {args.require_successes} required.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Kit's immediate shutdown otherwise turns a failed check into exit 0.
        app._app.post_quit(1)
        app.close(skip_cleanup=True)
        raise
    app.close(skip_cleanup=True)
