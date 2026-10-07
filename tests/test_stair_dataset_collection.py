"""Teacher scene splits, failure accounting, and exact PD command recording."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from legged_lab.scripts.collect_stair_dataset import assign_splits, canonical_scenario, build_parser, collect
from legged_lab.scripts import mujoco_stair_position_teacher as teacher


class DatasetCollectionTest(unittest.TestCase):
    def test_split_is_by_physical_scene_and_independent_of_input_order(self):
        scenes = [dict(direction=direction, initial_lateral_offset=i*.001)
                  for direction in (-1, 1) for i in range(4)]
        first = assign_splits(scenes)
        reverse = assign_splits(list(reversed(scenes)))
        self.assertEqual({s["scenario_id"]: s["dataset_split"] for s in first},
                         {s["scenario_id"]: s["dataset_split"] for s in reverse})
        self.assertEqual(sum(s["dataset_split"] == "validation" for s in first), 2)
        self.assertEqual({s["direction"] for s in first if s["dataset_split"] == "validation"}, {-1, 1})

    def test_renaming_and_deterministic_seed_do_not_manufacture_holdout(self):
        a, b = dict(direction="up", seed=1, name="a"), dict(direction=1, seed=9, name="b")
        self.assertEqual(canonical_scenario(a)["scenario_id"], canonical_scenario(b)["scenario_id"])
        with self.assertRaisesRegex(ValueError, "Duplicate physical"):
            assign_splits([a, b])
        a["initial_velocity_noise"], b["initial_velocity_noise"] = .001, .001
        self.assertNotEqual(canonical_scenario(a)["scenario_id"], canonical_scenario(b)["scenario_id"])

    def test_batch_noise_forwards_provenance_without_changing_scene_splits_or_approving_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            scenes_path = Path(directory)/"scenes.json"
            scenes = [dict(direction="up", dataset_split="train"), dict(direction="down", dataset_split="validation")]
            scenes_path.write_text(json.dumps(scenes))
            manifests = []
            for seed in (2, 7):
                passed = []

                def run_episode(args, direction):
                    passed.append(args)
                    path = Path(args.output_dir)/"trace.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    observations = np.zeros((3, 1000), np.float32)
                    observations[:, 963] = 1.
                    np.savez(path, observations=observations, action_sources=np.full(3, "teacher_perturbed"))
                    return dict(success=direction > 0, failure="" if direction > 0 else "recovery_failed",
                                failure_stage="" if direction > 0 else "position_physics", trajectory=str(path))

                args = build_parser().parse_args(["--scenarios", str(scenes_path), "--position_config", "unused",
                    "--output_dir", str(Path(directory)/str(seed)), "--teacher_action_noise_std", ".01",
                    "--teacher_action_noise_seed", str(seed)])
                with patch.object(teacher, "run_episode", side_effect=run_episode), contextlib.redirect_stdout(io.StringIO()):
                    manifest = collect(args)
                self.assertTrue(all(a.teacher_action_noise_std == .01 and a.teacher_action_noise_seed == seed for a in passed))
                self.assertTrue(all(a.seed == 42 for a in passed))
                self.assertEqual(manifest["teacher_action_noise_seed"], seed)
                self.assertEqual(manifest["summary"]["complete_teacher_demonstrations"], 0)
                self.assertEqual(manifest["summary"]["complete_perturbed_teacher_demonstrations"], 1)
                self.assertEqual(len(manifest["summary"]["paths"]["train"]), 1)
                self.assertEqual(manifest["summary"]["paths"]["validation"], [])
                self.assertEqual([r["training_eligible"] for r in manifest["episodes"]], [True, False])
                self.assertTrue(all(r["data_kind"] == "teacher_perturbed" for r in manifest["episodes"]))
                manifests.append(manifest)
            self.assertEqual(manifests[0]["scenarios"], manifests[1]["scenarios"])

    def test_failed_initialization_is_retained_without_a_demonstration(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = teacher.build_parser()
            args = parser.parse_args(["--supervisor", "dynamic", "--headless", "--position_config", "unused", "--output_dir", directory])
            with patch.object(teacher, "DynamicTeachingEpisode", side_effect=RuntimeError("No valid tread lock")):
                report = teacher.run_episode(args, -1)
            saved = json.loads((Path(directory)/"down_000.json").read_text())
            self.assertFalse(saved["physical_success"])
            self.assertFalse(saved["independent_policy_success"])
            self.assertEqual(saved["failure_stage"], "initialization")
            self.assertNotIn("trajectory", report)
            self.assertEqual(list(Path(directory).glob("*.npz")), [])

    def position_runner_fixture(self):
        # Use the real 1000-feature observation builder and runner, with only
        # physics replaced by a deterministic clock for this boundary test.
        interface = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.0025),
                                                         actuator_ctrlrange=np.tile([-100., 100.], (29, 1))),
            dt=.02, physics_steps=8, clip_actions=10., clip_obs=100., history_length=10,
            qpos=np.arange(7, 36), dofs=np.arange(6, 35), motors=np.arange(29),
            joint_names=tuple(f"joint_{i}" for i in range(29)), default=np.zeros(29),
            kp=np.ones(29)*100., kd=np.ones(29)*5., scales=np.ones(29)*.25,
            obs_scales=dict(ang_vel=1., projected_gravity=1., commands=1., joint_pos=1., joint_vel=1., actions=1.),
            torque_from_action=lambda data, action: np.zeros(29))

        class Episode:
            def __init__(self, direction, **kwargs):
                self.direction, self.start_time, self.model = direction, 0., interface.model
                self.data = SimpleNamespace(time=0., qpos=np.zeros(36), qvel=np.zeros(35), xmat=np.eye(3).reshape(1, 9))
                self.teacher = SimpleNamespace(reader=SimpleNamespace(root_id=0), control=lambda: None)
                self._context = SimpleNamespace(teacher=self.teacher)

            def actor_features(self):
                result = np.zeros(39)
                result[3], result[31] = 1., self.direction
                return result

            def measurement(self):
                return SimpleNamespace(root_angular_velocity=np.zeros(3))

            def _advance(self, motor):
                motor(self, None)
                self.data.time += .0025
                return dict(motion_time=self.data.time, phase="LIFT_LEAD", success=self.data.time >= .07999), None

        return interface, Episode

    def test_saved_executed_actions_preserve_live_pd_float64_bits_but_labels_remain_float32(self):
        interface, Episode = self.position_runner_fixture()
        base = np.linspace(-.123456789123456, .234567891234567, 29, dtype=np.float64)
        # Probe the recorder boundary, including post-clip semantics, without
        # treating this synthetic out-of-range label as a valid demonstration.
        base[0] = interface.clip_actions+1.
        labels = [base+index*1.e-10 for index in range(4)]
        held = []
        interface.torque_from_action = lambda data, action: held.append(action.copy()) or np.zeros(29)
        with tempfile.TemporaryDirectory() as directory:
            args = teacher.build_parser().parse_args(["--supervisor", "dynamic", "--headless",
                "--position_config", "unused", "--output_dir", directory])
            with patch.object(teacher, "DynamicTeachingEpisode", Episode), \
                 patch.object(teacher, "MujocoPositionInterface", return_value=interface), \
                 patch.object(teacher, "DynamicPositionBridge") as bridge:
                bridge.return_value.action.side_effect = labels
                report = teacher.run_episode(args, 1)
            self.assertTrue(report["success"])
            with np.load(report["trajectory"], allow_pickle=False) as trace:
                actions = trace["applied_actions"]
                self.assertEqual(actions.dtype, np.dtype("float64"))
                self.assertEqual(trace["actions"].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(trace["actions"], np.stack(labels).astype(np.float32))
                executed = np.stack(held).reshape(4, 8, 29)
                for substep in range(8):
                    np.testing.assert_array_equal(actions.view(np.uint64), executed[:, substep].view(np.uint64))
                np.testing.assert_array_equal(actions[:, 0], interface.clip_actions)
                self.assertFalse(np.array_equal(actions, actions.astype(np.float32).astype(np.float64)))
                self.assertEqual(trace["action_sources"].tolist(), ["teacher"]*4)
                np.testing.assert_array_equal(trace["executed_physics_steps"], 8)
                metadata = json.loads(str(trace["metadata_json"]))
                self.assertEqual(metadata["applied_actions_dtype"], "float64")
                self.assertEqual(metadata["applied_actions_semantics"], "post_clip_position_actions_held_by_live_pd")

    def test_noisy_teacher_records_held_action_and_history_but_preserves_clean_state_labels(self):
        base = np.linspace(-.123456789123456, .234567891234567, 29)
        saved = []
        with tempfile.TemporaryDirectory() as directory:
            for attempt, (std, noise_seed) in enumerate(((0., 2), (0., 7), (.01, 2), (.01, 2), (.01, 7))):
                interface, Episode = self.position_runner_fixture()
                held, labels, queried_states = [], [], []
                episode_holder = []

                class StateEpisode(Episode):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        episode_holder.append(self)

                    def _advance(self, motor):
                        sample, result = super()._advance(motor)
                        self.data.qpos[7:] += held[-1]*.001
                        return sample, result

                def clean_action():
                    actual = episode_holder[0].data.qpos[7:].copy()
                    queried_states.append(actual)
                    label = base-actual*.1
                    labels.append(label.copy())
                    return label

                interface.torque_from_action = lambda data, action: held.append(action.copy()) or np.zeros(29)
                args = teacher.build_parser().parse_args(["--supervisor", "dynamic", "--headless",
                    "--position_config", "unused", "--output_dir", str(Path(directory)/str(attempt)),
                    "--teacher_action_noise_std", str(std), "--teacher_action_noise_seed", str(noise_seed)])
                with patch.object(teacher, "DynamicTeachingEpisode", StateEpisode), \
                     patch.object(teacher, "MujocoPositionInterface", return_value=interface), \
                     patch.object(teacher, "DynamicPositionBridge", return_value=SimpleNamespace(action=clean_action)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if std == 0.:
                        with patch.object(teacher.np.random, "Generator", side_effect=AssertionError("zero noise must not create RNG")), \
                             patch.object(teacher.np.random, "PCG64", side_effect=AssertionError("zero noise must not seed RNG")):
                            report = teacher.run_episode(args, 1)
                    else:
                        report = teacher.run_episode(args, 1)
                with np.load(report["trajectory"], allow_pickle=False) as trace:
                    data = {k: trace[k].copy() for k in trace.files}
                saved.append(data)
                np.testing.assert_array_equal(data["qpos"][:, 7:], queried_states)
                np.testing.assert_array_equal(data["actions"], np.stack(labels).astype(np.float32))
                actual = np.stack(held).reshape(4, 8, 29)
                for step in range(8):
                    np.testing.assert_array_equal(data["applied_actions"].view(np.uint64), actual[:, step].view(np.uint64))
                np.testing.assert_array_equal(data["observations"][1:, 931:960], data["applied_actions"][:-1].astype(np.float32))
                meta = json.loads(str(data["metadata_json"]))
                self.assertFalse(meta["learned_policy"])
                self.assertFalse(meta["independent_policy_success"])
                if std:
                    self.assertEqual(meta["data_kind"], "teacher_perturbed")
                    self.assertEqual(meta["teacher_action_noise_seed"], noise_seed)
                    self.assertEqual(meta["teacher_action_noise_std"], std)
                    self.assertEqual(meta["teacher_label_semantics"], "clean_teacher_action_at_actual_state")
                    self.assertTrue((data["action_sources"] == "teacher_perturbed").all())
                    self.assertFalse(np.array_equal(data["applied_actions"].astype(np.float32), data["actions"]))
                    expected = np.stack(labels)+np.random.Generator(np.random.PCG64(noise_seed)).normal(0., std, (4, 29))
                    np.testing.assert_array_equal(data["applied_actions"], expected)
                else:
                    self.assertEqual(meta["data_kind"], "teacher")
                    np.testing.assert_array_equal(data["applied_actions"], labels)
            for key in ("applied_actions", "actions", "observations", "qpos", "qvel"):
                np.testing.assert_array_equal(saved[0][key], saved[1][key])
                np.testing.assert_array_equal(saved[2][key], saved[3][key])
            self.assertFalse(np.array_equal(saved[2]["applied_actions"], saved[4]["applied_actions"]))
            self.assertFalse(np.array_equal(saved[2]["qpos"], saved[4]["qpos"]))

    def test_batch_preserves_explicit_splits_geometry_supervisor_and_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            scenes_path = Path(directory)/"scenes.json"
            scenes = [dict(direction="up", dataset_split="train"),
                      dict(direction="down", dataset_split="validation")]
            scenes_path.write_text(json.dumps(scenes))
            args = build_parser().parse_args(["--scenarios", str(scenes_path), "--geometry_source", "depth",
                                             "--output_dir", str(Path(directory)/"output")])
            passed = []

            def run_episode(episode_args, direction):
                passed.append((episode_args, direction))
                return dict(success=False, failure="deliberate boundary stop", failure_stage="initialization")

            with patch.object(teacher, "run_episode", side_effect=run_episode), contextlib.redirect_stdout(io.StringIO()):
                manifest = collect(args)
            self.assertEqual(len(passed), 2)
            for episode_args, _ in passed:
                self.assertEqual(episode_args.supervisor, "event")
                self.assertEqual(episode_args.duration, 40.)
                self.assertEqual(episode_args.geometry_source, "depth")
                self.assertFalse(hasattr(episode_args, "policy_checkpoint"))
            self.assertEqual([e["dataset_split"] for e in manifest["episodes"]], ["train", "validation"])
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["summary"]["attempts"], 2)
            self.assertEqual(manifest["summary"]["paths"], {"train": [], "validation": []})
            saved = json.loads((Path(directory)/"output"/"manifest.json").read_text())
            self.assertEqual(saved, manifest)
            with self.assertRaises(FileExistsError):
                collect(args)

    def test_collection_interrupt_keeps_manifest_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            scenes_path = Path(directory)/"scenes.json"
            scenes_path.write_text(json.dumps([dict(direction="up")]))
            args = build_parser().parse_args(["--scenarios", str(scenes_path), "--output_dir", str(Path(directory)/"output")])
            with patch.object(teacher, "run_episode", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                collect(args)
            manifest = json.loads((Path(directory)/"output"/"manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["summary"]["physical_successes"], 0)

    def test_teacher_cli_rejects_removed_policy_modes_and_invalid_noise(self):
        parser = teacher.build_parser()
        base = ["--output_dir", "unused"]
        for extra in (["--mode", "evaluate"], ["--policy_checkpoint", "student.pt"],
                      ["--student_fraction", ".5"], ["--teacher_handoff_time", "1.4"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(base+extra)
        for extra in (["--teacher_action_noise_std", "nan"], ["--teacher_action_noise_std", "inf"],
                      ["--teacher_action_noise_std", "-.01"], ["--teacher_action_noise_seed", "-1"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                teacher.validate_args(parser.parse_args(base+extra), parser)
        args = parser.parse_args(base+["--teacher_action_noise_std", ".01", "--teacher_action_noise_seed", "13"])
        teacher.validate_args(args, parser)
        self.assertEqual(teacher.teacher_noise_settings(args, 2), (.01, 15))
        self.assertEqual(teacher.initial_conditions(args, 2)["seed"], args.seed+2)


if __name__ == "__main__":
    unittest.main()
