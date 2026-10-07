"""Named AMP contracts, split isolation, and chronological balanced sampling."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rsl_rl.utils.gmr_motion_loader import (
    AMP_JOINT_NAMES, FEATURE_SCHEMA, GMRMotionLoader, JOINT_NAMES, MANIFEST_SCHEMA, MOTION_SCHEMA,
)


class GMRMotionLoaderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "manifest.json"
        self.manifest = dict(schema_version=MANIFEST_SCHEMA, sample_dt=.02,
                             contract=dict(feature_schema=FEATURE_SCHEMA, joint_names=list(JOINT_NAMES),
                                           amp_joint_names=list(AMP_JOINT_NAMES)), clips=[])

    def clip(self, clip_id, *, marker=1., count=8, category="walk", split="train", eligible=True, group=None):
        observations = np.zeros((count, 70), dtype=np.float32)
        observations[:, 0] = marker
        observations[:, 1] = np.arange(count)
        data = dict(schema_version=MOTION_SCHEMA, feature_schema=FEATURE_SCHEMA, sample_dt=.02,
                    time=np.arange(count)*.02, amp_observations=observations,
                    joint_names=np.asarray(JOINT_NAMES), amp_joint_names=np.asarray(AMP_JOINT_NAMES))
        path = self.root / f"{clip_id}.npz"
        np.savez(path, **data)
        row = dict(id=clip_id, group=group or clip_id, output=path.name,
                   output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   category=category, split=split, eligible=eligible)
        self.manifest["clips"].append(row)
        return row, data

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def loader(self, **kwargs):
        self.save()
        return GMRMotionLoader("cpu", .02, self.path, **kwargs)

    def rewrite(self, row, arrays):
        path = self.root / row["output"]
        np.savez(path, **arrays)
        row["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_pairs_are_adjacent_within_the_selected_clip_and_keep_all_seventy_features(self):
        self.clip("up", marker=10, count=2, category="stairs_up")
        self.clip("walk", marker=20, count=13)
        loader = self.loader(seed=7)
        pairs = list(loader.feed_forward_generator(3, 101))
        for states, next_states in pairs:
            self.assertEqual(states.shape, (101, 70))
            self.assertEqual(states.dtype, torch.float32)
            self.assertEqual(states.device.type, "cpu")
            torch.testing.assert_close(states[:, 0], next_states[:, 0], rtol=0., atol=0.)
            torch.testing.assert_close(next_states[:, 1]-states[:, 1], torch.ones(101), rtol=0., atol=0.)
            self.assertTrue(bool((next_states[states[:, 0] == 10, 1] == 1).all()))
            self.assertFalse(states.requires_grad)
        self.assertEqual(loader.get_full_frame_batch(1).shape, (1, 70))
        self.assertEqual(loader.observation_dim, 70)
        self.assertEqual(loader.num_motions, 2)
        self.assertFalse(hasattr(loader, "preloaded_s"))
        self.assertEqual(sum(len(frames) for frames in loader.trajectories), 15)

    def test_sampling_balances_categories_then_clips_without_duration_bias(self):
        self.clip("up", marker=1, count=2, category="stairs_up")
        self.clip("short_walk", marker=2, count=3)
        self.clip("long_walk", marker=3, count=401)
        states, _ = self.loader(seed=123).sample_pairs(20000)
        counts = [(states[:, 0] == marker).float().mean().item() for marker in (1, 2, 3)]
        for value, expected in zip(counts, (.5, .25, .25)):
            self.assertAlmostEqual(value, expected, delta=.02)

    def test_only_eligible_requested_partition_is_loaded_and_rng_is_reproducible(self):
        self.clip("train", marker=1)
        self.clip("validation", marker=2, split="validation")
        excluded, _ = self.clip("excluded", marker=3, eligible=False)
        (self.root / excluded["output"]).unlink()  # Invalid/ineligible output is not a training dependency.
        first, second = self.loader(seed=12), self.loader(seed=12)
        for a, b in zip(first.sample_pairs(77), second.sample_pairs(77)):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)
            self.assertTrue(bool((a[:, 0] == 1).all()))
        validation = self.loader(split="validation", seed=3)
        self.assertEqual(validation.clip_ids, ["validation"])
        self.assertTrue(bool((validation.get_full_frame_batch(10)[:, 0] == 2).all()))
        self.assertEqual(first.provenance["manifest_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_cross_split_groups_duplicate_ids_and_empty_training_are_rejected(self):
        self.clip("one", group="same")
        self.clip("two", marker=2, group="same", split="validation")
        with self.assertRaisesRegex(ValueError, "group"):
            self.loader()
        self.manifest["clips"][1]["group"] = "other"
        self.manifest["clips"][1]["id"] = "one"
        with self.assertRaisesRegex(ValueError, "Duplicate clip id"):
            self.loader()
        self.manifest["clips"] = [self.manifest["clips"][1]]
        with self.assertRaisesRegex(ValueError, "No eligible train"):
            self.loader()

    def test_hash_validation_and_manifest_contained_paths_are_required(self):
        row, _ = self.clip("one")
        original_hash = row["output_sha256"]
        row["output_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            self.loader()
        row["output_sha256"] = original_hash
        for output in (str(self.root / "one.npz"), "../one.npz", "one.txt"):
            row["output"] = output
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "output"):
                self.loader()
        row["output"] = "one.npz"
        self.manifest["clips"].append(dict(row, id="duplicate", group="duplicate"))
        with self.assertRaisesRegex(ValueError, "Duplicate clip output or content"):
            self.loader()

    def test_manifest_and_clip_joint_names_and_feature_schema_must_match(self):
        row, arrays = self.clip("one")
        for key in ("joint_names", "amp_joint_names", "feature_schema"):
            original = self.manifest["contract"][key]
            self.manifest["contract"][key] = "visualization" if key == "feature_schema" else original[::-1]
            with self.subTest(source="manifest", key=key), self.assertRaises(ValueError):
                self.loader()
            self.manifest["contract"][key] = original
            changed = dict(arrays)
            changed[key] = "visualization" if key == "feature_schema" else arrays[key][::-1]
            self.rewrite(row, changed)
            with self.subTest(source="npz", key=key), self.assertRaises(ValueError):
                self.loader()
            self.rewrite(row, arrays)

    def test_bad_schema_shape_dtype_nonfinite_times_and_missing_frames_are_rejected(self):
        row, original = self.clip("one")
        changes = [dict(schema_version="visualization_v1"),
                   dict(amp_observations=np.zeros((8, 69), np.float32)),
                   dict(amp_observations=np.zeros((8, 70), np.float64)),
                   dict(amp_observations=np.full((8, 70), np.nan, np.float32)),
                   dict(time=np.arange(8)*.03), dict(time=np.arange(8)*.02+1.),
                   dict(time=np.array([0.])), dict(time=np.full(8, np.inf)), dict(sample_dt=.01)]
        for change in changes:
            self.rewrite(row, dict(original, **change))
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.loader()
        self.rewrite(row, {key: value for key, value in original.items() if key != "feature_schema"})
        with self.assertRaisesRegex(ValueError, "required arrays"):
            self.loader()

    def test_sampling_period_and_invalid_batch_sizes_fail_before_sampling(self):
        self.clip("one")
        self.save()
        with self.assertRaisesRegex(ValueError, "sample_dt"):
            GMRMotionLoader("cpu", .01, self.path)
        loader = self.loader()
        for count in (0, -1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                loader.sample_pairs(count)
        with self.assertRaises(ValueError):
            list(loader.feed_forward_generator(0, 2))
        self.manifest["sample_dt"] = .033
        with self.assertRaisesRegex(ValueError, "sample_dt"):
            self.loader()

    def test_runner_requires_explicit_manifest_and_preserves_legacy_loader_arguments(self):
        from rsl_rl.runners.amp_on_policy_runner import _create_amp_loader

        env = SimpleNamespace(step_dt=.02)
        legacy = dict(amp_motion_files=["old.txt"], amp_num_preload_transitions=13, seed=42)
        with patch("rsl_rl.runners.amp_on_policy_runner.AMPLoader") as old:
            result = _create_amp_loader(env, legacy, "cpu")
            old.assert_called_once_with("cpu", time_between_frames=.02, preload_transitions=True,
                                        num_preload_transitions=13, motion_files=["old.txt"])
            self.assertIs(result, old.return_value)
        self.clip("train")
        self.clip("validation", marker=2, split="validation")
        self.save()
        with patch("rsl_rl.runners.amp_on_policy_runner.AMPLoader", side_effect=AssertionError("legacy loader used")):
            result = _create_amp_loader(env, dict(amp_motion_manifest=str(self.path), seed=42), "cpu")
        self.assertIsInstance(result, GMRMotionLoader)
        self.assertEqual(result.split, "train")
        self.assertEqual(result.clip_ids, ["train"])


    def test_runner_persists_manifest_identity_only_for_logging_rank_and_attaches_checkpoint(self):
        from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner

        self.clip("train")
        loader = self.loader(seed=7)
        runner = AmpOnPolicyRunner.__new__(AmpOnPolicyRunner)
        runner.motion_prior_provenance = loader.provenance
        runner.disable_logs = True
        runner.log_dir = str(self.root / "worker")
        runner._write_motion_prior_provenance()
        self.assertFalse(Path(runner.log_dir).exists())
        runner.disable_logs = False
        runner.log_dir = None
        runner._write_motion_prior_provenance()
        runner.log_dir = str(self.root / "legacy")
        runner.motion_prior_provenance = None
        runner._write_motion_prior_provenance()
        self.assertFalse(Path(runner.log_dir).exists())
        runner.motion_prior_provenance = loader.provenance
        runner.log_dir = str(self.root / "run")
        runner._write_motion_prior_provenance()
        saved = json.loads((Path(runner.log_dir)/"motion_prior.json").read_text())
        self.assertEqual(saved, loader.provenance)
        self.assertEqual(saved["sources"][0]["output_sha256"], self.manifest["clips"][0]["output_sha256"])
        policy = torch.nn.Linear(2, 1)
        runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()),
                                     discriminator=torch.nn.Linear(2, 1), amp_normalizer=None, rnd=None)
        runner.current_learning_iteration = 3
        runner.empirical_normalization = False
        runner.logger_type = "tensorboard"
        with patch("rsl_rl.runners.amp_on_policy_runner.torch.save") as save:
            runner.save("unused.pt")
        self.assertEqual(save.call_args.args[0]["motion_prior_provenance"], saved)
        runner.motion_prior_provenance = None
        with patch("rsl_rl.runners.amp_on_policy_runner.torch.save") as save:
            runner.save("legacy.pt")
        self.assertNotIn("motion_prior_provenance", save.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
