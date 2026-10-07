"""Validated, category-balanced AMP transitions from ELF3 GMR manifests.

This loader consumes the explicit 70-D AMP contract, never visualization frames.
It retains the source clips on CPU and samples adjacent rows without wrapping or
materializing a large transition bank. Conversion and simulation are separate.
"""

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


from legged_lab.motion.elf3_contract import AMP_JOINT_NAMES, JOINT_NAMES


MANIFEST_SCHEMA = "elf3_gmr_dataset_v1"
MOTION_SCHEMA = "elf3_gmr_motion_v1"
FEATURE_SCHEMA = "elf3_amp70_absolute_joint_pos_vel_root_local_end_effectors_v1"
SAMPLE_DT = .02


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_dt(value, name):
    _require(type(value) in (int, float) and np.isfinite(value) and value > 0.,
             f"{name} must be a positive finite number.")
    return float(value)


def _names(value, expected, field):
    values = np.asarray(value)
    _require(values.shape == (29,) and values.dtype.kind in ("U", "S"),
             f"{field} must contain 29 explicit string names.")
    _require(tuple(values.astype(str)) == expected, f"{field} differs from the ELF3 joint order.")


class GMRMotionLoader:
    """Sample category -> clip -> valid adjacent frame, uniformly at each level."""

    def __init__(self, device, time_between_frames, manifest_path, *, split="train", seed=None):
        _require(split in ("train", "validation"), "split must be train or validation.")
        self.device = torch.device(device)
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.generator = np.random.default_rng(seed)
        payload = self.manifest_path.read_bytes()
        manifest = json.loads(payload)
        _require(isinstance(manifest, dict) and manifest.get("schema_version") == MANIFEST_SCHEMA,
                 "Expected an elf3_gmr_dataset_v1 manifest.")
        sample_dt = _positive_dt(manifest.get("sample_dt"), "manifest sample_dt")
        transition_dt = _positive_dt(time_between_frames, "time_between_frames")
        _require(np.isclose(sample_dt, SAMPLE_DT, rtol=0., atol=1.e-12)
                 and np.isclose(transition_dt, sample_dt, rtol=0., atol=1.e-12),
                 "GMR data and policy transitions must use the same 20 ms sample_dt.")
        self.time_between_frames = sample_dt
        contract = manifest.get("contract")
        _require(isinstance(contract, dict) and contract.get("feature_schema") == FEATURE_SCHEMA,
                 f"Unsupported AMP feature_schema; expected {FEATURE_SCHEMA}.")
        _names(contract.get("joint_names"), tuple(JOINT_NAMES), "contract joint_names")
        _names(contract.get("amp_joint_names"), tuple(AMP_JOINT_NAMES), "contract amp_joint_names")
        clips = manifest.get("clips")
        _require(isinstance(clips, list) and clips, "Manifest must contain clips.")
        self.trajectories = []
        self.clip_ids = []
        self.clip_paths = []
        categories = defaultdict(list)
        seen_ids, seen_paths, seen_hashes, group_splits = set(), set(), set(), {}
        for row in clips:
            _require(isinstance(row, dict), "Each clip must be an object.")
            for key in ("id", "group", "category"):
                _require(isinstance(row.get(key), str) and bool(row[key].strip()), f"Clip needs a nonempty {key}.")
            _require(row["id"] not in seen_ids, "Duplicate clip id in manifest.")
            seen_ids.add(row["id"])
            _require(type(row.get("eligible")) is bool and row.get("split") in ("train", "validation"),
                     "Every clip needs explicit eligible and train/validation split fields.")
            previous_split = group_splits.setdefault(row["group"], row["split"])
            _require(previous_split == row["split"], "A motion group cannot occur in both train and validation.")
            if not row["eligible"] or row["split"] != split:
                continue
            output = row.get("output")
            _require(isinstance(output, str) and bool(output) and not Path(output).is_absolute(),
                     "Eligible clip output must be a relative NPZ path.")
            path = (self.manifest_path.parent/output).resolve()
            _require(path.is_relative_to(self.manifest_path.parent) and path.suffix == ".npz",
                     "Eligible clip output must remain inside the manifest directory and end in .npz.")
            expected_hash = row.get("output_sha256")
            _require(isinstance(expected_hash, str) and len(expected_hash) == 64
                     and all(c in "0123456789abcdef" for c in expected_hash), "Clip needs a lowercase output_sha256.")
            _require(path not in seen_paths and expected_hash not in seen_hashes,
                     "Duplicate clip output or content would bias sampling.")
            seen_paths.add(path)
            seen_hashes.add(expected_hash)
            clip_payload = path.read_bytes()
            _require(hashlib.sha256(clip_payload).hexdigest() == expected_hash,
                     f"Clip sha256 mismatch: {row['id']}.")
            with np.load(io.BytesIO(clip_payload), allow_pickle=False) as archive:
                required = {"schema_version", "sample_dt", "time", "amp_observations", "joint_names", "amp_joint_names", "feature_schema"}
                _require(required <= set(archive.files), f"Clip {row['id']} lacks required arrays.")
                schema = archive["schema_version"]
                _require(schema.shape == () and str(schema) == MOTION_SCHEMA, "Unsupported clip schema_version.")
                feature = archive["feature_schema"]
                _require(feature.shape == () and str(feature) == FEATURE_SCHEMA, "Clip feature_schema differs from AMP contract.")
                dt = archive["sample_dt"]
                _require(dt.shape == () and dt.dtype.kind in "fi" and np.isfinite(dt)
                         and np.isclose(float(dt), sample_dt, rtol=0., atol=1.e-9), "Clip sample_dt differs from manifest.")
                _names(archive["joint_names"], tuple(JOINT_NAMES), "joint_names")
                _names(archive["amp_joint_names"], tuple(AMP_JOINT_NAMES), "amp_joint_names")
                time = archive["time"]
                observations = archive["amp_observations"]
                _require(time.ndim == 1 and len(time) >= 2 and time.dtype.kind == "f" and np.isfinite(time).all(),
                         "Clip needs at least two finite time samples.")
                _require(np.allclose(time, np.arange(len(time))*sample_dt, rtol=0., atol=1.e-6),
                         "Clip time must start at zero and contain consecutive sample_dt intervals.")
                _require(observations.shape == (len(time), 70) and observations.dtype == np.float32
                         and np.isfinite(observations).all(), "AMP observations must be finite float32 [N,70].")
                frames = torch.from_numpy(observations.copy())
            index = len(self.trajectories)
            self.trajectories.append(frames)
            self.clip_ids.append(row["id"])
            self.clip_paths.append(str(path))
            categories[row["category"]].append(index)
        _require(self.trajectories, f"No eligible {split} clips in manifest.")
        self.categories = tuple(sorted(categories))
        self.category_clip_indices = tuple(np.asarray(categories[category], dtype=np.int64) for category in self.categories)
        self.provenance = dict(manifest=str(self.manifest_path), manifest_sha256=hashlib.sha256(payload).hexdigest(),
                               split=split, sample_dt=sample_dt, feature_schema=FEATURE_SCHEMA,
                               clips=list(self.clip_ids), categories=list(self.categories),
                               sampling="uniform_category_then_clip_then_adjacent_frame", transitions_preloaded=False)

    @property
    def observation_dim(self):
        return 70

    @property
    def num_motions(self):
        return len(self.trajectories)

    def sample_pairs(self, batch_size):
        _require(type(batch_size) is int and batch_size > 0, "batch_size must be a positive integer.")
        groups = self.generator.integers(len(self.categories), size=batch_size)
        clips = np.empty(batch_size, dtype=np.int64)
        for category, choices in enumerate(self.category_clip_indices):
            selected = groups == category
            clips[selected] = self.generator.choice(choices, size=int(selected.sum()))
        states = torch.empty((batch_size, 70), dtype=torch.float32)
        next_states = torch.empty_like(states)
        for clip_index in np.unique(clips):
            selected = np.flatnonzero(clips == clip_index)
            frames = self.trajectories[int(clip_index)]
            frame_indices = self.generator.integers(len(frames)-1, size=len(selected))
            states[selected] = frames[frame_indices]
            next_states[selected] = frames[frame_indices+1]
        return states.to(self.device), next_states.to(self.device)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        _require(type(num_mini_batch) is int and num_mini_batch > 0, "num_mini_batch must be a positive integer.")
        for _ in range(num_mini_batch):
            yield self.sample_pairs(mini_batch_size)

    def get_full_frame_batch(self, num_frames):
        return self.sample_pairs(num_frames)[0]
