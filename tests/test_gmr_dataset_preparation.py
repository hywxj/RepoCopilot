"""Temporal and provenance tests; no source motion files or Isaac installation needed."""

import json
import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from legged_lab.motion.gmr_dataset import (
    DEFAULT_FILTERS, build_dataset, classify_motion, exclusion_reasons, resample_motion, subject_group, subject_splits,
)
from legged_lab.motion.elf3_contract import Elf3Kinematics, PROJECT_XML


def constant_motion(fps=29.97, frames=61):
    time = np.arange(frames) / fps
    return {"fps": fps, "root_pos": np.column_stack((time * .4, time * 0, time * 0 + 1)),
            "root_rot": Rotation.from_euler("z", time * .6).as_quat(),
            "dof_pos": np.repeat((time * .2)[:, None], 29, axis=1)}


def test_resampling_uses_source_seconds_and_derives_consistent_velocities():
    raw = constant_motion()
    out = resample_motion(raw)
    t = out["time"]
    np.testing.assert_allclose(np.diff(t), .02, atol=1e-14)
    assert t[-1] <= (len(raw["root_pos"]) - 1) / raw["fps"]
    assert (len(raw["root_pos"]) - 1) / raw["fps"] - t[-1] < .02
    np.testing.assert_allclose(out["root_position_world"][:, 0], t * .4, atol=1e-13)
    np.testing.assert_allclose(out["joint_positions"][:, 0], t * .2, atol=1e-13)
    np.testing.assert_allclose(out["joint_velocities"], .2, atol=1e-12)
    expected_world = np.tile([.4, 0, 0], (len(t), 1))
    np.testing.assert_allclose(out["root_linear_velocity_world"], expected_world, atol=1e-12)
    np.testing.assert_allclose(out["root_angular_velocity_local"], np.tile([0, 0, .6], (len(t), 1)), atol=1e-12)
    expected_local = Rotation.from_euler("z", t * .6).inv().apply(expected_world)
    np.testing.assert_allclose(out["root_linear_velocity_local"], expected_local, atol=1e-12)


def test_quaternion_sign_changes_do_not_create_spins_or_bad_interpolation():
    raw = constant_motion(fps=30, frames=61)
    expected = resample_motion(raw)
    raw["root_rot"][::2] *= -1
    actual = resample_motion(raw)
    np.testing.assert_allclose(actual["root_angular_velocity_local"], expected["root_angular_velocity_local"], atol=1e-12)
    rotations = Rotation.from_quat(actual["root_quaternion_wxyz"][:, [1, 2, 3, 0]])
    np.testing.assert_allclose(rotations.as_matrix(), Rotation.from_euler("z", actual["time"] * .6).as_matrix(), atol=1e-12)


def test_limited_joint_angles_are_not_unwrapped_or_clipped():
    raw = constant_motion(fps=25, frames=30)
    raw["dof_pos"][:, 0] = np.linspace(-2.8, 2.8, 30)
    out = resample_motion(raw)
    assert np.all(np.diff(out["joint_positions"][:, 0]) > 0)
    assert out["joint_positions"][0, 0] == -2.8
    assert out["joint_positions"][-1, 0] == 2.8


def test_piecewise_linear_resampling_does_not_invent_endpoint_speed_peaks():
    raw = constant_motion(fps=30, frames=4)
    raw["dof_pos"][:, 0] = [0., .5, .5, .5]
    out = resample_motion(raw)
    assert np.abs(out["joint_velocities"]).max() <= 15. + 1e-12


@pytest.mark.parametrize("field,value", [("fps", 0), ("fps", float("nan")), ("root_rot", np.zeros((61, 4))),
                                         ("dof_pos", np.zeros((60, 29))), ("root_pos", np.full((61, 3), np.nan))])
def test_invalid_time_pose_or_frame_alignment_is_rejected(field, value):
    raw = constant_motion()
    raw[field] = value
    with pytest.raises(ValueError):
        resample_motion(raw)


def test_stairs_and_terrain_share_subject_holdout_deterministically():
    stair = "elf3_kit_stairs_gmr/513/downstairs01_stageii.pkl"
    terrain = "elf3_loco_large_gmr/kit_terrain_mix/513/step_stones08_stageii.pkl"
    assert subject_group(stair) == subject_group(terrain) == "kit/513"
    groups = [f"kit/{i}" for i in range(10)] + [f"bmlrub/rub{i:03}" for i in range(10)]
    first = subject_splits(groups, seed=2)
    assert first == subject_splits(reversed(groups), seed=2)
    assert sum(s == "validation" for s in first.values()) == 4
    assert set(first.values()) == {"train", "validation"}


def test_subject_holdout_covers_categories_when_possible():
    groups = [f"kit/{i}" for i in range(5)]
    categories = {g: {"walk": 1} for g in groups}
    categories["kit/0"]["stairs"] = 3
    categories["kit/1"]["stairs"] = 3
    split = subject_splits(groups, categories_by_group=categories)
    for partition in ("train", "validation"):
        present = set().union(*(categories[g] for g in groups if split[g] == partition))
        assert present == {"walk", "stairs"}


@pytest.mark.parametrize("name,category", [("downstairs_b01_stageii.pkl", "backward"),
                                          ("downstairs_backwards02_stageii.pkl", "backward"),
                                          ("upstairs_downstairs03_stageii.pkl", "stairs_mixed"),
                                          ("0005_normal_walk1_stageii.pkl", "walk_stationary_root"),
                                          ("0000_treadmill_norm_stageii.pkl", "walk_treadmill"),
                                          ("jump01_stageii.pkl", "other")])
def test_categories_do_not_silently_mix_reverse_or_stationary_motion(name, category):
    assert classify_motion(name) == category


def test_resampling_cannot_hide_a_source_speed_spike():
    metrics = dict(source_duration_s=2., source_max_joint_speed_rad_s=17., target_max_joint_speed_rad_s=10.,
                   max_joint_limit_excess_rad=0., max_root_tilt_deg=20., mean_forward_speed_m_s=.3,
                   backward_frame_fraction=0., source_fk_error_m=1e-6)
    assert exclusion_reasons("stairs_up", metrics, DEFAULT_FILTERS) == ["joint_speed_spike"]


def test_trusted_raw_to_manifest_and_training_pairs_end_to_end(tmp_path):
    from rsl_rl.utils.gmr_motion_loader import GMRMotionLoader

    source, index, output = [tmp_path / name for name in ("source", "index", "output")]
    model = Elf3Kinematics()
    for subject in range(4):
        for direction in ("upstairs", "downstairs"):
            relative = f"elf3_kit_stairs_gmr/{subject}/{direction}01_stageii"
            raw = constant_motion()
            raw["dof_pos"][:] = 0
            raw["root_pos"][:, 0] *= 1 + subject / 10
            raw["root_pos"][:, 2] += (1 if direction == "upstairs" else -1) * np.arange(61) * .001
            fk = model.forward(raw["dof_pos"])
            raw.update(local_body_pos=fk["body_positions_local"], link_body_list=fk["body_names"])
            src, ref = source / (relative + ".pkl"), index / (relative + ".txt")
            src.parent.mkdir(parents=True, exist_ok=True)
            ref.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(pickle.dumps(raw))
            ref.write_text("Filename allowlist only; never used as motion content.")
    manifest = build_dataset(source, index, PROJECT_XML, output)
    assert manifest["summary"]["eligible_clips"] == 8
    assert manifest["summary"]["missing_split_categories"] == {"train": [], "validation": []}
    groups = manifest["summary"]["split_groups"]
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert json.loads((output / "manifest.json").read_text())["schema_version"] == "elf3_gmr_dataset_v1"
    for split in ("train", "validation"):
        loader = GMRMotionLoader("cpu", .02, output / "manifest.json", split=split, seed=5)
        assert loader.categories == ("stairs_down", "stairs_up")
        states, following = loader.sample_pairs(100)
        assert states.shape == following.shape == (100, 70)
        assert np.isfinite(states.numpy()).all()
    with pytest.raises(ValueError, match="empty"):
        build_dataset(source, index, PROJECT_XML, output)
