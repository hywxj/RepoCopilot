"""Train-only format conversion and the runtime ELF3 joint-name adapter."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from legged_lab.motion.elf3_contract import AMP_JOINT_NAMES, JOINT_NAMES, mirror_joint_positions
from legged_lab.scripts.prepare_hiking_motion import FEATURE_SCHEMA, export_hiking_motion


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    arrays = {
        "schema_version": np.array("elf3_gmr_motion_v1"), "feature_schema": np.array(FEATURE_SCHEMA),
        "joint_names": np.asarray(JOINT_NAMES), "amp_joint_names": np.asarray(AMP_JOINT_NAMES),
        "sample_dt": np.array(.02), "time": np.arange(12) * .02,
        "root_position_world": np.tile(np.array([.1, .2, 1.1], np.float32), (12, 1)),
        "root_quaternion_wxyz": np.tile(np.array([np.cos(.4), 0., np.sin(.4), 0.], np.float32), (12, 1)),
        "joint_positions": np.arange(12 * 29, dtype=np.float32).reshape(12, 29) / 1000.,
    }
    clips = []
    for index, category in enumerate(("stairs_up", "stairs_up", "terrain_slope")):
        values = {key: value.copy() for key, value in arrays.items()}
        values["joint_positions"] += index / 100.
        path = source / f"{index}.npz"
        np.savez_compressed(path, **values)
        clips.append(dict(id=str(index), group=f"train/{index}", category=category, eligible=True, split="train",
                          output=path.name, output_sha256=_hash(path)))
    # These intentionally absent files must never be opened or exported.
    clips += [dict(id="holdout", group="validation/0", category="stairs_up", eligible=True, split="validation",
                   output="validation_only.npz", output_sha256="a" * 64),
              dict(id="excluded", group="train/excluded", category="other", eligible=False, split="train")]
    manifest = dict(schema_version="elf3_gmr_dataset_v1", sample_dt=.02,
                    contract=dict(feature_schema=FEATURE_SCHEMA, quaternion_order="wxyz", root_frame="torso_link",
                                  world_axes="right_handed_z_up", joint_names=list(JOINT_NAMES),
                                  amp_joint_names=list(AMP_JOINT_NAMES)), clips=clips)
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, arrays


def test_official_export_preserves_poses_wxyz_time_and_subject_holdout(tmp_path):
    source, original, arrays = _source(tmp_path)
    output = tmp_path / "output"
    result = export_hiking_motion(source, output)
    assert result["source_manifest_sha256"] == _hash(source)
    assert result["summary"]["train_clips"] == 3
    assert result["summary"]["eligible_validation_clips_not_exported"] == 1
    assert result["validation_sources_not_exported"][0]["id"] == "holdout"
    assert {row["group"] for row in result["clips"]}.isdisjoint({"validation/0"})
    assert len(list((output / "clips").glob("*.npz"))) == 3
    assert not (output / "validation_only.npz").exists()
    selection = yaml.safe_load((output / "selection_train.yaml").read_text())
    assert selection["motion_weights"] == [.5, .5, 1.]
    assert result["selection_sha256"] == _hash(output / "selection_train.yaml")
    for index, (name, row) in enumerate(zip(selection["selected_files"], result["clips"])):
        assert name.endswith("_retargeted.npz")
        assert row["output_sha256"] == _hash(output / name)
        assert row["source_output_sha256"] == original["clips"][index]["output_sha256"]
        with np.load(output / name, allow_pickle=False) as data:
            assert set(data.files) == {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
            assert data["framerate"].item() == 50.
            assert tuple(data["joint_names"]) == JOINT_NAMES
            np.testing.assert_array_equal(data["base_pos_w"], arrays["root_position_world"])
            np.testing.assert_array_equal(data["base_quat_w"], arrays["root_quaternion_wxyz"])
            np.testing.assert_array_equal(data["joint_pos"], arrays["joint_positions"] + index / 100.)
            np.testing.assert_allclose(np.arange(len(data["joint_pos"])) / data["framerate"], arrays["time"])
    assert json.loads((output / "provenance.json").read_text()) == result
    with pytest.raises(ValueError, match="empty"):
        export_hiking_motion(source, output)


@pytest.mark.parametrize("change,error", [("hash", "sha256"), ("names", "joint_names"),
                                         ("time", "time"), ("dt", "sample_dt"), ("quaternion", "unit WXYZ")])
def test_invalid_training_clip_is_rejected_before_export(tmp_path, change, error):
    source, manifest, _ = _source(tmp_path)
    path = source.parent / "0.npz"
    with np.load(path, allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files}
    if change == "names":
        fields["joint_names"] = fields["joint_names"][::-1]
    elif change == "time":
        fields["time"][5] += .001
    elif change == "dt":
        fields["sample_dt"] = np.array(.01)
    elif change == "quaternion":
        fields["root_quaternion_wxyz"][0] *= 2
    else:
        fields["joint_positions"][0, 0] += .1
    np.savez_compressed(path, **fields)
    if change != "hash":
        manifest["clips"][0]["output_sha256"] = _hash(path)
        source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        export_hiking_motion(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_overlapping_subject_or_xyzw_manifest_is_rejected(tmp_path):
    source, manifest, _ = _source(tmp_path)
    manifest["clips"][3]["group"] = "train/0"
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        export_hiking_motion(source, tmp_path / "output")
    manifest["clips"][3]["group"] = "validation/0"
    manifest["contract"]["quaternion_order"] = "xyzw"
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quaternion_order"):
        export_hiking_motion(source, tmp_path / "output")


def test_reference_factory_resolves_mirror_in_runtime_order(monkeypatch):
    # Exercise our adapter without importing Isaac Sim; upstream runtime behavior
    # itself is deliberately outside the scope of this CPU test.
    class Config(SimpleNamespace):
        pass

    class OfficialManager:
        def _initialize_impl(self):
            self.isaac_joint_names = list(reversed(JOINT_NAMES))

    for module_name, members in {
        "instinctlab.motion_reference": {"MotionReferenceManagerCfg": Config},
        "instinctlab.motion_reference.motion_files.amass_motion_cfg": {"AmassMotionCfg": Config},
        "instinctlab.motion_reference.motion_reference_manager": {"MotionReferenceManager": OfficialManager},
    }.items():
        module = ModuleType(module_name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, module_name, module)
    path = Path(__file__).resolve().parents[1] / "legged_lab/hiking/motion_cfg.py"
    spec = importlib.util.spec_from_file_location("_hiking_motion_cfg_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.make_motion_reference_cfg()
    assert cfg.frame_interval_s == cfg.update_period == .02
    assert cfg.num_frames == 10
    motion = cfg.motion_buffers["elf3_gmr_train"]
    assert motion.retargetting_func is None and motion.motion_interpolate_func is None
    assert motion.velocity_estimation_method == "frontward"
    assert motion.filtered_motion_selection_filepath.endswith("selection_train.yaml")
    manager = module.Elf3MotionReferenceManager()
    manager.cfg = cfg
    manager._initialize_impl()
    assert cfg.symmetric_augmentation_joint_mapping is None  # No mutation of a shared config.
    q = np.arange(29) / 10.
    actual = q[::-1][manager.cfg.symmetric_augmentation_joint_mapping] * manager.cfg.symmetric_augmentation_joint_reverse_buf
    np.testing.assert_array_equal(actual, mirror_joint_positions(q)[::-1])
    for index, name in enumerate(cfg.link_of_interests):
        assert cfg.link_of_interests[cfg.symmetric_augmentation_link_mapping[index]] == module._other_side(name)
