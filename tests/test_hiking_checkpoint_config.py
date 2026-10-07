"""Checkpoint action semantics must survive changes to training defaults."""

import pytest
import yaml

from legged_lab.hiking.checkpoint_config import load_checkpoint_action_scale
from legged_lab.motion.elf3_contract import JOINT_NAMES


def snapshot(tmp_path):
    scales = dict.fromkeys(JOINT_NAMES, .25)
    for side in ("l", "r"):
        scales[f"{side}_hip_y_joint"] = .18571428571428572
        scales[f"{side}_knee_y_joint"] = .23170731707317074
    action = dict(joint_names=list(JOINT_NAMES), preserve_order=True, use_default_offset=True,
                  offset=0., clip=None, min_delay=0, max_delay=2, scale=scales)
    path = tmp_path / "params/env.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump({"actions": {"joint_pos": action}})
                    + "unused: !!python/object/apply:builtins.slice [null, null, null]\n")
    return tmp_path / "model_4300.pt", path, action


def test_recovers_original_named_scales_from_tagged_isaac_snapshot(tmp_path):
    checkpoint, _, action = snapshot(tmp_path)
    restored = load_checkpoint_action_scale(checkpoint)
    assert restored == action["scale"]
    assert restored["l_hip_y_joint"] == restored["r_hip_y_joint"] == .18571428571428572
    assert restored["l_knee_y_joint"] == restored["r_knee_y_joint"] == .23170731707317074


def test_missing_snapshot_cannot_silently_use_current_defaults(tmp_path):
    with pytest.raises(FileNotFoundError, match="training snapshot"):
        load_checkpoint_action_scale(tmp_path / "model_4300.pt")


@pytest.mark.parametrize("change", ["order", "missing_joint", "nan", "negative", "offset", "delay"])
def test_incompatible_action_mapping_fails_before_simulator_start(tmp_path, change):
    checkpoint, path, action = snapshot(tmp_path)
    if change == "order":
        action["joint_names"].reverse()
    elif change == "missing_joint":
        del action["scale"]["l_knee_y_joint"]
    elif change in ("nan", "negative"):
        action["scale"]["l_knee_y_joint"] = "nan" if change == "nan" else -.4
    elif change == "offset":
        action["use_default_offset"] = False
    else:
        action["max_delay"] = 3
    path.write_text(yaml.safe_dump({"actions": {"joint_pos": action}}))
    with pytest.raises(ValueError, match="Invalid checkpoint action configuration"):
        load_checkpoint_action_scale(checkpoint)
