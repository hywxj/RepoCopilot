"""Read saved action amplitudes without importing or executing simulator configs."""

import math
from pathlib import Path

import yaml

from legged_lab.motion.elf3_contract import JOINT_NAMES


def load_checkpoint_action_scale(checkpoint):
    """Keep replay/resume targets consistent with the checkpoint's training run.

    Isaac snapshots contain Python YAML tags. BaseLoader reads them as plain
    data; it never imports classes or invokes their constructors.
    """
    path = Path(checkpoint).expanduser().resolve().parent / "params/env.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint action scales require its training snapshot: {path}")
    try:
        saved = yaml.load(path.read_text(), Loader=yaml.BaseLoader)["actions"]["joint_pos"]
        if saved["joint_names"] != list(JOINT_NAMES) or saved["preserve_order"] != "true":
            raise ValueError("joint order differs from the named ELF3 Hiking action order")
        if saved["use_default_offset"] != "true" or float(saved["offset"]) != 0.:
            raise ValueError("joint-position offset differs from ELF3 Hiking")
        if saved["clip"] != "null" or (int(saved["min_delay"]), int(saved["max_delay"])) != (0, 2):
            raise ValueError("action clipping or delay differs from ELF3 Hiking")
        scales = {name: float(value) for name, value in saved["scale"].items()}
        if set(scales) != set(JOINT_NAMES) or not all(math.isfinite(v) and v > 0 for v in scales.values()):
            raise ValueError("expected 29 finite, positive named action scales")
    except (KeyError, TypeError, AttributeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid checkpoint action configuration in {path}: {exc}") from exc
    return scales
