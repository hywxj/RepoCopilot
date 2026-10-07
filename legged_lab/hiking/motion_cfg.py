"""ELF3 naming adapter around the official InstinctLab motion-reference loader."""

from copy import copy
from pathlib import Path

from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManager

from legged_lab.motion.elf3_contract import JOINT_NAMES


ROOT = Path(__file__).resolve().parents[2]
LINK_NAMES = (
    "waist_z_link", "torso_link", "l_shoulder_x_link", "r_shoulder_x_link",
    "l_elbow_y_link", "r_elbow_y_link", "l_wrist_z_link", "r_wrist_z_link",
    "l_hip_x_link", "r_hip_x_link", "l_knee_y_link", "r_knee_y_link", "l_ankle_x_link", "r_ankle_x_link",
)


def _other_side(name):
    return "r_" + name[2:] if name.startswith("l_") else "l_" + name[2:] if name.startswith("r_") else name


class Elf3MotionReferenceManager(MotionReferenceManager):
    """Resolve sagittal augmentation indices from the actual PhysX joint order."""

    def _initialize_impl(self):
        self.cfg = copy(self.cfg)
        super()._initialize_impl()
        names = list(self.isaac_joint_names)
        if len(names) != 29 or set(names) != set(JOINT_NAMES):
            raise ValueError("ELF3 motion reference requires exactly the named 29 ELF3 joints")
        self.cfg.symmetric_augmentation_joint_mapping = [names.index(_other_side(name)) for name in names]
        self.cfg.symmetric_augmentation_joint_reverse_buf = [1. if "_y_joint" in name else -1. for name in names]


def make_motion_reference_cfg(selection_path=None):
    """Use the official 50 Hz pose loader and its 10-frame reference manager."""
    selection = Path(selection_path) if selection_path is not None else ROOT / "logs/data_preparation/elf3_hiking_v1/selection_train.yaml"
    selection = selection.resolve()
    motion = AmassMotionCfg(
        path=str(selection.parent), filtered_motion_selection_filepath=str(selection),
        retargetting_func=None, motion_interpolate_func=None, motion_target_framerate=50.,
        velocity_estimation_method="frontward", motion_start_from_middle_range=[0., .9],
        motion_start_height_offset=0., ensure_link_below_zero_ground=False, buffer_device="output_device",
    )
    return MotionReferenceManagerCfg(
        class_type=Elf3MotionReferenceManager,
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        robot_model_path=str(ROOT / "legged_lab/assets/elf3_lite/urdf/elf3.urdf"),
        reference_prim_path=None,
        frame_interval_s=.02, update_period=.02, num_frames=10,
        motion_buffers={"elf3_gmr_train": motion}, link_of_interests=list(LINK_NAMES),
        symmetric_augmentation_link_mapping=[LINK_NAMES.index(_other_side(name)) for name in LINK_NAMES],
        # Resolved from PhysX names in Elf3MotionReferenceManager._initialize_impl.
        symmetric_augmentation_joint_mapping=None, symmetric_augmentation_joint_reverse_buf=None,
        mp_split_method="Even",
    )
