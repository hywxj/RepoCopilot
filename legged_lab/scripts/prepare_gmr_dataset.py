"""Prepare a versioned, subject-split ELF3 motion prior from trusted GMR files."""

import argparse
import json
from pathlib import Path

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.motion.gmr_dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Trusted GMR clean PKL tree; filenames must match the clean TXT index")
    parser.add_argument("--source-xml", type=Path, required=True, help="ELF3 XML used for GMR retargeting")
    parser.add_argument("--index-root", type=Path, default=Path(LEGGED_LAB_ROOT_DIR) /
                        "legged_lab/envs/elf3/datasets/motion_amp_expert/elf3_loco_clean_gmr")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty output directory")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--max-joint-speed", type=float, default=16.0,
                        help="Kinematic review cutoff in rad/s, not actuator certification")
    args = parser.parse_args()
    if args.max_joint_speed <= 0:
        parser.error("--max-joint-speed must be positive")
    manifest = build_dataset(args.source_root, args.index_root, args.source_xml, args.output_dir,
                             validation_fraction=args.validation_fraction, seed=args.seed,
                             filters={"max_joint_speed_rad_s": args.max_joint_speed})
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
