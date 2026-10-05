import pickle
import glob
import os
import numpy as np
import torch
import argparse
from scipy.spatial.transform import Rotation 


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def axis_angle_from_quat(q):
    eps = torch.finfo(q.dtype).eps
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    sin_half_angle = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, q[..., :1])
    scale = torch.where(sin_half_angle > eps, angle / sin_half_angle, 2.0 * torch.ones_like(sin_half_angle))
    return xyz * scale


def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str)
    parser.add_argument("--output_txt", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            raise ValueError("--output_dir is required when using --input_dir")
        input_pkls = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pkl"), recursive=True))
        print(f"Found {len(input_pkls)} pkl files under {args.input_dir}")
        converted = 0
        skipped = 0
        for idx, input_pkl in enumerate(input_pkls, start=1):
            rel_path = os.path.relpath(input_pkl, args.input_dir)
            output_txt = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + ".txt")
            if os.path.exists(output_txt) and not args.overwrite:
                skipped += 1
                continue
            convert_pkl_to_custom(input_pkl, output_txt, args.fps)
            converted += 1
            print(f"Processed {idx}/{len(input_pkls)}: {output_txt}")
        print(f"Done. converted={converted}, skipped={skipped}, output_dir={args.output_dir}")
    else:
        if not args.input_pkl or not args.output_txt:
            raise ValueError("--input_pkl and --output_txt are required for single-file conversion")
        convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)
