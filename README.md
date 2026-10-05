# TienKung-Lab
tglab for bxi-elf3

## Installation
TienKung-Lab is built with Cuda121，IsaacSim 4.5.0 and IsaacLab 2.1.0.

- Install Isaac Lab 

```bash
cd TienKung-Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.0

conda create -n tglab python=3.10
conda activate tglab

#cuda121
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
#pip install pillow==11.3.0 --force-reinstall

#isaacsim install
pip install --upgrade pip setuptools wheel
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

#isaacsim test
isaacsim isaacsim.exp.full.kit

#isaaclab install
sudo apt install cmake build-essential
./isaaclab.sh --install

#isaaclab test
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd TienKung-Lab
pip install -e .
```
- Install the rsl-rl library

```bash
cd TienKung-Lab/rsl_rl
pip install -e .
```


## Usage

### Train

Train the policy using AMP expert data from elf3/datasets/motion_amp_expert.

```bash
python legged_lab/scripts/train.py --task=walk_elf3 --headless --logger=tensorboard --num_envs=4096
```

Train the privileged ELF3 stair policy with a curriculum over 11-16 cm risers
and fixed 32 cm treads. The curriculum starts from terrain levels 0-1
and promotes or demotes each environment according to traversal performance.

```bash
python legged_lab/scripts/train.py --task=walk_elf3_stairs_curriculum --headless --logger=tensorboard --num_envs=1024
```

Inspect native 1280x720 D435i geometry extraction on 32 cm stairs before
training. The saved diagnostic combines depth, the forward height profile,
near/far edges, and the green safe landing interval.

```bash
python legged_lab/scripts/play.py --task=walk_elf3_geometry_debug --headless --num_envs=1 --terrain_mode=configured --terrain_type=stairs_up_32 --terrain_difficulty=0.8 --command_x=0.3 --max_steps=120 --skip_export --save_geometry_dir=logs/geometry_debug --save_depth_every=10 --save_depth_max=8 --load_run=2026-09-21_19-40-52_elf3_obstacle_course_teacher_v3_route_align_1024env_3k --checkpoint=model_52494.pt
```

The geometry-fusion actor receives D435i-derived stair geometry while only the
critic sees the simulator height scan. The ordered course has eight up steps,
eight down steps, an uphill ramp, a level platform, a downhill ramp and
pebbles. Stair risers vary from 11 to 16 cm and treads stay at 32 cm.
Training uses independent camera lanes; single-robot playback shows one lane.
The shared ascent/descent phase controller is now wired into separate
`walk_elf3_geometry_step_up` and `walk_elf3_geometry_step_down` single-tread tasks.
These are small-scale training infrastructure, not trained stair skills; see
[the second-stage implementation and limits](docs/stair_step_phase2.md).
Hardware currently provides joint position/velocity and IMU, not foot load.
Direct force inputs are excluded from the step actor, but the simulation phase
teacher still uses contact truth. See [the sensorless support limits and MuJoCo interface](docs/sensorless_support.md).
The optional 2-D plane and full-foot diagnostic now includes pose-compensated,
short-lived observed-surface memory. The final October 3 configuration found no
physical-margin or height violations in 186 replayed frames and 52 extra live
RTX frames. Controlled extra viewpoints produce both-foot targets on descent;
natural blind walking still often lacks the next target. Legacy tasks remain
unchanged; only the new single-tread tasks consume 2-D targets for control. See
[the refinement report and camera images](docs/stair_surface_refinement_report.md);
[the first-batch report](docs/stair_surface_validation_report.md) preserves the earlier failures.
Add `--validate_surfaces` to geometry playback, or `--surface_memory` for observed
history, keeping
the legacy policy observations unchanged. Diagnostic sampling is 640x360;
the legacy geometry sampling remains 160x90. Green dots are unverified candidate
sole centers, not evidence of stable contact or safe physical clearance.
The current checkpoint is not safe for stair descent or reliable on the entire course:
distance-to-end is not a success criterion until all eight treads have stable,
centered foot support. Use the checkpoint below for simulation diagnostics only.

```bash
/home/hamlet/miniconda3/envs/isaac_sim_env/bin/python legged_lab/scripts/play.py \
  --task=walk_elf3_geometry_stairs_down_bootstrap \
  --terrain_mode=configured --terrain_difficulty=0.8 --num_envs=1 \
  --command_x=0.3 --max_steps=0 --show_rgb --show_depth --show_geometry \
  --trace_footsteps --validate_surfaces \
  --skip_export --load_run=2026-09-30_13-59-17_elf3_correct_contacts_down15cm_v29 \
  --checkpoint=model_3197.pt
```

`--headless` is deliberately omitted for live viewing. Press `q` in a camera
window to exit. Geometry visualization marks swing, contact and confirmed
tread support separately. See [the perception and evaluation notes](docs/explicit_geometry_perception.md)
for current limitations and the terrain seam check.

Current stair-safety status and reproducible evaluation are summarized in
[docs/stair_safety_handoff.md](docs/stair_safety_handoff.md).

### Play

Run the trained policy.

```bash
python legged_lab/scripts/play.py --task=walk_elf3 --num_envs=200
```

### Sim2Sim(MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/run/[timestamp]/exported/policy.pt.
```bash
python legged_lab/scripts/amp_sim2sim_lite.py --policy logs/walk/2026-03-02_00-47-51/exported/policy.onnx
```

### TensorBoard

```bash
tensorboard --port=6006 --samples_per_plugin scalars=999999 --logdir logs/walk/
```


### Motion Retargeting

```bash
git clone https://github.com/MelodyAI/GMR.git
```

### gmr_to_visualization

```bash
python legged_lab/scripts/gmr_data_conversion.py --input_pkl legged_lab/envs/elf3/datasets/amp/walk_run.pkl --output_txt legged_lab/envs/elf3/datasets/motion_visualization/walk.txt
```

### visaul_to_amp_expert

```bash
python legged_lab/scripts/play_amp_animation.py --task=walk_elf3 --num_envs=1 --save_path legged_lab/envs/elf3/datasets/motion_amp_expert/walk.txt --fps 30.0
```
