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

