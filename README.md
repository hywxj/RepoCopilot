# TienKung-Lab

面向 bxi-elf3 的感知与运动控制研究。目标是**连续、安全、尽量像人类地上下楼**，允许自主调整步长、节奏及必要的并步，不要求每级停稳。

截至 2026-10-07，主线确定为“现有视觉规划 + Hiking 运动学习方法 + ELF3 GMR 先验”。**新连续运动任务尚未实现，尚无通过验收的连续上下楼策略。** 保留平面识别、安全落脚区域、单级 MuJoCo 教师和基础训练工具。

| 文档 | 内容 |
| --- | --- |
| [新主线](docs/hiking_mainline.md) | 设计、实施顺序、通行与自然性验收 |
| [感知](docs/perception.md) | 平面、安全区域、短时地图与复现 |
| [MuJoCo 教师](docs/mujoco_action_teacher.md) | 动态、深度与 20 ms 位置教师 |
| [数据与训练](docs/data_and_training.md) | 541 条 GMR、教师数据合同与工具 |
| [清理记录](docs/cleanup_20261007.md) | 保留资产与已移除实验 |

## 环境

现有基础栈：Python 3.10、PyTorch 2.5.1/cu121、Isaac Sim 4.5.0、Isaac Lab 2.1.0。本机环境为 `conda activate isaac_sim_env`；新机器可从仓库根目录按以下顺序安装：

```bash
conda create -n tglab python=3.10
conda activate tglab
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip setuptools wheel
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
sudo apt install cmake build-essential
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd ..
pip install -e .
pip install -e rsl_rl
pip install qpsolvers quadprog scipy pyyaml opencv-python
```

MuJoCo 教师使用 `mujoco==3.3.2`（项目依赖已声明），不需要启动 Isaac Sim。官方 InstinctLab/instinct_rl 当前依赖与此环境不同，整套复现应使用独立环境和相容提交。

## 常用入口

在仓库根目录和已配置的 Python 环境中运行：

```bash
# 现有基础 AMP 任务，不是新连续楼梯训练入口
python legged_lab/scripts/train.py --task=walk_elf3 --headless --logger=tensorboard --num_envs=256

# 查看单级物理教师；下楼改为 --direction down
python -m legged_lab.scripts.mujoco_stair_teacher --controller dynamic --direction up --loop

# 采集固定 20 ms 位置教师上下楼示范
python -m legged_lab.scripts.mujoco_stair_position_teacher --direction both --supervisor event --geometry_source known --headless --duration 40 --output_dir logs/teacher_runs/event_nominal
```

`--loop` 会重置后重复单级动作，不代表连续楼梯。感知回放、深度教师和数据转换入口见对应文档。新主线没有可运行的训练命令；旧学生试验入口及中间训练产物已按清理记录移除。
