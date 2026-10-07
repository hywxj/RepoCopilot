# TienKung-Lab

面向 bxi-elf3 的感知与运动控制研究。目标是**连续、安全、尽量像人类地上下楼**，允许自主调整步长、节奏及必要的并步，不要求每级停稳。

截至 2026-10-07，主线改为**直接迁移 Hiking 官方视觉训练链路到 ELF3**：深度历史、混合专家策略、PPO + AMP 和混合地形课程共同训练。本地 GMR 已完成数据转换；现有平面识别、安全区域和 MuJoCo 教师保留作辅助与对照。**迁移验证与学习能力分开记录，尚未证明连续安全上下楼。**

| 文档 | 内容 |
| --- | --- |
| [新主线](docs/hiking_mainline.md) | 设计、实施顺序、通行与自然性验收 |
| [感知](docs/perception.md) | 平面、安全区域、短时地图与复现 |
| [MuJoCo 教师](docs/mujoco_action_teacher.md) | 动态、深度与 20 ms 位置教师 |
| [数据与训练](docs/data_and_training.md) | 541 条 GMR、教师数据合同与工具 |
| [清理记录](docs/cleanup_20261007.md) | 保留资产与已移除实验 |

## 环境

本机使用 `conda activate isaac_sim_env`。2026-10-07 实测：Python 3.11.15、PyTorch 2.7.0/cu128、Isaac Sim 5.1.0.0、Isaac Lab 包元数据 0.54.4、MuJoCo 3.11.0，GPU 为 RTX 5060 Laptop 8 GB。旧安装说明中的版本不代表本轮运行环境；`setup.py` 仍保留旧依赖声明，尚未统一环境锁定文件。

MuJoCo 教师无需启动 Isaac Sim。训练从仓库根目录运行；Hiking 使用单独锁定的官方源码与可选依赖，入口不加载旧 rsl_rl。

## 常用入口

在仓库根目录和已配置的 Python 环境中运行：

```bash
# 准备锁定的 Hiking 依赖与本地动作格式（首次）
python -m legged_lab.scripts.setup_hiking
python -m legged_lab.scripts.prepare_hiking_motion

# 主线：深度感知与多地形运动共同训练
python -m legged_lab.scripts.train_hiking --headless --num_envs 256 --max_iterations 30000

# 查看单级物理教师；下楼改为 --direction down
python -m legged_lab.scripts.mujoco_stair_teacher --controller dynamic --direction up --loop

# 采集固定 20 ms 位置教师上下楼示范
python -m legged_lab.scripts.mujoco_stair_position_teacher --direction both --supervisor event --geometry_source known --headless --duration 40 --output_dir logs/teacher_runs/event_nominal
```

`--loop` 会重置后重复单级动作，不代表连续楼梯。感知回放、深度教师和数据转换入口见对应文档。旧 continuous 任务仅作真值诊断；新 Hiking 主线直接使用深度历史，楼梯课程为 5–18 cm。训练命令和验证范围见[数据与训练](docs/data_and_training.md)。
