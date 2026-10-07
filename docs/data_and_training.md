# 数据资产与训练入口

## ELF3 GMR 运动资产

2026-10-07 盘点 `legged_lab/envs/elf3/datasets/motion_amp_expert/elf3_loco_clean_gmr/`：

| 子目录 | 文件数 | 帧数 |
| --- | ---: | ---: |
| `elf3_loco_large_gmr` | 346 | 59,937 |
| `elf3_kit_stairs_gmr` | 195 | 33,425 |
| 合计 | 541 | 93,362 |

文件每帧均为 70 项，数值有限，未发现帧数组完全相同的重复文件。楼梯文件名初分 82 条上楼、79 条下楼、34 条组合，其中 5 条明确为倒退下楼；分类尚未逐条质量标注。一般运动含转向，不能将所有文件等权作为前进楼梯先验。

同名可视化轨迹位于 `legged_lab/envs/elf3/datasets/motion_visualization/elf3_loco_clean_gmr/`。本机原始文件位于 `/home/hamlet/GMR/outputs/elf3_kit_stairs_gmr/`（195 条）和 `/home/hamlet/GMR/outputs/elf3_loco_large_gmr/`（414 条），后一集合与筛选后的 346 条不相同；这些绝对路径不是仓库自带下载数据。

两种 70 列 TXT 语义不同：旧 AMP 是关节角、关节速度和末端位置；可视化文件保留根位置/姿态、关节角及速度。原始 GMR 抽查含 `fps`、`root_pos`、`root_rot`、29 维 `dof_pos`、`local_body_pos` 和 `link_body_list`。迁移优先从保留根状态的原始数据建立明确合同，不能因维数相同混用。

现有 TXT 标记 `FrameDuration=0.033`，源数据约 30 Hz 但逐文件不同。新主线需按真实源时间重采样到 50 Hz，同步处理旋转、速度和历史；只改帧率标签不构成重采样。还需检查关节限位、脚滑、接触与姿态，并按动作/来源划分验证集，避免相邻帧泄漏。

旧 Isaac 基础任务已使用 GMR；被移除的 MuJoCo 学生 AMP 仅用少量控制器教师示范，因此其失败不能解释为这 541 条人类先验已经验证无效。新连续主线的 GMR 特征适配、筛选 manifest 和训练尚未实现。

## 教师数据合同

历史资产在 `logs/retained/teachers/`，优先用 `event_position_float64/` 的精确执行动作记录；来源与哈希见[清理记录](cleanup_20261007.md)，采集命令见[教师文档](mujoco_action_teacher.md)。

| 位置 NPZ 字段 | 语义 |
| --- | --- |
| `schema_version` / `metadata_json` | 格式及来源、物理结果、控制合同 |
| `state_time` | 动作施加前时间，完整周期 0.02 s |
| `observations` | 历史教师合同下的 1000 维观测 |
| `actions` / `applied_actions` | 干净位置标签 / 裁剪后实际动作，可能因扰动不同 |
| `qpos` / `qvel` | 动作前完整 MuJoCo 状态 |
| `executed_physics_steps` | 完整周期 8 个 2.5 ms 步，失败末周期可能不足 |

1000 维为 960 维本体历史（10×96）、39 维阶段/参考几何特征和 1 维方向门控。阶段仍由接触真值驱动；这是保留教师数据格式，不是新连续 actor 最终观测定义。

加载时核对关节顺序、默认角度、动作尺度、PD、力矩限制、周期、历史顺序与几何来源。新 `applied_actions` 为 float64，旧文件可能为 float32；精确回放时不能先降精度。成功示范与失败诊断分开，manifest 中验证帧不能计作训练帧。

## 现有基础工具

以下是保留工具，新主线连续楼梯任务尚无训练入口。

```bash
# Isaac 基础 AMP 训练；按显存调整并行数
python legged_lab/scripts/train.py --task=walk_elf3 --headless --logger=tensorboard --num_envs=256

# GMR 转换为可视化格式，不是新主线 AMP 适配器
python legged_lab/scripts/gmr_data_conversion.py --input_dir /home/hamlet/GMR/outputs/elf3_kit_stairs_gmr --output_dir logs/data_preparation/stairs_visualization --fps 30

# 查看训练曲线
tensorboard --port=6006 --logdir logs/walk
```

`gmr_data_conversion.py` 的 `--fps` 用于输出帧率约定，不能替代逐源时间和状态重采样审核。旧布局见 [motion_loader.py](../rsl_rl/rsl_rl/utils/motion_loader.py)，转换实现见 [gmr_data_conversion.py](../legged_lab/scripts/gmr_data_conversion.py)。

基础权重在 `logs/retained/baselines/elf3_blind_v18/` 与 `elf3_standing/`，分别供盲走/感知采样及站稳对照。它们不代表新楼梯能力，也不能直接加载为未实现的新观测策略。新实验保留配置、数据清单、随机种子、完整物理评价及可追溯检查点，避免仅凭 reward 或末段接管汇报成功。
