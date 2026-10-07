# 数据资产与训练入口

## Hiking 视觉主线

入口直接使用锁定的官方 InstinctLab 环境、深度编码器、4 专家 MoE 与 WasabiPPO/AMP；ELF3 身体和数据适配在 `legged_lab/hiking/`。深度和本体历史为 8 帧，AMP 状态为 10×67 维。同一策略同时训练全部 10 类地形，普通楼梯课程 5–18 cm、踏面 32 cm。

```bash
# 首次准备；已有转换产物时跳过第二条
python -m legged_lab.scripts.setup_hiking
python -m legged_lab.scripts.prepare_hiking_motion

# 验证视觉/策略/判别器均更新
python -m legged_lab.scripts.train_hiking --headless --num_envs 256 --max_iterations 20 --check

# 正式训练；可用 --resume /绝对路径/model_N.pt 接续
python -m legged_lab.scripts.train_hiking --headless --num_envs 256 --max_iterations 30000
```

日志为 `logs/elf3_hiking/`，记录上游提交、配置、源代码/动作来源哈希、TensorBoard 与检查点。每 100 更新保存，自动只保留当前运行最近 3 份检查点；不会清理其他运行。`--check` 保存深度观测与各模块参数变化，证明训练链路，不代表已学会通行。

终端实时监控（本机无需先激活 conda）：

```bash
/home/hamlet/miniconda3/envs/isaac_sim_env/bin/python /home/hamlet/TienKung-Lab/legged_lab/scripts/monitor_hiking.py
```

每 5 秒刷新，自动跟随 `active_run.json`；显示进程、最新检查点、任务奖励、回合秒数、课程和速度跟踪等指标，以及前后各 100 更新的均值。`--once` 仅查看一次，`--interval 10 --window 200` 调整刷新和统计窗口。Ctrl+C 只退出监控。课程等级与 AMP 奖励不能代替通行成功率和步态回放。

带深度回放由使用者自行启动；默认自动选择最近检查点，4 个楼梯环境。`N/P` 切换机器人，深度窗口中 `Q/Esc` 退出；下楼地形用 `--terrain stairs_down`。左图是原始深度，右图直接取策略使用的最新深度帧。

```bash
cd /home/hamlet/TienKung-Lab
/home/hamlet/miniconda3/envs/isaac_sim_env/bin/python -m legged_lab.scripts.play_hiking --terrain stairs_up
```

回放每秒记录选中机器人的原始动作、关节目标、实际角度/速度和执行器报告力矩，保存在回放日志及正常退出后的 `state.json`。图形回放使用较小物理缓冲区和 640×360 视口，策略深度仍为 8×18×32。随机初态回放不能当作固定起点的楼梯通过率测试。

`logs/data_preparation/elf3_hiking_v1/` 为 349 条训练片段、96,158 帧，约 13.6 MB。原姿态、WXYZ 和时间完全保留，仅按官方格式映射字段；官方 loader 按名称重排并以前向差分计算速度。106 条合格验证片段未混入。`selection_train.yaml` 使用类别逆片段数权重；导出工具 8 项测试通过。

2026-10-07 实测：最终稳定控制＋完整视野配置以 256 环境完成 20 更新、122,880 transitions，耗时约 51 秒（不含初始化），约 2,400–2,500 transitions/s。10 类地形均被实际分配，视觉编码器、门控、4 个专家和 AMP 判别器全部有限且更新；记录在 `logs/elf3_hiking/2026-10-07_16-16-28_stable_control_full_depth_check/validation.json`。这是集成验收，尚无多地形通过率结论。

适配修正：保留 ELF3 原隐式 PD，位置目标延迟 0–10 ms。直接换成显式 PD 的早期检查出现零动作踝速度饱和，相关权重不用于新控制。现有 D435i 俯角较大，深度保留完整视野再缩小为 18×32，避免 G1 裁剪丢掉前方地形。数据与延迟缓冲共 11 项测试通过。

2026-10-07 修复 AMP 换片历史：官方静默换片未清理参考历史，会把前后片段混入同一 10 帧窗口。本地适配只清理换片环境的 `amp_reference` 历史，不重置机器人回合或其他观测；新首帧先填满窗口，随后 9 步逐步替换。数据、延迟与换片共 19 项测试通过；16 环境真实物理边界检查通过；从第 800 更新接续 20 更新、122,880 transitions，实际触发换片，各模块参数有限且更新。验收分别在 `logs/elf3_hiking/amp_history_fix/physics_validation.json` 和 `logs/elf3_hiking/2026-10-07_17-00-01_amp_history_fix_check/validation.json`，不代表通行能力已提升。

2026-10-07 已按用户要求停止训练和回放：训练日志到第 4328 更新，最新完整检查点为 `logs/elf3_hiking/2026-10-07_17-04-30_official_mainline_amp_history_fix/model_4300.pt`。楼梯回放确认前进不足：选中机器人两个回合各观察 19 秒，净位移约 0.29/0.73 m，命令速度约 0.48/0.76 m/s，按超时重置。奖励和回合时长提升不能解释为已经学会走路。后续先查策略输出、关节目标与实际响应，不能只延长同一训练；`active_run.json` 和 `active_playback.json` 均标记停止。

## ELF3 GMR 运动资产

2026-10-07 盘点 `legged_lab/envs/elf3/datasets/motion_amp_expert/elf3_loco_clean_gmr/`：

| 子目录 | 文件数 | 帧数 |
| --- | ---: | ---: |
| `elf3_loco_large_gmr` | 346 | 59,937 |
| `elf3_kit_stairs_gmr` | 195 | 33,425 |
| 合计 | 541 | 93,362 |

旧 TXT 每帧均为 70 项且有限。原始 PKL 与这 541 条一一匹配，共 94,444 帧，无重复文件哈希。进一步结合根部方向确认楼梯为 82 条上楼、71 条前向下楼、34 条组合及 **8 条倒退**，包括此前漏计的 `downstairs_b01–03`。组合动作包含转身；一般运动也含转向。

同名可视化轨迹位于 `legged_lab/envs/elf3/datasets/motion_visualization/elf3_loco_clean_gmr/`。本机原始文件位于 `/home/hamlet/GMR/outputs/elf3_kit_stairs_gmr/`（195 条）和 `/home/hamlet/GMR/outputs/elf3_loco_large_gmr/`（414 条），后一集合与筛选后的 346 条不相同；这些绝对路径不是仓库自带下载数据。

两种 70 列 TXT 语义不同：旧 AMP 是关节角、关节速度和末端位置；可视化文件保留根位置/姿态、关节角及速度。原始 GMR 抽查含 `fps`、`root_pos`、`root_rot`、29 维 `dof_pos`、`local_body_pos` 和 `link_body_list`。迁移优先从保留根状态的原始数据建立明确合同，不能因维数相同混用。

源 PKL 的帧率为 29.6907–30 Hz；旧 TXT 写成 `0.033 s`，并在转换中每条少了 2 帧。新适配器按源时间以线性位置/关节插值、四元数 SLERP 重采样到 50 Hz，再计算速度；不会循环拼接、延长或平滑源动作。

旧 Isaac 基础任务已使用 GMR；被移除的 MuJoCo 学生 AMP 仅用少量控制器教师示范，其失败不能解释为人类先验已经验证无效。现已完成数据适配、manifest 读取和连续任务初版，并运行了短训练；完整上下楼能力尚未验收。

## 首版标准化数据

产物为 `logs/data_preparation/elf3_gmr_v1/manifest.json` 与压缩 NPZ，约 72 MiB。455 条、123,303 帧通过初筛：上楼 76、下楼 69、组合 32、转弯/绕圈 214、坡道 38、踏石 26。按人物分为 349 条训练和 106 条验证，两边覆盖全部保留类别；KIT 同人物在楼梯与普通地形中共用分组。

86 条暂不进入缓存：59 条根部基本驻地的步行/跑步机、9 条实测倒退（8 楼梯、1 踏石）、18 条速度尖峰。原始文件全部保留。准入要求平均局部前向速度 ≥0.1 m/s、倒退帧占比 ≤15%、源与目标最大关节速度 ≤16 rad/s、根倾角 ≤60°、硬限位超出 ≤0.0001 rad；这是可复查的运动学筛选规则，不是执行器或自然性验收。

```bash
python -m legged_lab.scripts.prepare_gmr_dataset --source-root /home/hamlet/GMR/outputs/elf3_loco_clean_gmr --source-xml /home/hamlet/GMR/assets/elf3/xml/elf3.xml --output-dir logs/data_preparation/elf3_gmr_v1
```

输出目录须为空。清单逐条保存源真实路径、输入/输出哈希、分组、筛选原因及峰值所在关节/时间；源模型与项目模型先做 FK 合同核对，每条源文件再抽查最多 5 帧保存的局部 FK。两模型的碰撞和执行器定义不等价。

NPZ 保存时间、根世界位置、**wxyz** 四元数、29 关节绝对角/速度、根世界/局部线速度、局部角速度和显式关节名。原输入四元数为 **xyzw**。兼容 AMP 的 70 维是“右臂、左臂、腰、右腿、左腿”的角与速度共 58 维，加根局部四末端位置 12 维；双手为肘部偏置，双脚为踝原点，不是足底落脚点。左右镜像已通过 FK 检查，首版未扩增镜像。

新 runner 通过 `amp_motion_manifest` 显式启用 [GMRMotionLoader](../rsl_rl/rsl_rl/utils/gmr_motion_loader.py)，默认仅取 train，按类别→片段→相邻帧采样，不跨片段或回绕，也不预装百万对过渡。所有输出均校验格式、关节名、时间和 SHA256，来源写入训练日志及 checkpoint。37 项测试、25 项子测试通过；真实数据两分区各抽取 1,000 对，均为片段内连续帧，记录在产物目录的 `validation.json`。实现见 [gmr_dataset.py](../legged_lab/motion/gmr_dataset.py)。

当前仍无真实接触标签或楼梯几何；GMR 高度仅为整段最低 body 原点的偏移，不能推断脚底支撑或原地板高度。脚滑、冲击和自然性需后续回放/动力学审计；驻地动作如未来只用于关节风格，须单独定义采样规则。

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

## 真值诊断与历史工具

连续任务与检查入口：

```bash
# 正式连续上楼：4级、11cm高、32cm踏面；下楼改 elf3_continuous_down
python legged_lab/scripts/train.py --task elf3_continuous_up --stair_height 0.11 --headless --logger tensorboard --num_envs 32 --max_iterations 20

# 平地任务为 elf3_continuous_flat；可选课程高度4/8cm，目标15–16cm
# 独立短集成检查，刻意用0.4s回合触发reset，不是上下楼能力评价
python -m legged_lab.scripts.check_continuous_training --task elf3_continuous_up --num_envs 16 --iterations 2 --headless --output-dir logs/continuous_check_new

# GMR 转换为可视化格式，不是新主线 AMP 适配器
python legged_lab/scripts/gmr_data_conversion.py --input_dir /home/hamlet/GMR/outputs/elf3_kit_stairs_gmr --output_dir logs/data_preparation/stairs_visualization --fps 30

# 查看训练曲线
tensorboard --port=6006 --logdir logs/elf3_continuous
```

`gmr_data_conversion.py` 的 `--fps` 用于输出帧率约定，不能替代逐源时间和状态重采样审核。旧布局见 [motion_loader.py](../rsl_rl/rsl_rl/utils/motion_loader.py)，转换实现见 [gmr_data_conversion.py](../legged_lab/scripts/gmr_data_conversion.py)。

基础权重在 `logs/retained/baselines/elf3_blind_v18/` 与 `elf3_standing/`，分别供盲走/感知采样及站稳对照。新策略结构不同，不能直接加载这些模型作为连续策略。新实验保留配置、数据清单、随机种子、完整物理评价及可追溯检查点，避免仅凭 reward 或末段接管汇报成功。

连续任务 actor 为 4×96 维本体历史 + 45 个局部高度和 45 个有效位（474 维），critic 增加根速度/足接触历史（494 维）。当前高度网格来自真值，尚未接入既有平面/安全区域感知；也不等同于精确的整脚区域输入。动作尺度按真实关节名排列，50 Hz 位置目标由 200 Hz 物理 PD 执行，接触历史覆盖每个控制周期的 4 个子步。

任务无固定步态周期、逐级停稳或教师接管。到终点要求双脚越过末边且有接触；`endpoint_completion` 只表示到终点，安全与自然性另看承重脚边缘余量、足底滑移、碰撞、力峰及视频。边缘与滑移提供训练代价，正常卸载不会触发旧教师式阶段门控。短检查及报告位于 `logs/continuous_smoke_20261007/`；新训练保存到 `logs/elf3_continuous/`，记录任务/风格奖励的实际贡献与动作数据来源。

本阶段验证：65 项测试、80 项子测试通过；11 cm 上/下楼各完成 16 环境×2 更新的物理集成检查。正式训练入口完成 11 cm 上楼 32 环境×20 更新（15,360 transitions），参数与统计有限，但终点完成仍为 0。最终 128 环境平地入口和退出流程也通过。汇总为 `logs/continuous_smoke_20261007/summary.json`；该路线已退为诊断，不再作为 Hiking 视觉主线的前置训练阶段。
