# MuJoCo 单级动作教师

## 保留用途与结果

教师用于单级物理可行性、动作采集和诊断；新主线是[连续自然通行](hiking_mainline.md)。教师依赖仿真位姿、接触真值与已知边缘验收，尚未通过自然性、起点扰动或真机验收。

固定场景为 11 cm 高、32 cm 踏面，默认初始站位前移 5 cm。以下时间按锁定目标后的动作时间计算，不同控制器须分开解释。

| 教师 | 控制方式 | 历史名义上楼 / 下楼 |
| --- | --- | --- |
| Dynamic | 全身逆动力学与质心/足端参考，力矩执行 | 4.10 s / 4.44 s |
| Dynamic + depth | 渲染深度选区后执行动态控制 | 4.10 s / 4.44 s |
| Event + 20 ms position | 接触事件门控，位置目标由 PD 执行 | 8.16 s / 7.88 s |

这些是已有固定初态运行记录，不是泛化成功率。动态下楼源踝曾接近力矩与关节边界，前移量 45/55 mm 邻域试验曾失败。名义完成不能当作合格的人类风格专家。本轮清理后的验证范围另见[清理记录](cleanup_20261007.md)。

## 动态与深度教师

从仓库根目录运行，依赖见 [README](../README.md)。可视化上楼：

```bash
python -m legged_lab.scripts.mujoco_stair_teacher --controller dynamic --direction up --initial_forward_offset .05 --loop
```

下楼改 `--direction down`。循环每次明确重置，不是连续多级行走；无窗口时不使用 `--loop`。`--controller staged` 保留逐阶段诊断基线。

无窗口运行渲染深度下楼，图像来自仿真 D435i 参数，RGB 只供显示：

```bash
MUJOCO_GL=egl python -m legged_lab.scripts.mujoco_stair_teacher --controller dynamic --direction down --geometry_source depth --headless --duration 20 --output_dir logs/teacher_runs/dynamic_depth_down
```

深度目标来自平面提取与短时地图，不从真值点云选区；接触、位姿和边缘净空验收仍使用仿真信息。该链路证明的是“渲染深度→落脚区域→教师动作”，不是视觉神经策略或纯 IMU 部署。

## 20 ms 位置教师与批量采集

位置桥接在隔离预测状态中拟合目标，然后在真实轨迹按固定 20 ms 目标和原力矩限制执行。默认配置为 [elf3_stair_position.yaml](../legged_lab/configs/elf3_stair_position.yaml)。

```bash
python -m legged_lab.scripts.mujoco_stair_position_teacher --direction both --supervisor event --geometry_source known --headless --duration 40 --output_dir logs/teacher_runs/event_nominal

python -m legged_lab.scripts.collect_stair_dataset --scenarios legged_lab/configs/stair_dataset_scenarios.json --supervisor event --duration 40 --output_dir logs/teacher_runs/event_dataset
```

`--supervisor dynamic` 可选动态参考；`--geometry_source depth` 可启用深度目标，支持该组合不等于已通过其物理验收。场景清单声明待测起点，失败保留在报告中，不进入成功示范集合。入口只负责教师采集，不接受学生模型、PPO 或 DAgger 参数。

## 输出与验收

历史教师资产位于 `logs/retained/teachers/`，子目录来源见[清理记录](cleanup_20261007.md)。新采集写入命令指定的独立目录。

- 动态力矩教师成功时输出 JSON 与 `*_candidate.npz`。10 ms 一帧，含实际物理状态及 2.5 ms 子步力矩；12 维阶段 one-hot 不是完整策略观测，力矩也不是位置标签。
- 位置教师输出 JSON 和 `*_positions.npz`；失败轨迹为 `*_positions_failed.npz`。20 ms 状态/动作合同与来源写入元数据，详见[数据文档](data_and_training.md)。
- `force_oracle=True`、`learned_policy=False` 标明执行依赖；物理成功与 `motion_quality_validated` 分开记录，历史文件名中的 expert 不等于通过自然性认证。

教师以完整脚掌区域、越边净空、真实承重和末端稳定判定完成。Event 门控和动态教师阈值按各自协议保留；取消新主线逐级停稳要求，不改变教师历史结果。

动态轨迹可用 `python -m legged_lab.scripts.audit_mujoco_motion TRACE.npz --output audit.json` 重算姿态、腿线和限位余量，下楼加 `--direction down`。这只是运动学审计，原始载荷来自配套 JSON，不能把回放重算的接触当作原物理载荷。
