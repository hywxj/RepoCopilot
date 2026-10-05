# ELF3 楼梯几何感知与落脚控制：当前状态

更新时间：2026-10-03。本仓库当前是**研究中的仿真方案，不是可用于真机的安全下楼策略**。

当前地形已改为固定 32 cm 踏面、11–16 cm 高度课程。下面的 `v29`/`v39` 评测是此前约 22 cm 踏面的历史结果，不能用于说明新地形性能。逐级并步方案见 [stair_step_control_plan.md](stair_step_control_plan.md)。独立单级任务已接入新阶段和奖励，但真实初始化检查仍停在观察阶段，不能称为学会；见 [第二阶段状态](stair_step_phase2.md)。

新二维平面、完整脚掌候选与末端平台诊断已实现，新关键净距为上楼脚跟/下楼脚尖 2 cm。第一批实际保存的 58 帧修复后回放，净距违规从 17 个降到 0。新增短时观测地图、脚边点验证、稳定平面 ID、超时与重置清理。最终法向窗口 7 x 7 / 4 px 在 186 帧回放及额外 52 帧在线采集中均无候选净距或高度违规；受控补充视角下，11/15/16 cm 楼梯已得到左右同一级候选。自然盲走中仍常无下一阶目标；独立第二阶段任务已有观察参考、起脚门槛、逐级阶段与 Actor 接口，但尚未学会稳定观察及并步。见 [最新感知报告与实测图](stair_surface_refinement_report.md)和 [第二阶段实现](stair_step_phase2.md)；[第一批报告](stair_surface_validation_report.md)保留历史失败证据。

## 当前问题

仿真 D435i 深度图已经被转换为台阶近/远棱边、踏面宽度与高度。下楼时高置信踏面曾系统性偏前约 6 cm；针对当前仿真相机和 160 x 90 处理分辨率施加 -6.5 cm 边界标定后，平均中心绝对误差约 2 cm。该偏移并非由机器人运动导致的帧延迟：近乎静止时仍测得约 6.1 cm 偏差。真实相机必须单独标定。

更难的问题在落脚闭环。原盲走策略能前进，但在约 15 cm 高、22 cm 宽的下楼梯上经常越过相邻台阶、踩在边缘，或者以滑落方式到达末端。冻结盲走网络并叠加小幅几何残差，不足以可靠改变迈脚轨迹和支撑转换。更强的脚端 IK、预锁定目标、延长停顿、降低指令速度，分别减少了部分危险接触，却都未得到连续安全落脚。

## 验收口径与证据

固定种子 42、8 环境、900 控制步的下楼专项评测中，旧 `v29` 权重在未标定深度下有 5 次“走到末端”，但 **0 次严格成功**；物理台阶稳定承重事件 41 次，不安全首次触地 71 次。后续 `v39` 固定目标难度训练 900 轮，最后权重 `model_4096.pt` 同样为 **0 次严格成功**：5 次到达末端、物理承重事件 18 次、不安全首次触地 39 次。两者均不能称为学会下楼。完整赛道（上下楼梯、上下坡、鹅卵石）也未获可靠通过。

“严格成功”要求每一级台阶都有脚底居中、向上承重、低滑动速度并连续稳定 3 帧，最后还要在终点平台稳定承重且保持髋部净空。仅底座越过终点线会计入 `stair_distance_end_reached`，不会计入 `stair_goal_reached`。`play.py` 还打印 `stair_goal_physical_by_level` 和 `unsafe_tread_first_contacts`，用于定位漏级与踩边。

## 当前实现

- `legged_lab/perception/stair_geometry.py`：深度投影、踏面分割和仿真下楼边界标定。
- `legged_lab/perception/tread_surfaces.py`：可选二维平面、已观测覆盖、24 cm 完整脚掌候选和方向相关净距诊断，默认关闭，不控制旧 Actor。
- `legged_lab/perception/surface_memory.py`：最多 1 s / 25 帧的真实观测历史、位姿补偿、平面 ID、变化失效和重置；不是未知区补全。
- `legged_lab/perception/surface_audit.py`、`scripts/replay_stair_surfaces.py`：独立净距/高度核验、逐脚候选覆盖和离线回放，不使用地形真值选择策略目标。
- `legged_lab/scripts/inspect_stair_surfaces.py`：真实 RTX 相机帧与独立仿真地形核验、原始数据和图像采集。
- `legged_lab/perception/foothold_control.py`：逐脚目标选择、支撑确认、物理台阶覆盖审计。
- `legged_lab/envs/elf3/elf3_env.py`：感知历史、楼梯门控、脚端控制实验和严格终止判定。
- `legged_lab/envs/elf3/walk_terrain_teacher_cfg.py`：11–16 cm 台阶高度、固定 32 cm 踏面及奖励配置。
- `rsl_rl/rsl_rl/modules/actor_critic.py`：冻结盲走底座与受楼梯门控的可训练残差。
- `legged_lab/scripts/play.py`、`train.py`：可视化、同条件评测与训练入口。

新增的 `walk_elf3_geometry_stairs_down_full_control` 任务把楼梯分支动作权限从 0.18 提高到 0.80，非楼梯仍由原盲走策略控制。**这只是下一轮结构实验，尚未训练或验证，不能用作已解决的权重。**

## 复现和后续工作

环境是 `isaac_sim_env`。例如评测已训练的 `v39`（只用于仿真）：

```bash
conda activate isaac_sim_env
python legged_lab/scripts/play.py \
  --task=walk_elf3_geometry_stairs_down_bootstrap --headless \
  --terrain_mode=configured --terrain_difficulty=0.8 \
  --num_envs=8 --seed=42 --command_x=0.3 --max_steps=900 --skip_export \
  --enable_foothold_preview --leg_residual_multiplier=2.0 \
  --load_run=2026-10-02_01-53-00_elf3_signed_foothold_hardstairs_v39 \
  --checkpoint=model_4096.pt
```

训练输出和模型权重保留在本机 `logs/`，默认不纳入源代码上传。GMR 动作数据也是本机生成文件；新克隆若没有这些数据，配置会回退到仓库已有的 `walk.txt`，仅保证程序可以启动，不能据此声称复现上面的训练结果。没有模型权重时须重新训练；具体依赖及视觉原理见 [explicit_geometry_perception.md](explicit_geometry_perception.md)。

下一步使用独立单级任务定位观察与支撑条件，并训练停稳、卸载、落脚及承重转换；暂不恢复 22 cm 窄踏面。每一阶段都需多随机种子检查逐级双脚站稳、危险首次触地、滑移、非足部碰撞和非楼梯盲走退化。严格成功出现前，不能部署真机。
