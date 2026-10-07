# 可踩踏平面离线样例

5 帧原始 NPZ，约 15 MiB，摘自 2026-10-03 的 Isaac RTX 采集。一个上楼中段视角，加上四个按时间排序的下楼中段视角；字节、相机参数、位姿、时间戳和重置代次均按原始记录保留。来源和 SHA256 见 [manifest.json](manifest.json)。

每帧包含 `rgb`、`depth`、`intrinsic`、`camera_pos/quat`、`root_pos/quat`、`foot_pos/quat`、`timestamp_s`、`sequence_id`、`memory_generation`。`truth` 是独立审计所用的地形真值，不作为识别输入。

从仓库根目录执行：

```bash
python -m legged_lab.scripts.replay_stair_surfaces \
  --input_dirs examples/perception/stairs_15cm \
  --output_dir logs/perception_restore/demo \
  --surface_memory --save_images --normal_window_size 7 --normal_radius 4
```

预期 98 个候选、净距与高度违规均为 0。下楼四帧按时间依次积累，最后一帧下一阶有 29 个候选，左右各 8 个；这证明保存观测的融合效果，不代表机器人已能自主取得这些视角或安全落脚。逐帧结果见 [expected_summary.json](expected_summary.json)。
