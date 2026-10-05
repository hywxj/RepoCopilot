# 2026-10-05 旧图片清理

按用户要求，仅移除二维规划启用前的旧一维几何、旧 RGB/深度展示图片。以阶段和已确认目录筛选，不按图片修改时间批量删除。二维诊断第一批记录为 2026-10-02；本轮旧图来自 2026-09-23 至 2026-10-01 的一维展示。

共 24 张 PNG，8,406,040 字节，约 8.02 MiB，已使用 `gio trash` 移入系统回收站。可以从桌面回收站按原路径恢复；尚未永久释放磁盘空间。

| 原目录 | 数量 |
| --- | ---: |
| `docs/images/explicit_geometry/` | 8 |
| `docs/images/d435i_rgb_depth/` | 4 |
| `logs/trained_geometry_eval/` | 12 |

删除的文件（相对于仓库根目录）：

```text
docs/images/d435i_rgb_depth/depth_step000700.png
docs/images/d435i_rgb_depth/depth_step000900.png
docs/images/d435i_rgb_depth/rgb_step000700.png
docs/images/d435i_rgb_depth/rgb_step000900.png
docs/images/explicit_geometry/geometry_down8_feet_overlay.png
docs/images/explicit_geometry/geometry_env000_step000010.png
docs/images/explicit_geometry/geometry_env000_step000020.png
docs/images/explicit_geometry/geometry_env000_step000030.png
docs/images/explicit_geometry/geometry_env000_step000040.png
docs/images/explicit_geometry/geometry_env000_step000050.png
docs/images/explicit_geometry/geometry_env000_step000060.png
docs/images/explicit_geometry/trained_model_frame_020.png
logs/trained_geometry_eval/geometry_env000_step000010.png
logs/trained_geometry_eval/geometry_env000_step000020.png
logs/trained_geometry_eval/geometry_env000_step000030.png
logs/trained_geometry_eval/geometry_env000_step000040.png
logs/trained_geometry_eval/geometry_env000_step000050.png
logs/trained_geometry_eval/geometry_env000_step000060.png
logs/trained_geometry_eval/geometry_env000_step000070.png
logs/trained_geometry_eval/geometry_env000_step000080.png
logs/trained_geometry_eval/geometry_env000_step000090.png
logs/trained_geometry_eval/geometry_env000_step000100.png
logs/trained_geometry_eval/geometry_env000_step000110.png
logs/trained_geometry_eval/geometry_env000_step000120.png
```

二维感知、跨帧地图、第二阶段策略回放和 MuJoCo 动作教师图片共 1,032 张保留；模型权重、专家动作轨迹、原始 NPZ、报告、源码、历史打包文件和其他回收站内容均未删除。旧文档中的一维图片链接已改为指向保留的二维报告，不留下失效引用。
