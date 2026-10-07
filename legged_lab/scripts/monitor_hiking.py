"""Read-only terminal dashboard; does not import Isaac Sim or touch training."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RUN = ROOT / "logs/elf3_hiking/active_run.json"
METRICS = (
    ("任务奖励 ↑", "Train/mean_reward_0", 1.0),
    ("平均回合 / 秒 ↑", "Train/mean_episode_length", None),
    ("地形课程等级 ↑", "Episode/Curriculum/terrain_levels", 1.0),
    ("平移速度误差 ↓", "Episode/Metrics/base_velocity/error_vel_xy", 1.0),
    ("转向速度误差 ↓", "Episode/Metrics/base_velocity/error_vel_yaw", 1.0),
    ("速度跟踪每步奖励 ↑", "Episode_Reward/rewards_track_lin_vel_xy_exp/timestep", 1.0),
    ("AMP 动作奖励 ↑", "Step/discriminator_reward", 1.0),
    ("AMP 判别器损失", "Loss/discriminator_loss", 1.0),
    ("价值函数损失", "Loss/value_loss", 1.0),
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def number(value) -> str:
    if value is None:
        return "--"
    if not math.isfinite(value):
        return str(value).upper()
    return f"{value:.4f}"


def pad(text: str, width: int) -> str:
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - used)


def process_status(pid) -> str:
    if not pid:
        return "未记录 PID"
    try:
        process = Path("/proc") / str(int(pid))
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
        if state == "Z":
            return f"PID {pid} 已退出（僵尸进程）"
        if b"train_hiking" not in command:
            return f"PID {pid} 已被其他程序占用"
        return f"PID {pid} 训练进程存活"
    except (OSError, ValueError, IndexError):
        return f"PID {pid} 未运行或不可读取"


def step_duration(run_dir: Path) -> tuple[float, str]:
    """Read two plain YAML numbers without importing YAML or simulation packages."""
    try:
        lines = (run_dir / "params/env.yaml").read_text().splitlines()
        dt = decimation = None
        in_sim = False
        for line in lines:
            if line and not line[0].isspace() and not line.startswith("#"):
                in_sim = line == "sim:"
            if in_sim and re.match(r"^  dt:\s", line):
                dt = float(line.split(":", 1)[1].split("#", 1)[0])
            if line.startswith("decimation:"):
                decimation = int(line.split(":", 1)[1].split("#", 1)[0])
        if dt and decimation and dt > 0 and decimation > 0:
            return dt * decimation, f"配置 dt={dt:g} × decimation={decimation}"
    except (OSError, ValueError):
        pass
    return 0.02, "未读到配置，暂按默认 0.02 秒/步"


class ScalarHistory:
    def __init__(self, run_dir: Path, window: int, accumulator_class):
        self.window = window
        # A finite scalar size uses reservoir sampling, NOT the latest N points.
        # Load incrementally with sampling disabled, then discard old scalar rows.
        self.accumulator = accumulator_class(
            str(run_dir),
            size_guidance={"scalars": 0, "images": 1, "histograms": 1,
                           "compressedHistograms": 1, "tensors": 1, "audio": 1},
        )
        self.rows = {}

    def reload(self) -> None:
        self.accumulator.Reload()
        for tag in self.accumulator.Tags().get("scalars", []):
            events = self.accumulator.Scalars(tag)
            # One row per training update; retain the last row if a step repeats.
            by_step = {event.step: event for event in events}
            rows = sorted(by_step.values(), key=lambda event: event.step)[-2 * self.window:]
            self.rows[tag] = rows
            if rows:
                cutoff = rows[0].step
                self.accumulator.scalars.FilterItems(lambda event: event.step >= cutoff, key=tag)

    def latest(self, tag: str):
        rows = self.rows.get(tag, [])
        return rows[-1].value if rows else None

    def mean(self, tag: str, previous: bool = False):
        rows = self.rows.get(tag, [])
        rows = rows[-2 * self.window:-self.window] if previous else rows[-self.window:]
        # Do not silently filter non-finite metrics: NAN/INF must remain visible.
        return sum(row.value for row in rows) / len(rows) if rows else None


def render(run_dir: Path, active: dict, history: ScalarHistory, window: int, interval: float) -> str:
    now = time.time()
    progress = read_json(run_dir / "progress.json")
    policy_dt, dt_source = step_duration(run_dir)
    lines = [f"Hiking 训练监控  {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
             f"运行：{run_dir.name}", process_status(active.get("pid"))]
    updated = progress.get("updated_at")
    age = None
    if updated:
        try:
            stamp = datetime.fromisoformat(updated)
            age = now - stamp.timestamp()
            state = "（日志滞后）" if age > max(30, interval * 3) else ""
            lines.append(f"进度日志：{stamp.astimezone():%H:%M:%S}，{max(0, age):.1f} 秒前 {state}")
        except (TypeError, ValueError):
            lines.append(f"进度日志时间不可解析：{updated}")
    else:
        lines.append("尚无进度日志；等待训练启动或检查运行目录。")
    iteration = progress.get("iteration")
    target = progress.get("target_iteration", active.get("target_iteration"))
    lines.append(f"训练更新：{iteration if iteration is not None else '--'} / {target or '--'}")
    collection = history.mean("Perf/collection_time")
    learning = history.mean("Perf/learning_time")
    per_iteration = collection + learning if collection is not None and learning is not None else None
    eta = None
    if iteration is not None and target is not None and per_iteration is not None:
        eta = max(0, target - iteration - 1) * per_iteration
    lines.append(
        f"本次训练耗时 {duration(progress.get('training_seconds'))}；"
        f"近窗口吞吐 {number(history.mean('Perf/total_fps'))} transitions/s；"
        f"每次更新 {number(per_iteration)} 秒；剩余预算约 {duration(eta)}"
    )
    if age is not None and age > max(30, interval * 3):
        lines.append("日志已滞后，吞吐和剩余时间仅为最后记录，不能据此确认仍在训练。")
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path.name))
    lines.append(f"最新检查点：{max(checkpoints)[1] if checkpoints else '尚无'}；"
                 f"本次采样量：{progress.get('transitions_this_run', '--')}")
    lines.extend(["", pad("指标", 25) + f"{'最新':>12}  {f'前 {window} 均值':>14}  {f'近 {window} 均值':>14}"])
    for label, tag, scale in METRICS:
        scale = policy_dt if scale is None else scale
        values = (history.latest(tag), history.mean(tag, previous=True), history.mean(tag))
        lines.append(pad(label, 25) + "  ".join(f"{number(v * scale if v is not None else None):>14}" for v in values))
    reward_rows = history.rows.get("Train/mean_reward_0", [])
    if reward_rows:
        recent = reward_rows[-window:]
        previous = reward_rows[-2 * window:-window]
        describe = lambda rows: f"{rows[0].step}–{rows[-1].step}（{len(rows)} 点）" if rows else "尚无"
        lines.append(f"任务奖励窗口：前 {describe(previous)}；近 {describe(recent)}。最新值为日志值。")
    else:
        lines.append("TensorBoard 标量尚未写入，等待首次更新。")
    latest_step = max((rows[-1].step for rows in history.rows.values() if rows), default=0)
    nonfinite = []
    for tag, rows in history.rows.items():
        bad = [row for row in rows if row.step > latest_step - window and not math.isfinite(row.value)]
        if bad:
            nonfinite.append(f"{tag}（最近 step {bad[-1].step}，{len(bad)} 点）")
    if nonfinite:
        lines.append("警告：近窗口含非有限值：" + "；".join(nonfinite[:6]))
        if len(nonfinite) > 6:
            lines.append(f"另有 {len(nonfinite) - 6} 个标签含非有限值。")
    lines.extend([f"回合换算：{policy_dt:g} 秒/步（{dt_source}）。",
                  "前/近窗口为相邻更新区间；回合奖励和时长本身已是训练器的滑动统计。",
                  "课程等级、AMP 奖励和训练损失不等于上下楼成功率；能力需要独立回放验证。",
                  f"每 {interval:g} 秒刷新；Ctrl+C 只退出监控，训练继续。"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="只读 Hiking 实时终端监控，无需启动 Isaac Sim。")
    parser.add_argument("--once", action="store_true", help="输出一次后退出")
    parser.add_argument("--interval", type=float, default=5, help="刷新间隔秒数（默认 5）")
    parser.add_argument("--window", type=int, default=100, help="每个均值窗口的更新数（默认 100）")
    parser.add_argument("--run-dir", type=Path, help="固定查看指定目录，默认跟随 active_run.json")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0 or args.window <= 0:
        parser.error("--interval 和 --window 必须为正数")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.backend.event_processing.directory_watcher import DirectoryDeletedError
    except ImportError:
        print("缺少 tensorboard；请使用 /home/hamlet/miniconda3/envs/isaac_sim_env/bin/python 运行。", file=sys.stderr)
        return 1
    history = None
    current_run = None
    tty = sys.stdout.isatty()
    try:
        while True:
            active = read_json(ACTIVE_RUN)
            path = args.run_dir or active.get("run_dir")
            output = "尚无活跃训练记录，等待 " + str(ACTIVE_RUN)
            if path:
                run_dir = Path(path).expanduser().resolve()
                if run_dir != current_run:
                    history = ScalarHistory(run_dir, args.window, EventAccumulator)
                    current_run = run_dir
                matching_active = active if Path(active.get("run_dir", "")).resolve() == run_dir else {}
                try:
                    history.reload()
                    output = render(run_dir, matching_active, history, args.window, args.interval)
                except (OSError, ValueError, RuntimeError, DirectoryDeletedError) as exc:
                    output = f"读取训练日志暂时失败：{exc}\n运行：{run_dir}\n下次刷新重试，训练不受影响。"
            if tty and not args.once:
                print("\033[2J\033[H", end="")
            elif not args.once:
                print("\n" + "=" * 76)
            print(output, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出监控，训练继续。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
