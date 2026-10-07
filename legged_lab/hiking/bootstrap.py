"""Load pinned upstream packages without replacing Isaac or the legacy RL package."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name("upstream.lock.json")


def activate():
    versions = json.loads(LOCK.read_text())
    paths = [ROOT / "third_party/hiking_python"]
    for name, spec in versions.items():
        repo = ROOT / "third_party" / name
        if not (repo / ".git").exists():
            raise RuntimeError("Run python -m legged_lab.scripts.setup_hiking first")
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["commit"]:
            raise RuntimeError(f"{name} revision mismatch: {actual} != {spec['commit']}")
        paths.append(repo / spec["python_path"])
    sys.path[:0] = [str(p) for p in paths]
    return versions
