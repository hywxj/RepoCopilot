"""Fetch the pinned Hiking source and install only its isolated optional dependencies."""

import json
import subprocess
import sys

from legged_lab.hiking.bootstrap import LOCK, ROOT, activate


def main():
    for name, spec in json.loads(LOCK.read_text()).items():
        repo = ROOT / "third_party" / name
        if not repo.exists():
            repo.mkdir(parents=True)
            subprocess.run(["git", "init", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", spec["url"]], check=True)
            subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", spec["commit"]], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "--detach", spec["commit"]], check=True)
    activate()  # Fail on an unexpected existing checkout instead of resetting local work.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(ROOT / "third_party/hiking_python"),
        "-r", str(LOCK.with_name("requirements.txt")),
    ], check=True)
    print("Hiking sources and optional dependencies are ready; Isaac/PyTorch were not upgraded.")


if __name__ == "__main__":
    main()
