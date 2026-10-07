"""Official learner with bounded checkpoint retention for the local GPU run."""

import re
import json
from datetime import datetime, timezone
from pathlib import Path

from instinct_rl.runners import OnPolicyRunner


class HikingRunner(OnPolicyRunner):
    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        if self.log_dir is not None and not self.is_mp_rank_other_process():
            status = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "iteration": self.current_learning_iteration,
                "target_iteration": locs["tot_iter"],
                "transitions_this_run": self.tot_timesteps,
                "training_seconds": self.tot_time,
                "mean_episode_length_steps": sum(locs["lenbuffer"]) / max(1, len(locs["lenbuffer"])),
                "capability_validated": False,
            }
            path = Path(self.log_dir) / "progress.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(path)

    def save(self, path, infos=None):
        super().save(path, infos)
        # Only prune this runner's own numbered checkpoints after a successful save.
        checkpoints = []
        for candidate in Path(path).parent.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", candidate.name)
            if match:
                checkpoints.append((int(match.group(1)), candidate))
        for _, stale in sorted(checkpoints)[:-3]:
            stale.unlink()
