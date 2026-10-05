"""Live playback wrapper for geometry policies.

It reuses the normal playback path while keeping the requested velocity command
after resets and interval command updates. The original playback script already
provides the OpenCV depth/geometry windows.
"""

from __future__ import annotations

import importlib


def main() -> None:
    play_module = importlib.import_module("legged_lab.scripts.play")
    original_get_task_class = play_module.task_registry.get_task_class
    fixed_command = (
        float(play_module.args_cli.command_x),
        float(play_module.args_cli.command_y),
        float(play_module.args_cli.command_yaw),
    )

    def get_fixed_command_task(task_name):
        base_class = original_get_task_class(task_name)

        class FixedCommandEnv(base_class):
            def _keep_command(self):
                self.command_generator.command[:, 0] = fixed_command[0]
                self.command_generator.command[:, 1] = fixed_command[1]
                self.command_generator.command[:, 2] = fixed_command[2]

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._keep_command()

            def step(self, actions):
                self._keep_command()
                output = super().step(actions)
                self._keep_command()
                return output

        return FixedCommandEnv

    play_module.task_registry.get_task_class = get_fixed_command_task
    play_module.play()
    play_module.simulation_app.close()


if __name__ == "__main__":
    main()
