"""Delay position setpoints at the physics rate while retaining implicit PD."""

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class PhysicsDelayedJointPositionAction(JointPositionAction):
    """Keep Isaac's raw/scaled actions and delay only targets sent to the robot.

    ``process_actions`` still runs once per policy step. ``apply_actions`` runs
    once per physics substep, so a lag of two at 5ms means 10ms, not 40ms.
    The implicit actuator always uses current joint feedback. Isaac's standard
    DelayBuffer fills empty history with the first post-reset target, ensuring
    that no target from the previous episode is reused during buffer warmup.
    """

    def __init__(self, cfg, env):
        if (not isinstance(cfg.min_delay, int) or not isinstance(cfg.max_delay, int)
                or not 0 <= cfg.min_delay <= cfg.max_delay):
            raise ValueError("Position target delay must be ordered nonnegative integer physics steps")
        super().__init__(cfg, env)
        self._target_delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset()

    def apply_actions(self):
        delayed_targets = self._target_delay_buffer.compute(self.processed_actions)
        self._asset.set_joint_position_target(delayed_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = self._all_env_ids if env_ids is None else self._all_env_ids[env_ids]
        if len(ids) == 0:
            return
        super().reset(ids)
        # Zero raw action means the configured/default offset, not a zero pose.
        # Clear this pending target too in case apply_actions precedes the first
        # process_actions after reset; it must not revive an old episode target.
        neutral = self._offset[ids] if isinstance(self._offset, torch.Tensor) else self._offset
        self._processed_actions[ids] = neutral
        if self.cfg.clip is not None:
            self._processed_actions[ids] = torch.clamp(
                self._processed_actions[ids], min=self._clip[ids, :, 0], max=self._clip[ids, :, 1])
        lags = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1,
                             (len(ids),), dtype=torch.int, device=self.device)
        self._target_delay_buffer.reset(ids)
        self._target_delay_buffer.set_time_lag(lags, ids)


@configclass
class PhysicsDelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = PhysicsDelayedJointPositionAction
    min_delay: int = 0
    max_delay: int = 2
