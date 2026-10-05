"""Stateful gate for enabling stair-specific locomotion behavior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class StairMode(IntEnum):
    BLIND = 0
    STAIRS_UP = 1
    STAIRS_DOWN = -1


@dataclass
class StairModeGateCfg:
    """Thresholds for robust stair detection with temporal hysteresis."""

    enter_confidence: float = 0.65
    exit_confidence: float = 0.35
    enter_frames: int = 3
    # D435i frames can lose the current riser exactly at a stair crest. Hold
    # the active mode long enough to cross that blind transition instead of
    # handing control back to the flat-ground policy immediately.
    exit_frames: int = 40
    switch_frames: int = 3
    min_valid_treads: int = 1
    min_nearest_tread_m: float = 0.15
    max_nearest_tread_m: float = 1.25


class StairModeGate:
    """Convert noisy frame-wise geometry detections into a stable mode flag."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg: StairModeGateCfg | None = None):
        self.cfg = cfg or StairModeGateCfg()
        self._mode = torch.zeros(num_envs, dtype=torch.int8, device=device)
        self._candidate_direction = torch.zeros(num_envs, dtype=torch.int8, device=device)
        self._enter_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self._exit_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self._switch_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

    @property
    def active(self) -> torch.Tensor:
        return self._mode != StairMode.BLIND

    @property
    def mode(self) -> torch.Tensor:
        return self._mode

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._mode.zero_()
            self._candidate_direction.zero_()
            self._enter_count.zero_()
            self._exit_count.zero_()
            self._switch_count.zero_()
            return
        self._mode[env_ids] = StairMode.BLIND
        self._candidate_direction[env_ids] = StairMode.BLIND
        self._enter_count[env_ids] = 0
        self._exit_count[env_ids] = 0
        self._switch_count[env_ids] = 0

    def update(
        self,
        confidence: torch.Tensor,
        valid_treads: torch.Tensor,
        nearest_tread_m: torch.Tensor,
        direction: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update and return BLIND, STAIRS_UP, or STAIRS_DOWN per environment."""

        if update_mask is None:
            update_mask = torch.ones_like(confidence, dtype=torch.bool)

        detected_direction = torch.sign(direction).to(torch.int8)
        in_range = (nearest_tread_m >= self.cfg.min_nearest_tread_m) & (
            nearest_tread_m <= self.cfg.max_nearest_tread_m
        )
        enter_evidence = (
            (confidence >= self.cfg.enter_confidence)
            & (valid_treads >= self.cfg.min_valid_treads)
            & in_range
            & (detected_direction != StairMode.BLIND)
        )
        exit_evidence = (
            (confidence < self.cfg.exit_confidence)
            | (valid_treads < self.cfg.min_valid_treads)
            | ~in_range
        )

        inactive = ~self.active
        same_candidate = detected_direction == self._candidate_direction
        next_enter_count = torch.where(same_candidate, self._enter_count + 1, torch.ones_like(self._enter_count))
        decayed_enter_count = (self._enter_count - 1).clamp_min(0)
        # A walking camera can briefly lose a riser during torso pitch. Keep a
        # leaky evidence count so one missed frame does not discard an otherwise
        # consistent detection, while isolated false positives still decay.
        new_enter_count = torch.where(
            inactive & enter_evidence,
            next_enter_count,
            torch.where(inactive, decayed_enter_count, torch.zeros_like(self._enter_count)),
        )
        self._enter_count = torch.where(update_mask, new_enter_count, self._enter_count)
        new_candidate_direction = torch.where(
            inactive & enter_evidence,
            detected_direction,
            torch.where(
                inactive & (new_enter_count > 0),
                self._candidate_direction,
                torch.zeros_like(self._candidate_direction),
            ),
        )
        self._candidate_direction = torch.where(
            update_mask, new_candidate_direction, self._candidate_direction
        )

        enter_now = update_mask & inactive & (self._enter_count >= self.cfg.enter_frames)
        self._mode = torch.where(enter_now, self._candidate_direction, self._mode)

        active = self.active
        new_exit_count = torch.where(
            active & exit_evidence,
            self._exit_count + 1,
            torch.zeros_like(self._exit_count),
        )
        self._exit_count = torch.where(update_mask, new_exit_count, self._exit_count)
        exit_now = update_mask & active & (self._exit_count >= self.cfg.exit_frames)

        switch_evidence = active & enter_evidence & (detected_direction != self._mode)
        new_switch_count = torch.where(
            switch_evidence,
            self._switch_count + 1,
            torch.zeros_like(self._switch_count),
        )
        self._switch_count = torch.where(update_mask, new_switch_count, self._switch_count)
        switch_now = update_mask & switch_evidence & (self._switch_count >= self.cfg.switch_frames)
        self._mode = torch.where(switch_now, detected_direction, self._mode)

        self._mode = torch.where(exit_now, torch.zeros_like(self._mode), self._mode)
        return self._mode
