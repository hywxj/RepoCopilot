"""Keep AMP reference windows within one motion clip when the clip ends."""

import torch

from isaaclab.managers import SceneEntityCfg


def dataset_exhausted_with_reference_history_reset(
    env,
    reference_cfg=SceneEntityCfg("motion_reference"),
    reset_without_notice=False,
    print_reason=False,
):
    """Match upstream termination semantics and reset only switched expert rows.

    Called during termination computation, before the normal end-of-step
    observation update. Isaac's CircularBuffer fills the cleared rows with the
    first new frame, then replaces that padding over the next nine steps. We do
    not compute observations here: that would advance unrelated histories twice.
    No robot reset is introduced when ``reset_without_notice`` is enabled.
    """
    reference = env.scene[reference_cfg.name]
    exhausted = ~reference.data.validity[reference.ALL_INDICES, reference.aiming_frame_idx]
    if not reset_without_notice:
        return exhausted
    ids = exhausted.nonzero(as_tuple=True)[0]
    env.extras.setdefault("step", {})["amp_reference_clip_restarts"] = exhausted.sum().float()
    if ids.numel():
        manager = env.observation_manager
        # Pinned Isaac Lab exposes no public group-only history reset. Keep this
        # dependency in one place and fail before changing the reference state
        # if a future version changes it.
        histories = manager._group_obs_term_history_buffer["amp_reference"]
        if not histories:
            raise RuntimeError("AMP reference history buffers are missing")
        reference.reset(env_ids=ids)
        for history in histories.values():
            history.reset(batch_ids=ids)
        # Prevent readers of the optional cached pack from seeing old expert
        # observations; the normal end-of-step compute repopulates the cache.
        manager._obs_buffer = None
        if print_reason:
            print("AMP reference clips restarted:", ids.numel())
    # A reference clip boundary never terminates the robot's episode.
    return torch.zeros_like(exhausted)
