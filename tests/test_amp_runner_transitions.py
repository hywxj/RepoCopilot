"""Episode boundaries must reach discriminator reward and replay consistently."""

import pytest
import torch

from rsl_rl.runners.amp_on_policy_runner import _restore_terminal_amp_observations


def test_restore_uses_pre_reset_rows_and_preserves_new_episode_observations():
    post_reset = torch.tensor([[100., 101.], [2., 3.], [200., 201.]])
    original = post_reset.clone()
    infos = {"terminal_amp_env_ids": torch.tensor([2, 0]),
             "terminal_amp_observations": torch.tensor([[20., 21.], [10., 11.]])}
    restored = _restore_terminal_amp_observations(post_reset, torch.tensor([True, False, True]), infos)
    torch.testing.assert_close(restored, torch.tensor([[10., 11.], [2., 3.], [20., 21.]]))
    torch.testing.assert_close(post_reset, original)
    infos["terminal_amp_observations"].fill_(-1.)
    torch.testing.assert_close(restored[0], torch.tensor([10., 11.]))


def test_no_terminal_returns_owned_nonterminal_data():
    states = torch.zeros(3, 70)
    restored = _restore_terminal_amp_observations(states, torch.zeros(3, dtype=torch.bool), {})
    restored[0] = 1
    assert not states.any()


@pytest.mark.parametrize("ids", [torch.tensor([1]), torch.tensor([0, 0]), torch.tensor([], dtype=torch.long),
                                 torch.tensor([0.]), torch.tensor([[0]])])
def test_stale_duplicate_wrong_or_missing_terminal_ids_are_rejected(ids):
    infos = {"terminal_amp_env_ids": ids, "terminal_amp_observations": torch.zeros(ids.numel(), 70)}
    with pytest.raises(ValueError):
        _restore_terminal_amp_observations(torch.zeros(3, 70), torch.tensor([True, False, False]), infos)


def test_ending_episode_without_snapshot_is_an_error_instead_of_reset_transition():
    with pytest.raises(ValueError, match="pre-reset"):
        _restore_terminal_amp_observations(torch.zeros(3, 70), torch.tensor([True, False, False]), {})
