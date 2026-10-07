"""Run the production action term with real Isaac buffers, without launching Kit."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_action_class():
    isaac_spec = importlib.util.find_spec("isaaclab")
    if isaac_spec is None:
        raise unittest.SkipTest("Isaac Lab source is required for its actual DelayBuffer")
    isaac_root = Path(next(iter(isaac_spec.submodule_search_locations)))
    modules = {}

    def module(name):
        if name not in modules:
            result = ModuleType(name)
            result.__path__ = []
            modules[name] = result
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(module(parent), child, result)
        return modules[name]

    class ActionTerm:
        def __init__(self, cfg, env):
            self.cfg, self._asset = cfg, env.scene[cfg.asset_name]
            self.num_envs, self.device = env.num_envs, env.device

    module("isaaclab.managers.action_manager").ActionTerm = ActionTerm
    module("isaaclab.assets.articulation").Articulation = object
    module("isaaclab.utils.string")
    module("isaaclab.utils").configclass = lambda cls: cls
    action_package = module("isaaclab.envs.mdp.actions")
    action_package.JointPositionActionCfg = type("JointPositionActionCfg", (), {})

    def load(name, path, package=False):
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)] if package else None)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    with patch.dict(sys.modules, modules):
        # These complete production modules execute unchanged. Only their
        # simulator-facing base class and unused import dependencies are stubs.
        buffers = load("isaaclab.utils.buffers", isaac_root / "utils/buffers/__init__.py", package=True)
        modules["isaaclab.utils"].buffers = buffers
        joint_actions = load("isaaclab.envs.mdp.actions.joint_actions", isaac_root / "envs/mdp/actions/joint_actions.py")
        action_package.JointPositionAction = joint_actions.JointPositionAction
        loaded = load("_hiking_action_delay_under_test", ROOT / "legged_lab/hiking/actions.py")
        return loaded.PhysicsDelayedJointPositionAction


def make_action(cls):
    defaults = torch.tensor([[.1, -.2], [.3, -.4], [.5, -.6]])
    asset = SimpleNamespace(num_joints=2, data=SimpleNamespace(default_joint_pos=defaults), targets=None)
    asset.find_joints = lambda names, preserve_order: ([0, 1], ["joint0", "joint1"])

    def set_targets(targets, joint_ids):
        asset.targets = targets.clone()

    asset.set_joint_position_target = set_targets
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset})
    cfg = SimpleNamespace(asset_name="robot", joint_names=["joint0", "joint1"], preserve_order=True,
                          scale=2., offset=0., clip=None, use_default_offset=True, min_delay=0, max_delay=2)
    return cls(cfg, env), asset


class HikingActionDelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.action_class = load_action_class()

    def test_targets_delay_by_physics_calls_and_raw_processed_actions_stay_current(self):
        action, asset = make_action(self.action_class)
        defaults = asset.data.default_joint_pos
        action._target_delay_buffer.set_time_lag(torch.tensor([0, 1, 2]))
        action.process_actions(torch.zeros(3, 2))
        action.apply_actions()
        action.process_actions(torch.ones(3, 2))
        for ready in ([True, False, False], [True, True, False], [True, True, True]):
            action.apply_actions()
            expected = defaults + torch.tensor(ready)[:, None] * 2.
            torch.testing.assert_close(asset.targets, expected)
            torch.testing.assert_close(action.raw_actions, torch.ones(3, 2))
            torch.testing.assert_close(action.processed_actions, defaults + 2.)

    def test_partial_reset_clears_old_targets_without_erasing_other_environment_history(self):
        action, asset = make_action(self.action_class)
        defaults = asset.data.default_joint_pos
        action.process_actions(torch.ones(3, 2))
        for _ in range(3):
            action.apply_actions()
        action.reset(torch.tensor([1]))
        action._target_delay_buffer.set_time_lag(2)
        torch.testing.assert_close(action.raw_actions[1], torch.zeros(2))
        torch.testing.assert_close(action.processed_actions[1], defaults[1])
        action.process_actions(torch.tensor([[4., 4.], [9., 9.], [4., 4.]]))
        action.apply_actions()
        torch.testing.assert_close(asset.targets, defaults + torch.tensor([[2., 2.], [18., 18.], [2., 2.]]))

    def test_full_slice_reset_and_empty_reset(self):
        action, asset = make_action(self.action_class)
        action.process_actions(torch.ones(3, 2))
        action.apply_actions()
        action.reset([])
        torch.testing.assert_close(action.raw_actions, torch.ones(3, 2))
        action.reset(slice(None))
        action.apply_actions()
        torch.testing.assert_close(asset.targets, asset.data.default_joint_pos)
        torch.testing.assert_close(action.raw_actions, torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
