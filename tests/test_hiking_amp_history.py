"""Exercise real Isaac observation histories across an expert clip boundary.

Only simulator-facing imports and motion values are stand-ins. The old upstream
termination, new termination, ObservationManager methods and CircularBuffer all
execute their production implementations unchanged, without starting Kit.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_production_types():
    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        raise unittest.SkipTest("Isaac Lab source is required for its observation manager")
    isaac_root = Path(next(iter(spec.submodule_search_locations)))
    upstream = ROOT / "third_party/InstinctLab/source/instinctlab/instinctlab/envs/mdp/terminations/general.py"
    if not upstream.is_file():
        raise unittest.SkipTest("Pinned InstinctLab checkout is required for the regression baseline")
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

    class ManagerBase:
        @property
        def device(self):
            return self._env.device

    class SceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)

    managers = module("isaaclab.managers")
    managers.SceneEntityCfg = SceneEntityCfg
    managers.ManagerTermBase = type("ManagerTermBase", (), {})
    managers.ManagerTermBaseCfg = type("ManagerTermBaseCfg", (), {})
    bases = module("isaaclab.managers.manager_base")
    bases.ManagerBase, bases.ManagerTermBase = ManagerBase, managers.ManagerTermBase
    configs = module("isaaclab.managers.manager_term_cfg")
    configs.ObservationGroupCfg = type("ObservationGroupCfg", (), {})
    configs.ObservationTermCfg = type("ObservationTermCfg", (), {})
    module("isaaclab.sensors").ContactSensor = object
    utils = module("isaaclab.utils")
    utils.class_to_dict = vars
    utils.modifiers = module("isaaclab.utils.modifiers")
    utils.noise = module("isaaclab.utils.noise")
    utils.noise.NoiseCfg = type("NoiseCfg", (), {})
    utils.noise.NoiseModelCfg = type("NoiseModelCfg", (), {})

    def load(name, path, package=False):
        module_spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)] if package else None)
        loaded = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = loaded
        module_spec.loader.exec_module(loaded)
        return loaded

    with patch.dict(sys.modules, modules):
        buffers = load("isaaclab.utils.buffers", isaac_root / "utils/buffers/__init__.py", package=True)
        utils.buffers = buffers
        observations = load("isaaclab.managers.observation_manager", isaac_root / "managers/observation_manager.py")
        old = load("_hiking_original_exhaustion", upstream)
        new = load("_hiking_fixed_exhaustion", ROOT / "legged_lab/hiking/amp_history.py")
        return (observations.ObservationManager, buffers.CircularBuffer, old.dataset_exhausted,
                new.dataset_exhausted_with_reference_history_reset)


class ReferenceFrames:
    """Distinct clip identities and a mirror sign expose any stale sample."""

    def __init__(self):
        self.ALL_INDICES = torch.arange(3)
        self.aiming_frame_idx = torch.tensor([0, 1, 2])
        self.data = SimpleNamespace(validity=torch.ones(3, 3, dtype=torch.bool))
        self.clip = torch.tensor([100., 200., 300.])
        self.frame = torch.zeros(3)
        self.mirror = torch.ones(3)
        self.reset_calls = []

    def exhaust(self, ids):
        self.data.validity[ids, self.aiming_frame_idx[ids]] = False

    def reset(self, env_ids):
        self.reset_calls.append(env_ids.clone())
        self.clip[env_ids] += 1000.
        self.frame[env_ids] = 0.
        self.mirror[env_ids] *= -1.
        self.data.validity[env_ids] = True

    def value(self, width, signed):
        value = self.clip[:, None] + self.frame[:, None] + torch.arange(width)[None] * .001
        return value * self.mirror[:, None] if signed else value


def make_environment(manager_cls, buffer_cls, filled=True):
    reference = ReferenceFrames()
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"motion_reference": reference}, tick=0., extras={})
    env.camera_history = torch.arange(3 * 37 * 18 * 32).reshape(3, 37, 18, 32)
    manager = manager_cls.__new__(manager_cls)
    manager._env = env
    manager._obs_buffer = None
    manager._group_obs_term_names = {}
    manager._group_obs_term_cfgs = {}
    manager._group_obs_term_history_buffer = {}
    manager._group_obs_concatenate = {}
    manager._group_obs_concatenate_dim = {}
    manager._group_obs_class_term_cfgs = {}
    manager._group_obs_class_instances = []
    env.observation_manager = manager

    def add_group(name, terms):
        manager._group_obs_term_names[name] = list(terms)
        manager._group_obs_term_cfgs[name] = []
        manager._group_obs_term_history_buffer[name] = {}
        manager._group_obs_concatenate[name] = False
        manager._group_obs_concatenate_dim[name] = -1
        manager._group_obs_class_term_cfgs[name] = []
        for term_name, (width, scale, signed) in terms.items():
            func = (lambda env, width=width, signed=signed: env.scene["motion_reference"].value(width, signed)) if name == "amp_reference" else (
                lambda env, width=width: torch.arange(3.)[:, None].expand(3, width) + env.tick)
            manager._group_obs_term_cfgs[name].append(SimpleNamespace(
                func=func, params={}, modifiers=None, noise=None, clip=None, scale=scale,
                history_length=10, flatten_history_dim=term_name != "projected_gravity"))
            manager._group_obs_term_history_buffer[name][term_name] = buffer_cls(10, 3, "cpu")

    add_group("amp_reference", {"projected_gravity": (3, None, True), "joint_pos_rel": (29, None, True),
                                "joint_vel": (29, .05, True), "base_lin_vel": (3, None, False),
                                "base_ang_vel": (3, None, True)})
    for name in ("amp_policy", "policy", "critic"):
        add_group(name, {"state": (3, None, False)})
    if filled:
        for tick in range(12):
            env.tick = float(tick)
            reference.frame[:] = tick
            manager.compute(update_history=True)
    return env


def histories(env):
    return env.observation_manager._group_obs_term_history_buffer


class HikingAmpHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = load_production_types()

    def make_env(self, filled=True):
        return make_environment(*self.production[:2], filled=filled)

    def test_regression_old_upstream_mixes_clips_new_function_does_not(self):
        for label, reset_fn in (("old", self.production[2]), ("fixed", self.production[3])):
            with self.subTest(implementation=label):
                env = self.make_env()
                ref = env.scene["motion_reference"]
                ref.exhaust(torch.tensor([1]))
                done = reset_fn(env, reset_without_notice=True)
                self.assertFalse(done.any())
                pack = env.observation_manager.compute(update_history=True)
                window = pack["amp_reference"]["joint_pos_rel"].reshape(3, 10, 29)[1]
                if label == "old":
                    self.assertTrue((window[:-1] > 0).all())
                    self.assertTrue((window[-1] < -1000).all())
                else:
                    torch.testing.assert_close(window, ref.value(29, True)[1].expand(10, -1))

    def test_partial_switch_preserves_other_histories_and_builds_ten_real_new_frames(self):
        env = self.make_env()
        manager, ref = env.observation_manager, env.scene["motion_reference"]
        before = {group: {name: buffer.buffer.clone() for name, buffer in terms.items()}
                  for group, terms in histories(env).items()}
        camera = env.camera_history.clone()
        ref.exhaust(torch.tensor([1]))
        self.production[3](env, reset_without_notice=True)
        self.assertIsNone(manager._obs_buffer)
        torch.testing.assert_close(ref.reset_calls[0], torch.tensor([1]))
        for group in ("amp_policy", "policy", "critic"):
            torch.testing.assert_close(histories(env)[group]["state"].buffer, before[group]["state"])
        torch.testing.assert_close(env.camera_history, camera)
        pack = manager.compute(update_history=True)
        expected_frames = {name: [term.func(env).clone()] for name, term in zip(
            manager._group_obs_term_names["amp_reference"], manager._group_obs_term_cfgs["amp_reference"])}
        for name, buffer in histories(env)["amp_reference"].items():
            torch.testing.assert_close(buffer.buffer[[0, 2], :-1], before["amp_reference"][name][[0, 2], 1:])
        for _ in range(9):
            env.tick += 1.
            ref.frame += 1.
            pack = manager.compute(update_history=True)
            for name, term in zip(manager._group_obs_term_names["amp_reference"], manager._group_obs_term_cfgs["amp_reference"]):
                expected_frames[name].append(term.func(env).clone())
                self.assertTrue((histories(env)["amp_reference"][name].buffer[1].abs() > 50.).all())
        for name, term in zip(manager._group_obs_term_names["amp_reference"], manager._group_obs_term_cfgs["amp_reference"]):
            expected = torch.stack(expected_frames[name], dim=1)
            if term.scale is not None:
                expected *= term.scale
            torch.testing.assert_close(histories(env)["amp_reference"][name].buffer[1], expected[1])
        self.assertIs(manager._obs_buffer, pack)

    def test_compute_false_never_advances_existing_or_newly_reset_histories(self):
        env = self.make_env()
        ref, manager = env.scene["motion_reference"], env.observation_manager
        ref.exhaust(torch.tensor([0, 2]))
        self.production[3](env, reset_without_notice=True)
        counts = {group: {name: b._num_pushes.clone() for name, b in terms.items()} for group, terms in histories(env).items()}
        for _ in range(3):
            manager.compute(update_history=False)
        for group, terms in histories(env).items():
            for name, buffer in terms.items():
                torch.testing.assert_close(buffer._num_pushes, counts[group][name])
        pack = manager.compute(update_history=True)
        for index in (0, 2):
            torch.testing.assert_close(pack["amp_reference"]["joint_pos_rel"].reshape(3, 10, 29)[index],
                                       ref.value(29, True)[index].expand(10, -1))

    def test_unallocated_histories_use_nonpersistent_false_read_then_correct_first_frame(self):
        env = self.make_env(filled=False)
        manager, ref = env.observation_manager, env.scene["motion_reference"]
        ref.exhaust(torch.tensor([1]))
        self.production[3](env, reset_without_notice=True)
        manager.compute(update_history=False)
        manager.compute(update_history=False)
        for terms in histories(env).values():
            for buffer in terms.values():
                self.assertIsNone(buffer._buffer)
                self.assertFalse(buffer._num_pushes.any())
        manager.compute(update_history=True)
        for name, buffer in histories(env)["amp_reference"].items():
            self.assertTrue((buffer._num_pushes == 1).all())
            torch.testing.assert_close(buffer.buffer[:, 0], buffer.buffer[:, -1])

    def test_no_exhaustion_leaves_reference_histories_and_cache_untouched(self):
        env = self.make_env()
        manager, ref = env.observation_manager, env.scene["motion_reference"]
        cached = manager._obs_buffer
        before = histories(env)["amp_reference"]["joint_pos_rel"].buffer.clone()
        for _ in range(3):
            self.assertFalse(self.production[3](env, reset_without_notice=True).any())
        self.assertEqual(ref.reset_calls, [])
        self.assertIs(manager._obs_buffer, cached)
        torch.testing.assert_close(histories(env)["amp_reference"]["joint_pos_rel"].buffer, before)

    def test_all_switch_and_real_episode_reset_same_step_have_no_old_frames(self):
        for real_episode in (False, True):
            with self.subTest(real_episode_reset=real_episode):
                env = self.make_env()
                manager, ref = env.observation_manager, env.scene["motion_reference"]
                ref.exhaust(torch.arange(3))
                self.assertFalse(self.production[3](env, reset_without_notice=True).any())
                if real_episode:
                    # Match normal env reset: scene resets sensors, then the
                    # observation manager clears all groups for these rows.
                    ref.reset(torch.tensor([1]))
                    manager.reset(torch.tensor([1]))
                pack = manager.compute(update_history=True)
                expected = ref.value(29, True)[:, None].expand(-1, 10, -1)
                torch.testing.assert_close(pack["amp_reference"]["joint_pos_rel"].reshape(3, 10, 29), expected)

    def test_notice_mode_returns_termination_without_silent_reset(self):
        env = self.make_env()
        ref = env.scene["motion_reference"]
        ref.exhaust(torch.tensor([0, 2]))
        cached = env.observation_manager._obs_buffer
        done = self.production[3](env, reset_without_notice=False)
        torch.testing.assert_close(done, torch.tensor([True, False, True]))
        self.assertEqual(ref.reset_calls, [])
        self.assertIs(env.observation_manager._obs_buffer, cached)

    def test_missing_reference_history_fails_before_switching_motion(self):
        env = self.make_env()
        ref = env.scene["motion_reference"]
        histories(env)["amp_reference"] = {}
        ref.exhaust(torch.tensor([1]))
        with self.assertRaisesRegex(RuntimeError, "history buffers"):
            self.production[3](env, reset_without_notice=True)
        self.assertEqual(ref.reset_calls, [])


if __name__ == "__main__":
    unittest.main()
