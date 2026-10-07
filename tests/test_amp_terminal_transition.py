"""Execute the real environment step methods with small simulator stand-ins."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_environment_classes():
    """Stub import-only Isaac dependencies without starting an application.

    Both complete production modules are executed unchanged. Tests below call
    their actual step methods; simulator objects supply only the state changes.
    """
    modules = {}

    def module(name, symbols=()):
        result = modules.setdefault(name, ModuleType(name))
        for symbol in symbols:
            setattr(result, symbol, type(symbol, (), {}))
        return result

    dependencies = {
        "isaaclab.sim": ("PhysxCfg", "SimulationContext"),
        "isaacsim.core.utils.torch": (),
        "isaaclab.assets.articulation": ("Articulation",),
        "isaaclab.envs.mdp.commands": ("UniformVelocityCommand", "UniformVelocityCommandCfg"),
        "isaaclab.managers": ("EventManager", "RewardManager"),
        "isaaclab.managers.scene_entity_cfg": ("SceneEntityCfg",),
        "isaaclab.scene": ("InteractiveScene",),
        "isaaclab.sensors": ("ContactSensor", "RayCaster"),
        "isaaclab.sensors.camera": ("TiledCamera",),
        "isaaclab.utils.buffers": ("CircularBuffer", "DelayBuffer"),
        "isaaclab.utils.math": ("quat_apply", "quat_apply_inverse", "quat_conjugate", "quat_mul",
                               "quat_from_euler_xyz", "quat_rotate"),
        "legged_lab.envs.elf3.walk_cfg": ("Elf3WalkFlatEnvCfg",),
        "legged_lab.envs.tienkung.run_cfg": ("TienKungRunFlatEnvCfg",),
        "legged_lab.envs.tienkung.run_with_sensor_cfg": ("TienKungRunWithSensorFlatEnvCfg",),
        "legged_lab.envs.tienkung.walk_cfg": ("TienKungWalkFlatEnvCfg",),
        "legged_lab.envs.tienkung.walk_with_sensor_cfg": ("TienKungWalkWithSensorFlatEnvCfg",),
        "legged_lab.utils.env_utils.scene": ("SceneCfg",),
        "rsl_rl.env": ("VecEnv",),
        "rsl_rl.utils": ("AMPLoaderDisplay",),
    }
    for name, symbols in dependencies.items():
        module(name, symbols)
    for name in list(modules):
        if name.startswith(("isaaclab.", "isaacsim.")):
            parts = name.split(".")
            for end in range(1, len(parts)):
                parent = module(".".join(parts[:end]))
                parent.__path__ = []
                setattr(parent, parts[end], module(".".join(parts[:end + 1])))
    classes = []
    with patch.dict(sys.modules, modules):
        for relative, class_name in (("elf3/elf3_env.py", "Elf3Env"),
                                     ("tienkung/tienkung_env.py", "TienKungEnv")):
            spec = importlib.util.spec_from_file_location(
                f"_terminal_amp_test_{class_name}", ROOT / "legged_lab/envs" / relative)
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
            classes.append(getattr(loaded, class_name))
    return classes


def make_environment(cls, done_rows):
    env = cls.__new__(cls)
    env.num_envs, env.device, env.headless = 3, "cpu", True
    env.physics_dt = env.step_dt = .02
    env.clip_actions, env.action_scale, env.sim_step_counter = 1., 1., 0
    env.cfg = SimpleNamespace(sim=SimpleNamespace(decimation=1),
                              scene=SimpleNamespace(depth_camera=SimpleNamespace(enable_depth_camera=False)))
    env.stair_settle_enabled = env.stair_step_enabled = env.step_skill_pretrain = False
    env.action_buffer = SimpleNamespace(compute=lambda actions: actions)
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.feet_cfg = SimpleNamespace(body_ids=[0, 1])
    env.feet_body_ids = [0, 1]
    env.robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(3, 29),
                                                     body_lin_vel_w=torch.zeros(3, 2, 3)),
                                set_joint_position_target=lambda actions: None)
    env.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(3, 2, 3)))
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None)
    env.amp_state = torch.arange(3 * 70, dtype=torch.float32).reshape(3, 70)
    env.sim = SimpleNamespace(step=lambda **kwargs: env.amp_state.add_(1.))
    env.command_generator = SimpleNamespace(compute=lambda dt: None)
    env.event_manager = SimpleNamespace(available_modes=[])
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.full((3,), 2.))
    env._calculate_gait_para = lambda: None
    env._restore_settle_commands = lambda: None
    env.done_rows = done_rows

    def check_reset():
        dones = torch.zeros(3, dtype=torch.bool)
        dones[env.done_rows] = True
        return dones, torch.zeros_like(dones)

    env.check_reset = check_reset
    env.get_amp_obs_for_expert_trans = lambda: env.amp_state

    def reset(ids):
        # Replace state in place, exactly the alias hazard the snapshot avoids.
        env.amp_state[ids] = -100.

    env.reset = reset
    env.compute_observations = lambda **kwargs: (env.amp_state.clone(), env.amp_state.clone())
    env.extras = {}
    return env


class TestAMPTerminalTransition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment_classes = load_environment_classes()

    def test_real_step_preserves_terminal_rows_before_reset(self):
        for cls in self.environment_classes:
            with self.subTest(environment=cls.__name__):
                env = make_environment(cls, [0, 2])
                expected_terminal = env.amp_state[[0, 2]].clone() + 1.
                obs, rewards, dones, infos = env.step(torch.zeros(3, 29))
                torch.testing.assert_close(infos["terminal_amp_env_ids"], torch.tensor([0, 2]))
                torch.testing.assert_close(infos["terminal_amp_observations"], expected_terminal)
                torch.testing.assert_close(obs[[0, 2]], torch.full((2, 70), -100.))
                torch.testing.assert_close(dones, torch.tensor([True, False, True]))
                torch.testing.assert_close(rewards, torch.full((3,), 2.))
                self.assertTrue(env.supports_amp_terminal_states)

    def test_real_step_without_termination_publishes_empty_contract(self):
        for cls in self.environment_classes:
            with self.subTest(environment=cls.__name__):
                env = make_environment(cls, [])
                expected = env.amp_state.clone() + 1.
                obs, _, dones, infos = env.step(torch.zeros(3, 29))
                self.assertEqual(infos["terminal_amp_env_ids"].shape, (0,))
                self.assertEqual(infos["terminal_amp_observations"].shape, (0, 70))
                self.assertEqual(infos["terminal_amp_env_ids"].dtype, torch.long)
                self.assertFalse(dones.any())
                torch.testing.assert_close(obs, expected)

    def test_snapshot_is_independent_and_next_step_clears_stale_rows(self):
        for cls in self.environment_classes:
            with self.subTest(environment=cls.__name__):
                env = make_environment(cls, [1])
                expected = env.amp_state[[1]].clone() + 1.
                _, _, _, infos = env.step(torch.zeros(3, 29))
                terminal = infos["terminal_amp_observations"]
                ids = infos["terminal_amp_env_ids"]
                env.amp_state.fill_(999.)
                env.reset_env_ids.fill_(0)
                torch.testing.assert_close(terminal, expected)
                torch.testing.assert_close(ids, torch.tensor([1]))
                env.done_rows = []
                _, _, _, next_infos = env.step(torch.zeros(3, 29))
                self.assertEqual(next_infos["terminal_amp_env_ids"].numel(), 0)
                self.assertEqual(next_infos["terminal_amp_observations"].shape, (0, 70))
                torch.testing.assert_close(terminal, expected)
                torch.testing.assert_close(ids, torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
