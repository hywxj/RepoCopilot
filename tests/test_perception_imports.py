"""Standalone MuJoCo helpers must work without the policy training stack."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class PerceptionImportsTest(unittest.TestCase):
    def test_mujoco_helpers_do_not_import_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent("""
                import importlib
                import importlib.abc
                import sys

                class BlockTorch(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "torch" or fullname.startswith("torch."):
                            raise AssertionError("MuJoCo helpers must not import torch")

                sys.meta_path.insert(0, BlockTorch())
                for module, name in (
                    ("tread_surfaces", "SurfaceGeometryResult"),
                    ("stair_step_controller", "StepMeasurement"),
                    ("mujoco_support", "MujocoSupportReader"),
                ):
                    loaded = importlib.import_module("legged_lab.perception." + module)
                    assert getattr(loaded, name) is not None
                assert "torch" not in sys.modules
            """)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
