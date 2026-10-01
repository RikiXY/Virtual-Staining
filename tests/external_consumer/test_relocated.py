"""Copy only the consumer; run its full contract suite outside the repository."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def test_relocated_consumer():
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="mexina-relocated-", dir="/tmp") as directory:
        root = Path(directory)
        assert not root.is_relative_to(repo)
        consumer = root / "consumer"
        shutil.copytree(
            repo / "examples/external_consumer",
            consumer,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        # PYTHONPATH contains only the relocated consumer; MEXINA comes from the
        # managed environment's normal installation (which may be editable).
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTHON", "VS_", "MEXINA_"))
        }
        env.update(
            PYTHONPATH=str(consumer),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(root / "mpl"),
            CONSUMER_ORIGINAL_REPO=str(repo),
        )
        code = r"""
import os, sys, pathlib, unittest
repo = pathlib.Path(os.environ["CONSUMER_ORIGINAL_REPO"])
allowed = (repo / "virtual_staining", repo / ".venv")
def audit(event, args):
    if event in {"open", "os.listdir", "os.scandir"} and args:
        if isinstance(args[0], (str, bytes, os.PathLike)):
            path = pathlib.Path(os.path.abspath(os.fsdecode(args[0])))
            if path.is_relative_to(repo) and not any(path.is_relative_to(p) for p in allowed):
                raise AssertionError(f"Consumer accessed original repository data: {path}")
sys.addaudithook(audit)
import virtual_staining
print("Installed framework:", virtual_staining.__file__, flush=True)
suite = unittest.defaultTestLoader.discover("checks")
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert not any(name == "tests" or name.startswith("tests.") for name in sys.modules)
sys.exit(not result.wasSuccessful())
"""
        completed = subprocess.run(
            [sys.executable, "-P", "-c", code],
            cwd=consumer,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Ran 7 tests" in completed.stderr
