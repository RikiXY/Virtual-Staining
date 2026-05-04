from __future__ import annotations

import subprocess
import sys


def collect_environment() -> dict:
    """Return a snapshot of the current git commit hash and key dependency versions."""
    return {
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "torch": _pkg_version("torch"),
        "numpy": _pkg_version("numpy"),
        "opencv": _pkg_version("cv2"),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        import importlib
        return getattr(importlib.import_module(name), "__version__", None)
    except ImportError:
        return None
