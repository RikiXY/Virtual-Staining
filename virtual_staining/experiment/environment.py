from __future__ import annotations

import importlib
import platform
import subprocess
import sys


def collect_environment() -> dict:
    """Return a snapshot of the current runtime, dependency, and hardware environment."""
    cuda_available = _cuda_available()
    cuda_version = _torch_cuda_version()
    gpu_name = _gpu_name() if cuda_available else None

    return {
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "numpy": _pkg_version("numpy"),
        "opencv": _pkg_version("cv2"),
        "albumentations": _pkg_version("albumentations"),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
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
        return getattr(importlib.import_module(name), "__version__", None)
    except ImportError:
        return None


def _torch_cuda_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except ImportError:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return None
