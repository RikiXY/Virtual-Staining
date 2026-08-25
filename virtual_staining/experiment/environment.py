from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeInfo:
    git_commit: str | None
    git_dirty: bool | None
    python: str
    platform: str
    packages: Mapping[str, str | None]
    cuda_available: bool
    cuda_version: str | None
    gpu_names: tuple[str, ...]

    @classmethod
    def collect(cls, package_names: Sequence[str]) -> RuntimeInfo:
        git_commit, git_dirty = _git_state()
        packages = {name: _pkg_version(name) for name in package_names}
        cuda_available, cuda_version, gpu_names = _cuda_state()
        return cls(
            git_commit=git_commit,
            git_dirty=git_dirty,
            python=sys.version.split()[0],
            platform=platform.platform(),
            packages=packages,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_names=gpu_names,
        )


def collect_environment() -> dict[str, object]:
    runtime = RuntimeInfo.collect(("torch", "numpy", "cv2", "albumentations"))
    return {
        "git_commit": runtime.git_commit,
        "python": runtime.python,
        "platform": runtime.platform,
        "torch": runtime.packages.get("torch"),
        "numpy": runtime.packages.get("numpy"),
        "opencv": runtime.packages.get("cv2"),
        "albumentations": runtime.packages.get("albumentations"),
        "cuda_available": runtime.cuda_available,
        "cuda_version": runtime.cuda_version,
        "gpu_name": runtime.gpu_names[0] if runtime.gpu_names else None,
    }


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        if commit.returncode != 0 or dirty.returncode != 0:
            return None, None
        return commit.stdout.strip() or None, bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _git_commit() -> str | None:
    return _git_state()[0]


def _pkg_version(name: str) -> str | None:
    try:
        return getattr(importlib.import_module(name), "__version__", None)
    except (ImportError, AttributeError):
        return None


def _cuda_state() -> tuple[bool, str | None, tuple[str, ...]]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        names = tuple(
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        )
        return available, torch.version.cuda, names
    except Exception:
        return False, None, ()


def _torch_cuda_version() -> str | None:
    return _cuda_state()[1]


def _cuda_available() -> bool:
    return _cuda_state()[0]


def _gpu_name() -> str | None:
    names = _cuda_state()[2]
    return names[0] if names else None
