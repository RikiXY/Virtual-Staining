from __future__ import annotations

import contextlib
import csv
import ctypes
import importlib
import io
import os
import platform
import shutil
import struct
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from virtual_staining.experiment.environment import RuntimeInfo

REQUIRED_PACKAGES = (
    ("Albumentations", "albumentations", "albumentations"),
    ("Matplotlib", "matplotlib", "matplotlib.pyplot"),
    ("NumPy", "numpy", "numpy"),
    ("OpenCV", "opencv-python-headless", "cv2"),
    ("pandas", "pandas", "pandas"),
    ("Pillow", "pillow", "PIL.Image"),
    ("PyYAML", "pyyaml", "yaml"),
    ("scikit-image", "scikit-image", "skimage.metrics"),
    ("SciPy", "scipy", "scipy.stats"),
    ("PyTorch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision.transforms"),
)


def _error_text(exc: BaseException, captured: str = "") -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        lines = [line.strip() for line in captured.splitlines() if line.strip()]
    return f"{type(exc).__name__}: {lines[-1] if lines else 'unknown error'}"


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _required_packages() -> tuple[list[dict[str, Any]], dict[str, ModuleType]]:
    results: list[dict[str, Any]] = []
    modules: dict[str, ModuleType] = {}
    for label, distribution, module_name in REQUIRED_PACKAGES:
        version = _distribution_version(distribution)
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                module = importlib.import_module(module_name)
            modules[module_name] = module
            error = None if version is not None else "Distribution metadata not found"
        except Exception as exc:
            error = _error_text(exc, captured.getvalue())
        results.append(
            {"name": label, "distribution": distribution, "version": version, "error": error}
        )
    return results, modules


def _os_status() -> dict[str, str]:
    system = platform.system() or "Unknown"
    release = platform.release() or "unknown"
    if system == "Linux":
        try:
            distribution = platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
        except OSError:
            distribution = "Linux"
        is_wsl = bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))
        is_wsl = is_wsl or "microsoft" in release.lower()
        name = f"WSL ({distribution})" if is_wsl else distribution
    elif system == "Darwin":
        name = f"macOS {platform.mac_ver()[0]}".strip()
    elif system == "Windows":
        windows_version = " ".join(value for value in platform.win32_ver()[:2] if value)
        name = f"Windows {windows_version}".strip()
    else:
        name = f"{system} {platform.version()}".strip()
    return {
        "name": name,
        "kernel": f"{system} {release}",
        "architecture": platform.machine() or "unknown",
        "bitness": f"{struct.calcsize('P') * 8}-bit",
    }


def _linux_memory() -> tuple[int, int] | None:
    try:
        values = {
            key: int(value.split()[0]) * 1024
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                if ":" in line
            )
            if key in {"MemTotal", "MemAvailable"}
        }
        return values["MemTotal"], values["MemAvailable"]
    except (KeyError, OSError, ValueError):
        return None


def _windows_memory() -> tuple[int, int] | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong),
            ("available", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        kernel32 = ctypes.windll.kernel32  # pyright: ignore[reportAttributeAccessIssue]
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.total), int(status.available)
    except (AttributeError, OSError):
        return None


def _unix_memory() -> tuple[int, int] | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (
            page_size * os.sysconf("SC_PHYS_PAGES"),
            page_size * os.sysconf("SC_AVPHYS_PAGES"),
        )
    except (OSError, ValueError):
        return None


def _memory_status() -> dict[str, int | float] | None:
    system = platform.system()
    values = (
        _windows_memory()
        if system == "Windows"
        else _linux_memory() or _unix_memory()
        if system == "Linux"
        else _unix_memory()
    )
    if values is None:
        return None
    total, available = values
    if total <= 0 or available < 0:
        return None
    used = max(0, total - available)
    return {
        "total": total,
        "available": available,
        "used": used,
        "percent_used": used / total * 100,
    }


def _openslide_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": _distribution_version("openslide-python"),
        "library_version": None,
        "usable": False,
        "error": None,
    }
    try:
        openslide = importlib.import_module("openslide")
        result["library_version"] = getattr(openslide, "__library_version__", None)
        openslide.OpenSlide.detect_format(str(Path(__file__)))
        result["usable"] = True
    except Exception as exc:
        result["error"] = _error_text(exc)
    return result


def _nvidia_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {"executable": executable, "usable": False, "gpus": [], "error": None}
    if executable is None:
        result["error"] = "nvidia-smi not found"
        return result
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0:
            output = completed.stderr or completed.stdout
            result["error"] = " ".join(output.split()) or "query failed"
            return result
        result["gpus"] = [
            {
                "index": int(index.strip()),
                "name": name.strip(),
                "driver": driver.strip(),
                "memory_mib": int(memory.strip()),
            }
            for index, name, driver, memory in csv.reader(completed.stdout.splitlines())
        ]
        result["usable"] = True
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["error"] = _error_text(exc)
    return result


def _cuda_status(torch: Any | None, import_error: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "build_version": None,
        "available": False,
        "cudnn_version": None,
        "devices": [],
        "error": import_error,
    }
    if torch is None:
        return result
    try:
        result["build_version"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["available"] = bool(torch.cuda.is_available())
        if result["available"]:
            result["devices"] = [
                {
                    "index": index,
                    "name": properties.name,
                    "memory": properties.total_memory,
                    "capability": f"{properties.major}.{properties.minor}",
                }
                for index in range(torch.cuda.device_count())
                for properties in [torch.cuda.get_device_properties(index)]
            ]
    except Exception as exc:
        result["error"] = _error_text(exc)
    return result


def collect_status() -> dict[str, Any]:
    """Collect a complete, non-mutating health report for the active runtime."""
    runtime = RuntimeInfo.collect(("torch", "numpy", "cv2", "albumentations"))
    packages, modules = _required_packages()
    package_modules = {
        "torch": "torch",
        "numpy": "numpy",
        "opencv-python-headless": "cv2",
        "albumentations": "albumentations",
    }
    for item in packages:
        module_name = package_modules.get(item["distribution"])
        if module_name is not None and module_name in runtime.packages:
            item["version"] = runtime.packages[module_name]
    torch_error = next(item["error"] for item in packages if item["distribution"] == "torch")
    try:
        package_version = metadata.version("virtual-staining")
    except metadata.PackageNotFoundError:
        from virtual_staining import __version__ as package_version

    cuda = _cuda_status(modules.get("torch"), torch_error)
    cuda["build_version"] = runtime.cuda_version
    cuda["available"] = runtime.cuda_available
    return {
        "healthy": all(item["error"] is None for item in packages),
        "virtual_staining": package_version,
        "python": runtime.python,
        "python_executable": sys.executable,
        "os": _os_status(),
        "memory": _memory_status(),
        "git": {"commit": runtime.git_commit, "dirty": runtime.git_dirty},
        "packages": packages,
        "openslide": _openslide_status(),
        "nvidia": _nvidia_status(),
        "cuda": cuda,
    }


__all__ = ["collect_status"]
