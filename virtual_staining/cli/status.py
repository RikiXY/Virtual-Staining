from __future__ import annotations

import argparse
from typing import Any

from virtual_staining.applications.status import collect_status
from virtual_staining.cli._output import print_info, print_section, style

_LEVEL_COLORS = {"OK": "green", "WARN": "yellow", "ERROR": "red", "INFO": "cyan"}


def _bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _line(level: str, label: str, value: str) -> None:
    marker = style(f"[{level}]", "bold", _LEVEL_COLORS[level])
    print(f"{marker} {style(label + ':', 'bold')} {value}")


def _print_report(report: dict[str, Any]) -> None:
    os_status = report["os"]
    print_section("System")
    print_info("Virtual Staining", report["virtual_staining"])
    print_info("Python", f"{report['python']} ({report['python_executable']})")
    print_info("Operating system", os_status["name"])
    print_info("Kernel", os_status["kernel"])
    print_info("Architecture", f"{os_status['architecture']}, {os_status['bitness']}")

    memory = report["memory"]
    if memory is None:
        _line("WARN", "RAM", "unavailable")
    else:
        print_info("RAM total", _bytes(memory["total"]))
        print_info("RAM used", f"{_bytes(memory['used'])} ({memory['percent_used']:.1f}%)")
        print_info("RAM available", _bytes(memory["available"]))

    git = report["git"]
    if git["commit"] is None:
        _line("INFO", "Git", "not a Git checkout or Git is unavailable")
    else:
        print_info("Git commit", git["commit"])
        print_info("Working tree", "dirty" if git["dirty"] else "clean")

    print_section("Required dependencies")
    for package in report["packages"]:
        version = package["version"] or "not installed"
        if package["error"] is None:
            _line("OK", package["name"], version)
        else:
            _line("ERROR", package["name"], f"{version} — {package['error']}")

    print_section("OpenSlide (optional)")
    openslide = report["openslide"]
    if openslide["usable"]:
        _line(
            "OK",
            "OpenSlide",
            f"Python {openslide['version']}, native {openslide['library_version'] or 'unknown'}",
        )
    else:
        version = openslide["version"] or "not installed"
        _line("WARN", "OpenSlide", f"Python {version} — {openslide['error']}")

    print_section("GPU and drivers (optional)")
    nvidia = report["nvidia"]
    if nvidia["usable"]:
        _line("OK", "NVIDIA driver", nvidia["executable"])
        for gpu in nvidia["gpus"]:
            print_info(
                f"GPU {gpu['index']}",
                f"{gpu['name']} — driver {gpu['driver']}, {gpu['memory_mib']} MiB",
            )
    else:
        _line("WARN", "NVIDIA driver", nvidia["error"] or "unavailable")

    cuda = report["cuda"]
    if cuda["available"]:
        _line("OK", "PyTorch CUDA", f"build {cuda['build_version']}")
        print_info("cuDNN", str(cuda["cudnn_version"] or "unavailable"))
        for device in cuda["devices"]:
            print_info(
                f"CUDA device {device['index']}",
                f"{device['name']} — {_bytes(device['memory'])}, capability {device['capability']}",
            )
    else:
        detail = cuda["error"] or (
            f"unavailable (PyTorch CUDA build {cuda['build_version']})"
            if cuda["build_version"]
            else "unavailable (CPU-only PyTorch build)"
        )
        _line("WARN", "PyTorch CUDA", detail)

    print_section("Result")
    if report["healthy"]:
        _line("OK", "Runtime", "required CPU pipeline dependencies are usable")
    else:
        _line("ERROR", "Runtime", "one or more required dependencies are unusable")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vs status", description="Check runtime dependencies and optional hardware support."
    )
    parser.parse_args(argv)
    report = collect_status()
    _print_report(report)
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
