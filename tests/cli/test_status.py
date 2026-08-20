from __future__ import annotations

from types import SimpleNamespace

import pytest

from virtual_staining.applications import status as status_app
from virtual_staining.cli import status as status_cli


def _report(*, healthy: bool = True) -> dict:
    return {
        "healthy": healthy,
        "virtual_staining": "0.1.0",
        "python": "3.11.0",
        "python_executable": "/python",
        "os": {
            "name": "WSL (Ubuntu)",
            "kernel": "Linux 6.6",
            "architecture": "x86_64",
            "bitness": "64-bit",
        },
        "memory": {
            "total": 16 * 1024**3,
            "available": 10 * 1024**3,
            "used": 6 * 1024**3,
            "percent_used": 37.5,
        },
        "git": {"commit": "abc123", "dirty": False},
        "packages": [
            {
                "name": "NumPy",
                "distribution": "numpy",
                "version": "2.0",
                "error": None if healthy else "ImportError: broken native library",
            }
        ],
        "openslide": {
            "version": "1.4",
            "library_version": None,
            "usable": False,
            "error": "native library not found",
        },
        "nvidia": {
            "executable": None,
            "usable": False,
            "gpus": [],
            "error": "nvidia-smi not found",
        },
        "cuda": {
            "build_version": None,
            "available": False,
            "cudnn_version": None,
            "devices": [],
            "error": None,
        },
    }


def test_status_prints_complete_report_and_only_required_failures_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(status_cli, "collect_status", lambda: _report())
    status_cli.main([])
    output = capsys.readouterr().out
    assert "WSL (Ubuntu)" in output
    assert "RAM available: 10.0 GiB" in output
    assert "OpenSlide" in output
    assert "NVIDIA driver" in output
    assert "required CPU pipeline dependencies are usable" in output

    monkeypatch.setattr(status_cli, "collect_status", lambda: _report(healthy=False))
    with pytest.raises(SystemExit) as exc:
        status_cli.main([])
    assert exc.value.code == 1


@pytest.mark.parametrize(
    ("system", "release", "expected"),
    [
        ("Linux", "6.6.0", "Ubuntu 24.04"),
        ("Linux", "6.6.0-microsoft-standard-WSL2", "WSL (Ubuntu 24.04)"),
        ("Darwin", "25.0.0", "macOS 15.0"),
        ("Windows", "11", "Windows 11 10.0.26100"),
    ],
)
def test_os_detection(
    system: str,
    release: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(status_app.platform, "system", lambda: system)
    monkeypatch.setattr(status_app.platform, "release", lambda: release)
    monkeypatch.setattr(
        status_app.platform, "freedesktop_os_release", lambda: {"PRETTY_NAME": "Ubuntu 24.04"}
    )
    monkeypatch.setattr(status_app.platform, "mac_ver", lambda: ("15.0", (), ""))
    monkeypatch.setattr(status_app.platform, "win32_ver", lambda: ("11", "10.0.26100", "", ""))
    monkeypatch.setattr(status_app.platform, "machine", lambda: "x86_64")

    result = status_app._os_status()

    assert result["name"] == expected
    assert result["architecture"] == "x86_64"
    assert result["bitness"].endswith("-bit")


def test_required_import_failure_is_reported_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(status_app, "REQUIRED_PACKAGES", (("Broken", "broken", "broken"),))
    monkeypatch.setattr(status_app, "_distribution_version", lambda name: "1.0")

    def fail_import(name: str) -> None:
        raise ImportError("libexample.so: cannot open shared object file")

    monkeypatch.setattr(status_app.importlib, "import_module", fail_import)

    packages, modules = status_app._required_packages()

    assert modules == {}
    assert packages[0]["error"] == ("ImportError: libexample.so: cannot open shared object file")


def test_openslide_and_cuda_smoke_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    detected: list[str] = []
    openslide = SimpleNamespace(
        __library_version__="4.0.0",
        OpenSlide=SimpleNamespace(detect_format=lambda path: detected.append(path)),
    )
    monkeypatch.setattr(status_app, "_distribution_version", lambda name: "1.4.0")
    monkeypatch.setattr(status_app.importlib, "import_module", lambda name: openslide)

    assert status_app._openslide_status()["usable"] is True
    assert detected

    properties = SimpleNamespace(name="GPU", total_memory=8 * 1024**3, major=8, minor=6)
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9100)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        ),
    )
    cuda = status_app._cuda_status(torch, None)
    assert cuda["available"] is True
    assert cuda["devices"] == [
        {"index": 0, "name": "GPU", "memory": 8 * 1024**3, "capability": "8.6"}
    ]
