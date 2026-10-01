from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
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
                "name": label,
                "distribution": distribution,
                "version": "1.0",
                "library_version": "4.0" if module in {"openslide", "pyvips"} else None,
                "error": None if healthy else "ImportError: broken native library",
            }
            for label, distribution, module in status_app.REQUIRED_PACKAGES
        ],
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
    assert "[OK] OpenSlide: 1.0, native 4.0" in output
    assert "[OK] pyvips: 1.0, native 4.0" in output
    assert "OpenSlide (optional)" not in output
    assert "GPU and drivers (optional)" in output
    assert "NVIDIA driver" in output
    assert "required Python dependencies and native WSI libraries are usable" in output

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


def test_cuda_smoke_check() -> None:
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


def test_required_package_checks_match_project_dependencies() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert {item[1] for item in status_app.REQUIRED_PACKAGES} == {
        requirement.split(">=")[0] for requirement in project["dependencies"]
    }
    assert "wsi" not in project.get("optional-dependencies", {})


@pytest.mark.parametrize(
    ("broken", "failure"),
    [
        (None, None),
        ("albumentations", "import"),
        ("openslide", "import"),
        ("pyvips", "import"),
        ("openslide", "native"),
        ("pyvips", "native"),
        ("pyvips", "metadata"),
    ],
)
def test_runtime_health_requires_wsi_but_not_gpu(
    broken: str | None,
    failure: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from virtual_staining.experiment import environment

    checked: list[str] = []

    def native_check(name: str) -> None:
        checked.append(name)
        if name == broken and failure == "native":
            raise OSError("native call failed")

    def import_module(name: str) -> object:
        if name == broken and failure == "import":
            raise ImportError("required package missing")
        if name == "openslide":
            return SimpleNamespace(
                __library_version__="4.0.0",
                OpenSlide=SimpleNamespace(detect_format=lambda path: native_check(name)),
            )
        if name == "pyvips":
            return SimpleNamespace(
                version=lambda index: (8, 18, 0)[index],
                Image=SimpleNamespace(
                    black=lambda w, h: SimpleNamespace(avg=lambda: native_check(name))
                ),
            )
        if name == "torch":
            return SimpleNamespace(
                version=SimpleNamespace(cuda=None),
                backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None)),
                cuda=SimpleNamespace(is_available=lambda: False),
            )
        return SimpleNamespace()

    monkeypatch.setattr(status_app.importlib, "import_module", import_module)
    monkeypatch.setattr(
        status_app,
        "_distribution_version",
        lambda name: None if name == broken and failure == "metadata" else "1.0",
    )
    monkeypatch.setattr(status_app.shutil, "which", lambda name: None)
    monkeypatch.setattr(environment, "_cuda_state", lambda: (False, None, ()))
    report = status_app.collect_status()
    assert report["healthy"] is (broken is None)
    assert report["cuda"]["available"] is False
    assert report["nvidia"]["usable"] is False
    if broken is None:
        assert checked == ["openslide", "pyvips"]
    else:
        package = next(item for item in report["packages"] if item["error"])
        assert package["name"] in {"Albumentations", "OpenSlide", "pyvips"}
        assert failure != "native" or "native call failed" in package["error"]
    monkeypatch.setattr(status_cli, "collect_status", lambda: report)
    if broken is None:
        status_cli.main([])
    else:
        with pytest.raises(SystemExit) as exc:
            status_cli.main([])
        assert exc.value.code == 1
        assert "[ERROR] Runtime" in capsys.readouterr().out


def test_status_entrypoint_reports_missing_packages_before_pipeline_imports() -> None:
    modules = {item[2].split(".")[0] for item in status_app.REQUIRED_PACKAGES}
    script = f"""
import sys
class BrokenDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {modules!r}:
            raise OSError('simulated broken required dependency')
sys.meta_path.insert(0, BrokenDependencies())
from virtual_staining.cli import main
main(['status'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 1
    assert "[ERROR] OpenSlide:" in result.stdout
    assert "[ERROR] pyvips:" in result.stdout
    assert "[ERROR] Runtime:" in result.stdout
    assert "Traceback" not in result.stderr
