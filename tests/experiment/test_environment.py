from __future__ import annotations

from importlib import metadata

import pytest

from virtual_staining.experiment import environment


def test_environment_records_required_wsi_versions_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, "_cuda_state", lambda: (False, None, ()))
    result = environment.collect_environment()
    assert result["openslide"] == metadata.version("openslide-python")
    assert result["pyvips"] == metadata.version("pyvips")
    assert result["cuda_available"] is False
    assert result["cuda_version"] is None
    assert result["gpu_name"] is None


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_package_provenance_remains_best_effort(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    def fail(name: str) -> None:
        raise error_type("broken runtime")

    monkeypatch.setattr(environment.importlib, "import_module", fail)
    assert environment._pkg_version("pyvips") is None
