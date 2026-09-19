from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from virtual_staining.applications import prepare as prepare_app
from virtual_staining.data.slide_sets import SlideAsset, SlideSet


def _config(root: Path, backend: str = "auto", *, tiled: bool = True) -> Any:
    return SimpleNamespace(
        preprocessing=SimpleNamespace(
            dataset_root=root,
            io=SimpleNamespace(tiled=tiled, backend=backend),
        )
    )


def _sets(root: Path) -> tuple[SlideSet, ...]:
    assets = (
        Path("raw/source.tif"),
        Path("raw/aux.tif"),
        Path("raw/target.tif"),
    )
    for path in assets:
        full_path = root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(b"image")
    return (
        SlideSet(
            "P1",
            (
                SlideAsset("source", assets[0], already_aligned=True),
                SlideAsset("aux", assets[1], already_aligned=True),
            ),
            SlideAsset("target", assets[2], already_aligned=True),
            "source",
        ),
    )


@pytest.mark.parametrize("backend", ["auto", "openslide"])
@pytest.mark.parametrize("error_type", [ImportError, OSError, RuntimeError])
def test_backend_check_propagates_broken_openslide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, error_type: type[Exception]
) -> None:
    slide_sets = _sets(tmp_path)

    def fail(path: Path) -> None:
        raise error_type("broken OpenSlide runtime")

    monkeypatch.setattr(prepare_app, "detect_openslide_format", fail)
    with pytest.raises(error_type, match="broken OpenSlide runtime"):
        prepare_app._warn_image_backend(_config(tmp_path, backend), slide_sets)


def test_forced_openslide_warns_for_incompatible_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    slide_sets = _sets(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: None)

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path, "openslide"), slide_sets)

    assert not caplog.messages[-1].startswith("\x1b")
    assert "cannot use the requested backend" in caplog.messages[-1]


@pytest.mark.parametrize(
    ("backend", "tiled"),
    [("auto", False), ("auto", True)],
)
def test_backend_warning_is_suppressed_for_compatible_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    backend: str,
    tiled: bool,
) -> None:
    slide_sets = _sets(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: "generic-tiff")

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path, backend, tiled=tiled), slide_sets)

    assert not caplog.messages


def test_auto_backend_accepts_unsupported_formats_without_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    slide_sets = _sets(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: None)
    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path), slide_sets)
    assert not caplog.messages
