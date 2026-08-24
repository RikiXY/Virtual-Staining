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


def test_auto_backend_warns_when_openslide_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    slide_sets = _sets(tmp_path)
    monkeypatch.setattr(
        prepare_app,
        "detect_openslide_format",
        lambda path: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    monkeypatch.setattr(prepare_app, "style", lambda text, color: f"{color}:{text}")

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path), slide_sets)

    assert caplog.messages[-1].startswith("yellow:")
    assert "Pillow because OpenSlide is unavailable" in caplog.messages[-1]


def test_forced_openslide_warns_for_incompatible_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    slide_sets = _sets(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: None)
    monkeypatch.setattr(prepare_app, "style", lambda text, color: f"{color}:{text}")

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path, "openslide"), slide_sets)

    assert caplog.messages[-1].startswith("yellow:")
    assert "cannot use the requested backend" in caplog.messages[-1]


@pytest.mark.parametrize(
    ("backend", "tiled"),
    [("auto", False), ("auto", True)],
)
def test_backend_warning_is_suppressed_when_no_fallback_is_needed(
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
