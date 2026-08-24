from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from virtual_staining.applications import prepare as prepare_app
from virtual_staining.data.pairs import SlidePair


def _config(root: Path, backend: str = "auto", *, tiled: bool = True) -> Any:
    return SimpleNamespace(
        preprocessing=SimpleNamespace(
            dataset_root=root,
            io=SimpleNamespace(tiled=tiled, backend=backend),
        )
    )


def _pairs(root: Path, count: int = 1) -> tuple[SlidePair, ...]:
    pairs = []
    for index in range(count):
        source = Path(f"raw/source_{index}.tif")
        target = Path(f"raw/target_{index}.tif")
        for path in (source, target):
            full_path = root / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(b"image")
        pairs.append(SlidePair(f"P{index}", source, target))
    return tuple(pairs)


def test_auto_backend_warns_when_pillow_will_be_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pairs = _pairs(tmp_path, 2)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: None)
    monkeypatch.setattr(prepare_app, "style", lambda text, color: f"{color}:{text}")

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path), pairs)

    message = caplog.messages[-1]
    assert message.startswith("yellow:")
    assert "Pillow" in message
    assert "vs convert /path/to/slides" in message
    assert "(+1 more)" in message


def test_explicit_pillow_recommends_openslide_for_compatible_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pairs = _pairs(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: "generic-tiff")

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path, "pillow"), pairs)

    assert "using Pillow" in caplog.messages[-1]
    assert "io.backend: openslide" in caplog.messages[-1]


def test_missing_openslide_explains_auto_pillow_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pairs = _pairs(tmp_path)
    monkeypatch.setattr(
        prepare_app,
        "detect_openslide_format",
        lambda path: (_ for _ in ()).throw(RuntimeError("missing")),
    )

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path), pairs)

    assert "Pillow because OpenSlide is unavailable" in caplog.messages[-1]
    assert "uv sync --extra wsi" in caplog.messages[-1]


def test_forced_openslide_warns_before_rejecting_incompatible_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pairs = _pairs(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: None)

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path, "openslide"), pairs)

    assert "cannot read them with OpenSlide" in caplog.messages[-1]
    assert "vs convert" in caplog.messages[-1]


@pytest.mark.parametrize(
    ("backend", "tiled"),
    [("openslide", True), ("auto", False)],
)
def test_backend_warning_is_suppressed_when_no_fallback_is_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    backend: str,
    tiled: bool,
) -> None:
    pairs = _pairs(tmp_path)
    monkeypatch.setattr(prepare_app, "detect_openslide_format", lambda path: "generic-tiff")

    with caplog.at_level(logging.WARNING, logger="virtual_staining.applications.prepare"):
        prepare_app._warn_image_backend(_config(tmp_path, backend, tiled=tiled), pairs)

    assert not caplog.messages
