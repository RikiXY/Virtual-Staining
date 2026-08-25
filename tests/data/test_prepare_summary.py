from __future__ import annotations

import logging
from pathlib import Path

import pytest

from virtual_staining.applications.prepare import _log_prepare_summary
from virtual_staining.config.data import InputConfig, PreprocessingConfig
from virtual_staining.data.slide_sets import SlideAsset, SlideSet


def _config(root: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(
            Path("manifests/slide_sets.csv"),
            ("source", "aux"),
            "source",
            "target",
        ),
    )


def _sets() -> tuple[SlideSet, ...]:
    return (
        SlideSet(
            "P1",
            (
                SlideAsset("source", Path("slides/p1-source.tif"), already_aligned=True),
                SlideAsset("aux", Path("slides/p1-aux.tif"), already_aligned=True),
            ),
            SlideAsset("target", Path("slides/p1-target.tif"), already_aligned=True),
            "source",
        ),
        SlideSet(
            "P2",
            (
                SlideAsset("source", Path("slides/p2-source.tif"), already_aligned=True),
                SlideAsset("aux", Path("slides/p2-aux.tif"), already_aligned=False),
            ),
            SlideAsset("target", Path("slides/p2-target.tif"), already_aligned=True),
            "source",
        ),
    )


def test_prepare_summary_reports_slide_sets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.prepare"):
        _log_prepare_summary(_config(tmp_path), _sets(), reused=False)

    assert caplog.messages[0] == f"Prepare summary | dataset={tmp_path} | sets=2 | action=build"
    assert caplog.messages[1] == "Set P1 | inputs=source,aux | target=target | reference=source"
    assert caplog.messages[2] == "Set P2 | inputs=source,aux | target=target | reference=source"


def test_prepare_summary_reports_reuse(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.prepare"):
        _log_prepare_summary(_config(tmp_path), (_sets()[0],), reused=True)

    assert caplog.messages[0].endswith("sets=1 | action=reuse")
