from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from virtual_staining.applications.prepare import _log_prepare_summary
from virtual_staining.data.config import (
    AlignmentConfig,
    InputConfig,
    IOConfig,
    MaskConfig,
    PatchingConfig,
    PreprocessingConfig,
)
from virtual_staining.data.pairs import SlidePair


def _config(root: Path, **changes: object) -> PreprocessingConfig:
    config = PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(Path("inputs/pairs.csv"), "source", "target"),
        patching=PatchingConfig(patch_size=(256, 128), grid_movement=(128, 64), margin=12),
        io=IOConfig(tiled=True, backend="openslide"),
        masks=MaskConfig(generation="if_missing", strategy="connected_components", scale=0.25),
        alignment=AlignmentConfig(mode="auto"),
    )
    return dataclasses.replace(config, **changes)


def test_prepare_summary_reports_mixed_pair_actions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pairs = (
        SlidePair("P1", Path("s1.tif"), Path("t1.tif"), already_aligned=False),
        SlidePair(
            "P2",
            Path("s2.tif"),
            Path("t2.tif"),
            already_aligned=True,
            shared_mask_path=Path("m2.tif"),
        ),
        SlidePair(
            "P3",
            Path("s3.tif"),
            Path("t3.tif"),
            source_mask_path=Path("s3-mask.tif"),
            target_mask_path=Path("t3-mask.tif"),
        ),
    )

    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.prepare"):
        _log_prepare_summary(_config(tmp_path), pairs, reused=False)

    assert "pairs=3 | action=build | io=openslide/tiled" in caplog.messages[0]
    assert "patch=256x128 | stride=128x64 | margin=12" in caplog.messages[0]
    assert (
        "aligned=no | alignment=affine_sift | masks=generate connected_components@0.25x"
        in caplog.messages[1]
    )
    assert "aligned=yes | alignment=identity | masks=shared m2.tif" in caplog.messages[2]
    assert "masks=separate source=s3-mask.tif,target=t3-mask.tif" in caplog.messages[3]


@pytest.mark.parametrize(
    ("mode", "declared", "expected"),
    [
        ("always", True, "affine_sift"),
        ("auto", True, "identity"),
        ("auto", None, "affine_sift"),
        ("never", False, "identity"),
    ],
)
def test_prepare_summary_reports_alignment_action(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    declared: bool | None,
    expected: str,
) -> None:
    config = _config(tmp_path, alignment=AlignmentConfig(mode=mode))
    pair = SlidePair("P1", Path("source.tif"), Path("target.tif"), declared)

    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.prepare"):
        _log_prepare_summary(config, (pair,), reused=True)

    assert "action=reuse" in caplog.messages[0]
    assert f"alignment={expected}" in caplog.messages[1]


def test_prepare_summary_reports_maskless_and_conflicting_inputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    maskless = _config(tmp_path, masks=MaskConfig(generation="never"))
    conflicting = _config(tmp_path, masks=MaskConfig(generation="always"))
    plain = SlidePair("P1", Path("source.tif"), Path("target.tif"))
    supplied = dataclasses.replace(plain, shared_mask_path=Path("mask.tif"))

    with caplog.at_level(logging.INFO, logger="virtual_staining.applications.prepare"):
        _log_prepare_summary(maskless, (plain,), reused=False)
        _log_prepare_summary(conflicting, (supplied,), reused=False)

    assert any(message.endswith("masks=none") for message in caplog.messages)
    assert any(
        message.endswith("masks=conflict: supplied with generation=always")
        for message in caplog.messages
    )
