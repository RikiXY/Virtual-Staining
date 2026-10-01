from __future__ import annotations

from typing import Any

import numpy as np

from virtual_staining.data.filtering import foreground_ratios, is_valid_patch_pair


def _solid_bgr(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _solid_mask(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def _call(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    min_fg: float = 0.25,
    max_white: float = 0.7,
    white_threshold: int = 250,
    max_lw: float = 0.20,
) -> tuple[bool, dict[str, Any]]:
    return is_valid_patch_pair(
        source_img=src_bgr,
        target_img=tgt_bgr,
        source_mask=src_mask,
        target_mask=tgt_mask,
        min_foreground_ratio=min_fg,
        max_white_ratio=max_white,
        white_threshold=white_threshold,
        max_largest_white_component_ratio=max_lw,
    )


def test_foreground_ratios_reports_modalities_and_combined_policies() -> None:
    left = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    right = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    ratios = foreground_ratios({"left": left, "right": right})

    assert ratios == {
        "left": 0.5,
        "right": 0.5,
        "all": 0.5,
        "intersection": 0.25,
        "union": 0.75,
    }


def test_valid_pair_is_accepted() -> None:
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)

    valid, info = _call(tissue, tissue, mask, mask)

    assert valid is True
    assert info["reasons"] == []


def test_rejects_low_source_foreground() -> None:
    tissue = _solid_bgr(80)

    valid, info = _call(tissue, tissue, _solid_mask(0), _solid_mask(255))

    assert valid is False
    assert "low_source_foreground" in info["reasons"]


def test_rejects_low_target_foreground() -> None:
    tissue = _solid_bgr(80)

    valid, info = _call(tissue, tissue, _solid_mask(255), _solid_mask(0))

    assert valid is False
    assert "low_target_foreground" in info["reasons"]


def test_rejects_high_source_white_ratio() -> None:
    white = _solid_bgr(255)
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)

    valid, info = _call(white, tissue, mask, mask, max_white=0.3)

    assert valid is False
    assert "high_source_white_ratio" in info["reasons"]


def test_rejects_high_target_white_ratio() -> None:
    white = _solid_bgr(255)
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)

    valid, info = _call(tissue, white, mask, mask, max_white=0.3)

    assert valid is False
    assert "high_target_white_ratio" in info["reasons"]


def test_debug_info_contains_filter_metrics_and_reasons() -> None:
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)

    _, info = _call(tissue, tissue, mask, mask)

    assert set(info) == {
        "source_foreground_ratio",
        "target_foreground_ratio",
        "source_white_ratio",
        "target_white_ratio",
        "source_largest_white_component_ratio",
        "target_largest_white_component_ratio",
        "reasons",
    }
