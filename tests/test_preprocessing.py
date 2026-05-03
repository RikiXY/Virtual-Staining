from __future__ import annotations

import numpy as np
import pytest

from virtual_staining.data.preprocessing import is_valid_patch_pair, split_items


# ---------------------------------------------------------------------------
# split_items
# ---------------------------------------------------------------------------

def test_split_items_covers_all_items():
    items = list(range(100))
    parts = split_items(items, [0.7, 0.15, 0.15])
    assert sum(len(p) for p in parts) == 100


def test_split_items_respects_ratios():
    items = list(range(100))
    parts = split_items(items, [0.8, 0.1, 0.1])
    assert len(parts[0]) == 80
    assert len(parts[1]) == 10
    assert len(parts[2]) == 10


def test_split_items_raises_on_single_ratio():
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [1.0])


def test_split_items_raises_on_sum_exceeds_one():
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [0.6, 0.6])


def test_split_items_raises_on_negative_ratio():
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [0.8, -0.1, 0.3])


# ---------------------------------------------------------------------------
# is_valid_patch_pair helpers
# ---------------------------------------------------------------------------

def _solid_bgr(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _solid_mask(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def _call(
    src_bgr,
    tgt_bgr,
    src_mask,
    tgt_mask,
    min_fg=0.25,
    max_white=0.7,
    white_threshold=250,
    max_lw=0.20,
):
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


# ---------------------------------------------------------------------------
# is_valid_patch_pair — acceptance
# ---------------------------------------------------------------------------

def test_valid_pair_is_accepted():
    tissue = _solid_bgr(80)     # dark BGR → not white
    mask = _solid_mask(255)     # fully foreground
    valid, info = _call(tissue, tissue, mask, mask)
    assert valid is True
    assert info["reasons"] == []


# ---------------------------------------------------------------------------
# is_valid_patch_pair — rejection reasons
# ---------------------------------------------------------------------------

def test_rejects_low_source_foreground():
    tissue = _solid_bgr(80)
    background_mask = _solid_mask(0)   # no foreground at all
    valid, info = _call(tissue, tissue, background_mask, _solid_mask(255))
    assert valid is False
    assert "low_source_foreground" in info["reasons"]


def test_rejects_low_target_foreground():
    tissue = _solid_bgr(80)
    valid, info = _call(tissue, tissue, _solid_mask(255), _solid_mask(0))
    assert valid is False
    assert "low_target_foreground" in info["reasons"]


def test_rejects_high_source_white_ratio():
    white = _solid_bgr(255)
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)
    valid, info = _call(white, tissue, mask, mask, max_white=0.3)
    assert valid is False
    assert "high_source_white_ratio" in info["reasons"]


def test_rejects_high_target_white_ratio():
    white = _solid_bgr(255)
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)
    valid, info = _call(tissue, white, mask, mask, max_white=0.3)
    assert valid is False
    assert "high_target_white_ratio" in info["reasons"]


# ---------------------------------------------------------------------------
# is_valid_patch_pair — debug_info keys
# ---------------------------------------------------------------------------

def test_debug_info_contains_required_keys():
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)
    _, info = _call(tissue, tissue, mask, mask)
    expected_keys = {
        "source_foreground_ratio",
        "target_foreground_ratio",
        "source_white_ratio",
        "target_white_ratio",
        "source_largest_white_component_ratio",
        "target_largest_white_component_ratio",
        "reasons",
    }
    assert expected_keys.issubset(info.keys())
