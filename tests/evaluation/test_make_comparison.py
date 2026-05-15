from __future__ import annotations

from virtual_staining.evaluation.panels import select_representative_rows


def test_select_representative_rows_uses_higher_is_better_direction() -> None:
    rows = [
        {"sample_id": "low", "ssim": "0.10"},
        {"sample_id": "mid", "ssim": "0.50"},
        {"sample_id": "high", "ssim": "0.90"},
    ]
    selected = select_representative_rows(
        "ssim",
        {"median": 0.5, "min": 0.1, "max": 0.9},
        rows,
    )

    assert selected["best"]["sample_id"] == "high"
    assert selected["median"]["sample_id"] == "mid"
    assert selected["worst"]["sample_id"] == "low"


def test_select_representative_rows_uses_lower_is_better_direction() -> None:
    rows = [
        {"sample_id": "low", "mae": "0.10"},
        {"sample_id": "mid", "mae": "0.50"},
        {"sample_id": "high", "mae": "0.90"},
    ]
    selected = select_representative_rows(
        "mae",
        {"median": 0.5, "min": 0.1, "max": 0.9},
        rows,
    )

    assert selected["best"]["sample_id"] == "low"
    assert selected["median"]["sample_id"] == "mid"
    assert selected["worst"]["sample_id"] == "high"
