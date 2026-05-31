from __future__ import annotations

from pathlib import Path

import pytest

from tests.image_helpers import write_rgb_image, write_rgb_pair
from virtual_staining.evaluation.panels import (
    DiagnosticEntry,
    build_metric_case_artifacts,
    save_comparison_panel,
    save_metric_diagnostics_summary,
    save_residual_heatmap,
    select_representative_rows,
)


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


def test_build_metric_case_artifacts_saves_panel_without_metric_suptitle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_id = "00000_00000"
    source_path, target_path = write_rgb_pair(tmp_path / "splits" / "test", sample_id)
    generated_path = write_rgb_image(
        tmp_path / "generated" / f"{sample_id}_target_generated.png",
        color=(32, 64, 96),
    )
    seen_suptitles: list[str | None] = []

    def _recording_save_comparison_panel(*args, **kwargs) -> Path:
        seen_suptitles.append(kwargs["suptitle"])
        return save_comparison_panel(*args, **kwargs)

    monkeypatch.setattr(
        "virtual_staining.evaluation.panels.save_comparison_panel",
        _recording_save_comparison_panel,
    )

    selection_row, diagnostic_entry = build_metric_case_artifacts(
        metric_name="mae",
        kind="best",
        row={
            "sample_id": sample_id,
            "mae": "0.125",
            "source_path": str(source_path),
            "target_path": str(target_path),
            "generated_path": str(generated_path),
        },
        metric_summary={"min": 0.125, "median": 0.5, "max": 0.9},
        metric_dir=tmp_path / "comparisons" / "metrics" / "mae",
    )

    assert seen_suptitles == [None]
    assert Path(str(selection_row["comparison_path"])).is_file()
    assert diagnostic_entry["comparison_path"].is_file()
    assert diagnostic_entry["error_histogram_path"].is_file()


def test_save_residual_heatmap_writes_file_for_same_size_pair(tmp_path: Path) -> None:
    target_path = write_rgb_image(tmp_path / "target.png", color=(0, 0, 0))
    generated_path = write_rgb_image(tmp_path / "generated.png", color=(255, 255, 255))

    saved_path = save_residual_heatmap(
        target_path=target_path,
        generated_path=generated_path,
        save_path=tmp_path / "heatmaps" / "sample_residual_heatmap.png",
    )

    assert saved_path.is_file()


def test_save_residual_heatmap_raises_for_size_mismatch(tmp_path: Path) -> None:
    target_path = write_rgb_image(tmp_path / "target.png", size=(16, 16))
    generated_path = write_rgb_image(tmp_path / "generated.png", size=(8, 8))

    with pytest.raises(ValueError, match="same size"):
        save_residual_heatmap(
            target_path=target_path,
            generated_path=generated_path,
            save_path=tmp_path / "heatmap.png",
        )


def test_save_metric_diagnostics_summary_labels_best_median_worst_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries: list[DiagnosticEntry] = []
    for kind, value in (("best", 0.1), ("median", 0.5), ("worst", 0.9)):
        image_path = write_rgb_image(tmp_path / f"{kind}.png")
        entries.append(
            {
                "kind": kind,
                "sample_id": f"{kind}_sample",
                "metric_value": value,
                "comparison_path": image_path,
                "error_histogram_path": image_path,
                "intensity_overlay_histogram_path": image_path,
                "target_vs_generated_scatter_by_channel_path": image_path,
            }
        )
    seen_row_titles: list[list[str] | None] = []

    def _recording_save_stacked_image_panel(*, image_paths, save_path, row_titles, suptitle):
        seen_row_titles.append(row_titles)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"placeholder")
        return save_path

    monkeypatch.setattr(
        "virtual_staining.evaluation.panels.save_stacked_image_panel",
        _recording_save_stacked_image_panel,
    )

    saved_paths = save_metric_diagnostics_summary(
        metric_name="mae",
        metric_dir=tmp_path / "metric",
        diagnostic_entries=entries,
    )

    assert len(saved_paths) == 4
    assert all(path.is_file() for path in saved_paths)
    assert all(
        row_titles
        == [
            "BEST | sample=best_sample | mae=0.100000",
            "MEDIAN | sample=median_sample | mae=0.500000",
            "WORST | sample=worst_sample | mae=0.900000",
        ]
        for row_titles in seen_row_titles
    )
