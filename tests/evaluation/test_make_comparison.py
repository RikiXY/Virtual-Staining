from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.image_helpers import write_rgb_image, write_rgb_pair
from virtual_staining.applications.compare_panels import (
    ComparePanelsRequest,
    FromMetricsResult,
    compare_panels,
)
from virtual_staining.evaluation import diagnostics
from virtual_staining.evaluation.comparison import plot_paired_delta_histogram
from virtual_staining.evaluation.evaluator import EvaluationSample, evaluate_samples
from virtual_staining.evaluation.panels import (
    DiagnosticEntry,
    build_metric_case_artifacts,
    save_comparison_panel,
    save_metric_diagnostics_summary,
)
from virtual_staining.evaluation.selection import (
    infer_source_path_from_row,
    select_representative_rows,
)
from virtual_staining.metrics import (
    BUILTIN_METRIC_DEFINITIONS,
    MetricDefinition,
    MetricResult,
    resolve_metrics,
)


def test_save_diagnostic_plots_delegates_to_canonical_plotters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_id = "00000_00000"
    target = write_rgb_image(tmp_path / f"{sample_id}_target.png")
    generated = write_rgb_image(tmp_path / "HE" / f"{sample_id}_generated.png")
    called: list[str] = []

    def _record(name: str):
        def save(_target, _generated, output_path):
            called.append(name)
            return Path(output_path)

        return save

    monkeypatch.setattr(diagnostics, "_make_error_histogram", _record("error"))
    monkeypatch.setattr(diagnostics, "_make_scatter_by_channel", _record("scatter"))
    monkeypatch.setattr(diagnostics, "_make_intensity_overlay_histogram", _record("intensity"))

    paths = diagnostics.save_diagnostic_plots(generated, target, tmp_path / "diagnostics")

    assert called == ["error", "scatter", "intensity"]
    # Diagnostics are labelled by (sample_id, output_name), never by sample alone.
    assert [path.name for path in paths] == [
        f"{sample_id}__HE_error_histogram.png",
        f"{sample_id}__HE_target_vs_generated_scatter_by_channel.png",
        f"{sample_id}__HE_intensity_overlay_histogram.png",
    ]


def test_paired_delta_histogram_accepts_constant_small_deltas(tmp_path: Path) -> None:
    import numpy as np

    plot_paired_delta_histogram(np.array([0.1, 0.1]), "ssim", tmp_path)

    assert (tmp_path / "paired_delta_histogram.png").is_file()


def test_select_representative_rows_uses_higher_is_better_direction() -> None:
    rows = [
        {"sample_id": "low", "ssim": "0.10", "ssim_status": "finite"},
        {"sample_id": "mid", "ssim": "0.50", "ssim_status": "finite"},
        {"sample_id": "high", "ssim": "0.90", "ssim_status": "finite"},
        {"sample_id": "none", "ssim": "", "ssim_status": "unavailable"},
    ]
    selected = select_representative_rows(
        "ssim",
        {"finite_median": 0.5, "finite_min": 0.1, "finite_max": 0.9},
        rows,
        higher_is_better=True,
    )

    assert selected["best"]["sample_id"] == "high"
    assert selected["median"]["sample_id"] == "mid"
    assert selected["worst"]["sample_id"] == "low"


def test_select_representative_rows_uses_lower_is_better_direction() -> None:
    rows = [
        {"sample_id": "low", "mae": "0.10", "mae_status": "finite"},
        {"sample_id": "mid", "mae": "0.50", "mae_status": "finite"},
        {"sample_id": "high", "mae": "0.90", "mae_status": "finite"},
    ]
    selected = select_representative_rows(
        "mae",
        {"finite_median": 0.5, "finite_min": 0.1, "finite_max": 0.9},
        rows,
        higher_is_better=False,
    )

    assert selected["best"]["sample_id"] == "low"
    assert selected["median"]["sample_id"] == "mid"
    assert selected["worst"]["sample_id"] == "high"


def test_infer_source_path_supports_named_manifest_input_patches(tmp_path: Path) -> None:
    sample_id = "S1__x00000000_y00000000"
    split_dir = tmp_path / "splits" / "test" / "S1"
    source = write_rgb_image(split_dir / f"{sample_id}__input__unstained.png")
    target = write_rgb_image(split_dir / f"{sample_id}__target.png")

    assert (
        infer_source_path_from_row({"sample_id": sample_id, "target_path": str(target)}) == source
    )


def test_build_metric_case_artifacts_saves_panel_without_metric_suptitle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_id = "00000_00000"
    source_path, target_path = write_rgb_pair(tmp_path / "splits" / "test", sample_id)
    generated_path = write_rgb_image(
        tmp_path / "generated" / "HE" / f"{sample_id}_generated.png",
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
        metric_summary={"finite_min": 0.125, "finite_median": 0.5, "finite_max": 0.9},
        metric_dir=tmp_path / "comparisons" / "metrics" / "mae",
        higher_is_better=False,
    )

    assert seen_suptitles == [None]
    assert Path(str(selection_row["comparison_path"])).is_file()
    assert diagnostic_entry["comparison_path"].is_file()
    assert diagnostic_entry["error_histogram_path"].is_file()


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
        "virtual_staining.evaluation.panels._save_stacked_image_panel",
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


def _brightness(
    target: np.ndarray,
    generated: np.ndarray,
    support: np.ndarray | None,
    requested: Mapping[str, Mapping[str, Any]],
) -> dict[str, MetricResult]:
    return {"brightness": MetricResult.of(float(generated.mean()))}


def test_panels_rank_each_metric_by_its_recorded_direction(tmp_path: Path) -> None:
    run = tmp_path / "run"
    samples = []
    for index, shade in enumerate((10, 60, 120)):
        sample_id = f"s{index}"
        _, target = write_rgb_pair(tmp_path / "pairs", sample_id)
        generated = write_rgb_image(
            tmp_path / "generated" / "HE" / f"{sample_id}_generated.png",
            color=(shade, shade, shade),
        )
        samples.append(EvaluationSample(sample_id, "HE", "S1", target, generated))
    definitions = {
        "brightness": MetricDefinition("brightness", "1", "tests", _brightness, True),
        "mae": BUILTIN_METRIC_DEFINITIONS["mae"],
    }
    metrics = resolve_metrics([{"name": "brightness"}, {"name": "mae"}], definitions)
    evaluate_samples(samples, run / "evaluation", metrics=metrics)

    result = compare_panels(ComparePanelsRequest(mode="from_metrics", run_path=run))

    assert isinstance(result, FromMetricsResult)
    selected = result.per_metric_representative_rows
    assert result.available_metrics == ["HE/brightness", "HE/mae"]
    assert (
        selected["HE/brightness"]["best"]["sample_id"],
        selected["HE/mae"]["best"]["sample_id"],
    ) == ("s2", "s0")
