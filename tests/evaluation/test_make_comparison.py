from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from tests.image_helpers import write_rgb_image, write_rgb_pair
from virtual_staining.applications.render_panels import (
    FromMetricsResult,
    RenderPanelsRequest,
    render_panels,
)
from virtual_staining.evaluation.panels import (
    DiagnosticEntry,
    build_metric_case_artifacts,
    save_comparison_panel,
    save_metric_diagnostics_summary,
    save_residual_heatmap,
    select_representative_rows,
)
from virtual_staining.evaluation.ranking import organize_metric
from virtual_staining.evaluation.reports import write_per_image_metrics_csv
from virtual_staining.evaluation.summaries import write_summary_csv


def _metric_row(
    sample_id: str,
    target_path: Path,
    generated_path: Path,
    *,
    ssim: float,
    mae: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "width": 16,
        "height": 16,
        "channels": 3,
        "mae": mae,
        "mse": mae * mae,
        "rmse": mae,
        "psnr": 30.0,
        "ssim": ssim,
        "pcc_gray": 0.9,
        "pcc_r": 0.9,
        "pcc_g": 0.9,
        "pcc_b": 0.9,
        "pcc_rgb_mean": 0.9,
    }


def _write_render_panel_run(
    tmp_path: Path,
    metric_values: list[tuple[str, float, float]],
) -> Path:
    run_path = tmp_path / "results" / "ranked_run"
    target_dir = tmp_path / "data" / "splits" / "test"
    generated_dir = run_path / "artifacts" / "output_test"
    evaluation_dir = run_path / "evaluation"
    rows: list[dict[str, object]] = []

    for index, (sample_id, ssim, mae) in enumerate(metric_values, start=1):
        _, target_path = write_rgb_pair(target_dir, sample_id, color=(0, 0, 0))
        generated_path = write_rgb_image(
            generated_dir / f"{sample_id}_target_generated.png",
            color=(index * 32, index * 32, index * 32),
        )
        rows.append(
            _metric_row(
                sample_id,
                target_path,
                generated_path,
                ssim=ssim,
                mae=mae,
            )
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_per_image_metrics_csv(rows, evaluation_dir / "per_image_metrics.csv")
    write_summary_csv(rows, evaluation_dir)
    return run_path


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


def test_render_panels_top_k_one_preserves_default_panel_filenames(tmp_path: Path) -> None:
    run_path = _write_render_panel_run(
        tmp_path,
        [
            ("low", 0.10, 0.90),
            ("mid", 0.50, 0.50),
            ("high", 0.90, 0.10),
        ],
    )

    result = render_panels(
        RenderPanelsRequest(
            mode="from_metrics",
            run_path=run_path,
            metrics=("ssim",),
            top_k=1,
        )
    )

    assert isinstance(result, FromMetricsResult)
    metric_dir = run_path / "comparisons" / "metrics" / "ssim"
    assert (metric_dir / "best_high_comparison.png").is_file()
    assert (metric_dir / "median_mid_comparison.png").is_file()
    assert (metric_dir / "worst_low_comparison.png").is_file()
    assert not (metric_dir / "best_001_high_comparison.png").exists()

    with (metric_dir / "selection_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["kind"] for row in rows] == ["best", "median", "worst"]
    assert [row["rank"] for row in rows] == ["1", "1", "1"]
    assert [row["sample_id"] for row in rows] == ["high", "mid", "low"]


def test_render_panels_top_k_ranks_by_metric_direction(tmp_path: Path) -> None:
    run_path = _write_render_panel_run(
        tmp_path,
        [
            ("weak", 0.10, 0.10),
            ("strong", 0.90, 0.90),
            ("runner_up", 0.80, 0.80),
        ],
    )

    result = render_panels(
        RenderPanelsRequest(
            mode="from_metrics",
            run_path=run_path,
            metrics=("ssim", "mae"),
            kinds=("best", "worst"),
            top_k=2,
        )
    )

    assert isinstance(result, FromMetricsResult)

    ssim_dir = run_path / "comparisons" / "metrics" / "ssim"
    assert (ssim_dir / "best_001_strong_comparison.png").is_file()
    assert (ssim_dir / "best_002_runner_up_comparison.png").is_file()
    with (ssim_dir / "selection_summary.csv").open(encoding="utf-8", newline="") as handle:
        ssim_rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in ssim_rows if row["kind"] == "best"] == [
        "strong",
        "runner_up",
    ]

    mae_dir = run_path / "comparisons" / "metrics" / "mae"
    assert (mae_dir / "worst_001_strong_comparison.png").is_file()
    assert (mae_dir / "worst_002_runner_up_comparison.png").is_file()
    with (mae_dir / "selection_summary.csv").open(encoding="utf-8", newline="") as handle:
        mae_rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in mae_rows if row["kind"] == "worst"] == [
        "strong",
        "runner_up",
    ]

    with (run_path / "comparisons" / "metrics" / "metrics_selection_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        global_rows = list(csv.DictReader(handle))

    assert "rank" in global_rows[0]
    assert "comparison_path" in global_rows[0]
    assert "error_histogram_path" in global_rows[0]


def test_render_panels_from_metrics_registers_secondary_artifacts(tmp_path: Path) -> None:
    run_path = _write_render_panel_run(
        tmp_path,
        [
            ("weak", 0.10, 0.90),
            ("strong", 0.90, 0.10),
        ],
    )

    result = render_panels(
        RenderPanelsRequest(
            mode="from_metrics",
            run_path=run_path,
            metrics=("ssim",),
            kinds=("best",),
            top_k=1,
        )
    )

    assert isinstance(result, FromMetricsResult)
    artifact_manifest_path = result.artifact_manifest_path
    assert artifact_manifest_path is not None
    assert artifact_manifest_path == run_path / "evaluation" / "artifacts.json"

    manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    artifact_types = [artifact["artifact_type"] for artifact in artifacts]

    assert "selection_summary" in artifact_types
    assert "comparison_panel" in artifact_types
    assert "diagnostic_image" in artifact_types
    assert "diagnostic_panel" in artifact_types
    assert all(artifact["stage"] == "render_panels" for artifact in artifacts)
    assert all(artifact["path_type"] == "run_relative" for artifact in artifacts)

    global_summary = next(
        artifact
        for artifact in artifacts
        if artifact["artifact_type"] == "selection_summary"
        and artifact["path"] == "comparisons/metrics/metrics_selection_summary.csv"
    )
    assert global_summary["metadata"]["command"] == "vs-render-panels from-metrics"
    assert global_summary["metadata"]["source_run"] == "ranked_run"
    assert global_summary["metadata"]["selected_metrics"] == ["ssim"]

    comparison_panel = next(
        artifact for artifact in artifacts if artifact["artifact_type"] == "comparison_panel"
    )
    assert comparison_panel["path"] == "comparisons/metrics/ssim/best_strong_comparison.png"
    assert comparison_panel["metric"] == "ssim"
    assert comparison_panel["sample_id"] == "strong"
    assert comparison_panel["metadata"]["kind"] == "best"
    assert comparison_panel["metadata"]["rank"] == 1


def test_render_panels_and_organize_share_top_k_selection(tmp_path: Path) -> None:
    run_path = _write_render_panel_run(
        tmp_path,
        [
            ("weak", 0.10, 0.90),
            ("strong", 0.90, 0.10),
            ("runner_up", 0.80, 0.20),
        ],
    )

    panel_result = render_panels(
        RenderPanelsRequest(
            mode="from_metrics",
            run_path=run_path,
            metrics=("ssim",),
            kinds=("best",),
            top_k=2,
        )
    )
    assert isinstance(panel_result, FromMetricsResult)

    organize_dir = run_path / "evaluation" / "sorted_by_metrics"
    organization = organize_metric(
        df=pd.read_csv(run_path / "evaluation" / "per_image_metrics.csv"),
        metric="ssim",
        output_dir=organize_dir,
        image_columns=["generated_path"],
        top_k=2,
        mode="copy",
        overwrite=False,
        include_all_ranked=False,
    )

    assert organization is not None
    panel_sample_ids = [
        row["sample_id"] for row in panel_result.per_metric_ranked_rows["ssim"]["best"]
    ]
    organized_sample_ids = [
        path.name.split("_generated", maxsplit=1)[0].split("_", maxsplit=1)[1]
        for path in sorted((organize_dir / "ssim" / "best").glob("*_generated.png"))
    ]
    assert organized_sample_ids == panel_sample_ids


def test_render_panels_missing_source_image_fails_clearly(tmp_path: Path) -> None:
    run_path = tmp_path / "results" / "missing_source_run"
    target_dir = tmp_path / "data" / "splits" / "test"
    generated_dir = run_path / "artifacts" / "output_test"
    evaluation_dir = run_path / "evaluation"
    target_path = write_rgb_image(target_dir / "sample_target.png")
    generated_path = write_rgb_image(generated_dir / "sample_target_generated.png")
    rows = [
        _metric_row(
            "sample",
            target_path,
            generated_path,
            ssim=0.90,
            mae=0.10,
        )
    ]
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_per_image_metrics_csv(rows, evaluation_dir / "per_image_metrics.csv")
    write_summary_csv(rows, evaluation_dir)

    with pytest.raises(FileNotFoundError, match="Could not infer source path"):
        render_panels(
            RenderPanelsRequest(
                mode="from_metrics",
                run_path=run_path,
                metrics=("ssim",),
                kinds=("best",),
            )
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
