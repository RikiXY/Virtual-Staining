from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.evaluation.artifact_readers import (
    MalformedArtifactError,
    read_comparison_artifacts,
    read_evaluation_run_artifacts,
    read_paired_evaluation_artifacts,
)
from virtual_staining.evaluation.artifacts import EvaluationArtifact, write_artifact_manifest
from virtual_staining.evaluation.comparison import (
    DECISION_BREAKDOWN_COLUMNS,
    PAIRED_DELTA_SUMMARY_COLUMNS,
)
from virtual_staining.evaluation.reports import METRIC_FIELDNAMES, write_per_image_metrics_csv
from virtual_staining.evaluation.summaries import write_summary_csv, write_weak_tail_csv


def _metric_row(sample_id: str = "sample_a") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "target_path": f"/data/{sample_id}_target.png",
        "generated_path": f"/runs/{sample_id}_generated.png",
        "width": 16,
        "height": 16,
        "channels": 3,
        "mae": 0.05,
        "mse": 0.0025,
        "rmse": 0.05,
        "psnr": 28.0,
        "ssim": 0.91,
        "pcc_gray": 0.94,
        "pcc_r": 0.95,
        "pcc_g": 0.93,
        "pcc_b": 0.92,
        "pcc_rgb_mean": 0.93,
    }


def _write_csv(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",".join(fieldnames)
        + "\n"
        + ",".join(str(row.get(fieldname, "")) for fieldname in fieldnames)
        + "\n",
        encoding="utf-8",
    )


def _write_run(tmp_path: Path, run_name: str = "reader_run") -> Path:
    run_root = tmp_path / "results" / run_name
    evaluation_dir = run_root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    rows = [_metric_row()]
    metrics_csv = write_per_image_metrics_csv(rows, evaluation_dir / "per_image_metrics.csv")
    summary_csv = write_summary_csv(rows, evaluation_dir)
    weak_tail_csv = write_weak_tail_csv(rows, evaluation_dir)
    write_artifact_manifest(
        [
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="per_image_metrics_csv",
                path=metrics_csv,
                description="Per-image evaluation metrics.",
            ),
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="summary_csv",
                path=summary_csv,
                description="Aggregate summary.",
            ),
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="weak_tail_csv",
                path=weak_tail_csv,
                description="Weak-tail summary.",
            ),
        ],
        evaluation_dir / "artifacts.json",
        run_root=run_root,
        created_at="2026-06-01T08:00:00+00:00",
    )
    return run_root


def _tree_snapshot(root: Path) -> list[tuple[str, bool, int, int]]:
    snapshot: list[tuple[str, bool, int, int]] = []
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        snapshot.append(
            (
                path.relative_to(root).as_posix(),
                path.is_file(),
                stat.st_size if path.is_file() else 0,
                stat.st_mtime_ns,
            )
        )
    return snapshot


def test_read_evaluation_run_artifacts_loads_core_metrics_and_manifest(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path)

    artifacts = read_evaluation_run_artifacts(run_root)

    assert artifacts.manifest.status == "present"
    assert artifacts.manifest.created_at == "2026-06-01T08:00:00+00:00"
    assert artifacts.per_image_metrics.status == "present"
    assert artifacts.per_image_metrics.rows[0]["sample_id"] == "sample_a"
    assert artifacts.summary.status == "present"
    assert {row["metric"] for row in artifacts.summary.rows} >= {"mae", "ssim"}
    assert artifacts.weak_tail.status == "present"
    assert artifacts.residual_heatmaps.status == "missing"

    metrics_record = next(
        record
        for record in artifacts.manifest.records
        if record.artifact_type == "per_image_metrics_csv"
    )
    assert metrics_record.path == "evaluation/per_image_metrics.csv"
    assert (
        metrics_record.resolved_path
        == (run_root / "evaluation" / "per_image_metrics.csv").resolve()
    )


def test_read_evaluation_run_artifacts_reports_missing_optional_artifacts(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "minimal_run"
    evaluation_dir = run_root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    write_per_image_metrics_csv([_metric_row()], evaluation_dir / "per_image_metrics.csv")

    artifacts = read_evaluation_run_artifacts(run_root)

    assert artifacts.manifest.status == "missing"
    assert artifacts.summary.status == "missing"
    assert artifacts.weak_tail.status == "missing"
    assert artifacts.residual_heatmaps.status == "missing"
    assert artifacts.skipped.status == "missing"
    assert artifacts.metric_selection_summaries[0].status == "missing"
    assert artifacts.organization_summary.status == "missing"


def test_read_evaluation_run_artifacts_handles_empty_metrics_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "empty_run"
    evaluation_dir = run_root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    write_per_image_metrics_csv([], evaluation_dir / "per_image_metrics.csv")

    artifacts = read_evaluation_run_artifacts(run_root)

    assert artifacts.per_image_metrics.status == "empty"
    assert artifacts.per_image_metrics.rows == ()
    assert artifacts.per_image_metrics.fieldnames == tuple(METRIC_FIELDNAMES)


def test_read_evaluation_run_artifacts_rejects_malformed_required_metrics_csv(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "malformed_run"
    evaluation_dir = run_root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "per_image_metrics.csv").write_text(
        "sample_id,ssim\nsample_a,0.91\n",
        encoding="utf-8",
    )

    with pytest.raises(MalformedArtifactError, match="missing required columns"):
        read_evaluation_run_artifacts(run_root)


def test_artifact_readers_do_not_write_to_run_directory(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path)
    before = _tree_snapshot(run_root)

    read_evaluation_run_artifacts(run_root)

    assert _tree_snapshot(run_root) == before


def test_read_comparison_artifacts_reports_present_and_missing_reports(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparisons" / "run_a_vs_run_b" / "paired_mae"
    _write_csv(
        comparison_dir / "paired_decision_breakdown.csv",
        DECISION_BREAKDOWN_COLUMNS,
        {
            "mode": "paired",
            "criterion": "mean_signed_delta",
            "description": "Mean direction-aware paired delta.",
            "label_a": "run_a",
            "label_b": "run_b",
            "value_a": 0.0,
            "value_b": 0.1,
            "signed_difference": 0.1,
            "weight": 1.0,
            "favors": "run_b",
            "score_a": 0.0,
            "score_b": 1.0,
        },
    )
    _write_csv(
        comparison_dir / "paired_delta_summary.csv",
        PAIRED_DELTA_SUMMARY_COLUMNS,
        {
            "mode": "paired",
            "metric": "mae",
            "direction": "lower_is_better",
            "label_a": "run_a",
            "label_b": "run_b",
            "n_pairs": 1,
            "tolerance": 0.0,
            "mean_signed_delta": 0.1,
            "median_signed_delta": 0.1,
            "signed_delta_q10": 0.1,
            "signed_delta_q25": 0.1,
            "signed_delta_q50": 0.1,
            "signed_delta_q75": 0.1,
            "signed_delta_q90": 0.1,
            "mean_relative_signed_delta": 0.5,
            "median_relative_signed_delta": 0.5,
            "relative_delta_count": 1,
            "share_b_better": 1.0,
            "share_a_better": 0.0,
            "share_equal": 0.0,
            "score_a": 0.0,
            "score_b": 5.0,
            "score_difference": 5.0,
            "total_score": 5.0,
            "decision_strength": "strong",
            "decision_reason": "run_b has a strong score-based advantage.",
            "better_label": "run_b",
        },
    )

    artifacts = read_comparison_artifacts(comparison_dir)

    assert artifacts.paired_decision_breakdown.status == "present"
    assert artifacts.paired_decision_breakdown.rows[0]["favors"] == "run_b"
    assert artifacts.paired_delta_summary.status == "present"
    assert artifacts.comparison_summary.status == "missing"
    assert artifacts.unpaired_decision_breakdown.status == "missing"


def test_read_paired_evaluation_artifacts_can_attach_comparison_reports(
    tmp_path: Path,
) -> None:
    run_a = _write_run(tmp_path, run_name="run_a")
    run_b = _write_run(tmp_path, run_name="run_b")
    comparison_dir = tmp_path / "comparisons" / "run_a_vs_run_b" / "paired_all_metrics"
    _write_csv(
        comparison_dir / "paired_metric_delta_summary.csv",
        [
            "metric",
            "direction",
            "label_a",
            "label_b",
            "tolerance",
            "total_common_count",
            "finite_pair_count",
            "missing_pair_count",
            "improved_count",
            "worsened_count",
            "equal_count",
            "improved_share",
            "worsened_share",
            "equal_share",
            "mean_raw_delta_b_minus_a",
            "median_raw_delta_b_minus_a",
            "mean_signed_delta",
            "median_signed_delta",
        ],
        {
            "metric": "ssim",
            "direction": "higher_is_better",
            "label_a": "run_a",
            "label_b": "run_b",
            "tolerance": 0.0,
            "total_common_count": 1,
            "finite_pair_count": 1,
            "missing_pair_count": 0,
            "improved_count": 1,
            "worsened_count": 0,
            "equal_count": 0,
            "improved_share": 1.0,
            "worsened_share": 0.0,
            "equal_share": 0.0,
            "mean_raw_delta_b_minus_a": 0.1,
            "median_raw_delta_b_minus_a": 0.1,
            "mean_signed_delta": 0.1,
            "median_signed_delta": 0.1,
        },
    )

    paired = read_paired_evaluation_artifacts(
        run_a,
        run_b,
        comparison_dir=comparison_dir,
    )

    assert paired.run_a.per_image_metrics.status == "present"
    assert paired.run_b.per_image_metrics.status == "present"
    assert paired.comparison is not None
    assert paired.comparison.paired_metric_delta_summary.status == "present"
