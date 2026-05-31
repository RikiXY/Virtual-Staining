from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from virtual_staining.applications.organize import OrganizeRequest, organize
from virtual_staining.evaluation.ranking import (
    ORGANIZATION_SUMMARY_FIELDNAMES,
    organize_by_metrics,
    organize_metric,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("image", encoding="utf-8")


def test_organize_metric_exports_best_and_worst_for_higher_metric(
    tmp_path: Path,
) -> None:
    low_path = tmp_path / "images" / "low.png"
    high_path = tmp_path / "images" / "high.png"
    _touch(low_path)
    _touch(high_path)

    df = pd.DataFrame(
        [
            {"sample_id": "low", "generated_path": str(low_path), "ssim": 0.1},
            {"sample_id": "high", "generated_path": str(high_path), "ssim": 0.9},
        ]
    )
    output_dir = tmp_path / "sorted"

    result = organize_metric(
        df=df,
        metric="ssim",
        output_dir=output_dir,
        image_columns=["generated_path"],
        top_k=1,
        mode="copy",
        overwrite=False,
        include_all_ranked=False,
    )

    assert result is not None
    assert result["best_files"] == 1
    assert result["worst_files"] == 1
    assert (output_dir / "ssim" / "best" / "0001_high_generated.png").exists()
    assert (output_dir / "ssim" / "worst" / "0001_low_generated.png").exists()


def test_organize_metric_exports_best_and_worst_for_lower_metric(
    tmp_path: Path,
) -> None:
    low_path = tmp_path / "images" / "low.png"
    high_path = tmp_path / "images" / "high.png"
    _touch(low_path)
    _touch(high_path)

    df = pd.DataFrame(
        [
            {"sample_id": "low", "generated_path": str(low_path), "mae": 0.1},
            {"sample_id": "high", "generated_path": str(high_path), "mae": 0.9},
        ]
    )
    output_dir = tmp_path / "sorted"

    result = organize_metric(
        df=df,
        metric="mae",
        output_dir=output_dir,
        image_columns=["generated_path"],
        top_k=1,
        mode="copy",
        overwrite=False,
        include_all_ranked=False,
    )

    assert result is not None
    assert result["best_files"] == 1
    assert result["worst_files"] == 1
    assert (output_dir / "mae" / "best" / "0001_low_generated.png").exists()
    assert (output_dir / "mae" / "worst" / "0001_high_generated.png").exists()


def test_organize_metric_uses_sample_id_tie_breaker(tmp_path: Path) -> None:
    a_path = tmp_path / "images" / "a.png"
    b_path = tmp_path / "images" / "b.png"
    _touch(a_path)
    _touch(b_path)

    df = pd.DataFrame(
        [
            {"sample_id": "b", "generated_path": str(b_path), "ssim": 0.9},
            {"sample_id": "a", "generated_path": str(a_path), "ssim": 0.9},
        ]
    )
    output_dir = tmp_path / "sorted"

    result = organize_metric(
        df=df,
        metric="ssim",
        output_dir=output_dir,
        image_columns=["generated_path"],
        top_k=2,
        mode="copy",
        overwrite=False,
        include_all_ranked=False,
    )

    assert result is not None
    assert (output_dir / "ssim" / "best" / "0001_a_generated.png").exists()
    assert (output_dir / "ssim" / "best" / "0002_b_generated.png").exists()


@pytest.mark.parametrize("mode", ["copy", "symlink", "hardlink"])
def test_organize_metric_preserves_file_placement_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    low_path = tmp_path / "images" / "low.png"
    high_path = tmp_path / "images" / "high.png"
    _touch(low_path)
    _touch(high_path)

    df = pd.DataFrame(
        [
            {"sample_id": "low", "generated_path": str(low_path), "ssim": 0.1},
            {"sample_id": "high", "generated_path": str(high_path), "ssim": 0.9},
        ]
    )
    output_dir = tmp_path / "sorted"

    result = organize_metric(
        df=df,
        metric="ssim",
        output_dir=output_dir,
        image_columns=["generated_path"],
        top_k=1,
        mode=mode,
        overwrite=False,
        include_all_ranked=False,
    )

    assert result is not None
    assert result["best_files"] == 1
    assert result["worst_files"] == 1
    assert (output_dir / "ssim" / "best" / "0001_high_generated.png").exists()
    assert (output_dir / "ssim" / "worst" / "0001_low_generated.png").exists()


def test_organize_by_metrics_writes_ranked_export_summary(tmp_path: Path) -> None:
    rows = []
    for sample_id, metric_value in [("low", 0.1), ("high", 0.9)]:
        source_path = tmp_path / "images" / f"{sample_id}_source.png"
        target_path = tmp_path / "images" / f"{sample_id}_target.png"
        generated_path = tmp_path / "images" / f"{sample_id}_generated.png"
        _touch(source_path)
        _touch(target_path)
        _touch(generated_path)
        rows.append(
            {
                "sample_id": sample_id,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "generated_path": str(generated_path),
                "ssim": metric_value,
            }
        )

    csv_path = tmp_path / "per_image_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    output_dir = tmp_path / "sorted"

    organize_by_metrics(
        csv_path=csv_path,
        output_dir=output_dir,
        top_n=1,
        metrics=["ssim"],
        mode="copy",
        include_all_ranked=True,
    )

    summary = pd.read_csv(output_dir / "organization_summary.csv")

    assert list(summary.columns) == ORGANIZATION_SUMMARY_FIELDNAMES
    assert list(summary["kind"]) == ["best", "worst", "all_ranked"]
    assert list(summary["metric"]) == ["ssim", "ssim", "ssim"]
    assert list(summary["rank_count"]) == [1, 1, 2]
    assert list(summary["export_mode"]) == ["copy", "copy", "copy"]
    assert list(summary["selected_file_roles"]) == [
        "generated,target,source",
        "generated,target,source",
        "generated,target,source",
    ]
    assert list(summary["output_dir"]) == [
        str(output_dir / "ssim" / "best"),
        str(output_dir / "ssim" / "worst"),
        str(output_dir / "ssim" / "all_ranked"),
    ]
    assert list(summary["files_exported"]) == [3, 3, 6]


def test_organize_registers_ranked_export_artifacts(tmp_path: Path) -> None:
    run_path = tmp_path / "results" / "organize_run"
    evaluation_dir = run_path / "evaluation"
    image_dir = tmp_path / "images"
    rows = []
    for sample_id, metric_value in [("low", 0.1), ("high", 0.9)]:
        source_path = image_dir / f"{sample_id}_source.png"
        target_path = image_dir / f"{sample_id}_target.png"
        generated_path = image_dir / f"{sample_id}_generated.png"
        _touch(source_path)
        _touch(target_path)
        _touch(generated_path)
        rows.append(
            {
                "sample_id": sample_id,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "generated_path": str(generated_path),
                "ssim": metric_value,
            }
        )

    evaluation_dir.mkdir(parents=True)
    metrics_csv = evaluation_dir / "per_image_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    organize(
        OrganizeRequest(
            metrics_csv=metrics_csv,
            output_dir=evaluation_dir / "sorted_by_metrics",
            top_k=1,
            metrics=("ssim",),
            mode="copy",
            overwrite=False,
            include_all_ranked=True,
            run_path=run_path,
        )
    )

    manifest = json.loads((evaluation_dir / "artifacts.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    summary_artifact = next(
        artifact for artifact in artifacts if artifact["artifact_type"] == "organization_summary"
    )
    export_artifacts = [
        artifact for artifact in artifacts if artifact["artifact_type"] == "ranked_sample_export"
    ]

    assert summary_artifact["stage"] == "organize"
    assert summary_artifact["path"] == "evaluation/sorted_by_metrics/organization_summary.csv"
    assert summary_artifact["path_type"] == "run_relative"
    assert summary_artifact["metadata"]["command"] == "vs-organize"
    assert summary_artifact["metadata"]["source_run"] == "organize_run"
    assert summary_artifact["metadata"]["selected_metrics"] == ["ssim"]
    assert {artifact["metadata"]["kind"] for artifact in export_artifacts} == {
        "best",
        "worst",
        "all_ranked",
    }
    assert all(artifact["metric"] == "ssim" for artifact in export_artifacts)
    assert all(artifact["path_type"] == "run_relative" for artifact in export_artifacts)
    assert all(artifact["metadata"]["export_mode"] == "copy" for artifact in export_artifacts)


def test_organize_standalone_csv_skips_manifest_registration(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.png"
    _touch(image_path)
    metrics_csv = tmp_path / "per_image_metrics.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "sample",
                "generated_path": str(image_path),
                "ssim": 0.9,
            }
        ]
    ).to_csv(metrics_csv, index=False)

    organize(
        OrganizeRequest(
            metrics_csv=metrics_csv,
            output_dir=tmp_path / "exports",
            top_k=1,
            metrics=("ssim",),
            mode="copy",
            overwrite=False,
            include_all_ranked=False,
            run_path=None,
        )
    )

    assert not (tmp_path / "artifacts.json").exists()
    assert not (tmp_path / "exports" / "artifacts.json").exists()
