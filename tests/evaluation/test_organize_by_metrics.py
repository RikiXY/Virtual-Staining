from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from virtual_staining.evaluation.ranking import organize_by_metrics


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

    csv_path = tmp_path / "metrics.csv"
    df.to_csv(csv_path, index=False)
    results, summary_path, image_columns = organize_by_metrics(
        csv_path,
        output_dir,
        top_n=1,
        metrics=["ssim"],
        mode="copy",
    )

    assert results[0]["best_files"] == 1
    assert results[0]["worst_files"] == 1
    assert summary_path is not None and summary_path.exists()
    assert image_columns == ("generated_path",)
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

    csv_path = tmp_path / "metrics.csv"
    df.to_csv(csv_path, index=False)
    results, summary_path, image_columns = organize_by_metrics(
        csv_path,
        output_dir,
        top_n=1,
        metrics=["mae"],
        mode="copy",
    )

    assert results[0]["best_files"] == 1
    assert results[0]["worst_files"] == 1
    assert summary_path is not None and summary_path.exists()
    assert image_columns == ("generated_path",)
    assert (output_dir / "mae" / "best" / "0001_low_generated.png").exists()
    assert (output_dir / "mae" / "worst" / "0001_high_generated.png").exists()


def _custom_metric_csv(tmp_path: Path, *, with_metadata: bool) -> Path:
    paths = {name: tmp_path / "images" / f"{name}.png" for name in ("a", "b")}
    for path in paths.values():
        _touch(path)
    run_dir = tmp_path / "evaluation"
    run_dir.mkdir()
    csv_path = run_dir / "per_image_metrics.csv"
    pd.DataFrame(
        [
            {"sample_id": "a", "generated_path": str(paths["a"]), "bias": 0.1},
            {"sample_id": "b", "generated_path": str(paths["b"]), "bias": 0.9},
        ]
    ).to_csv(csv_path, index=False)
    if with_metadata:
        (run_dir / "evaluation_result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metrics": [{"name": "bias", "higher_is_better": False}],
                }
            ),
            encoding="utf-8",
        )
    return csv_path


def test_custom_metric_direction_comes_from_result_metadata(tmp_path: Path) -> None:
    csv_path = _custom_metric_csv(tmp_path, with_metadata=True)
    output_dir = tmp_path / "sorted"

    results, _, _ = organize_by_metrics(csv_path, output_dir, top_n=1, mode="copy")

    assert [result["metric"] for result in results] == ["bias"]
    assert (output_dir / "bias" / "best" / "0001_a_generated.png").exists()
    assert (output_dir / "bias" / "worst" / "0001_b_generated.png").exists()


def test_unknown_metric_without_metadata_needs_an_explicit_direction(tmp_path: Path) -> None:
    csv_path = _custom_metric_csv(tmp_path, with_metadata=False)
    output_dir = tmp_path / "sorted"

    with pytest.raises(ValueError, match="Ranking direction of 'bias' is unknown"):
        organize_by_metrics(csv_path, output_dir, metrics=["bias"], mode="copy")

    organize_by_metrics(
        csv_path, output_dir, top_n=1, metrics=["bias"], mode="copy", directions={"bias": True}
    )
    assert (output_dir / "bias" / "best" / "0001_b_generated.png").exists()
