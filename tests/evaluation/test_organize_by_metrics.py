from __future__ import annotations

from pathlib import Path

import pandas as pd

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
