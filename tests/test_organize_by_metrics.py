from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.organize_by_metrics import organize_metric


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
