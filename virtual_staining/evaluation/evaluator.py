from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from virtual_staining.evaluation.reports import build_metric_row, write_per_image_metrics_csv
from virtual_staining.metrics import (
    compute_mae,
    compute_mse,
    compute_pcc_gray,
    compute_pcc_rgb,
    compute_psnr,
    compute_ssim,
)
from virtual_staining.utils.image_io import load_rgb_image, to_float01

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    output_dir: Path
    metrics_csv: Path | None = None
    summary_csv: Path | None = None
    num_evaluated: int = 0
    num_skipped: int = 0
    rows: list[dict[str, object]] = field(default_factory=list)
    skipped_rows: list[dict[str, str]] = field(default_factory=list)


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Loads an image pair and computes the standard metrics."""
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)
    if target.shape != generated.shape:
        raise ValueError(
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )

    target_float = to_float01(target)
    generated_float = to_float01(generated)
    mse = compute_mse(target_float, generated_float)
    pcc_r, pcc_g, pcc_b, pcc_rgb_mean = compute_pcc_rgb(target_float, generated_float)
    return {
        "mae": compute_mae(target_float, generated_float),
        "mse": mse,
        "rmse": float(mse**0.5),
        "psnr": compute_psnr(target_float, generated_float),
        "ssim": compute_ssim(target_float, generated_float),
        "pcc_gray": compute_pcc_gray(target_float, generated_float),
        "pcc_r": pcc_r,
        "pcc_g": pcc_g,
        "pcc_b": pcc_b,
        "pcc_rgb_mean": pcc_rgb_mean,
    }, target.shape


def evaluate_pairs(
    pairs: list[tuple[Path, Path, str]],
    output_dir: Path,
) -> EvaluationResult:
    """Computes image quality metrics for a list of target/generated pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = EvaluationResult(output_dir=output_dir)

    for target_path, generated_path, sample_id in pairs:
        if not generated_path.exists():
            logger.warning("Generated image not found, skipping: %s", generated_path)
            result.num_skipped += 1
            result.skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": "missing_generated",
                    "target_path": str(target_path),
                    "generated_path": str(generated_path),
                }
            )
            continue

        try:
            metrics, shape = evaluate_pair(target_path, generated_path)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", sample_id, exc)
            result.num_skipped += 1
            result.skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": str(exc),
                    "target_path": str(target_path),
                    "generated_path": str(generated_path),
                }
            )
            continue

        row = build_metric_row(sample_id, target_path, generated_path, shape, metrics)
        result.rows.append(row)
        result.num_evaluated += 1

    result.metrics_csv = write_per_image_metrics_csv(
        result.rows, output_dir / "per_image_metrics.csv"
    )
    return result
