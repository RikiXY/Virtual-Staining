from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from virtual_staining.evaluation.reports import (
    build_metric_row,
    write_per_image_metrics_csv,
    write_skipped_csv,
)
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.metrics import compute_standard_metrics
from virtual_staining.utils.image_io import load_rgb_image, to_float01

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    set_id: str
    target_path: Path
    generated_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics_csv: Path
    summary_csv: Path | None
    skipped_csv: Path | None
    num_evaluated: int
    num_skipped: int
    rows: tuple[dict[str, object], ...]
    skipped_rows: tuple[dict[str, str], ...]


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Load one image pair and compute the standard metrics."""
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)
    if target.shape != generated.shape:
        raise ValueError(
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )
    return compute_standard_metrics(to_float01(target), to_float01(generated)), target.shape


def evaluate_samples(samples: Sequence[EvaluationSample], output_dir: Path) -> EvaluationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for sample in samples:
        if not sample.target_path.exists() or not sample.generated_path.exists():
            reason = "missing_target" if not sample.target_path.exists() else "missing_generated"
            skipped.append(
                {
                    "sample_id": sample.sample_id,
                    "reason": reason,
                    "target_path": str(sample.target_path),
                    "generated_path": str(sample.generated_path),
                }
            )
            continue
        try:
            metrics, shape = evaluate_pair(sample.target_path, sample.generated_path)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", sample.sample_id, exc)
            skipped.append(
                {
                    "sample_id": sample.sample_id,
                    "reason": str(exc),
                    "target_path": str(sample.target_path),
                    "generated_path": str(sample.generated_path),
                }
            )
            continue
        rows.append(
            build_metric_row(
                sample.sample_id,
                sample.target_path,
                sample.generated_path,
                shape,
                metrics,
                set_id=sample.set_id,
            )
        )

    metrics_csv = write_per_image_metrics_csv(rows, output_dir / "per_image_metrics.csv")
    summary_csv = write_summary_csv(rows, output_dir) if rows else None
    skipped_csv = write_skipped_csv(skipped, output_dir / "skipped.csv") if skipped else None
    return EvaluationResult(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
        skipped_csv=skipped_csv,
        num_evaluated=len(rows),
        num_skipped=len(skipped),
        rows=tuple(rows),
        skipped_rows=tuple(skipped),
    )
