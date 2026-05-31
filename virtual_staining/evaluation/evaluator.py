from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from virtual_staining.evaluation.metrics import evaluate_pair
from virtual_staining.evaluation.reports import build_metric_row, write_per_image_metrics_csv

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    output_dir: Path
    metrics_csv: Path | None = None
    summary_csv: Path | None = None
    weak_tail_csv: Path | None = None
    residual_heatmaps_csv: Path | None = None
    num_evaluated: int = 0
    num_skipped: int = 0
    rows: list[dict[str, object]] = field(default_factory=list)
    skipped_rows: list[dict[str, str]] = field(default_factory=list)


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
