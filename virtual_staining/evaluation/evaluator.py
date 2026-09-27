"""Paired evaluation of explicit target/generated records with a resolved metric request.

Only known input problems (:class:`EvaluationInputError`) are per-sample coverage events:
``strict`` (default) records them and fails the operation, ``permissive`` excludes those
samples and continues. Any other exception (metric defects, backend or programming
errors) propagates in both modes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.config.evaluation import InputFailureMode
from virtual_staining.evaluation.reports import (
    COVERAGE_CSV,
    PER_IMAGE_METRICS_CSV,
    build_metric_row,
    metric_fieldnames,
    write_coverage_csv,
    write_evaluation_result,
    write_per_image_metrics_csv,
)
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.metrics import (
    MetricResult,
    ResolvedMetric,
    check_valid_region_support,
    compute_metrics,
    default_metrics,
)
from virtual_staining.utils.image_io import to_float01

logger = logging.getLogger(__name__)


class EvaluationInputError(ValueError):
    """A known per-sample input problem; ``reason`` is a stable machine-readable code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class EvaluationCoverageError(RuntimeError):
    """The evaluation did not produce a publishable result; see ``coverage_csv``."""

    def __init__(self, message: str, coverage_csv: Path) -> None:
        super().__init__(f"{message}. Coverage diagnostics: {coverage_csv}")
        self.coverage_csv = coverage_csv


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    set_id: str
    target_path: Path
    generated_path: Path
    # Optional binary valid-region mask on the target/generated pixel grid.
    support_path: Path | None = None


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics_csv: Path
    summary_csv: Path
    coverage_csv: Path
    result_json: Path
    metrics: tuple[ResolvedMetric, ...]
    num_requested: int
    num_evaluated: int
    num_excluded: int
    rows: tuple[dict[str, object], ...]
    coverage_rows: tuple[dict[str, str], ...]


def _read_image(path: Path, role: str) -> Image.Image:
    if not path.is_file():
        raise EvaluationInputError(f"missing_{role}", f"{role} file not found: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()
    except OSError as exc:
        raise EvaluationInputError(
            f"unreadable_{role}", f"cannot read {role} {path}: {exc}"
        ) from exc


def load_rgb(path: Path, role: str) -> np.ndarray:
    """Load an 8-bit RGB image as-is; other modes are rejected, never converted."""
    image = _read_image(path, role)
    if image.mode != "RGB":
        raise EvaluationInputError(
            f"unsupported_{role}_mode", f"{role} {path} has mode {image.mode!r}, expected 'RGB'"
        )
    return np.asarray(image)


def load_support(path: Path, grid: tuple[int, int]) -> np.ndarray:
    """Load an explicitly binary support mask: mode ``1``, or mode ``L`` holding only 0/255."""
    image = _read_image(path, "support")
    array = np.asarray(image)
    if image.mode == "1":
        support = array.astype(bool)
    elif image.mode == "L" and np.isin(array, (0, 255)).all():
        support = array == 255
    else:
        raise EvaluationInputError(
            "malformed_support",
            f"support {path} must be a binary mask (mode '1', or mode 'L' with only 0 and "
            f"255); got mode {image.mode!r}. Soft or continuous masks are not thresholded.",
        )
    if support.shape != grid:
        raise EvaluationInputError(
            "support_shape_mismatch",
            f"support {path} is {support.shape[::-1]} but the images are {grid[::-1]} (W, H)",
        )
    return support


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
    *,
    metrics: Sequence[ResolvedMetric] | None = None,
    support_path: str | Path | None = None,
) -> tuple[dict[str, MetricResult], tuple[int, int, int]]:
    """Compute the requested metrics (default: the built-in default set) for one pair."""
    requested = tuple(metrics) if metrics is not None else default_metrics()
    if support_path is not None:
        check_valid_region_support(requested)
    target = load_rgb(Path(target_path), "target")
    generated = load_rgb(Path(generated_path), "generated")
    if target.shape != generated.shape:
        raise EvaluationInputError(
            "shape_mismatch",
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}.",
        )
    support = load_support(Path(support_path), target.shape[:2]) if support_path else None
    shape = (target.shape[0], target.shape[1], target.shape[2])
    return compute_metrics(requested, to_float01(target), to_float01(generated), support), shape


def evaluate_samples(
    samples: Sequence[EvaluationSample],
    output_dir: Path,
    *,
    metrics: Sequence[ResolvedMetric] | None = None,
    input_failures: InputFailureMode = "strict",
) -> EvaluationResult:
    """Evaluate every sample and write per-image, summary, coverage and result metadata.

    Raises :class:`EvaluationCoverageError` (after writing ``coverage.csv``) when a known
    input failure occurs in ``strict`` mode or no sample could be evaluated.
    """
    requested = tuple(metrics) if metrics is not None else default_metrics()
    if not requested:
        raise ValueError("Evaluation requires at least one requested metric")
    if not samples:
        raise ValueError("Evaluation requires at least one sample")
    if input_failures not in ("strict", "permissive"):
        raise ValueError(f"input_failures must be 'strict' or 'permissive', got {input_failures!r}")
    with_support = {sample.support_path is not None for sample in samples}
    if len(with_support) > 1:
        raise ValueError("Valid-region support must be supplied for every sample or for none")
    use_support = with_support == {True}
    if use_support:
        check_valid_region_support(requested)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    coverage: list[dict[str, str]] = []
    for sample in samples:
        entry = {
            "sample_id": sample.sample_id,
            "set_id": sample.set_id,
            "status": "evaluated",
            "reason": "",
            "detail": "",
            "target_path": str(sample.target_path),
            "generated_path": str(sample.generated_path),
            "support_path": str(sample.support_path) if sample.support_path else "",
        }
        try:
            results, shape = evaluate_pair(
                sample.target_path,
                sample.generated_path,
                metrics=requested,
                support_path=sample.support_path,
            )
        except EvaluationInputError as exc:
            logger.warning("Evaluation input failure for %s: %s", sample.sample_id, exc)
            status = "failed" if input_failures == "strict" else "excluded"
            coverage.append({**entry, "status": status, "reason": exc.reason, "detail": str(exc)})
            continue
        coverage.append(entry)
        rows.append(
            build_metric_row(
                sample.sample_id,
                sample.target_path,
                sample.generated_path,
                shape,
                results,
                set_id=sample.set_id,
                support_path=sample.support_path,
            )
        )

    coverage_csv = write_coverage_csv(coverage, output_dir / COVERAGE_CSV)
    counts = {
        status: sum(entry["status"] == status for entry in coverage)
        for status in ("evaluated", "excluded", "failed")
    }
    if counts["failed"]:
        raise EvaluationCoverageError(
            f"{counts['failed']} of {len(samples)} samples had input failures "
            "(input_failures='strict')",
            coverage_csv,
        )
    if not rows:
        raise EvaluationCoverageError(
            f"No sample could be evaluated ({counts['excluded']} excluded)", coverage_csv
        )

    names = [metric.name for metric in requested]
    metrics_csv = write_per_image_metrics_csv(
        rows, metric_fieldnames(names, support=use_support), output_dir / PER_IMAGE_METRICS_CSV
    )
    summary_csv = write_summary_csv(rows, names, output_dir)
    result_json = write_evaluation_result(
        output_dir,
        requested,
        counts={"requested": len(samples), **counts},
        input_failures=input_failures,
        valid_region_support=use_support,
    )
    return EvaluationResult(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
        coverage_csv=coverage_csv,
        result_json=result_json,
        metrics=requested,
        num_requested=len(samples),
        num_evaluated=counts["evaluated"],
        num_excluded=counts["excluded"],
        rows=tuple(rows),
        coverage_rows=tuple(coverage),
    )
