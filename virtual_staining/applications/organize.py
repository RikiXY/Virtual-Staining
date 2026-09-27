from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from virtual_staining.evaluation.ranking import organize_by_metrics
from virtual_staining.experiment.run_layout import RunLayout


@dataclass(frozen=True)
class OrganizeRequest:
    run_path: Path | None = None
    metrics_csv: Path | None = None
    output_dir: Path | None = None
    top_k: int = 20
    # None ranks the metrics recorded with the evaluation result (or the built-in defaults).
    metrics: tuple[str, ...] | None = None
    # Explicit ranking directions (True: higher is better); required for unknown metrics.
    directions: Mapping[str, bool] = field(default_factory=dict)
    mode: str = "hardlink"
    overwrite: bool = False
    include_all_ranked: bool = False


@dataclass(frozen=True)
class OrganizeResult:
    metrics_csv: Path
    output_dir: Path
    mode: str
    top_k: int
    metric_summaries: tuple[dict[str, Any], ...]
    summary_csv: Path | None
    image_columns: tuple[str, ...] = ()


def organize(request: OrganizeRequest) -> OrganizeResult:
    metrics_csv, output_dir = _resolve_paths(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, summary_csv, image_columns = organize_by_metrics(
        csv_path=metrics_csv,
        output_dir=output_dir,
        top_n=request.top_k,
        metrics=list(request.metrics) if request.metrics is not None else None,
        mode=request.mode,
        overwrite=request.overwrite,
        include_all_ranked=request.include_all_ranked,
        directions=request.directions,
    )
    return OrganizeResult(
        metrics_csv=metrics_csv,
        output_dir=output_dir,
        mode=request.mode,
        top_k=request.top_k,
        metric_summaries=tuple(summaries),
        summary_csv=summary_csv,
        image_columns=image_columns,
    )


def _resolve_paths(request: OrganizeRequest) -> tuple[Path, Path]:
    layout = RunLayout(request.run_path.resolve()) if request.run_path is not None else None
    metrics_csv = (
        request.metrics_csv.resolve()
        if request.metrics_csv is not None
        else layout.per_image_metrics
        if layout is not None
        else None
    )
    if metrics_csv is None:
        raise ValueError("You must provide either --run-path or --metrics-csv.")
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Could not find per_image_metrics.csv. Expected: {metrics_csv}")
    if request.output_dir is not None:
        return metrics_csv, request.output_dir.resolve()
    if layout is None:
        try:
            layout = RunLayout.from_evaluation_path(metrics_csv)
        except ValueError:
            raise ValueError(
                "Could not infer output directory. Please provide --output-dir explicitly."
            ) from None
    return metrics_csv, layout.evaluation_dir / "sorted_by_metrics"
