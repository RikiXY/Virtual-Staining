from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from virtual_staining.evaluation.comparison import (
    plot_distribution_ecdf,
    plot_distribution_histogram,
    plot_paired_delta_histogram,
    plot_paired_scatter,
    save_paired_comparison_summary,
    save_paired_report_txt,
    save_paired_sample_deltas,
    save_paired_summary_json,
    save_unpaired_comparison_summary,
    save_unpaired_group_statistics,
    save_unpaired_report_txt,
    save_unpaired_summary_json,
)
from virtual_staining.evaluation.plotting import get_metric_plot_range
from virtual_staining.evaluation.statistics import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    align_paired_frames,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
    load_metric_values,
    resolve_input_csv,
)
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.metrics import get_metric_thresholds, is_higher_better_metric


@dataclass(frozen=True)
class CompareRequest:
    mode: Literal["paired", "unpaired"]
    run_a: Path | None = None
    run_b: Path | None = None
    csv_a: str | Path | None = None
    csv_b: str | Path | None = None
    label_a: str | None = None
    label_b: str | None = None
    column: str = "ssim"
    output_dir: Path | None = None
    higher_is_better: bool | None = None
    bins: int = 30
    min_value: float | None = None
    max_value: float | None = None
    thresholds: tuple[float, ...] | None = None
    tolerance: float = 0.0
    sample_id_column: str = "sample_id"


@dataclass(frozen=True)
class _ResolvedCompareRequest:
    mode: Literal["paired", "unpaired"]
    csv_a: Path
    csv_b: Path
    label_a: str
    label_b: str
    column: str
    output_dir: Path
    higher_is_better: bool
    bins: int
    min_value: float
    max_value: float
    thresholds: tuple[float, ...]
    tolerance: float
    sample_id_column: str


@dataclass
class CompareResult:
    mode: Literal["paired", "unpaired"]
    output_dir: Path
    column: str
    higher_is_better: bool
    group_a: UnpairedGroupStats | None = None
    group_b: UnpairedGroupStats | None = None
    unpaired_comparison: UnpairedComparison | None = None
    paired_summary: PairedSummary | None = None


def compare(request: CompareRequest) -> CompareResult:
    """Run the full comparison pipeline for paired or unpaired metric distributions."""
    if request.mode not in {"paired", "unpaired"}:
        raise ValueError(f"Unsupported comparison mode: {request.mode}")
    resolved = _resolve_request(request)
    resolved.output_dir.mkdir(parents=True, exist_ok=True)
    return _compare_unpaired(resolved) if resolved.mode == "unpaired" else _compare_paired(resolved)


def _resolve_request(request: CompareRequest) -> _ResolvedCompareRequest:
    csv_a = _resolve_csv(request.run_a, request.csv_a, "A")
    csv_b = _resolve_csv(request.run_b, request.csv_b, "B")
    label_a = request.label_a or _infer_label(request.run_a, request.csv_a, "A")
    label_b = request.label_b or _infer_label(request.run_b, request.csv_b, "B")
    default_min, default_max = get_metric_plot_range(request.column)
    min_value = request.min_value if request.min_value is not None else default_min
    max_value = request.max_value if request.max_value is not None else default_max
    if min_value == max_value:
        padding = 0.5 if min_value == 0 else abs(min_value) * 0.05
        min_value -= padding
        max_value += padding
    output_dir = request.output_dir or _default_output_dir(request, csv_a, label_a, label_b)
    return _ResolvedCompareRequest(
        mode=request.mode,
        csv_a=csv_a,
        csv_b=csv_b,
        label_a=label_a,
        label_b=label_b,
        column=request.column,
        output_dir=output_dir,
        higher_is_better=(
            request.higher_is_better
            if request.higher_is_better is not None
            else is_higher_better_metric(request.column)
        ),
        bins=request.bins,
        min_value=float(min_value),
        max_value=float(max_value),
        thresholds=(
            request.thresholds
            if request.thresholds is not None
            else tuple(get_metric_thresholds(request.column))
        ),
        tolerance=request.tolerance,
        sample_id_column=request.sample_id_column,
    )


def _resolve_csv(run_path: Path | None, csv_path: str | Path | None, label: str) -> Path:
    if run_path is None and csv_path is None:
        raise ValueError(f"You must provide either --run-{label.lower()} or --csv-{label.lower()}.")
    if run_path is None:
        assert csv_path is not None
        return resolve_input_csv(csv_path)
    run_layout = RunLayout(run_path.resolve())
    path = run_layout.per_image_metrics
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find per_image_metrics.csv for run '{run_layout.root.name}'. "
            f"Expected: {path}"
        )
    return path


def _infer_label(run_path: Path | None, csv_path: str | Path | None, fallback: str) -> str:
    if run_path is not None:
        return run_path.resolve().name
    if csv_path is None:
        return fallback
    path = Path(csv_path).resolve()
    if path.name == "per_image_metrics.csv" and path.parent.name in {"metrics", "evaluation"}:
        return path.parent.parent.name
    return path.stem


def _run_root_for_csv(path: Path) -> Path | None:
    resolved = path.resolve()
    if resolved.name == "per_image_metrics.csv" and resolved.parent.name in {
        "metrics",
        "evaluation",
    }:
        return resolved.parent.parent
    return None


def _default_output_dir(request: CompareRequest, csv_a: Path, label_a: str, label_b: str) -> Path:
    for run_path in (request.run_a, request.run_b):
        if run_path is not None:
            return (
                RunLayout(run_path.resolve().parent).comparisons_dir
                / f"{label_a}_vs_{label_b}"
                / f"{request.mode}_{request.column}"
            )
    run_root = _run_root_for_csv(csv_a)
    if run_root is not None:
        return (
            RunLayout(run_root).comparisons_dir
            / f"{label_a}_vs_{label_b}"
            / f"{request.mode}_{request.column}"
        )
    return (
        Path("local_workspace")
        / "results"
        / "comparisons"
        / f"{label_a}_vs_{label_b}"
        / f"{request.mode}_{request.column}"
    )


def _compare_unpaired(request: _ResolvedCompareRequest) -> CompareResult:
    values_a = load_metric_values(request.csv_a, request.column)
    values_b = load_metric_values(request.csv_b, request.column)
    thresholds = list(request.thresholds)

    group_a = compute_unpaired_group_stats(
        values=values_a,
        label=request.label_a,
        thresholds=thresholds,
        higher_is_better=request.higher_is_better,
    )
    group_b = compute_unpaired_group_stats(
        values=values_b,
        label=request.label_b,
        thresholds=thresholds,
        higher_is_better=request.higher_is_better,
    )
    comparison = compute_unpaired_comparison(
        a=values_a,
        b=values_b,
        group_a=group_a,
        group_b=group_b,
        higher_is_better=request.higher_is_better,
    )

    save_unpaired_group_statistics(group_a, group_b, request.output_dir)
    save_unpaired_comparison_summary(
        group_a,
        group_b,
        comparison,
        request.column,
        request.higher_is_better,
        request.output_dir,
    )
    save_unpaired_summary_json(group_a, group_b, comparison, request.output_dir)
    save_unpaired_report_txt(
        group_a,
        group_b,
        comparison,
        request.column,
        request.higher_is_better,
        request.output_dir,
    )

    edges = np.linspace(request.min_value, request.max_value, request.bins + 1)
    plot_distribution_histogram(
        values_a,
        values_b,
        edges,
        request.label_a,
        request.label_b,
        request.column,
        request.output_dir,
    )
    plot_distribution_ecdf(
        values_a,
        values_b,
        request.label_a,
        request.label_b,
        request.column,
        request.output_dir,
    )

    return CompareResult(
        mode="unpaired",
        output_dir=request.output_dir,
        column=request.column,
        higher_is_better=request.higher_is_better,
        group_a=group_a,
        group_b=group_b,
        unpaired_comparison=comparison,
    )


def _compare_paired(request: _ResolvedCompareRequest) -> CompareResult:
    merged = align_paired_frames(
        csv_a=request.csv_a,
        csv_b=request.csv_b,
        sample_id_column=request.sample_id_column,
        metric_column=request.column,
    )
    summary = compute_paired_summary(
        merged=merged,
        label_a=request.label_a,
        label_b=request.label_b,
        tolerance=request.tolerance,
        higher_is_better=request.higher_is_better,
    )

    save_paired_comparison_summary(
        summary,
        request.column,
        request.higher_is_better,
        request.output_dir,
    )
    save_paired_sample_deltas(
        merged,
        request.higher_is_better,
        request.tolerance,
        request.label_a,
        request.label_b,
        request.output_dir,
    )
    save_paired_summary_json(summary, request.output_dir)
    save_paired_report_txt(summary, request.column, request.higher_is_better, request.output_dir)

    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)
    signed_delta = raw_delta if request.higher_is_better else -raw_delta
    values_a = merged["value_a"].to_numpy(dtype=float)
    values_b = merged["value_b"].to_numpy(dtype=float)

    edges = np.linspace(request.min_value, request.max_value, request.bins + 1)
    plot_distribution_histogram(
        values_a,
        values_b,
        edges,
        request.label_a,
        request.label_b,
        request.column,
        request.output_dir,
    )
    plot_distribution_ecdf(
        values_a,
        values_b,
        request.label_a,
        request.label_b,
        request.column,
        request.output_dir,
    )
    plot_paired_delta_histogram(signed_delta, request.column, request.output_dir)
    plot_paired_scatter(
        merged,
        request.label_a,
        request.label_b,
        request.column,
        request.output_dir,
    )

    return CompareResult(
        mode="paired",
        output_dir=request.output_dir,
        column=request.column,
        higher_is_better=request.higher_is_better,
        paired_summary=summary,
    )
