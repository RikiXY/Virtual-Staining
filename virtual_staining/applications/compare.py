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
from virtual_staining.evaluation.statistics import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    align_paired_frames,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
    load_metric_values,
)


@dataclass(frozen=True)
class CompareRequest:
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
    group_a: UnpairedGroupStats | None = None
    group_b: UnpairedGroupStats | None = None
    unpaired_comparison: UnpairedComparison | None = None
    paired_summary: PairedSummary | None = None


def compare(request: CompareRequest) -> CompareResult:
    """Run the full comparison pipeline for paired or unpaired metric distributions."""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    if request.mode == "unpaired":
        return _compare_unpaired(request)
    if request.mode == "paired":
        return _compare_paired(request)
    raise ValueError(f"Unsupported comparison mode: {request.mode}")


def _compare_unpaired(request: CompareRequest) -> CompareResult:
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
        group_a=group_a,
        group_b=group_b,
        unpaired_comparison=comparison,
    )


def _compare_paired(request: CompareRequest) -> CompareResult:
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
        paired_summary=summary,
    )
