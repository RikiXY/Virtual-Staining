from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance, wilcoxon

from virtual_staining.utils.metrics import (
    DEFAULT_METRICS,
    get_metric_thresholds,
    is_higher_better_metric,
)
from virtual_staining.utils.metrics import (
    get_metric_plot_range as get_default_metric_plot_range,
)

DECISION_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


class DecisionSummary(TypedDict):
    better_label: str
    score_a: float
    score_b: float
    score_difference: float
    total_score: float
    decision_strength: str
    decision_reason: str


@dataclass
class UnpairedGroupStats:
    label: str
    n: int
    mean: float
    median: float
    iqr: float
    threshold_shares: dict[str, float]
    decision_score: float = 0.0
    decision_favored_count: int = 0
    decision_counter_count: int = 0


@dataclass
class UnpairedComparison:
    better_label: str
    mean_favors: str
    median_favors: str
    threshold_favors: str
    wasserstein_between_groups: float
    ks_statistic: float
    ks_pvalue: float
    mannwhitney_u: float
    mannwhitney_pvalue: float
    score_a: float = 0.0
    score_b: float = 0.0
    score_difference: float = 0.0
    total_score: float = 0.0
    decision_strength: str = "tie"
    decision_reason: str = "No score-based suggestion; criteria are tied."
    common_language_a_better: float = 0.5
    common_language_b_better: float = 0.5


@dataclass
class PairedSummary:
    label_a: str
    label_b: str
    n_pairs: int
    tolerance: float
    mean_signed_delta: float
    median_signed_delta: float
    share_b_better: float
    share_a_better: float
    share_equal: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    better_label: str
    score_a: float = 0.0
    score_b: float = 0.0
    score_difference: float = 0.0
    total_score: float = 0.0
    decision_strength: str = "tie"
    decision_reason: str = "No score-based suggestion; criteria are tied."
    signed_delta_q10: float = 0.0
    signed_delta_q25: float = 0.0
    signed_delta_q50: float = 0.0
    signed_delta_q75: float = 0.0
    signed_delta_q90: float = 0.0
    mean_relative_signed_delta: float = float("nan")
    median_relative_signed_delta: float = float("nan")
    relative_delta_count: int = 0


def metric_direction_name(higher_is_better: bool) -> str:
    """Return the stable direction label used in compare artifacts."""
    return "higher_is_better" if higher_is_better else "lower_is_better"


def signed_metric_difference(
    value_a: float,
    value_b: float,
    higher_is_better: bool,
) -> float:
    """Return a direction-aware B-vs-A difference where positive favors B."""
    return value_b - value_a if higher_is_better else value_a - value_b


def _favor_from_signed_difference(
    signed_difference: float,
    label_a: str,
    label_b: str,
    *,
    tolerance: float = 0.0,
) -> str:
    if signed_difference > tolerance:
        return label_b
    if signed_difference < -tolerance:
        return label_a
    return "tie"


def _decision_breakdown_row(
    *,
    mode: str,
    criterion: str,
    description: str,
    label_a: str,
    label_b: str,
    value_a: float,
    value_b: float,
    signed_difference: float,
    weight: float = 1.0,
    tolerance: float = 0.0,
) -> dict[str, object]:
    favors = _favor_from_signed_difference(
        signed_difference,
        label_a,
        label_b,
        tolerance=tolerance,
    )
    return {
        "mode": mode,
        "criterion": criterion,
        "description": description,
        "label_a": label_a,
        "label_b": label_b,
        "value_a": value_a,
        "value_b": value_b,
        "signed_difference": signed_difference,
        "weight": weight,
        "favors": favors,
        "score_a": weight if favors == label_a else 0.0,
        "score_b": weight if favors == label_b else 0.0,
    }


def _decision_row_float(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"Decision row field '{key}' must be numeric.")


def classify_decision_strength(score_difference: float, total_score: float) -> str:
    """Classify score margin as tie, weak, moderate, or strong."""
    if total_score <= 0 or score_difference == 0:
        return "tie"

    margin = abs(score_difference) / total_score
    if margin >= 0.75:
        return "strong"
    if margin >= 0.40:
        return "moderate"
    return "weak"


def _decision_from_rows(
    rows: Iterable[dict[str, object]],
    *,
    label_a: str,
    label_b: str,
) -> DecisionSummary:
    row_list = list(rows)
    score_a = float(sum(_decision_row_float(row, "score_a") for row in row_list))
    score_b = float(sum(_decision_row_float(row, "score_b") for row in row_list))
    total_score = float(sum(_decision_row_float(row, "weight") for row in row_list))
    score_difference = abs(score_b - score_a)

    if score_b > score_a:
        better_label = label_b
        favored_count = sum(1 for row in row_list if row["favors"] == label_b)
        counter_count = sum(1 for row in row_list if row["favors"] == label_a)
    elif score_a > score_b:
        better_label = label_a
        favored_count = sum(1 for row in row_list if row["favors"] == label_a)
        counter_count = sum(1 for row in row_list if row["favors"] == label_b)
    else:
        better_label = "tie"
        favored_count = 0
        counter_count = 0

    decision_strength = classify_decision_strength(score_difference, total_score)
    if better_label == "tie":
        reason = f"No score-based suggestion; criteria are tied ({score_a:.1f} vs {score_b:.1f})."
    else:
        reason = (
            f"{better_label} has a {decision_strength} score-based advantage "
            f"({score_a:.1f} vs {score_b:.1f}); {favored_count} criteria favored "
            f"{better_label} and {counter_count} opposed it."
        )

    return {
        "better_label": better_label,
        "score_a": score_a,
        "score_b": score_b,
        "score_difference": score_difference,
        "total_score": total_score,
        "decision_strength": decision_strength,
        "decision_reason": reason,
    }


def resolve_input_csv(path_like: str | Path) -> Path:
    """Resolves a direct CSV path or a directory containing per_image_metrics.csv."""
    path = Path(path_like)

    if path.is_dir():
        candidate = path / "per_image_metrics.csv"
        if candidate.exists():
            return candidate
        raise ValueError(f"Directory {path} does not contain per_image_metrics.csv")

    if path.is_file():
        return path

    raise ValueError(f"Input path does not exist: {path}")


def load_metric_frame(csv_path: str | Path) -> pd.DataFrame:
    """Loads a metrics CSV as a DataFrame."""
    resolved_csv = resolve_input_csv(csv_path)
    return pd.read_csv(resolved_csv)


def load_metric_values(csv_path: str | Path, column: str) -> np.ndarray:
    """Loads a numeric column from a CSV, discarding missing or invalid values."""
    df = load_metric_frame(csv_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError(f"No valid numeric values found in column '{column}'")

    return values


def choose_threshold_favors(
    shares_a: dict[str, float],
    shares_b: dict[str, float],
    label_a: str,
    label_b: str,
) -> str:
    """Chooses the favoured group by comparing the mean of the above/below-threshold shares."""
    mean_a = float(np.mean(list(shares_a.values()))) if shares_a else 0.0
    mean_b = float(np.mean(list(shares_b.values()))) if shares_b else 0.0

    if mean_b > mean_a:
        return label_b
    if mean_a > mean_b:
        return label_a
    return "tie"


def choose_unpaired_better_label(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
) -> str:
    """Chooses the better group by combining the main signals from the comparison."""
    if comparison.score_b > comparison.score_a:
        return group_b.label
    if comparison.score_a > comparison.score_b:
        return group_a.label

    score_a = 0
    score_b = 0

    for favored in [
        comparison.mean_favors,
        comparison.median_favors,
        comparison.threshold_favors,
    ]:
        if favored == group_a.label:
            score_a += 1
        elif favored == group_b.label:
            score_b += 1

    if score_b > score_a:
        return group_b.label
    if score_a > score_b:
        return group_a.label
    return "tie"


def choose_paired_better_label(
    mean_signed_delta: float,
    median_signed_delta: float,
    share_b_better: float,
    share_a_better: float,
    label_a: str,
    label_b: str,
) -> str:
    """Chooses the better group in a paired comparison from the main signals."""
    score_a = 0
    score_b = 0

    if mean_signed_delta > 0:
        score_b += 1
    elif mean_signed_delta < 0:
        score_a += 1

    if median_signed_delta > 0:
        score_b += 1
    elif median_signed_delta < 0:
        score_a += 1

    if share_b_better > share_a_better:
        score_b += 1
    elif share_a_better > share_b_better:
        score_a += 1

    if score_b > score_a:
        return label_b
    if score_a > score_b:
        return label_a
    return "tie"


def common_language_effect_size(
    a: np.ndarray,
    b: np.ndarray,
    higher_is_better: bool,
) -> float:
    """Return P(B beats A) + 0.5 * P(tie), respecting metric direction."""
    sorted_a = np.sort(a)
    left = np.searchsorted(sorted_a, b, side="left")
    right = np.searchsorted(sorted_a, b, side="right")
    ties = right - left

    wins = left if higher_is_better else sorted_a.size - right

    denominator = float(sorted_a.size * b.size)
    if denominator == 0.0:
        return float("nan")
    return float((np.sum(wins) + 0.5 * np.sum(ties)) / denominator)


def build_unpaired_quantile_comparison_rows(
    a: np.ndarray,
    b: np.ndarray,
    *,
    label_a: str,
    label_b: str,
    higher_is_better: bool,
    quantiles: Iterable[float] = DECISION_QUANTILES,
) -> list[dict[str, object]]:
    """Build direction-aware quantile comparison rows for unpaired distributions."""
    rows: list[dict[str, object]] = []
    direction = metric_direction_name(higher_is_better)

    for quantile in quantiles:
        value_a = float(np.quantile(a, quantile))
        value_b = float(np.quantile(b, quantile))
        raw_delta = value_b - value_a
        signed_delta = signed_metric_difference(value_a, value_b, higher_is_better)
        rows.append(
            {
                "quantile": f"q{int(round(quantile * 100)):02d}",
                "quantile_probability": float(quantile),
                "direction": direction,
                "value_a": value_a,
                "value_b": value_b,
                "raw_delta_b_minus_a": raw_delta,
                "signed_delta": signed_delta,
                "favors": _favor_from_signed_difference(
                    signed_delta,
                    label_a,
                    label_b,
                ),
            }
        )

    return rows


def build_unpaired_threshold_share_rows(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    *,
    higher_is_better: bool,
) -> list[dict[str, object]]:
    """Build threshold share rows using the favorable side of each threshold."""
    rows: list[dict[str, object]] = []
    operator = ">=" if higher_is_better else "<="

    for threshold_name, share_a in group_a.threshold_shares.items():
        share_b = group_b.threshold_shares.get(threshold_name, float("nan"))
        threshold = threshold_name.split("_", maxsplit=1)[1]
        share_delta = share_b - share_a
        rows.append(
            {
                "threshold_name": threshold_name,
                "rule": f"value {operator} {threshold}",
                "threshold": threshold,
                "share_a": share_a,
                "share_b": share_b,
                "share_delta_b_minus_a": share_delta,
                "favors": _favor_from_signed_difference(
                    share_delta,
                    group_a.label,
                    group_b.label,
                ),
            }
        )

    return rows


def build_unpaired_decision_breakdown_rows(
    a: np.ndarray,
    b: np.ndarray,
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    higher_is_better: bool,
) -> list[dict[str, object]]:
    """Build deterministic score criteria for an unpaired comparison."""
    tail_quantile = 0.10 if higher_is_better else 0.90
    tail_a = float(np.quantile(a, tail_quantile))
    tail_b = float(np.quantile(b, tail_quantile))
    threshold_mean_a = (
        float(np.mean(list(group_a.threshold_shares.values())))
        if group_a.threshold_shares
        else float("nan")
    )
    threshold_mean_b = (
        float(np.mean(list(group_b.threshold_shares.values())))
        if group_b.threshold_shares
        else float("nan")
    )
    common_language_b = common_language_effect_size(a, b, higher_is_better)
    common_language_a = 1.0 - common_language_b

    rows = [
        _decision_breakdown_row(
            mode="unpaired",
            criterion="mean",
            description="Direction-aware difference between group means.",
            label_a=group_a.label,
            label_b=group_b.label,
            value_a=group_a.mean,
            value_b=group_b.mean,
            signed_difference=signed_metric_difference(
                group_a.mean,
                group_b.mean,
                higher_is_better,
            ),
        ),
        _decision_breakdown_row(
            mode="unpaired",
            criterion="median",
            description="Direction-aware difference between group medians.",
            label_a=group_a.label,
            label_b=group_b.label,
            value_a=group_a.median,
            value_b=group_b.median,
            signed_difference=signed_metric_difference(
                group_a.median,
                group_b.median,
                higher_is_better,
            ),
        ),
        _decision_breakdown_row(
            mode="unpaired",
            criterion=f"worst_tail_q{int(round(tail_quantile * 100)):02d}",
            description="Worst-tail quantile comparison using metric direction.",
            label_a=group_a.label,
            label_b=group_b.label,
            value_a=tail_a,
            value_b=tail_b,
            signed_difference=signed_metric_difference(tail_a, tail_b, higher_is_better),
        ),
    ]

    if group_a.threshold_shares:
        rows.append(
            _decision_breakdown_row(
                mode="unpaired",
                criterion="threshold_mean",
                description="Mean favorable-threshold share difference.",
                label_a=group_a.label,
                label_b=group_b.label,
                value_a=threshold_mean_a,
                value_b=threshold_mean_b,
                signed_difference=threshold_mean_b - threshold_mean_a,
            )
        )

    rows.append(
        _decision_breakdown_row(
            mode="unpaired",
            criterion="common_language_effect",
            description="Probability that a random B sample beats a random A sample.",
            label_a=group_a.label,
            label_b=group_b.label,
            value_a=common_language_a,
            value_b=common_language_b,
            signed_difference=common_language_b - 0.5,
        )
    )

    return rows


def compute_unpaired_group_stats(
    values: np.ndarray,
    label: str,
    thresholds: Iterable[float],
    higher_is_better: bool,
) -> UnpairedGroupStats:
    """Computes the essential descriptive statistics of an unpaired group."""
    p25, p75 = np.percentile(values, [25, 75])

    if higher_is_better:
        shares = {
            f"ge_{threshold:.2f}": float(np.mean(values >= threshold)) for threshold in thresholds
        }
    else:
        shares = {
            f"le_{threshold:.2f}": float(np.mean(values <= threshold)) for threshold in thresholds
        }

    return UnpairedGroupStats(
        label=label,
        n=int(values.size),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        iqr=float(p75 - p25),
        threshold_shares=shares,
    )


def compute_unpaired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    higher_is_better: bool,
) -> UnpairedComparison:
    """Computes the main comparisons between two unpaired distributions."""
    mann_whitney = mannwhitneyu(a, b, alternative="two-sided")
    ks = ks_2samp(a, b, alternative="two-sided")

    if higher_is_better:
        mean_favors = (
            group_b.label
            if group_b.mean > group_a.mean
            else group_a.label
            if group_a.mean > group_b.mean
            else "tie"
        )
        median_favors = (
            group_b.label
            if group_b.median > group_a.median
            else group_a.label
            if group_a.median > group_b.median
            else "tie"
        )
    else:
        mean_favors = (
            group_b.label
            if group_b.mean < group_a.mean
            else group_a.label
            if group_a.mean < group_b.mean
            else "tie"
        )
        median_favors = (
            group_b.label
            if group_b.median < group_a.median
            else group_a.label
            if group_a.median < group_b.median
            else "tie"
        )

    threshold_favors = choose_threshold_favors(
        group_a.threshold_shares,
        group_b.threshold_shares,
        group_a.label,
        group_b.label,
    )
    common_language_b = common_language_effect_size(a, b, higher_is_better)
    decision_rows = build_unpaired_decision_breakdown_rows(
        a,
        b,
        group_a,
        group_b,
        higher_is_better,
    )
    decision = _decision_from_rows(
        decision_rows,
        label_a=group_a.label,
        label_b=group_b.label,
    )
    group_a.decision_score = float(decision["score_a"])
    group_b.decision_score = float(decision["score_b"])
    group_a.decision_favored_count = sum(
        1 for row in decision_rows if row["favors"] == group_a.label
    )
    group_b.decision_favored_count = sum(
        1 for row in decision_rows if row["favors"] == group_b.label
    )
    group_a.decision_counter_count = group_b.decision_favored_count
    group_b.decision_counter_count = group_a.decision_favored_count

    comparison = UnpairedComparison(
        better_label=str(decision["better_label"]),
        mean_favors=mean_favors,
        median_favors=median_favors,
        threshold_favors=threshold_favors,
        wasserstein_between_groups=float(wasserstein_distance(a, b)),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        mannwhitney_u=float(mann_whitney.statistic),
        mannwhitney_pvalue=float(mann_whitney.pvalue),
        score_a=float(decision["score_a"]),
        score_b=float(decision["score_b"]),
        score_difference=float(decision["score_difference"]),
        total_score=float(decision["total_score"]),
        decision_strength=str(decision["decision_strength"]),
        decision_reason=str(decision["decision_reason"]),
        common_language_a_better=1.0 - common_language_b,
        common_language_b_better=common_language_b,
    )
    return comparison


def align_paired_frames(
    csv_a: str | Path,
    csv_b: str | Path,
    sample_id_column: str,
    metric_column: str,
) -> pd.DataFrame:
    """Aligns two CSVs on the same sample_id for the paired comparison."""
    frame_a = load_metric_frame(csv_a)
    frame_b = load_metric_frame(csv_b)

    for frame_name, frame in [("A", frame_a), ("B", frame_b)]:
        if sample_id_column not in frame.columns:
            raise ValueError(f"Column '{sample_id_column}' not found in CSV {frame_name}")
        if metric_column not in frame.columns:
            raise ValueError(f"Column '{metric_column}' not found in CSV {frame_name}")

    subset_a = frame_a[[sample_id_column, metric_column]].rename(columns={metric_column: "value_a"})
    subset_b = frame_b[[sample_id_column, metric_column]].rename(columns={metric_column: "value_b"})
    merged = subset_a.merge(subset_b, on=sample_id_column, how="inner")
    merged["value_a"] = pd.to_numeric(merged["value_a"], errors="coerce")
    merged["value_b"] = pd.to_numeric(merged["value_b"], errors="coerce")
    merged = merged.dropna(subset=["value_a", "value_b"]).copy()

    if merged.empty:
        raise ValueError("No paired samples found after aligning the two CSV files.")

    return merged


def align_paired_metric_frames(
    csv_a: str | Path,
    csv_b: str | Path,
    sample_id_column: str,
    metric_columns: Iterable[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Align two metric CSVs by sample id for multi-metric paired reporting."""
    frame_a = load_metric_frame(csv_a)
    frame_b = load_metric_frame(csv_b)
    metrics = tuple(metric_columns)

    if not metrics:
        raise ValueError("At least one metric column is required for paired reporting.")

    for frame_name, frame in [("A", frame_a), ("B", frame_b)]:
        if sample_id_column not in frame.columns:
            raise ValueError(f"Column '{sample_id_column}' not found in CSV {frame_name}")

        missing_metrics = [metric for metric in metrics if metric not in frame.columns]
        if missing_metrics:
            raise ValueError(
                f"Metric columns not found in CSV {frame_name}: {', '.join(missing_metrics)}"
            )

    subset_a = frame_a[[sample_id_column, *metrics]].rename(
        columns={metric: f"{metric}_a" for metric in metrics}
    )
    subset_b = frame_b[[sample_id_column, *metrics]].rename(
        columns={metric: f"{metric}_b" for metric in metrics}
    )
    merged = subset_a.merge(subset_b, on=sample_id_column, how="inner")

    if merged.empty:
        raise ValueError("No paired samples found after aligning the two CSV files.")

    for metric in metrics:
        merged[f"{metric}_a"] = pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
        merged[f"{metric}_b"] = pd.to_numeric(merged[f"{metric}_b"], errors="coerce")

    return merged


def build_paired_multi_metric_delta_reports(
    merged: pd.DataFrame,
    metrics: Iterable[str],
    *,
    sample_id_column: str,
    label_a: str,
    label_b: str,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-sample and per-metric paired delta reports."""
    metric_names = tuple(metrics)
    sample_report = pd.DataFrame({sample_id_column: merged[sample_id_column]})
    summary_rows: list[dict[str, object]] = []
    total_common_count = int(merged.shape[0])

    for metric in metric_names:
        higher_is_better = is_higher_better_metric(metric)
        direction = "higher_is_better" if higher_is_better else "lower_is_better"
        values_a = merged[f"{metric}_a"]
        values_b = merged[f"{metric}_b"]
        raw_delta = values_b - values_a
        signed_delta = raw_delta if higher_is_better else -raw_delta
        valid_mask = pd.Series(
            np.isfinite(signed_delta.to_numpy(dtype=float)),
            index=merged.index,
        )
        valid_signed_delta = signed_delta[valid_mask]
        valid_raw_delta = raw_delta[valid_mask]

        winners = pd.Series("", index=merged.index, dtype=object)
        winners[signed_delta > tolerance] = label_b
        winners[signed_delta < -tolerance] = label_a
        winners[(signed_delta.abs() <= tolerance) & valid_mask] = "equal"

        sample_report[f"{metric}_a"] = values_a
        sample_report[f"{metric}_b"] = values_b
        sample_report[f"{metric}_raw_delta_b_minus_a"] = raw_delta
        sample_report[f"{metric}_signed_delta"] = signed_delta
        sample_report[f"{metric}_winner"] = winners

        finite_pair_count = int(valid_mask.sum())
        improved_count = int((valid_signed_delta > tolerance).sum())
        worsened_count = int((valid_signed_delta < -tolerance).sum())
        equal_count = int((valid_signed_delta.abs() <= tolerance).sum())
        denominator = finite_pair_count or float("nan")

        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "label_a": label_a,
                "label_b": label_b,
                "tolerance": tolerance,
                "total_common_count": total_common_count,
                "finite_pair_count": finite_pair_count,
                "missing_pair_count": total_common_count - finite_pair_count,
                "improved_count": improved_count,
                "worsened_count": worsened_count,
                "equal_count": equal_count,
                "improved_share": improved_count / denominator,
                "worsened_share": worsened_count / denominator,
                "equal_share": equal_count / denominator,
                "mean_raw_delta_b_minus_a": float(valid_raw_delta.mean()),
                "median_raw_delta_b_minus_a": float(valid_raw_delta.median()),
                "mean_signed_delta": float(valid_signed_delta.mean()),
                "median_signed_delta": float(valid_signed_delta.median()),
            }
        )

    return sample_report, pd.DataFrame(summary_rows)


def _paired_delta_arrays(
    merged: pd.DataFrame,
    higher_is_better: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_a = merged["value_a"].to_numpy(dtype=float)
    values_b = merged["value_b"].to_numpy(dtype=float)
    raw_delta = values_b - values_a
    signed_delta = raw_delta if higher_is_better else -raw_delta
    relative_delta = np.full_like(signed_delta, np.nan, dtype=float)
    denominator = np.abs(values_a)
    valid_denominator = denominator > 0.0
    relative_delta[valid_denominator] = (
        signed_delta[valid_denominator] / denominator[valid_denominator]
    )
    return raw_delta, signed_delta, relative_delta


def build_paired_decision_breakdown_rows(summary: PairedSummary) -> list[dict[str, object]]:
    """Build deterministic score criteria for a paired comparison."""
    return [
        _decision_breakdown_row(
            mode="paired",
            criterion="mean_signed_delta",
            description="Mean direction-aware paired delta.",
            label_a=summary.label_a,
            label_b=summary.label_b,
            value_a=0.0,
            value_b=summary.mean_signed_delta,
            signed_difference=summary.mean_signed_delta,
        ),
        _decision_breakdown_row(
            mode="paired",
            criterion="median_signed_delta",
            description="Median direction-aware paired delta.",
            label_a=summary.label_a,
            label_b=summary.label_b,
            value_a=0.0,
            value_b=summary.median_signed_delta,
            signed_difference=summary.median_signed_delta,
        ),
        _decision_breakdown_row(
            mode="paired",
            criterion="share_improved",
            description="Share of paired samples improved versus worsened.",
            label_a=summary.label_a,
            label_b=summary.label_b,
            value_a=summary.share_a_better,
            value_b=summary.share_b_better,
            signed_difference=summary.share_b_better - summary.share_a_better,
        ),
        _decision_breakdown_row(
            mode="paired",
            criterion="q25_signed_delta",
            description="Lower-middle quantile of direction-aware paired deltas.",
            label_a=summary.label_a,
            label_b=summary.label_b,
            value_a=0.0,
            value_b=summary.signed_delta_q25,
            signed_difference=summary.signed_delta_q25,
        ),
        _decision_breakdown_row(
            mode="paired",
            criterion="worst_tail_q10_signed_delta",
            description="Worst-tail quantile of direction-aware paired deltas.",
            label_a=summary.label_a,
            label_b=summary.label_b,
            value_a=0.0,
            value_b=summary.signed_delta_q10,
            signed_difference=summary.signed_delta_q10,
        ),
    ]


def build_paired_delta_summary_row(
    summary: PairedSummary,
    *,
    metric: str,
    higher_is_better: bool,
) -> dict[str, object]:
    """Build the one-row paired_delta_summary.csv payload."""
    return {
        "mode": "paired",
        "metric": metric,
        "direction": metric_direction_name(higher_is_better),
        "label_a": summary.label_a,
        "label_b": summary.label_b,
        "n_pairs": summary.n_pairs,
        "tolerance": summary.tolerance,
        "mean_signed_delta": summary.mean_signed_delta,
        "median_signed_delta": summary.median_signed_delta,
        "signed_delta_q10": summary.signed_delta_q10,
        "signed_delta_q25": summary.signed_delta_q25,
        "signed_delta_q50": summary.signed_delta_q50,
        "signed_delta_q75": summary.signed_delta_q75,
        "signed_delta_q90": summary.signed_delta_q90,
        "mean_relative_signed_delta": summary.mean_relative_signed_delta,
        "median_relative_signed_delta": summary.median_relative_signed_delta,
        "relative_delta_count": summary.relative_delta_count,
        "share_b_better": summary.share_b_better,
        "share_a_better": summary.share_a_better,
        "share_equal": summary.share_equal,
        "score_a": summary.score_a,
        "score_b": summary.score_b,
        "score_difference": summary.score_difference,
        "total_score": summary.total_score,
        "decision_strength": summary.decision_strength,
        "decision_reason": summary.decision_reason,
        "better_label": summary.better_label,
    }


def compute_paired_summary(
    merged: pd.DataFrame,
    label_a: str,
    label_b: str,
    tolerance: float,
    higher_is_better: bool,
) -> PairedSummary:
    """Computes the main summary for the paired comparison."""
    _, signed_delta, relative_delta = _paired_delta_arrays(merged, higher_is_better)

    share_b_better = float(np.mean(signed_delta > tolerance))
    share_a_better = float(np.mean(signed_delta < -tolerance))
    share_equal = float(np.mean(np.abs(signed_delta) <= tolerance))

    non_zero_delta = signed_delta[np.abs(signed_delta) > tolerance]
    if non_zero_delta.size == 0:
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    else:
        wilcoxon_result = wilcoxon(non_zero_delta, alternative="two-sided")
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)

    mean_signed_delta = float(np.mean(signed_delta))
    median_signed_delta = float(np.median(signed_delta))
    q10, q25, q50, q75, q90 = np.quantile(signed_delta, DECISION_QUANTILES)
    finite_relative_delta = relative_delta[np.isfinite(relative_delta)]
    mean_relative_delta = (
        float(np.mean(finite_relative_delta)) if finite_relative_delta.size else float("nan")
    )
    median_relative_delta = (
        float(np.median(finite_relative_delta)) if finite_relative_delta.size else float("nan")
    )
    preliminary_summary = PairedSummary(
        label_a=label_a,
        label_b=label_b,
        n_pairs=int(merged.shape[0]),
        tolerance=tolerance,
        mean_signed_delta=mean_signed_delta,
        median_signed_delta=median_signed_delta,
        share_b_better=share_b_better,
        share_a_better=share_a_better,
        share_equal=share_equal,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_pvalue=wilcoxon_pvalue,
        better_label="tie",
        signed_delta_q10=float(q10),
        signed_delta_q25=float(q25),
        signed_delta_q50=float(q50),
        signed_delta_q75=float(q75),
        signed_delta_q90=float(q90),
        mean_relative_signed_delta=mean_relative_delta,
        median_relative_signed_delta=median_relative_delta,
        relative_delta_count=int(finite_relative_delta.size),
    )
    decision = _decision_from_rows(
        build_paired_decision_breakdown_rows(preliminary_summary),
        label_a=label_a,
        label_b=label_b,
    )

    return PairedSummary(
        label_a=label_a,
        label_b=label_b,
        n_pairs=int(merged.shape[0]),
        tolerance=tolerance,
        mean_signed_delta=mean_signed_delta,
        median_signed_delta=median_signed_delta,
        share_b_better=share_b_better,
        share_a_better=share_a_better,
        share_equal=share_equal,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_pvalue=wilcoxon_pvalue,
        better_label=str(decision["better_label"]),
        score_a=float(decision["score_a"]),
        score_b=float(decision["score_b"]),
        score_difference=float(decision["score_difference"]),
        total_score=float(decision["total_score"]),
        decision_strength=str(decision["decision_strength"]),
        decision_reason=str(decision["decision_reason"]),
        signed_delta_q10=float(q10),
        signed_delta_q25=float(q25),
        signed_delta_q50=float(q50),
        signed_delta_q75=float(q75),
        signed_delta_q90=float(q90),
        mean_relative_signed_delta=mean_relative_delta,
        median_relative_signed_delta=median_relative_delta,
        relative_delta_count=int(finite_relative_delta.size),
    )


def resolve_plot_range(args: argparse.Namespace) -> tuple[float, float]:
    """Resolves a metric plot range from CLI overrides or metric defaults."""
    default_min, default_max = get_default_metric_plot_range(args.column)
    min_value = args.min_value if args.min_value is not None else default_min
    max_value = args.max_value if args.max_value is not None else default_max

    if min_value == max_value:
        padding = 0.5 if min_value == 0 else abs(min_value) * 0.05
        min_value -= padding
        max_value += padding

    return float(min_value), float(max_value)


def resolve_thresholds(args: argparse.Namespace) -> list[float]:
    """Resolves comparison thresholds from CLI overrides or metric defaults."""
    if getattr(args, "thresholds", None) is not None:
        return list(args.thresholds)
    return get_metric_thresholds(args.column)


def resolve_run_path(run_path: str | Path) -> Path:
    """Resolves and validates a run directory."""
    path = Path(run_path).resolve()

    if not path.is_dir():
        raise NotADirectoryError(f"Run directory not found: {path}")

    return path


def resolve_metrics_csv_from_run(run_path: str | Path) -> Path:
    """Returns evaluation/per_image_metrics.csv for a run directory."""
    run_dir = resolve_run_path(run_path)
    csv_path = run_dir / "evaluation" / "per_image_metrics.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find per_image_metrics.csv for run '{run_dir.name}'. Expected: {csv_path}"
        )

    return csv_path


def infer_label_from_input(
    run_path: str | Path | None,
    csv_path: str | Path | None,
    fallback: str,
) -> str:
    """Infers a readable label from a run path or metrics CSV path."""
    if run_path is not None:
        return Path(run_path).resolve().name

    if csv_path is not None:
        path = Path(csv_path).resolve()
        if path.name == "per_image_metrics.csv" and path.parent.name == "evaluation":
            return path.parent.parent.name
        return path.stem

    return fallback


def resolve_comparison_inputs(args: argparse.Namespace) -> None:
    """Resolves comparison CSVs and labels from run paths or explicit CSVs."""
    if args.run_a is None and args.csv_a is None:
        raise ValueError("You must provide either --run-a or --csv-a.")

    if args.run_b is None and args.csv_b is None:
        raise ValueError("You must provide either --run-b or --csv-b.")

    args.resolved_csv_a = (
        resolve_metrics_csv_from_run(args.run_a)
        if args.run_a is not None
        else resolve_input_csv(args.csv_a)
    )
    args.resolved_csv_b = (
        resolve_metrics_csv_from_run(args.run_b)
        if args.run_b is not None
        else resolve_input_csv(args.csv_b)
    )
    args.resolved_label_a = args.label_a or infer_label_from_input(
        run_path=args.run_a,
        csv_path=args.csv_a,
        fallback="A",
    )
    args.resolved_label_b = args.label_b or infer_label_from_input(
        run_path=args.run_b,
        csv_path=args.csv_b,
        fallback="B",
    )


def infer_results_root_from_inputs(args: argparse.Namespace) -> Path:
    """Infers the results root from resolved comparison inputs."""
    for run_attr in ("run_a", "run_b"):
        run_path = getattr(args, run_attr)
        if run_path is not None:
            run_dir = Path(run_path).resolve()
            if run_dir.parent.name == "results":
                return run_dir.parent

    csv_a = Path(args.resolved_csv_a).resolve()
    parts = csv_a.parts

    if "results" in parts:
        results_index = parts.index("results")
        return Path(*parts[: results_index + 1])

    return Path("local_workspace") / "results"


def resolve_comparison_output_dir(args: argparse.Namespace) -> Path:
    """Resolves the output directory for a comparison run."""
    if args.output_dir is not None:
        return Path(args.output_dir)

    results_root = infer_results_root_from_inputs(args)
    comparison_name = f"{args.resolved_label_a}_vs_{args.resolved_label_b}"
    if args.mode == "paired" and (
        getattr(args, "resolved_metrics", None) is not None or args.column == "all"
    ):
        metric_dir_name = "paired_all_metrics"
    else:
        metric_dir_name = f"{args.mode}_{args.column}"
    return results_root / "comparisons" / comparison_name / metric_dir_name


def flatten_unpaired_group_stats(group: UnpairedGroupStats) -> dict[str, Any]:
    """Converts grouped unpaired stats into a flat CSV row."""
    row: dict[str, Any] = {
        "label": group.label,
        "n": group.n,
        "mean": group.mean,
        "median": group.median,
        "iqr": group.iqr,
    }

    for threshold_name, share in group.threshold_shares.items():
        row[f"share_{threshold_name}"] = share

    return row


def resolve_metric_direction(args: argparse.Namespace) -> bool:
    """Resolves metric direction from explicit flags or known metric defaults."""
    if args.higher_is_better and args.lower_is_better:
        raise ValueError("Choose at most one between --higher-is-better and --lower-is-better.")
    if args.higher_is_better:
        return True
    if args.lower_is_better:
        return False
    return is_higher_better_metric(args.column)
