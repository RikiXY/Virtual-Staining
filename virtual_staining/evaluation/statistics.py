from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance, wilcoxon


@dataclass
class UnpairedGroupStats:
    label: str
    n: int
    mean: float
    median: float
    iqr: float
    threshold_shares: dict[str, float]


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
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

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
    mean_signed_delta: float, label_a: str, label_b: str
) -> str:
    """Chooses the better group in the paired comparison based on the mean signed delta."""
    if mean_signed_delta > 0:
        return label_b
    if mean_signed_delta < 0:
        return label_a
    return "tie"


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
            f"ge_{threshold:.2f}": float(np.mean(values >= threshold))
            for threshold in thresholds
        }
    else:
        shares = {
            f"le_{threshold:.2f}": float(np.mean(values <= threshold))
            for threshold in thresholds
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

    comparison = UnpairedComparison(
        better_label="tie",
        mean_favors=mean_favors,
        median_favors=median_favors,
        threshold_favors=threshold_favors,
        wasserstein_between_groups=float(wasserstein_distance(a, b)),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        mannwhitney_u=float(mann_whitney.statistic),
        mannwhitney_pvalue=float(mann_whitney.pvalue),
    )
    comparison.better_label = choose_unpaired_better_label(group_a, group_b, comparison)
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
            raise ValueError(
                f"Column '{sample_id_column}' not found in CSV {frame_name}"
            )
        if metric_column not in frame.columns:
            raise ValueError(f"Column '{metric_column}' not found in CSV {frame_name}")

    subset_a = frame_a[[sample_id_column, metric_column]].rename(
        columns={metric_column: "value_a"}
    )
    subset_b = frame_b[[sample_id_column, metric_column]].rename(
        columns={metric_column: "value_b"}
    )
    merged = subset_a.merge(subset_b, on=sample_id_column, how="inner")
    merged["value_a"] = pd.to_numeric(merged["value_a"], errors="coerce")
    merged["value_b"] = pd.to_numeric(merged["value_b"], errors="coerce")
    merged = merged.dropna(subset=["value_a", "value_b"]).copy()

    if merged.empty:
        raise ValueError("No paired samples found after aligning the two CSV files.")

    return merged


def compute_paired_summary(
    merged: pd.DataFrame,
    label_a: str,
    label_b: str,
    tolerance: float,
    higher_is_better: bool,
) -> PairedSummary:
    """Computes the main summary for the paired comparison."""
    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(
        dtype=float
    )
    signed_delta = raw_delta if higher_is_better else -raw_delta

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
        better_label=choose_paired_better_label(mean_signed_delta, label_a, label_b),
    )
