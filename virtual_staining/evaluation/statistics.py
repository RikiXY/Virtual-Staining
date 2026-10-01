from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    path = Path(path_like)

    if path.is_dir():
        candidate = path / "per_image_metrics.csv"
        if candidate.exists():
            return candidate
        raise ValueError(f"Directory {path} does not contain per_image_metrics.csv")

    if path.is_file():
        return path

    raise ValueError(f"Input path does not exist: {path}")


OUTPUT_NAME_COLUMN = "output_name"


def _load_metric_frame(csv_path: str | Path, output_name: str | None = None) -> pd.DataFrame:
    """Read per-image rows of exactly one model output; outputs are never pooled.

    A CSV with an ``output_name`` column holding several outputs requires ``output_name``.
    """
    resolved_csv = resolve_input_csv(csv_path)
    frame = pd.read_csv(resolved_csv)
    if OUTPUT_NAME_COLUMN not in frame.columns:
        if output_name is not None:
            raise ValueError(f"{resolved_csv} has no {OUTPUT_NAME_COLUMN!r} column")
        return frame
    frame[OUTPUT_NAME_COLUMN] = frame[OUTPUT_NAME_COLUMN].astype(str)
    names = list(dict.fromkeys(frame[OUTPUT_NAME_COLUMN]))
    if output_name is None:
        if len(names) > 1:
            raise ValueError(
                f"{resolved_csv} holds outputs {names}; select one output_name, since a "
                "comparison never pools different outputs"
            )
        return frame
    if output_name not in names:
        raise ValueError(f"{resolved_csv} has no rows for output {output_name!r}; found {names}")
    return frame[frame[OUTPUT_NAME_COLUMN] == output_name]


def load_metric_values(
    csv_path: str | Path, column: str, output_name: str | None = None
) -> np.ndarray:
    df = _load_metric_frame(csv_path, output_name)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError(f"No valid numeric values found in column '{column}'")

    return values


def _choose_threshold_favors(
    shares_a: dict[str, float],
    shares_b: dict[str, float],
    label_a: str,
    label_b: str,
) -> str:
    mean_a = float(np.mean(list(shares_a.values()))) if shares_a else 0.0
    mean_b = float(np.mean(list(shares_b.values()))) if shares_b else 0.0

    if mean_b > mean_a:
        return label_b
    if mean_a > mean_b:
        return label_a
    return "tie"


def _choose_unpaired_better_label(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
) -> str:
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


def _choose_paired_better_label(
    mean_signed_delta: float,
    median_signed_delta: float,
    share_b_better: float,
    share_a_better: float,
    label_a: str,
    label_b: str,
) -> str:
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


def compute_unpaired_group_stats(
    values: np.ndarray,
    label: str,
    thresholds: Iterable[float],
    higher_is_better: bool,
) -> UnpairedGroupStats:
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

    threshold_favors = _choose_threshold_favors(
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
    comparison.better_label = _choose_unpaired_better_label(group_a, group_b, comparison)
    return comparison


def align_paired_frames(
    csv_a: str | Path,
    csv_b: str | Path,
    sample_id_column: str,
    metric_column: str,
    output_name: str | None = None,
) -> pd.DataFrame:
    """Align two per-image CSVs by ``(sample_id, output_name)`` for one output.

    Each alignment key must be unique in each CSV; rows are never matched by sample ID
    alone across different outputs.
    """
    frame_a = _load_metric_frame(csv_a, output_name)
    frame_b = _load_metric_frame(csv_b, output_name)

    for frame_name, frame in [("A", frame_a), ("B", frame_b)]:
        if sample_id_column not in frame.columns:
            raise ValueError(f"Column '{sample_id_column}' not found in CSV {frame_name}")
        if metric_column not in frame.columns:
            raise ValueError(f"Column '{metric_column}' not found in CSV {frame_name}")
    keys = [sample_id_column]
    if OUTPUT_NAME_COLUMN in frame_a.columns and OUTPUT_NAME_COLUMN in frame_b.columns:
        keys.append(OUTPUT_NAME_COLUMN)
        if set(frame_a[OUTPUT_NAME_COLUMN]) != set(frame_b[OUTPUT_NAME_COLUMN]):
            raise ValueError(
                f"CSV A output {sorted(set(frame_a[OUTPUT_NAME_COLUMN]))} differs from CSV B "
                f"output {sorted(set(frame_b[OUTPUT_NAME_COLUMN]))}"
            )
    elif OUTPUT_NAME_COLUMN in frame_a.columns or OUTPUT_NAME_COLUMN in frame_b.columns:
        raise ValueError(f"Only one CSV has an {OUTPUT_NAME_COLUMN!r} column")
    for frame_name, frame in [("A", frame_a), ("B", frame_b)]:
        if frame.duplicated(subset=keys).any():
            raise ValueError(f"CSV {frame_name} has duplicate {keys} rows")

    subset_a = frame_a[[*keys, metric_column]].rename(columns={metric_column: "value_a"})
    subset_b = frame_b[[*keys, metric_column]].rename(columns={metric_column: "value_b"})
    merged = subset_a.merge(subset_b, on=keys, how="inner")
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
    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)
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
        better_label=_choose_paired_better_label(
            mean_signed_delta=mean_signed_delta,
            median_signed_delta=median_signed_delta,
            share_b_better=share_b_better,
            share_a_better=share_a_better,
            label_a=label_a,
            label_b=label_b,
        ),
    )


def flatten_unpaired_group_stats(group: UnpairedGroupStats) -> dict[str, Any]:
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
