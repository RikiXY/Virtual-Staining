from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from utils.cli import ANSI, use_color, style, print_section, print_info
from utils.metrics import color_metric_value

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance, wilcoxon


def color_distance(value: float, good: float, warn: float) -> str:
    """Colours a distance: the smaller, the better."""
    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    if value <= warn * 1.5:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_pvalue(value: float) -> str:
    """Colours a p-value according to the strength of evidence for a difference."""
    if value < 0.001:
        return style(f"{value:.6g}", "green")
    if value < 0.01:
        return style(f"{value:.6g}", "yellow")
    if value < 0.05:
        return style(f"{value:.6g}", "orange")
    return style(f"{value:.6g}", "red")


def color_share(value: float) -> str:
    """Colours a share between 0 and 1."""
    if value >= 0.70:
        return style(f"{value:.6f}", "green")
    if value >= 0.40:
        return style(f"{value:.6f}", "yellow")
    if value >= 0.20:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_signed_delta(value: float) -> str:
    """Colours a signed delta: positive favours B, negative favours A."""
    if value > 0:
        return style(f"{value:.6f}", "green")
    if value < 0:
        return style(f"{value:.6f}", "red")
    return style(f"{value:.6f}", "yellow")


# =========================================
# Section dedicated to result structures
# =========================================
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


# ==========================
# Section dedicated to the parser
# ==========================
def add_direction_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the arguments that define the metric direction."""
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Use for metrics like SSIM and PSNR.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Use for metrics like MAE and RMSE.",
    )


def add_common_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the arguments common to both comparison modes."""
    parser.add_argument(
        "--csv-a",
        required=True,
        help="First CSV file or directory containing per_image_metrics.csv.",
    )
    parser.add_argument(
        "--csv-b",
        required=True,
        help="Second CSV file or directory containing per_image_metrics.csv.",
    )
    parser.add_argument(
        "--label-a",
        default="A",
        help="Label shown in reports and plots for the first group.",
    )
    parser.add_argument(
        "--label-b",
        default="B",
        help="Label shown in reports and plots for the second group.",
    )
    parser.add_argument(
        "--column",
        default="ssim",
        help="Metric column to compare.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs will be saved.",
    )
    add_direction_arguments(parser)


def add_unpaired_subparser(subparsers: Any) -> None:
    """Adds the subcommand for unpaired distributions."""
    parser = subparsers.add_parser(
        "unpaired",
        help="Compare two independent metric distributions.",
        description=(
            "Compare two independent metric distributions from per-image CSV files. "
            "Useful when the two runs do not share exactly the same samples."
        ),
    )
    add_common_comparison_arguments(parser)
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.0,
        help="Minimum plausible metric value used for shared histogram bins.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=1.0,
        help="Maximum plausible metric value used for shared histogram bins.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the common histogram.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.65, 0.75, 0.85],
        help="Thresholds used for share-above or share-below statistics.",
    )
    parser.set_defaults(func=run_unpaired)


def add_paired_subparser(subparsers: Any) -> None:
    """Adds the subcommand for paired distributions on the same samples."""
    parser = subparsers.add_parser(
        "paired",
        help="Compare two paired metric distributions on the same samples.",
        description=(
            "Compare two paired metric distributions by aligning rows on the same sample_id. "
            "Useful when the two runs share the same test samples."
        ),
    )
    add_common_comparison_arguments(parser)

    parser.add_argument(
        "--sample-id-column",
        default="sample_id",
        help="Column used to align the two CSV files.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Tolerance below which two values are considered equal.",
    )

    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help="Minimum metric value used for shared histogram bins. If omitted, inferred from paired values.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Maximum metric value used for shared histogram bins. If omitted, inferred from paired values.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the comparison histogram.",
    )

    parser.set_defaults(func=run_paired)


def build_parser() -> argparse.ArgumentParser:
    """Builds the main parser and registers the available subcommands."""
    parser = argparse.ArgumentParser(
        prog="python tools/compare_distributions.py",
        description=(
            "Compare metric distributions from per-image CSV files. "
            "Supports both unpaired and paired comparisons."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/compare_distributions.py unpaired \\\n"
            "      --csv-a local_workspace/results/run_a/evaluation \\\n"
            "      --csv-b local_workspace/results/run_b/evaluation \\\n"
            "      --label-a P-256 \\\n"
            "      --label-b P-512 \\\n"
            "      --column ssim \\\n"
            "      --higher-is-better \\\n"
            "      --output-dir local_workspace/results/comparisons/unpaired_ssim\n"
            "\n"
            "  python tools/compare_distributions.py paired \\\n"
            "      --csv-a local_workspace/results/run_a/evaluation \\\n"
            "      --csv-b local_workspace/results/run_b/evaluation \\\n"
            "      --label-a L1-25 \\\n"
            "      --label-b L1-37 \\\n"
            "      --column ssim \\\n"
            "      --higher-is-better \\\n"
            "      --output-dir local_workspace/results/comparisons/paired_ssim\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="mode")
    add_unpaired_subparser(subparsers)
    add_paired_subparser(subparsers)
    return parser


def validate_direction(args: argparse.Namespace) -> None:
    """Verifies that exactly one metric direction has been chosen."""
    if args.higher_is_better == args.lower_is_better:
        raise SystemExit("Choose exactly one between --higher-is-better and --lower-is-better.")


# ========================================
# Section dedicated to utility functions
# ========================================
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


def choose_unpaired_better_label(group_a: UnpairedGroupStats, group_b: UnpairedGroupStats, comparison: UnpairedComparison) -> str:
    """Chooses the better group by combining the main signals from the comparison."""
    score_a = 0
    score_b = 0

    for favored in [comparison.mean_favors, comparison.median_favors, comparison.threshold_favors]:
        if favored == group_a.label:
            score_a += 1
        elif favored == group_b.label:
            score_b += 1

    if score_b > score_a:
        return group_b.label
    if score_a > score_b:
        return group_a.label
    return "tie"


def choose_paired_better_label(mean_signed_delta: float, label_a: str, label_b: str) -> str:
    """Chooses the better group in the paired comparison based on the mean signed delta."""
    if mean_signed_delta > 0:
        return label_b
    if mean_signed_delta < 0:
        return label_a
    return "tie"


# ==========================================
# Section dedicated to metric computation
# ==========================================
def compute_unpaired_group_stats(
    values: np.ndarray,
    label: str,
    thresholds: Iterable[float],
    higher_is_better: bool,
) -> UnpairedGroupStats:
    """Computes the essential descriptive statistics of an unpaired group."""
    p25, p75 = np.percentile(values, [25, 75])

    if higher_is_better:
        shares = {f"ge_{threshold:.2f}": float(np.mean(values >= threshold)) for threshold in thresholds}
    else:
        shares = {f"le_{threshold:.2f}": float(np.mean(values <= threshold)) for threshold in thresholds}

    return UnpairedGroupStats(
        label=label,
        n=int(values.size),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        iqr=float(p75 - p25),
        threshold_shares=shares,
    )


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
        mean_favors = group_b.label if group_b.mean > group_a.mean else group_a.label if group_a.mean > group_b.mean else "tie"
        median_favors = group_b.label if group_b.median > group_a.median else group_a.label if group_a.median > group_b.median else "tie"
    else:
        mean_favors = group_b.label if group_b.mean < group_a.mean else group_a.label if group_a.mean < group_b.mean else "tie"
        median_favors = group_b.label if group_b.median < group_a.median else group_a.label if group_a.median < group_b.median else "tie"

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


def compute_paired_summary(
    merged: pd.DataFrame,
    label_a: str,
    label_b: str,
    tolerance: float,
    higher_is_better: bool,
) -> PairedSummary:
    """Computes the main summary for the paired comparison."""
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
        better_label=choose_paired_better_label(mean_signed_delta, label_a, label_b),
    )


# =======================================
# Section dedicated to output writing
# =======================================
def save_unpaired_group_statistics(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    output_dir: Path,
) -> None:
    """Saves group_statistics.csv with one row per group."""
    pd.DataFrame([asdict(group_a), asdict(group_b)]).to_csv(output_dir / "group_statistics.csv", index=False)


def save_unpaired_summary_json(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    output_dir: Path,
) -> None:
    """Saves a JSON summary of the unpaired comparison."""
    payload = {
        "group_a": asdict(group_a),
        "group_b": asdict(group_b),
        "comparison": asdict(comparison),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_paired_summary_json(summary: PairedSummary, output_dir: Path) -> None:
    """Saves a JSON summary of the paired comparison."""
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def save_unpaired_report_txt(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Saves report.txt for the unpaired comparison."""
    lines = [
        f"Metric: {args.column}",
        f"Direction: {'higher is better' if args.higher_is_better else 'lower is better'}",
        "",
        f"{group_a.label}: n={group_a.n}, mean={group_a.mean:.6f}, median={group_a.median:.6f}, IQR={group_a.iqr:.6f}",
        f"{group_b.label}: n={group_b.n}, mean={group_b.mean:.6f}, median={group_b.median:.6f}, IQR={group_b.iqr:.6f}",
        "",
        f"Mean favors: {comparison.mean_favors}",
        f"Median favors: {comparison.median_favors}",
        f"Threshold favors: {comparison.threshold_favors}",
        f"Wasserstein between groups: {comparison.wasserstein_between_groups:.6f}",
        f"KS statistic: {comparison.ks_statistic:.6f}",
        f"KS p-value: {comparison.ks_pvalue:.6g}",
        f"Mann-Whitney U: {comparison.mannwhitney_u:.6f}",
        f"Mann-Whitney p-value: {comparison.mannwhitney_pvalue:.6g}",
        "",
        f"Overall comparison favors: {comparison.better_label}",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def save_paired_report_txt(summary: PairedSummary, args: argparse.Namespace, output_dir: Path) -> None:
    """Saves report.txt for the paired comparison."""
    lines = [
        f"Metric: {args.column}",
        f"Direction: {'higher is better' if args.higher_is_better else 'lower is better'}",
        f"Paired samples: {summary.n_pairs}",
        f"Tolerance: {summary.tolerance:.6f}",
        "",
        f"Mean signed delta: {summary.mean_signed_delta:.6f}",
        f"Median signed delta: {summary.median_signed_delta:.6f}",
        f"Share {summary.label_b} better: {summary.share_b_better:.6f}",
        f"Share {summary.label_a} better: {summary.share_a_better:.6f}",
        f"Share equal: {summary.share_equal:.6f}",
        f"Wilcoxon statistic: {summary.wilcoxon_statistic:.6f}",
        f"Wilcoxon p-value: {summary.wilcoxon_pvalue:.6g}",
        "",
        f"Overall paired comparison favors: {summary.better_label}",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


# ===========================
# Section dedicated to plots
# ===========================
def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Builds the empirical cumulative distribution function of the sample."""
    x = np.sort(values)
    y = np.arange(1, values.size + 1) / values.size
    return x, y


def plot_distribution_histogram(
    a: np.ndarray,
    b: np.ndarray,
    edges: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Saves the comparison histogram between two distributions."""
    plt.figure(figsize=(9, 5))
    plt.hist(a, bins=edges, density=True, alpha=0.45, label=label_a)
    plt.hist(b, bins=edges, density=True, alpha=0.45, label=label_b)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"Histogram comparison - {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_comparison.png", dpi=180)
    plt.close()


def plot_distribution_ecdf(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Saves the comparison between the empirical cumulative distributions."""
    xa, ya = ecdf(a)
    xb, yb = ecdf(b)

    plt.figure(figsize=(9, 5))
    plt.step(xa, ya, where="post", label=label_a)
    plt.step(xb, yb, where="post", label=label_b)
    plt.xlabel(column)
    plt.ylabel("ECDF")
    plt.title(f"ECDF comparison - {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ecdf_comparison.png", dpi=180)
    plt.close()


def plot_paired_delta_histogram(signed_delta: np.ndarray, column: str, output_dir: Path) -> None:
    """Saves the histogram of signed deltas for the paired comparison."""
    plt.figure(figsize=(9, 5))
    plt.hist(signed_delta, bins=30)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(f"Signed delta of {column}")
    plt.ylabel("Count")
    plt.title(f"Paired signed delta histogram - {column}")
    plt.tight_layout()
    plt.savefig(output_dir / "paired_delta_histogram.png", dpi=180)
    plt.close()


def plot_paired_scatter(
    merged: pd.DataFrame,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Saves the paired scatter plot A vs B with a parity diagonal."""
    values_a = merged["value_a"].to_numpy(dtype=float)
    values_b = merged["value_b"].to_numpy(dtype=float)
    min_value = min(float(values_a.min()), float(values_b.min()))
    max_value = max(float(values_a.max()), float(values_b.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(values_a, values_b, s=12, alpha=0.45)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1)
    plt.xlabel(f"{label_a} {column}")
    plt.ylabel(f"{label_b} {column}")
    plt.title(f"Paired scatter - {column}")
    plt.tight_layout()
    plt.savefig(output_dir / "paired_scatter.png", dpi=180)
    plt.close()


# ====================================
# Section dedicated to the text report
# ====================================
def print_unpaired_group_summary(group: UnpairedGroupStats, metric_name: str) -> None:
    """Prints the CLI summary of an unpaired group."""
    print_section(f"Group {group.label}")
    print_info("Samples", str(group.n))
    print_info("Mean", color_metric_value(metric_name, group.mean))
    print_info("Median", color_metric_value(metric_name, group.median))
    print_info("IQR", color_distance(group.iqr, 0.05, 0.10))

    for key, value in group.threshold_shares.items():
        print_info(key, color_share(value))


def print_unpaired_cli_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Prints the CLI summary of the unpaired comparison."""
    print_section("Input")
    print_info("Metric", args.column)
    print_info("Direction", "higher is better" if args.higher_is_better else "lower is better")
    print_info("Output dir", str(output_dir))

    print_unpaired_group_summary(group_a, args.column)
    print_unpaired_group_summary(group_b, args.column)

    print_section("Distribution comparison")
    comparison_color = "green" if comparison.better_label != "tie" else "yellow"
    print_info("Mean favors", style(comparison.mean_favors, comparison_color) if comparison.mean_favors != "tie" else style("tie", "yellow"))
    print_info("Median favors", style(comparison.median_favors, comparison_color) if comparison.median_favors != "tie" else style("tie", "yellow"))
    print_info("Threshold favors", style(comparison.threshold_favors, comparison_color) if comparison.threshold_favors != "tie" else style("tie", "yellow"))
    print_info("Wasserstein between groups", color_distance(comparison.wasserstein_between_groups, 0.03, 0.08))
    print_info("KS statistic", color_distance(comparison.ks_statistic, 0.08, 0.18))
    print_info("KS p-value", color_pvalue(comparison.ks_pvalue))
    print_info("Mann-Whitney U", f"{comparison.mannwhitney_u:.6f}")
    print_info("Mann-Whitney p-value", color_pvalue(comparison.mannwhitney_pvalue))

    print_section("Conclusion")
    print(style(f"Overall unpaired comparison favors: {comparison.better_label}", "bold", comparison_color))
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


def print_paired_cli_summary(summary: PairedSummary, args: argparse.Namespace, output_dir: Path) -> None:
    """Prints the CLI summary of the paired comparison."""
    print_section("Input")
    print_info("Metric", args.column)
    print_info("Direction", "higher is better" if args.higher_is_better else "lower is better")
    print_info("Output dir", str(output_dir))

    print_section("Paired comparison")
    print_info("Paired samples", str(summary.n_pairs))
    print_info("Tolerance", f"{summary.tolerance:.6f}")
    print_info("Mean signed delta", color_signed_delta(summary.mean_signed_delta))
    print_info("Median signed delta", color_signed_delta(summary.median_signed_delta))
    print_info(f"Share {summary.label_b} better", color_share(summary.share_b_better))
    print_info(f"Share {summary.label_a} better", color_share(summary.share_a_better))
    print_info("Share equal", color_share(summary.share_equal))
    print_info("Wilcoxon statistic", f"{summary.wilcoxon_statistic:.6f}")
    print_info("Wilcoxon p-value", color_pvalue(summary.wilcoxon_pvalue))

    conclusion_color = "green" if summary.better_label != "tie" else "yellow"
    print_section("Conclusion")
    print(style(f"Overall paired comparison favors: {summary.better_label}", "bold", conclusion_color))
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


# =====================================
# Section dedicated to the main flow
# =====================================
def run_unpaired(args: argparse.Namespace) -> None:
    """Runs the complete flow for the comparison between unpaired distributions."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values_a = load_metric_values(args.csv_a, args.column)
    values_b = load_metric_values(args.csv_b, args.column)

    group_a = compute_unpaired_group_stats(
        values=values_a,
        label=args.label_a,
        thresholds=args.thresholds,
        higher_is_better=args.higher_is_better,
    )
    group_b = compute_unpaired_group_stats(
        values=values_b,
        label=args.label_b,
        thresholds=args.thresholds,
        higher_is_better=args.higher_is_better,
    )
    comparison = compute_unpaired_comparison(
        a=values_a,
        b=values_b,
        group_a=group_a,
        group_b=group_b,
        higher_is_better=args.higher_is_better,
    )

    save_unpaired_group_statistics(group_a, group_b, output_dir)
    save_unpaired_summary_json(group_a, group_b, comparison, output_dir)
    save_unpaired_report_txt(group_a, group_b, comparison, args, output_dir)

    edges = np.linspace(args.min_value, args.max_value, args.bins + 1)
    plot_distribution_histogram(values_a, values_b, edges, args.label_a, args.label_b, args.column, output_dir)
    plot_distribution_ecdf(values_a, values_b, args.label_a, args.label_b, args.column, output_dir)
    print_unpaired_cli_summary(group_a, group_b, comparison, args, output_dir)


def run_paired(args: argparse.Namespace) -> None:
    """Runs the complete flow for the paired comparison on the same samples."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = align_paired_frames(
        csv_a=args.csv_a,
        csv_b=args.csv_b,
        sample_id_column=args.sample_id_column,
        metric_column=args.column,
    )
    summary = compute_paired_summary(
        merged=merged,
        label_a=args.label_a,
        label_b=args.label_b,
        tolerance=args.tolerance,
        higher_is_better=args.higher_is_better,
    )

    save_paired_summary_json(summary, output_dir)
    save_paired_report_txt(summary, args, output_dir)

    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)
    signed_delta = raw_delta if args.higher_is_better else -raw_delta
    values_a = merged["value_a"].to_numpy(dtype=float)
    values_b = merged["value_b"].to_numpy(dtype=float)

    if args.min_value is None:
        min_value = min(float(values_a.min()), float(values_b.min()))
    else:
        min_value = args.min_value

    if args.max_value is None:
        max_value = max(float(values_a.max()), float(values_b.max()))
    else:
        max_value = args.max_value

    if min_value == max_value:
        padding = 0.5 if min_value == 0 else abs(min_value) * 0.05
        min_value -= padding
        max_value += padding

    edges = np.linspace(min_value, max_value, args.bins + 1)

    plot_distribution_histogram(
        values_a,
        values_b,
        edges,
        args.label_a,
        args.label_b,
        args.column,
        output_dir,
    )

    plot_distribution_ecdf(
        values_a,
        values_b,
        args.label_a,
        args.label_b,
        args.column,
        output_dir,
    )
    plot_paired_delta_histogram(signed_delta, args.column, output_dir)
    plot_paired_scatter(merged, args.label_a, args.label_b, args.column, output_dir)
    print_paired_cli_summary(summary, args, output_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    validate_direction(args)
    args.func(args)


if __name__ == "__main__":
    main()
