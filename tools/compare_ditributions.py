from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance


# =====================================
# Sezione dedicata alla colorazione CLI
# =====================================
ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "orange": "\033[38;5;208m",
}


def use_color() -> bool:
    """Restituisce True se ha senso usare colori ANSI in console."""
    return os.environ.get("NO_COLOR") is None and sys.stdout.isatty()


def style(text: str, *names: str) -> str:
    """Applica uno stile ANSI al testo, se la colorazione e abilitata."""
    if not use_color():
        return text
    prefix = "".join(ANSI[name] for name in names if name in ANSI)
    return prefix + text + ANSI["reset"]


def print_section(title: str) -> None:
    """Stampa un'intestazione di sezione leggibile in CLI."""
    print()
    print(style(f"=== {title} ===", "bold", "cyan"))


def print_info(label: str, value: str) -> None:
    """Stampa una singola riga etichetta."""
    print(f"{style(label + ':', 'bold', 'blue')} {value}")


def color_score(value: float, higher_is_better: bool, good: float, warn: float) -> str:
    """Colora un punteggio tenendo conto della direzione della metrica."""
    if higher_is_better:
        if value >= good:
            return style(f"{value:.6f}", "green")
        if value >= warn:
            return style(f"{value:.6f}", "yellow")
        return style(f"{value:.6f}", "red")

    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    return style(f"{value:.6f}", "red")


def color_distance(value: float, good: float = 0.05, warn: float = 0.10) -> str:
    """Colora una distanza: piu e piccola, meglio e."""
    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    return style(f"{value:.6f}", "red")


def color_pvalue(value: float) -> str:
    """Colora un p-value per facilitarne la lettura in CLI."""
    if value < 0.05:
        return style(f"{value:.6g}", "green")
    if value < 0.10:
        return style(f"{value:.6g}", "yellow")
    return style(f"{value:.6g}", "red")


# =========================================
# Sezione dedicata alle strutture risultati
# =========================================
@dataclass
class GroupStats:
    label: str
    n: int
    mean: float
    std: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float
    iqr: float
    share_above_thresholds: dict[str, float]
    gap_mean_from_ideal: float
    gap_median_from_ideal: float


@dataclass
class DistanceResults:
    wasserstein_between_groups: float
    wasserstein_a_to_ideal: float
    wasserstein_b_to_ideal: float
    jensen_shannon_distance: float


@dataclass
class TestResults:
    mannwhitney_u: float
    mannwhitney_pvalue: float
    ks_statistic: float
    ks_pvalue: float


# ==========================
# Sezione dedicata al parser
# ==========================
def add_direction_arguments(parser: argparse.ArgumentParser) -> None:
    """Aggiunge gli argomenti che definiscono la direzione della metrica."""
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


def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser principale dello script."""
    parser = argparse.ArgumentParser(
        prog="python tools/compare_distributions.py",
        description=(
            "Compare two independent metric distributions from CSV files. "
            "Inputs can be per_image_metrics.csv files or directories containing them."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/compare_distributions.py \\\n"
            "      --csv-a local_workspace/results/run_a/evaluation \\\n"
            "      --csv-b local_workspace/results/run_b/evaluation \\\n"
            "      --label-a P-256 \\\n"
            "      --label-b P-512 \\\n"
            "      --column ssim \\\n"
            "      --higher-is-better \\\n"
            "      --output-dir local_workspace/results/comparisons/ssim\n"
            "\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
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
        "--ideal-value",
        type=float,
        default=1.0,
        help="Ideal target value used for gap analysis and Wasserstein-to-ideal.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the common histogram used by Jensen-Shannon.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.8, 0.9, 0.95],
        help="Thresholds used for share-above or share-below statistics.",
    )
    add_direction_arguments(parser)
    return parser


def validate_direction(args: argparse.Namespace) -> None:
    """Verifica che sia stata scelta esattamente una direzione della metrica."""
    if args.higher_is_better == args.lower_is_better:
        raise SystemExit(
            "Choose exactly one between --higher-is-better and --lower-is-better."
        )


# ========================================
# Sezione dedicata alle funzioni utilities
# ========================================
def resolve_input_csv(path_like: str | Path) -> Path:
    """Risolve un CSV diretto o una directory contenente per_image_metrics.csv."""
    path = Path(path_like)

    if path.is_dir():
        candidate = path / "per_image_metrics.csv"
        if candidate.exists():
            return candidate
        raise ValueError(f"Directory {path} does not contain per_image_metrics.csv")

    if path.is_file():
        return path

    raise ValueError(f"Input path does not exist: {path}")


def load_metric_values(csv_path: str | Path, column: str) -> np.ndarray:
    """Carica una colonna numerica da CSV scartando valori mancanti o non validi."""
    resolved_csv = resolve_input_csv(csv_path)
    df = pd.read_csv(resolved_csv)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {resolved_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError(
            f"No valid numeric values found in column '{column}' of {resolved_csv}"
        )

    return values


def choose_better_label(group_a: GroupStats, group_b: GroupStats) -> str:
    """Restituisce il gruppo piu vicino all'ideale in base al gap medio."""
    if group_b.gap_mean_from_ideal < group_a.gap_mean_from_ideal:
        return group_b.label
    if group_b.gap_mean_from_ideal > group_a.gap_mean_from_ideal:
        return group_a.label
    return "tie"


# ==========================================
# Sezione dedicata al calcolo delle metriche
# ==========================================
def compute_group_stats(
    values: np.ndarray,
    label: str,
    thresholds: Iterable[float],
    ideal_value: float,
    higher_is_better: bool,
) -> GroupStats:
    """Calcola le statistiche descrittive di un singolo gruppo."""
    p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])

    if higher_is_better:
        shares = {f"ge_{t:.2f}": float(np.mean(values >= t)) for t in thresholds}
        gap = ideal_value - values
    else:
        shares = {f"le_{t:.2f}": float(np.mean(values <= t)) for t in thresholds}
        gap = values - ideal_value

    return GroupStats(
        label=label,
        n=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        median=float(np.median(values)),
        p10=float(p10),
        p25=float(p25),
        p75=float(p75),
        p90=float(p90),
        iqr=float(p75 - p25),
        share_above_thresholds=shares,
        gap_mean_from_ideal=float(np.mean(gap)),
        gap_median_from_ideal=float(np.median(gap)),
    )


def common_probability_histograms(
    a: np.ndarray,
    b: np.ndarray,
    bins: int,
    min_value: float,
    max_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Costruisce due istogrammi di probabilita sulla stessa griglia di bin."""
    edges = np.linspace(min_value, max_value, bins + 1)
    hist_a, _ = np.histogram(a, bins=edges, density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)

    prob_a = hist_a.astype(float)
    prob_b = hist_b.astype(float)
    prob_a /= prob_a.sum()
    prob_b /= prob_b.sum()

    return prob_a, prob_b, edges


def compute_distances(
    a: np.ndarray,
    b: np.ndarray,
    bins: int,
    min_value: float,
    max_value: float,
    ideal_value: float,
) -> DistanceResults:
    """Calcola le distanze tra gruppi e la distanza di ciascuno dall'ideale."""
    prob_a, prob_b, _ = common_probability_histograms(a, b, bins, min_value, max_value)
    ideal_a = np.full_like(a, ideal_value, dtype=float)
    ideal_b = np.full_like(b, ideal_value, dtype=float)

    return DistanceResults(
        wasserstein_between_groups=float(wasserstein_distance(a, b)),
        wasserstein_a_to_ideal=float(wasserstein_distance(a, ideal_a)),
        wasserstein_b_to_ideal=float(wasserstein_distance(b, ideal_b)),
        jensen_shannon_distance=float(jensenshannon(prob_a, prob_b, base=2.0)),
    )


def compute_tests(a: np.ndarray, b: np.ndarray) -> TestResults:
    """Calcola i test statistici tra i due gruppi."""
    mw = mannwhitneyu(a, b, alternative="two-sided")
    ks = ks_2samp(a, b, alternative="two-sided")

    return TestResults(
        mannwhitney_u=float(mw.statistic),
        mannwhitney_pvalue=float(mw.pvalue),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
    )


# =======================================
# Sezione dedicata alla scrittura output
# =======================================
def save_group_statistics(group_a: GroupStats, group_b: GroupStats, output_dir: Path) -> None:
    """Salva group_statistics.csv con una riga per gruppo."""
    pd.DataFrame([asdict(group_a), asdict(group_b)]).to_csv(
        output_dir / "group_statistics.csv",
        index=False,
    )


def save_summary_json(
    group_a: GroupStats,
    group_b: GroupStats,
    distances: DistanceResults,
    tests: TestResults,
    output_dir: Path,
) -> None:
    """Salva un riepilogo JSON con tutte le strutture principali."""
    payload = {
        "group_a": asdict(group_a),
        "group_b": asdict(group_b),
        "distances": asdict(distances),
        "tests": asdict(tests),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def save_report_txt(
    group_a: GroupStats,
    group_b: GroupStats,
    distances: DistanceResults,
    tests: TestResults,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Salva report.txt con un riepilogo testuale leggibile."""
    better = choose_better_label(group_a, group_b)
    direction = "higher is better" if args.higher_is_better else "lower is better"
    lines = [
        f"Metric: {args.column}",
        f"Direction: {direction}",
        "",
        (
            f"{group_a.label}: n={group_a.n}, mean={group_a.mean:.6f}, "
            f"std={group_a.std:.6f}, median={group_a.median:.6f}, "
            f"IQR={group_a.iqr:.6f}, gap_mean={group_a.gap_mean_from_ideal:.6f}"
        ),
        (
            f"{group_b.label}: n={group_b.n}, mean={group_b.mean:.6f}, "
            f"std={group_b.std:.6f}, median={group_b.median:.6f}, "
            f"IQR={group_b.iqr:.6f}, gap_mean={group_b.gap_mean_from_ideal:.6f}"
        ),
        "",
        f"Wasserstein between groups: {distances.wasserstein_between_groups:.6f}",
        f"Wasserstein to ideal ({group_a.label}): {distances.wasserstein_a_to_ideal:.6f}",
        f"Wasserstein to ideal ({group_b.label}): {distances.wasserstein_b_to_ideal:.6f}",
        f"Jensen-Shannon distance: {distances.jensen_shannon_distance:.6f}",
        "",
        f"Mann-Whitney U statistic: {tests.mannwhitney_u:.6f}",
        f"Mann-Whitney U p-value: {tests.mannwhitney_pvalue:.6g}",
        f"KS statistic: {tests.ks_statistic:.6f}",
        f"KS p-value: {tests.ks_pvalue:.6g}",
        "",
        f"Closer to the ideal value according to mean gap: {better}",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


# ===========================
# Sezione dedicata ai grafici
# ===========================
def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce la funzione di distribuzione empirica del campione."""
    x = np.sort(values)
    y = np.arange(1, values.size + 1) / values.size
    return x, y


def plot_histogram(
    a: np.ndarray,
    b: np.ndarray,
    edges: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Salva l'istogramma di confronto tra i due gruppi."""
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


def plot_ecdf(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Salva il confronto tra le distribuzioni empiriche cumulative."""
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


def plot_boxplot(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Salva il boxplot dei due gruppi."""
    plt.figure(figsize=(8, 5))
    plt.boxplot([a, b], tick_labels=[label_a, label_b], vert=True)
    plt.ylabel(column)
    plt.title(f"Boxplot - {column}")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot.png", dpi=180)
    plt.close()


# ====================================
# Sezione dedicata al report testuale
# ====================================
def format_score_thresholds(higher_is_better: bool) -> dict[str, tuple[float, float]]:
    """Restituisce le soglie CLI da usare per le statistiche descrittive."""
    if higher_is_better:
        return {
            "mean": (0.90, 0.80),
            "median": (0.90, 0.80),
            "p10": (0.80, 0.70),
            "p25": (0.85, 0.75),
            "p75": (0.92, 0.85),
            "p90": (0.95, 0.90),
        }

    return {
        "mean": (0.05, 0.10),
        "median": (0.05, 0.10),
        "p10": (0.10, 0.20),
        "p25": (0.08, 0.15),
        "p75": (0.04, 0.08),
        "p90": (0.03, 0.06),
    }


def print_group_summary(group: GroupStats, higher_is_better: bool) -> None:
    """Stampa il riepilogo CLI di un singolo gruppo."""
    thresholds = format_score_thresholds(higher_is_better)

    print_section(f"Group {group.label}")
    print_info("Samples", str(group.n))
    print_info("Mean", color_score(group.mean, higher_is_better, *thresholds["mean"]))
    print_info("Std", color_distance(group.std, 0.03, 0.08))
    print_info("Median", color_score(group.median, higher_is_better, *thresholds["median"]))
    print_info("P10", color_score(group.p10, higher_is_better, *thresholds["p10"]))
    print_info("P25", color_score(group.p25, higher_is_better, *thresholds["p25"]))
    print_info("P75", color_score(group.p75, higher_is_better, *thresholds["p75"]))
    print_info("P90", color_score(group.p90, higher_is_better, *thresholds["p90"]))
    print_info("IQR", color_distance(group.iqr, 0.05, 0.10))

    for key, value in group.share_above_thresholds.items():
        print_info(key, color_score(value, True, 0.70, 0.40))

    print_info("Gap mean", color_distance(group.gap_mean_from_ideal, 0.05, 0.10))
    print_info("Gap median", color_distance(group.gap_median_from_ideal, 0.05, 0.10))


def print_cli_summary(
    group_a: GroupStats,
    group_b: GroupStats,
    distances: DistanceResults,
    tests: TestResults,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Stampa in CLI il riepilogo finale di tutto il confronto."""
    higher_is_better = args.higher_is_better

    print_section("Input")
    print_info("Metric", args.column)
    print_info("Direction", "higher is better" if higher_is_better else "lower is better")
    print_info("Output dir", str(output_dir))

    print_group_summary(group_a, higher_is_better)
    print_group_summary(group_b, higher_is_better)

    print_section("Distances")
    print_info(
        "Wasserstein between groups",
        color_distance(distances.wasserstein_between_groups, 0.03, 0.08),
    )
    print_info(
        f"Wasserstein {group_a.label} to ideal",
        color_distance(distances.wasserstein_a_to_ideal, 0.05, 0.10),
    )
    print_info(
        f"Wasserstein {group_b.label} to ideal",
        color_distance(distances.wasserstein_b_to_ideal, 0.05, 0.10),
    )
    print_info(
        "Jensen-Shannon",
        color_distance(distances.jensen_shannon_distance, 0.08, 0.18),
    )

    print_section("Statistical tests")
    print_info("Mann-Whitney U", f"{tests.mannwhitney_u:.6f}")
    print_info("Mann-Whitney p-value", color_pvalue(tests.mannwhitney_pvalue))
    print_info("KS statistic", color_distance(tests.ks_statistic, 0.08, 0.18))
    print_info("KS p-value", color_pvalue(tests.ks_pvalue))

    better = choose_better_label(group_a, group_b)
    summary_color = "green" if better != "tie" else "yellow"

    print_section("Conclusion")
    print(
        style(
            f"Closer to the ideal value according to mean gap: {better}",
            "bold",
            summary_color,
        )
    )
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


# =====================================
# Sezione dedicata al flusso principale
# =====================================
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_direction(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a = load_metric_values(args.csv_a, args.column)
    b = load_metric_values(args.csv_b, args.column)

    group_a = compute_group_stats(
        a,
        args.label_a,
        args.thresholds,
        args.ideal_value,
        args.higher_is_better,
    )
    group_b = compute_group_stats(
        b,
        args.label_b,
        args.thresholds,
        args.ideal_value,
        args.higher_is_better,
    )
    distances = compute_distances(
        a,
        b,
        args.bins,
        args.min_value,
        args.max_value,
        args.ideal_value,
    )
    tests = compute_tests(a, b)

    save_group_statistics(group_a, group_b, output_dir)
    save_summary_json(group_a, group_b, distances, tests, output_dir)
    save_report_txt(group_a, group_b, distances, tests, args, output_dir)

    _, _, edges = common_probability_histograms(
        a,
        b,
        args.bins,
        args.min_value,
        args.max_value,
    )
    plot_histogram(a, b, edges, args.label_a, args.label_b, args.column, output_dir)
    plot_ecdf(a, b, args.label_a, args.label_b, args.column, output_dir)
    plot_boxplot(a, b, args.label_a, args.label_b, args.column, output_dir)
    print_cli_summary(group_a, group_b, distances, tests, args, output_dir)


if __name__ == "__main__":
    main()
