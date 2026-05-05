from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance, wilcoxon


METRIC_CONFIGS = {
    "ssim": {
        "higher_is_better": True,
        "min_value": 0.0,
        "max_value": 1.0,
        "thresholds": [0.65, 0.75, 0.85],
    },
    "psnr": {
        "higher_is_better": True,
        "min_value": 0.0,
        "max_value": 60.0,
        "thresholds": [15.0, 20.0, 25.0],
    },
    "mae": {
        "higher_is_better": False,
        "min_value": 0.0,
        "max_value": 1.0,
        "thresholds": [0.06, 0.10, 0.16],
    },
    "rmse": {
        "higher_is_better": False,
        "min_value": 0.0,
        "max_value": 1.0,
        "thresholds": [0.08, 0.12, 0.20],
    },
    "mse": {
        "higher_is_better": False,
        "min_value": 0.0,
        "max_value": 0.1,
        "thresholds": [0.0036, 0.0100, 0.0256],
    },
    "pcc_gray": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.80, 0.90, 0.95],
    },
    "pcc_rgb_mean": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.80, 0.90, 0.95],
    },
    "pcc_r": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.80, 0.90, 0.95],
    },
    "pcc_g": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.80, 0.90, 0.95],
    },
    "pcc_b": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.80, 0.90, 0.95],
    },
}


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


def is_higher_better_metric(metric_name: str) -> bool:
    """Restituisce True se per la metrica valori più alti sono migliori."""
    if metric_name not in METRIC_CONFIGS:
        raise ValueError(
            f"Unsupported metric column '{metric_name}'. "
            f"Supported metrics: {', '.join(METRIC_CONFIGS)}"
        )

    return bool(METRIC_CONFIGS[metric_name]["higher_is_better"])


def get_metric_config(column: str) -> dict[str, object]:
    """Restituisce la configurazione predefinita di una metrica."""
    if column not in METRIC_CONFIGS:
        raise ValueError(
            f"Unsupported metric column '{column}'. "
            f"Supported metrics: {', '.join(METRIC_CONFIGS)}"
        )

    return METRIC_CONFIGS[column]


def resolve_plot_range(args: argparse.Namespace) -> tuple[float, float]:
    """Risolvi il range dei grafici da CLI o configurazione metrica."""
    config = get_metric_config(args.column)

    min_value = args.min_value if args.min_value is not None else float(config["min_value"])
    max_value = args.max_value if args.max_value is not None else float(config["max_value"])

    if min_value == max_value:
        padding = 0.5 if min_value == 0 else abs(min_value) * 0.05
        min_value -= padding
        max_value += padding

    return min_value, max_value


def resolve_thresholds(args: argparse.Namespace) -> list[float]:
    """Risolvi le soglie da CLI o configurazione metrica."""
    if getattr(args, "thresholds", None) is not None:
        return args.thresholds

    return list(get_metric_config(args.column)["thresholds"])


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


def load_metric_frame(csv_path: str | Path) -> pd.DataFrame:
    """Carica un CSV di metriche come DataFrame."""
    resolved_csv = resolve_input_csv(csv_path)
    return pd.read_csv(resolved_csv)


def load_metric_values(csv_path: str | Path, column: str) -> np.ndarray:
    """Carica una colonna numerica da CSV scartando valori mancanti o non validi."""
    df = load_metric_frame(csv_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError(f"No valid numeric values found in column '{column}'")

    return values


def choose_unpaired_better_label(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
) -> str:
    """Sceglie il gruppo migliore combinando media, mediana e quote su soglia."""
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
    """Sceglie il gruppo migliore nel paired combinando delta medio, mediano e share better."""
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
    """Calcola le statistiche descrittive essenziali di un gruppo non appaiato."""
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
    """Sceglie il gruppo favorito confrontando la media delle quote sopra/sotto soglia."""
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
    """Calcola i confronti principali tra due distribuzioni non appaiate."""
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
    """Allinea due CSV sullo stesso sample_id per il confronto paired."""
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
    """Calcola il riepilogo principale per il confronto paired."""
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
        better_label=choose_paired_better_label(
            mean_signed_delta=mean_signed_delta,
            median_signed_delta=median_signed_delta,
            share_b_better=share_b_better,
            share_a_better=share_a_better,
            label_a=label_a,
            label_b=label_b,
        ),
    )


def flatten_unpaired_group_stats(group: UnpairedGroupStats) -> dict[str, Any]:
    """Converte le statistiche di gruppo in una riga tabellare CSV."""
    row = {
        "label": group.label,
        "n": group.n,
        "mean": group.mean,
        "median": group.median,
        "iqr": group.iqr,
    }

    for threshold_name, share in group.threshold_shares.items():
        row[f"share_{threshold_name}"] = share

    return row


def save_unpaired_group_statistics(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    output_dir: Path,
) -> None:
    """Salva group_statistics.csv con una riga per ciascun gruppo."""
    rows = [
        flatten_unpaired_group_stats(group_a),
        flatten_unpaired_group_stats(group_b),
    ]
    pd.DataFrame(rows).to_csv(output_dir / "group_statistics.csv", index=False)


def save_unpaired_comparison_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Salva un riepilogo tabellare del confronto unpaired."""
    row = {
        "mode": "unpaired",
        "metric": args.column,
        "direction": "higher_is_better" if args.resolved_higher_is_better else "lower_is_better",
        "label_a": group_a.label,
        "label_b": group_b.label,
        "n_a": group_a.n,
        "n_b": group_b.n,
        "mean_a": group_a.mean,
        "mean_b": group_b.mean,
        "median_a": group_a.median,
        "median_b": group_b.median,
        "iqr_a": group_a.iqr,
        "iqr_b": group_b.iqr,
        "mean_favors": comparison.mean_favors,
        "median_favors": comparison.median_favors,
        "threshold_favors": comparison.threshold_favors,
        "wasserstein_between_groups": comparison.wasserstein_between_groups,
        "ks_statistic": comparison.ks_statistic,
        "ks_pvalue": comparison.ks_pvalue,
        "mannwhitney_u": comparison.mannwhitney_u,
        "mannwhitney_pvalue": comparison.mannwhitney_pvalue,
        "better_label": comparison.better_label,
    }

    pd.DataFrame([row]).to_csv(output_dir / "comparison_summary.csv", index=False)


def save_paired_comparison_summary(
    summary: PairedSummary,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Salva un riepilogo tabellare del confronto paired."""
    row = {
        "mode": "paired",
        "metric": args.column,
        "direction": "higher_is_better" if args.resolved_higher_is_better else "lower_is_better",
        "label_a": summary.label_a,
        "label_b": summary.label_b,
        "n_pairs": summary.n_pairs,
        "tolerance": summary.tolerance,
        "mean_signed_delta": summary.mean_signed_delta,
        "median_signed_delta": summary.median_signed_delta,
        "share_b_better": summary.share_b_better,
        "share_a_better": summary.share_a_better,
        "share_equal": summary.share_equal,
        "wilcoxon_statistic": summary.wilcoxon_statistic,
        "wilcoxon_pvalue": summary.wilcoxon_pvalue,
        "better_label": summary.better_label,
    }

    pd.DataFrame([row]).to_csv(output_dir / "comparison_summary.csv", index=False)


def save_paired_sample_deltas(
    merged: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Salva un CSV con il confronto paired campione per campione."""
    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)

    if args.resolved_higher_is_better:
        signed_delta = raw_delta
    else:
        signed_delta = -raw_delta

    result = merged.copy()
    result["raw_delta_b_minus_a"] = raw_delta
    result["signed_delta"] = signed_delta

    result["winner"] = np.where(
        signed_delta > args.tolerance,
        args.resolved_label_b,
        np.where(
            signed_delta < -args.tolerance,
            args.resolved_label_a,
            "equal",
        ),
    )

    result.to_csv(output_dir / "paired_sample_deltas.csv", index=False)
    

def resolve_run_path(run_path: str | Path) -> Path:
    """Risolve e valida la directory di un run."""
    path = Path(run_path).resolve()

    if not path.is_dir():
        raise NotADirectoryError(f"Run directory not found: {path}")

    return path


def resolve_metrics_csv_from_run(run_path: str | Path) -> Path:
    """Restituisce evaluation/per_image_metrics.csv a partire dalla directory di un run."""
    run_dir = resolve_run_path(run_path)
    csv_path = run_dir / "evaluation" / "per_image_metrics.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find per_image_metrics.csv for run '{run_dir.name}'. "
            f"Expected: {csv_path}"
        )

    return csv_path


def infer_label_from_input(run_path: str | Path | None, csv_path: str | Path | None, fallback: str) -> str:
    """Inferisce una label leggibile da run path o CSV path."""
    if run_path is not None:
        return Path(run_path).resolve().name

    if csv_path is not None:
        path = Path(csv_path).resolve()

        if path.name == "per_image_metrics.csv" and path.parent.name == "evaluation":
            return path.parent.parent.name

        return path.stem

    return fallback


def resolve_comparison_inputs(args: argparse.Namespace) -> None:
    """
    Risolve CSV e label a partire da --run-a/--run-b oppure --csv-a/--csv-b.

    La funzione modifica args aggiungendo:
    - args.resolved_csv_a
    - args.resolved_csv_b
    - args.resolved_label_a
    - args.resolved_label_b
    """
    if args.run_a is None and args.csv_a is None:
        raise ValueError("You must provide either --run-a or --csv-a.")

    if args.run_b is None and args.csv_b is None:
        raise ValueError("You must provide either --run-b or --csv-b.")

    if args.run_a is not None:
        args.resolved_csv_a = resolve_metrics_csv_from_run(args.run_a)
    else:
        args.resolved_csv_a = resolve_input_csv(args.csv_a)

    if args.run_b is not None:
        args.resolved_csv_b = resolve_metrics_csv_from_run(args.run_b)
    else:
        args.resolved_csv_b = resolve_input_csv(args.csv_b)

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
    """
    Prova a ricavare la cartella results/ a partire dai run o dai CSV risolti.
    """
    if args.run_a is not None:
        run_a = Path(args.run_a).resolve()
        if run_a.parent.name == "results":
            return run_a.parent

    if args.run_b is not None:
        run_b = Path(args.run_b).resolve()
        if run_b.parent.name == "results":
            return run_b.parent

    csv_a = Path(args.resolved_csv_a).resolve()
    parts = csv_a.parts

    if "results" in parts:
        results_index = parts.index("results")
        return Path(*parts[: results_index + 1])

    return Path("local_workspace") / "results"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """
    Risolve la directory di output.

    Se --output-dir è specificato, usa quello.
    Altrimenti salva in:
    results/comparisons/LABEL_A_vs_LABEL_B/MODE_METRIC/
    """
    if args.output_dir is not None:
        return Path(args.output_dir)

    results_root = infer_results_root_from_inputs(args)
    comparison_name = f"{args.resolved_label_a}_vs_{args.resolved_label_b}"
    metric_dir_name = f"{args.mode}_{args.column}"

    return results_root / "comparisons" / comparison_name / metric_dir_name

