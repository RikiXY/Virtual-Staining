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
        "thresholds": [0.85, 0.90, 0.95],
    },
    "pcc_rgb_mean": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.85, 0.90, 0.95],
    },
    "pcc_r": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.85, 0.90, 0.95],
    },
    "pcc_g": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.85, 0.90, 0.95],
    },
    "pcc_b": {
        "higher_is_better": True,
        "min_value": -1.0,
        "max_value": 1.0,
        "thresholds": [0.85, 0.90, 0.95],
    },
}


QUANTILE_LEVELS = {
    "q10": 10,
    "q25": 25,
    "q50": 50,
    "q75": 75,
    "q90": 90,
}

P_VALUE_SIGNIFICANCE_THRESHOLD = 0.05
COMMON_LANGUAGE_DEAD_ZONE_LOW = 0.48
COMMON_LANGUAGE_DEAD_ZONE_HIGH = 0.52


@dataclass
class UnpairedGroupStats:
    label: str
    n: int
    mean: float
    median: float
    std: float
    min_value: float
    max_value: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    iqr: float
    worst_tail_value: float
    threshold_shares: dict[str, float]


@dataclass
class UnpairedComparison:
    better_label: str
    decision_strength: str
    reason: str

    score_a: float
    score_b: float
    score_diff: float

    signed_quantile_shift: float
    quantile_shift_favors: str
    quantile_improvement_rate_a: float
    quantile_improvement_rate_b: float
    quantile_improvement_favors: str

    threshold_favors: str
    threshold_share_mean_a: float
    threshold_share_mean_b: float

    worst_tail_favors: str

    common_language_b_better: float
    common_language_favors: str

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
    std_signed_delta: float
    min_signed_delta: float
    max_signed_delta: float
    q10_signed_delta: float
    q25_signed_delta: float
    q50_signed_delta: float
    q75_signed_delta: float
    q90_signed_delta: float

    share_b_better: float
    share_a_better: float
    share_equal: float

    wilcoxon_statistic: float
    wilcoxon_pvalue: float

    score_a: float
    score_b: float
    score_diff: float

    median_delta_favors: str
    share_improvement_favors: str
    worst_delta_favors: str
    mean_delta_favors: str
    wilcoxon_favors: str

    better_label: str
    decision_strength: str
    reason: str


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


def get_quantiles(values: np.ndarray) -> dict[str, float]:
    """Calcola i quantili principali usati nei confronti."""
    return {
        name: float(np.percentile(values, percentile))
        for name, percentile in QUANTILE_LEVELS.items()
    }


def oriented_difference(
    value_a: float,
    value_b: float,
    higher_is_better: bool,
) -> float:
    """
    Restituisce una differenza orientata.

    Positivo = B migliora A.
    Negativo = A migliora B.
    """
    if higher_is_better:
        return value_b - value_a
    return value_a - value_b


def favor_from_signed_value(
    signed_value: float,
    label_a: str,
    label_b: str,
    tolerance: float = 0.0,
) -> str:
    """Converte un valore signed in label favorita."""
    if signed_value > tolerance:
        return label_b
    if signed_value < -tolerance:
        return label_a
    return "tie"


def mean_threshold_share(shares: dict[str, float]) -> float:
    """Media delle threshold shares."""
    if not shares:
        return 0.0
    return float(np.mean(list(shares.values())))


def compute_common_language_b_better(
    a: np.ndarray,
    b: np.ndarray,
    higher_is_better: bool,
) -> float:
    """
    Calcola la common language effect size.

    Indica la probabilità che un campione casuale di B sia migliore
    di un campione casuale di A.
    """
    a_values = np.asarray(a, dtype=float)
    b_values = np.asarray(b, dtype=float)

    if higher_is_better:
        wins = b_values[:, None] > a_values[None, :]
        ties = b_values[:, None] == a_values[None, :]
    else:
        wins = b_values[:, None] < a_values[None, :]
        ties = b_values[:, None] == a_values[None, :]

    return float(np.mean(wins) + 0.5 * np.mean(ties))


def choose_common_language_favors(
    common_language_b_better: float,
    label_a: str,
    label_b: str,
) -> str:
    """Decide quale run è favorito dal common language effect size."""
    if common_language_b_better > COMMON_LANGUAGE_DEAD_ZONE_HIGH:
        return label_b
    if common_language_b_better < COMMON_LANGUAGE_DEAD_ZONE_LOW:
        return label_a
    return "tie"


def build_reason(
    better_label: str,
    decision_strength: str,
    favored_criteria: list[str],
    counter_criteria: list[str],
) -> str:
    """Costruisce una spiegazione testuale breve della decisione."""
    if better_label == "tie":
        return (
            "No clear overall winner: the criteria are balanced or the observed "
            "differences are too small."
        )

    if favored_criteria:
        favored_text = ", ".join(favored_criteria)
    else:
        favored_text = "no single dominant criterion"

    reason = (
        f"{better_label} is favored by {favored_text}. "
        f"Decision strength is {decision_strength}."
    )

    if counter_criteria:
        reason += f" Counter-signals: {', '.join(counter_criteria)}."

    return reason


def score_unpaired_comparison(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    signed_quantile_shift: float,
    quantile_shift_favors: str,
    threshold_favors: str,
    worst_tail_favors: str,
    quantile_improvement_favors: str,
    quantile_improvement_rate_a: float,
    quantile_improvement_rate_b: float,
    common_language_favors: str,
) -> tuple[float, float, list[str], list[str]]:
    """
    Applica lo schema di punteggio unpaired.

    Totale massimo teorico: 7.5 punti.
    """
    score_a = 0.0
    score_b = 0.0
    favored_criteria: list[str] = []
    counter_criteria: list[str] = []

    def add_points(favored_label: str, points: float, criterion_name: str) -> None:
        nonlocal score_a, score_b

        if favored_label == group_a.label:
            score_a += points
        elif favored_label == group_b.label:
            score_b += points
        else:
            return

        favored_criteria.append(criterion_name)

    add_points(quantile_shift_favors, 2.0, "signed quantile shift")
    add_points(threshold_favors, 2.0, "threshold shares")
    add_points(worst_tail_favors, 1.5, "worst-tail behavior")

    # Ogni quantile migliorato vale 0.3 punti.
    score_a += quantile_improvement_rate_a * 1.5
    score_b += quantile_improvement_rate_b * 1.5

    if quantile_improvement_favors != "tie":
        favored_criteria.append("quantile improvement rate")

    add_points(common_language_favors, 0.5, "common language effect size")

    return score_a, score_b, favored_criteria, counter_criteria


def choose_unpaired_decision_strength(
    score_diff: float,
    wasserstein_value: float,
    signed_quantile_shift: float,
) -> str:
    """
    Valuta la forza della decisione unpaired.

    La Wasserstein distance non decide il vincitore, ma aiuta a capire
    quanto le distribuzioni siano separate.
    """
    abs_shift = abs(signed_quantile_shift)

    if score_diff < 2.0 or wasserstein_value < 1e-6 or abs_shift < 1e-6:
        return "weak"

    if score_diff >= 4.0 and wasserstein_value >= 0.03 and abs_shift > 0:
        return "strong"

    return "moderate"


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
    """Calcola statistiche descrittive ricche di un gruppo non appaiato."""
    quantiles = get_quantiles(values)

    if higher_is_better:
        shares = {
            f"ge_{threshold:.4g}": float(np.mean(values >= threshold))
            for threshold in thresholds
        }
        worst_tail_value = quantiles["q10"]
    else:
        shares = {
            f"le_{threshold:.4g}": float(np.mean(values <= threshold))
            for threshold in thresholds
        }
        worst_tail_value = quantiles["q90"]

    return UnpairedGroupStats(
        label=label,
        n=int(values.size),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        q10=quantiles["q10"],
        q25=quantiles["q25"],
        q50=quantiles["q50"],
        q75=quantiles["q75"],
        q90=quantiles["q90"],
        iqr=float(quantiles["q75"] - quantiles["q25"]),
        worst_tail_value=float(worst_tail_value),
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
    """Calcola il confronto completo tra due distribuzioni non appaiate."""
    mann_whitney = mannwhitneyu(a, b, alternative="two-sided")
    ks = ks_2samp(a, b, alternative="two-sided")
    wasserstein_value = float(wasserstein_distance(a, b))

    quantile_deltas: dict[str, float] = {}
    b_quantile_wins = 0
    a_quantile_wins = 0

    for name in QUANTILE_LEVELS:
        value_a = getattr(group_a, name)
        value_b = getattr(group_b, name)
        delta = oriented_difference(value_a, value_b, higher_is_better)
        quantile_deltas[name] = delta

        if delta > 0:
            b_quantile_wins += 1
        elif delta < 0:
            a_quantile_wins += 1

    signed_quantile_shift = float(np.mean(list(quantile_deltas.values())))
    quantile_shift_favors = favor_from_signed_value(
        signed_quantile_shift,
        group_a.label,
        group_b.label,
    )

    quantile_improvement_rate_b = b_quantile_wins / len(QUANTILE_LEVELS)
    quantile_improvement_rate_a = a_quantile_wins / len(QUANTILE_LEVELS)

    if quantile_improvement_rate_b > quantile_improvement_rate_a:
        quantile_improvement_favors = group_b.label
    elif quantile_improvement_rate_a > quantile_improvement_rate_b:
        quantile_improvement_favors = group_a.label
    else:
        quantile_improvement_favors = "tie"

    threshold_share_mean_a = mean_threshold_share(group_a.threshold_shares)
    threshold_share_mean_b = mean_threshold_share(group_b.threshold_shares)

    if threshold_share_mean_b > threshold_share_mean_a:
        threshold_favors = group_b.label
    elif threshold_share_mean_a > threshold_share_mean_b:
        threshold_favors = group_a.label
    else:
        threshold_favors = "tie"

    worst_tail_delta = oriented_difference(
        group_a.worst_tail_value,
        group_b.worst_tail_value,
        higher_is_better,
    )
    worst_tail_favors = favor_from_signed_value(
        worst_tail_delta,
        group_a.label,
        group_b.label,
    )

    common_language_b_better = compute_common_language_b_better(
        a=a,
        b=b,
        higher_is_better=higher_is_better,
    )
    common_language_favors = choose_common_language_favors(
        common_language_b_better,
        group_a.label,
        group_b.label,
    )

    score_a, score_b, favored_criteria, counter_criteria = score_unpaired_comparison(
        group_a=group_a,
        group_b=group_b,
        signed_quantile_shift=signed_quantile_shift,
        quantile_shift_favors=quantile_shift_favors,
        threshold_favors=threshold_favors,
        worst_tail_favors=worst_tail_favors,
        quantile_improvement_favors=quantile_improvement_favors,
        quantile_improvement_rate_a=quantile_improvement_rate_a,
        quantile_improvement_rate_b=quantile_improvement_rate_b,
        common_language_favors=common_language_favors,
    )

    if score_b > score_a:
        better_label = group_b.label
    elif score_a > score_b:
        better_label = group_a.label
    else:
        better_label = "tie"

    score_diff = abs(score_a - score_b)

    decision_strength = choose_unpaired_decision_strength(
        score_diff=score_diff,
        wasserstein_value=wasserstein_value,
        signed_quantile_shift=signed_quantile_shift,
    )

    reason = build_reason(
        better_label=better_label,
        decision_strength=decision_strength,
        favored_criteria=favored_criteria,
        counter_criteria=counter_criteria,
    )

    return UnpairedComparison(
        better_label=better_label,
        decision_strength=decision_strength,
        reason=reason,
        score_a=score_a,
        score_b=score_b,
        score_diff=score_diff,
        signed_quantile_shift=signed_quantile_shift,
        quantile_shift_favors=quantile_shift_favors,
        quantile_improvement_rate_a=quantile_improvement_rate_a,
        quantile_improvement_rate_b=quantile_improvement_rate_b,
        quantile_improvement_favors=quantile_improvement_favors,
        threshold_favors=threshold_favors,
        threshold_share_mean_a=threshold_share_mean_a,
        threshold_share_mean_b=threshold_share_mean_b,
        worst_tail_favors=worst_tail_favors,
        common_language_b_better=common_language_b_better,
        common_language_favors=common_language_favors,
        wasserstein_between_groups=wasserstein_value,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        mannwhitney_u=float(mann_whitney.statistic),
        mannwhitney_pvalue=float(mann_whitney.pvalue),
    )


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


def score_paired_comparison(
    label_a: str,
    label_b: str,
    median_signed_delta: float,
    mean_signed_delta: float,
    q10_signed_delta: float,
    share_b_better: float,
    share_a_better: float,
    wilcoxon_pvalue: float,
    tolerance: float,
) -> tuple[float, float, str, str, str, str, str, list[str], list[str]]:
    """
    Applica lo schema di punteggio paired.

    Totale massimo teorico: 7 punti.
    """
    score_a = 0.0
    score_b = 0.0
    favored_criteria: list[str] = []
    counter_criteria: list[str] = []

    median_delta_favors = favor_from_signed_value(
        median_signed_delta,
        label_a,
        label_b,
        tolerance,
    )
    if median_delta_favors == label_b:
        score_b += 2.0
        favored_criteria.append("median delta")
    elif median_delta_favors == label_a:
        score_a += 2.0
        favored_criteria.append("median delta")

    if share_b_better > share_a_better:
        share_improvement_favors = label_b
        score_b += 2.0
        favored_criteria.append("share of improved samples")
    elif share_a_better > share_b_better:
        share_improvement_favors = label_a
        score_a += 2.0
        favored_criteria.append("share of improved samples")
    else:
        share_improvement_favors = "tie"

    worst_delta_favors = favor_from_signed_value(
        q10_signed_delta,
        label_a,
        label_b,
        tolerance,
    )
    if worst_delta_favors == label_b:
        score_b += 1.5
        favored_criteria.append("worst delta / negative tail")
    elif worst_delta_favors == label_a:
        score_a += 1.5
        favored_criteria.append("worst delta / negative tail")

    mean_delta_favors = favor_from_signed_value(
        mean_signed_delta,
        label_a,
        label_b,
        tolerance,
    )
    if mean_delta_favors == label_b:
        score_b += 1.0
        favored_criteria.append("mean delta")
    elif mean_delta_favors == label_a:
        score_a += 1.0
        favored_criteria.append("mean delta")

    if wilcoxon_pvalue < P_VALUE_SIGNIFICANCE_THRESHOLD:
        if median_signed_delta > tolerance:
            wilcoxon_favors = label_b
            score_b += 0.5
            favored_criteria.append("Wilcoxon signed-rank test")
        elif median_signed_delta < -tolerance:
            wilcoxon_favors = label_a
            score_a += 0.5
            favored_criteria.append("Wilcoxon signed-rank test")
        else:
            wilcoxon_favors = "tie"
    else:
        wilcoxon_favors = "tie"

    return (
        score_a,
        score_b,
        median_delta_favors,
        share_improvement_favors,
        worst_delta_favors,
        mean_delta_favors,
        wilcoxon_favors,
        favored_criteria,
        counter_criteria,
    )


def choose_paired_decision_strength(
    score_diff: float,
    median_signed_delta: float,
    share_b_better: float,
    share_a_better: float,
    q10_signed_delta: float,
    wilcoxon_pvalue: float,
    better_label: str,
    label_a: str,
    label_b: str,
    tolerance: float,
) -> str:
    """Valuta la forza della decisione paired."""
    if better_label == "tie":
        return "weak"

    share_gap = abs(share_b_better - share_a_better)

    if score_diff < 2.0:
        return "weak"

    if abs(median_signed_delta) <= tolerance and share_gap < 0.10:
        return "weak"

    wilcoxon_supports = wilcoxon_pvalue < P_VALUE_SIGNIFICANCE_THRESHOLD

    if better_label == label_b:
        median_supports = median_signed_delta > tolerance
        share_supports = share_b_better > share_a_better
        tail_supports = q10_signed_delta >= -tolerance
    else:
        median_supports = median_signed_delta < -tolerance
        share_supports = share_a_better > share_b_better
        tail_supports = q10_signed_delta <= tolerance

    if (
        score_diff >= 4.0
        and median_supports
        and share_supports
        and tail_supports
        and wilcoxon_supports
    ):
        return "strong"

    return "moderate"


def compute_paired_summary(
    merged: pd.DataFrame,
    label_a: str,
    label_b: str,
    tolerance: float,
    higher_is_better: bool,
) -> PairedSummary:
    """Calcola il riepilogo completo per il confronto paired."""
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

    delta_quantiles = get_quantiles(signed_delta)

    mean_signed_delta = float(np.mean(signed_delta))
    median_signed_delta = float(np.median(signed_delta))
    q10_signed_delta = delta_quantiles["q10"]

    (
        score_a,
        score_b,
        median_delta_favors,
        share_improvement_favors,
        worst_delta_favors,
        mean_delta_favors,
        wilcoxon_favors,
        favored_criteria,
        counter_criteria,
    ) = score_paired_comparison(
        label_a=label_a,
        label_b=label_b,
        median_signed_delta=median_signed_delta,
        mean_signed_delta=mean_signed_delta,
        q10_signed_delta=q10_signed_delta,
        share_b_better=share_b_better,
        share_a_better=share_a_better,
        wilcoxon_pvalue=wilcoxon_pvalue,
        tolerance=tolerance,
    )

    if score_b > score_a:
        better_label = label_b
    elif score_a > score_b:
        better_label = label_a
    else:
        better_label = "tie"

    score_diff = abs(score_a - score_b)

    decision_strength = choose_paired_decision_strength(
        score_diff=score_diff,
        median_signed_delta=median_signed_delta,
        share_b_better=share_b_better,
        share_a_better=share_a_better,
        q10_signed_delta=q10_signed_delta,
        wilcoxon_pvalue=wilcoxon_pvalue,
        better_label=better_label,
        label_a=label_a,
        label_b=label_b,
        tolerance=tolerance,
    )

    reason = build_reason(
        better_label=better_label,
        decision_strength=decision_strength,
        favored_criteria=favored_criteria,
        counter_criteria=counter_criteria,
    )

    return PairedSummary(
        label_a=label_a,
        label_b=label_b,
        n_pairs=int(merged.shape[0]),
        tolerance=tolerance,
        mean_signed_delta=mean_signed_delta,
        median_signed_delta=median_signed_delta,
        std_signed_delta=float(np.std(signed_delta, ddof=1)) if signed_delta.size > 1 else 0.0,
        min_signed_delta=float(np.min(signed_delta)),
        max_signed_delta=float(np.max(signed_delta)),
        q10_signed_delta=delta_quantiles["q10"],
        q25_signed_delta=delta_quantiles["q25"],
        q50_signed_delta=delta_quantiles["q50"],
        q75_signed_delta=delta_quantiles["q75"],
        q90_signed_delta=delta_quantiles["q90"],
        share_b_better=share_b_better,
        share_a_better=share_a_better,
        share_equal=share_equal,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_pvalue=wilcoxon_pvalue,
        score_a=score_a,
        score_b=score_b,
        score_diff=score_diff,
        median_delta_favors=median_delta_favors,
        share_improvement_favors=share_improvement_favors,
        worst_delta_favors=worst_delta_favors,
        mean_delta_favors=mean_delta_favors,
        wilcoxon_favors=wilcoxon_favors,
        better_label=better_label,
        decision_strength=decision_strength,
        reason=reason,
    )


def flatten_unpaired_group_stats(group: UnpairedGroupStats) -> dict[str, Any]:
    """Converte le statistiche di gruppo in una riga tabellare CSV."""
    row = {
        "label": group.label,
        "n": group.n,
        "mean": group.mean,
        "median": group.median,
        "std": group.std,
        "min": group.min_value,
        "max": group.max_value,
        "q10": group.q10,
        "q25": group.q25,
        "q50": group.q50,
        "q75": group.q75,
        "q90": group.q90,
        "iqr": group.iqr,
        "worst_tail_value": group.worst_tail_value,
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
    pd.DataFrame(rows).to_csv(output_dir / "unpaired_group_statistics.csv", index=False)


def save_unpaired_comparison_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Salva tabelle ricche del confronto unpaired."""
    summary_row = {
        "mode": "unpaired",
        "metric": args.column,
        "direction": "higher_is_better" if args.resolved_higher_is_better else "lower_is_better",
        "label_a": group_a.label,
        "label_b": group_b.label,
        "n_a": group_a.n,
        "n_b": group_b.n,
        "score_a": comparison.score_a,
        "score_b": comparison.score_b,
        "score_diff": comparison.score_diff,
        "better_label": comparison.better_label,
        "decision_strength": comparison.decision_strength,
        "reason": comparison.reason,
        "signed_quantile_shift": comparison.signed_quantile_shift,
        "common_language_b_better": comparison.common_language_b_better,
        "wasserstein_between_groups": comparison.wasserstein_between_groups,
        "ks_statistic": comparison.ks_statistic,
        "ks_pvalue": comparison.ks_pvalue,
        "mannwhitney_u": comparison.mannwhitney_u,
        "mannwhitney_pvalue": comparison.mannwhitney_pvalue,
    }

    pd.DataFrame([summary_row]).to_csv(
        output_dir / "unpaired_comparison_summary.csv",
        index=False,
    )

    breakdown_rows = [
        {
            "criterion": "signed_quantile_shift",
            "points": 2.0,
            "favors": comparison.quantile_shift_favors,
            "value": comparison.signed_quantile_shift,
        },
        {
            "criterion": "threshold_shares",
            "points": 2.0,
            "favors": comparison.threshold_favors,
            "value_a": comparison.threshold_share_mean_a,
            "value_b": comparison.threshold_share_mean_b,
        },
        {
            "criterion": "worst_tail",
            "points": 1.5,
            "favors": comparison.worst_tail_favors,
            "value_a": group_a.worst_tail_value,
            "value_b": group_b.worst_tail_value,
        },
        {
            "criterion": "quantile_improvement_rate",
            "points": 1.5,
            "favors": comparison.quantile_improvement_favors,
            "value_a": comparison.quantile_improvement_rate_a,
            "value_b": comparison.quantile_improvement_rate_b,
        },
        {
            "criterion": "common_language_effect_size",
            "points": 0.5,
            "favors": comparison.common_language_favors,
            "value": comparison.common_language_b_better,
        },
    ]

    pd.DataFrame(breakdown_rows).to_csv(
        output_dir / "unpaired_decision_breakdown.csv",
        index=False,
    )

    quantile_rows = []
    for name in QUANTILE_LEVELS:
        value_a = getattr(group_a, name)
        value_b = getattr(group_b, name)
        signed_delta = oriented_difference(
            value_a,
            value_b,
            args.resolved_higher_is_better,
        )
        quantile_rows.append(
            {
                "quantile": name,
                "value_a": value_a,
                "value_b": value_b,
                "signed_delta_positive_means_b_better": signed_delta,
                "favors": favor_from_signed_value(
                    signed_delta,
                    group_a.label,
                    group_b.label,
                ),
            }
        )

    pd.DataFrame(quantile_rows).to_csv(
        output_dir / "unpaired_quantile_comparison.csv",
        index=False,
    )

    threshold_rows = []
    for threshold_name in group_a.threshold_shares:
        share_a = group_a.threshold_shares[threshold_name]
        share_b = group_b.threshold_shares[threshold_name]
        threshold_rows.append(
            {
                "threshold": threshold_name,
                "share_a": share_a,
                "share_b": share_b,
                "delta_share_b_minus_a": share_b - share_a,
                "favors": (
                    group_b.label
                    if share_b > share_a
                    else group_a.label
                    if share_a > share_b
                    else "tie"
                ),
            }
        )

    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "unpaired_threshold_shares.csv",
        index=False,
    )


def save_paired_comparison_summary(
    summary: PairedSummary,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Salva tabelle ricche del confronto paired."""
    summary_row = {
        "mode": "paired",
        "metric": args.column,
        "direction": "higher_is_better" if args.resolved_higher_is_better else "lower_is_better",
        "label_a": summary.label_a,
        "label_b": summary.label_b,
        "n_pairs": summary.n_pairs,
        "tolerance": summary.tolerance,
        "score_a": summary.score_a,
        "score_b": summary.score_b,
        "score_diff": summary.score_diff,
        "better_label": summary.better_label,
        "decision_strength": summary.decision_strength,
        "reason": summary.reason,
        "mean_signed_delta": summary.mean_signed_delta,
        "median_signed_delta": summary.median_signed_delta,
        "share_b_better": summary.share_b_better,
        "share_a_better": summary.share_a_better,
        "share_equal": summary.share_equal,
        "q10_signed_delta": summary.q10_signed_delta,
        "wilcoxon_statistic": summary.wilcoxon_statistic,
        "wilcoxon_pvalue": summary.wilcoxon_pvalue,
    }

    pd.DataFrame([summary_row]).to_csv(
        output_dir / "paired_comparison_summary.csv",
        index=False,
    )

    breakdown_rows = [
        {
            "criterion": "median_delta",
            "points": 2.0,
            "favors": summary.median_delta_favors,
            "value": summary.median_signed_delta,
        },
        {
            "criterion": "share_improved_samples",
            "points": 2.0,
            "favors": summary.share_improvement_favors,
            "share_b_better": summary.share_b_better,
            "share_a_better": summary.share_a_better,
        },
        {
            "criterion": "worst_delta_negative_tail",
            "points": 1.5,
            "favors": summary.worst_delta_favors,
            "value": summary.q10_signed_delta,
        },
        {
            "criterion": "mean_delta",
            "points": 1.0,
            "favors": summary.mean_delta_favors,
            "value": summary.mean_signed_delta,
        },
        {
            "criterion": "wilcoxon_signed_rank",
            "points": 0.5,
            "favors": summary.wilcoxon_favors,
            "pvalue": summary.wilcoxon_pvalue,
        },
    ]

    pd.DataFrame(breakdown_rows).to_csv(
        output_dir / "paired_decision_breakdown.csv",
        index=False,
    )

    delta_summary_row = {
        "mean_signed_delta": summary.mean_signed_delta,
        "median_signed_delta": summary.median_signed_delta,
        "std_signed_delta": summary.std_signed_delta,
        "min_signed_delta": summary.min_signed_delta,
        "max_signed_delta": summary.max_signed_delta,
        "q10_signed_delta": summary.q10_signed_delta,
        "q25_signed_delta": summary.q25_signed_delta,
        "q50_signed_delta": summary.q50_signed_delta,
        "q75_signed_delta": summary.q75_signed_delta,
        "q90_signed_delta": summary.q90_signed_delta,
        "share_b_better": summary.share_b_better,
        "share_a_better": summary.share_a_better,
        "share_equal": summary.share_equal,
    }

    pd.DataFrame([delta_summary_row]).to_csv(
        output_dir / "paired_delta_summary.csv",
        index=False,
    )


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
    
    result["abs_signed_delta"] = np.abs(signed_delta)
    result["relative_signed_delta"] = np.where(
        np.abs(result["value_a"]) > 1e-12,
        signed_delta / np.abs(result["value_a"]),
        np.nan,
    )

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

