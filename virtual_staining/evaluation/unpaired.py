"""Collection-level appearance diagnostics for unpaired generated vs real images.

No sample correspondence is assumed: each image is reduced to a few scalar features and
only the per-image feature distributions of the two collections are compared.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from virtual_staining.evaluation.plotting import save_unpaired_feature_plot
from virtual_staining.utils.image_io import load_rgb_image, to_float01

UNPAIRED_IMAGE_STATISTICS_CSV = "unpaired_image_statistics.csv"
UNPAIRED_FEATURE_COMPARISON_CSV = "unpaired_feature_comparison.csv"
UNPAIRED_FEATURE_PLOT = "unpaired_feature_distributions.png"

# Same ITU-R BT.601 weights as the grayscale PCC metric.
LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)
FEATURE_DEFINITIONS: dict[str, str] = {
    "mean_r": "mean of the red channel, RGB scaled to [0, 1]",
    "mean_g": "mean of the green channel, RGB scaled to [0, 1]",
    "mean_b": "mean of the blue channel, RGB scaled to [0, 1]",
    "std_r": "population standard deviation of the red channel",
    "std_g": "population standard deviation of the green channel",
    "std_b": "population standard deviation of the blue channel",
    "mean_luminance": "mean of Y = 0.299 R + 0.587 G + 0.114 B",
    "std_luminance": "population standard deviation of Y = 0.299 R + 0.587 G + 0.114 B",
}
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_DEFINITIONS)
UNPAIRED_LIMITATIONS: tuple[str, ...] = (
    "Generated and real images are not assumed to correspond; no pairs are formed.",
    "Metrics are low-order appearance/distribution diagnostics of per-image scalar features.",
    "They do not establish sample-level reconstruction fidelity.",
    "They do not establish biological correctness.",
    "They do not establish clinical validity.",
    "KS p-values are exploratory and descriptive, not confirmatory tests.",
    "Patch-level observations may be dependent because multiple patches can come from the "
    "same specimen or slide.",
)
_STAT_NAMES = ("count", "mean", "std", "median", "min", "max")
COMPARISON_FIELDS: tuple[str, ...] = (
    "feature",
    *(f"generated_{stat}" for stat in _STAT_NAMES),
    *(f"reference_{stat}" for stat in _STAT_NAMES),
    "wasserstein_distance",
    "ks_statistic",
    "ks_pvalue",
)


@dataclass(frozen=True)
class UnpairedEvaluationResult:
    generated_count: int
    reference_count: int
    image_statistics_csv: Path
    feature_comparison_csv: Path
    graph_path: Path | None


def image_features(path: Path) -> dict[str, float]:
    """Reduce one RGB image to its scalar appearance features."""
    image = to_float01(load_rgb_image(path)).astype(np.float64)
    luminance = image @ np.asarray(LUMINANCE_WEIGHTS)
    means = image.mean(axis=(0, 1))
    stds = image.std(axis=(0, 1))
    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
    }


def _describe(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compare_feature_distributions(
    generated: Mapping[str, Sequence[float]], reference: Mapping[str, Sequence[float]]
) -> list[dict[str, object]]:
    """Describe each feature per collection and measure distribution distance; no ranking."""
    rows: list[dict[str, object]] = []
    for feature in FEATURE_NAMES:
        gen = np.asarray(generated[feature], dtype=np.float64)
        ref = np.asarray(reference[feature], dtype=np.float64)
        ks = ks_2samp(gen, ref, alternative="two-sided")
        rows.append(
            {
                "feature": feature,
                **{f"generated_{k}": v for k, v in _describe(gen).items()},
                **{f"reference_{k}": v for k, v in _describe(ref).items()},
                "wasserstein_distance": float(wasserstein_distance(gen, ref)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return rows


def evaluate_unpaired_collections(
    generated_paths: Sequence[Path],
    reference_paths: Sequence[Path],
    output_dir: Path,
    *,
    save_graphs: bool,
) -> UnpairedEvaluationResult:
    """Write per-image features and their generated-vs-reference distribution comparison.

    Images are loaded one at a time; only the scalar features are retained.
    """
    if not generated_paths or not reference_paths:
        raise ValueError("Unpaired evaluation requires non-empty generated and reference images")
    output_dir.mkdir(parents=True, exist_ok=True)
    values: dict[str, dict[str, list[float]]] = {
        "generated": {name: [] for name in FEATURE_NAMES},
        "reference": {name: [] for name in FEATURE_NAMES},
    }
    statistics_csv = output_dir / UNPAIRED_IMAGE_STATISTICS_CSV
    with statistics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["collection", "path", *FEATURE_NAMES])
        writer.writeheader()
        for collection, paths in (("generated", generated_paths), ("reference", reference_paths)):
            for path in paths:
                features = image_features(path)
                writer.writerow({"collection": collection, "path": str(path), **features})
                for name, value in features.items():
                    values[collection][name].append(value)

    comparison_csv = output_dir / UNPAIRED_FEATURE_COMPARISON_CSV
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_FIELDS))
        writer.writeheader()
        writer.writerows(compare_feature_distributions(values["generated"], values["reference"]))

    graph_path = None
    if save_graphs:
        graph_path = save_unpaired_feature_plot(
            values["generated"], values["reference"], output_dir / UNPAIRED_FEATURE_PLOT
        )
    return UnpairedEvaluationResult(
        generated_count=len(generated_paths),
        reference_count=len(reference_paths),
        image_statistics_csv=statistics_csv,
        feature_comparison_csv=comparison_csv,
        graph_path=graph_path,
    )
