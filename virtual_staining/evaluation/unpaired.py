from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from virtual_staining.utils.image_io import load_rgb_image, to_float01

FEATURE_NAMES = (
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "luminance_mean",
    "luminance_std",
)


@dataclass(frozen=True)
class UnpairedEvaluationResult:
    metadata_path: Path
    statistics_csv: Path
    graph_paths: tuple[Path, ...]
    generated_count: int
    real_target_count: int
    summary: dict[str, object]


def _image_features(path: Path) -> dict[str, float]:
    image = to_float01(load_rgb_image(path))
    means = np.mean(image, axis=(0, 1))
    stds = np.std(image, axis=(0, 1))
    luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "luminance_mean": float(np.mean(luminance)),
        "luminance_std": float(np.std(luminance)),
    }


def _feature_rows(paths: Sequence[Path], group: str) -> list[dict[str, object]]:
    return [{"group": group, "path": str(path), **_image_features(path)} for path in paths]


def _write_statistics_csv(rows: Sequence[dict[str, object]], path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "path", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _scalar(value: object) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"Expected scalar feature value, got {type(value).__name__}")


def _values(rows: Sequence[dict[str, object]], group: str, feature: str) -> np.ndarray:
    return np.asarray(
        [_scalar(row[feature]) for row in rows if row["group"] == group], dtype=np.float64
    )


def _distribution_summary(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for feature in FEATURE_NAMES:
        generated = _values(rows, "generated", feature)
        real = _values(rows, "real_target", feature)
        ks = ks_2samp(generated, real, alternative="two-sided")
        summary[feature] = {
            "generated": {
                "mean": float(np.mean(generated)),
                "median": float(np.median(generated)),
                "std": float(np.std(generated)),
            },
            "real_target": {
                "mean": float(np.mean(real)),
                "median": float(np.median(real)),
                "std": float(np.std(real)),
            },
            "wasserstein_distance": float(wasserstein_distance(generated, real)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
    return summary


def _save_distribution_plots(
    rows: Sequence[dict[str, object]],
    summary: dict[str, dict[str, object]],
    output_dir: Path,
) -> tuple[Path, ...]:
    distributions_path = output_dir / "unpaired_feature_distributions.png"
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, feature in zip(axes.flat, FEATURE_NAMES, strict=True):
        generated = _values(rows, "generated", feature)
        real = _values(rows, "real_target", feature)
        axis.hist(real, bins=30, density=True, alpha=0.55, label="real target")
        axis.hist(generated, bins=30, density=True, alpha=0.55, label="generated")
        axis.set_title(feature)
        axis.set_xlim(0.0, 1.0)
    axes.flat[0].legend()
    figure.suptitle("Unpaired image-feature distributions")
    figure.tight_layout()
    figure.savefig(distributions_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    distances_path = output_dir / "unpaired_feature_wasserstein.png"
    distances = [_scalar(summary[name]["wasserstein_distance"]) for name in FEATURE_NAMES]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(FEATURE_NAMES, distances)
    axis.set_ylabel("Wasserstein distance")
    axis.set_title("Generated vs real-target feature distributions (lower is closer)")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(distances_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return distributions_path, distances_path


def evaluate_unpaired_distributions(
    generated_paths: Sequence[Path],
    real_target_paths: Sequence[Path],
    output_dir: Path,
    *,
    method: str,
    direction: str | None,
    save_graphs: bool,
) -> UnpairedEvaluationResult:
    """Compare simple image-feature distributions without inventing image pairs."""
    if not generated_paths:
        raise ValueError("Unpaired evaluation found no generated images")
    if not real_target_paths:
        raise ValueError("Unpaired evaluation found no real target images")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        *_feature_rows(generated_paths, "generated"),
        *_feature_rows(real_target_paths, "real_target"),
    ]
    statistics_csv = _write_statistics_csv(rows, output_dir / "unpaired_image_statistics.csv")
    feature_summary = _distribution_summary(rows)
    graph_paths = _save_distribution_plots(rows, feature_summary, output_dir) if save_graphs else ()
    metadata: dict[str, object] = {
        "schema_version": 2,
        "method": method,
        "protocol": "unpaired",
        "pairing": "unpaired",
        "direction": direction,
        "paired_metrics_available": False,
        "interpretation": (
            "Dataset-level RGB and luminance distribution diagnostics. They do not measure "
            "sample-level fidelity or biological correctness."
        ),
        "generated_count": len(generated_paths),
        "real_target_count": len(real_target_paths),
        "generated_paths": [str(path) for path in generated_paths],
        "real_target_paths": [str(path) for path in real_target_paths],
        "statistics_csv": str(statistics_csv),
        "graph_paths": [str(path) for path in graph_paths],
        "features": feature_summary,
    }
    metadata_path = output_dir / "unpaired_evaluation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return UnpairedEvaluationResult(
        metadata_path=metadata_path,
        statistics_csv=statistics_csv,
        graph_paths=graph_paths,
        generated_count=len(generated_paths),
        real_target_count=len(real_target_paths),
        summary=metadata,
    )
