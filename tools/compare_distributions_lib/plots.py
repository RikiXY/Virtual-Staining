from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce la funzione di distribuzione empirica del campione."""
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
    """Salva l'istogramma di confronto tra due distribuzioni."""
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


def plot_paired_delta_histogram(signed_delta: np.ndarray, column: str, output_dir: Path) -> None:
    """Salva l'istogramma dei delta signed del confronto paired."""
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
    """Salva lo scatter paired A vs B con diagonale di parità."""
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