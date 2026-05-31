from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from virtual_staining.utils.metrics import is_higher_better_metric

_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "dataset_root",
        "results_path",
        "run_name",
        "save_graphs",
        "save_residual_heatmaps",
        "residual_heatmap_metric",
        "residual_heatmap_top_k",
        "target_dir",
        "generated_dir",
        "output_dir",
    }
)


@dataclass(frozen=True)
class EvaluationConfig:
    save_graphs: bool = False
    save_residual_heatmaps: bool = False
    residual_heatmap_metric: str = "ssim"
    residual_heatmap_top_k: int = 25
    target_dir: Path | None = None
    generated_dir: Path | None = None
    output_dir: Path | None = None

    def validate(self) -> None:
        if self.residual_heatmap_top_k <= 0:
            raise ValueError("residual_heatmap_top_k must be a positive integer.")
        is_higher_better_metric(self.residual_heatmap_metric)
