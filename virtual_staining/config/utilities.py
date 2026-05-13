from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_COMPARE_KEYS: frozenset[str] = frozenset(
    {
        "mode",
        "run_a",
        "run_b",
        "csv_a",
        "csv_b",
        "label_a",
        "label_b",
        "column",
        "output_dir",
        "higher_is_better",
        "lower_is_better",
        "bins",
        "min_value",
        "max_value",
        "thresholds",
        "tolerance",
        "sample_id_column",
    }
)

_COMPARE_PANELS_KEYS: frozenset[str] = frozenset(
    {
        "mode",
        "run_path",
        "hide_graphs_path",
        "source_image",
        "generated_image",
        "target_image",
        "save_path",
        "with_diagnostics",
    }
)

_ORGANIZE_KEYS: frozenset[str] = frozenset(
    {
        "run_path",
        "metrics_csv",
        "output_dir",
        "metrics",
        "top_k",
        "mode",
        "include_all_ranked",
        "overwrite",
    }
)


@dataclass(frozen=True)
class CompareConfig:
    mode: Literal["paired", "unpaired"] = "paired"
    run_a: Path | None = None
    run_b: Path | None = None
    csv_a: Path | None = None
    csv_b: Path | None = None
    label_a: str | None = None
    label_b: str | None = None
    column: str = "ssim"
    output_dir: Path | None = None
    higher_is_better: bool | None = None
    lower_is_better: bool | None = None
    bins: int = 30
    min_value: float | None = None
    max_value: float | None = None
    thresholds: tuple[float, ...] | None = None
    tolerance: float = 0.0
    sample_id_column: str = "sample_id"


@dataclass(frozen=True)
class ComparePanelsConfig:
    mode: Literal["single", "from_metrics"] = "from_metrics"
    run_path: Path | None = None
    hide_graphs_path: bool = False
    source_image: Path | None = None
    generated_image: Path | None = None
    target_image: Path | None = None
    save_path: Path | None = None
    with_diagnostics: bool = False


@dataclass(frozen=True)
class OrganizeConfig:
    run_path: Path | None = None
    metrics_csv: Path | None = None
    output_dir: Path | None = None
    metrics: tuple[str, ...] | None = None
    top_k: int = 20
    mode: Literal["hardlink", "symlink", "copy"] = "hardlink"
    include_all_ranked: bool = False
    overwrite: bool = False
