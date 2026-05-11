from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from virtual_staining.evaluation.ranking import organize_by_metrics


@dataclass(frozen=True)
class OrganizeRequest:
    metrics_csv: Path
    output_dir: Path
    top_k: int
    metrics: tuple[str, ...]
    mode: str
    overwrite: bool
    include_all_ranked: bool


def organize(request: OrganizeRequest) -> None:
    """Organize generated, target, and source images by metric ranking."""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    organize_by_metrics(
        csv_path=request.metrics_csv,
        output_dir=request.output_dir,
        top_n=request.top_k,
        metrics=list(request.metrics),
        mode=request.mode,
        overwrite=request.overwrite,
        include_all_ranked=request.include_all_ranked,
    )
