from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virtual_staining.evaluation.artifacts import (
    EvaluationArtifact,
    append_artifacts_to_manifest,
)
from virtual_staining.evaluation.ranking import organize_by_metrics
from virtual_staining.utils.console import print_info


@dataclass(frozen=True)
class OrganizeRequest:
    metrics_csv: Path
    output_dir: Path
    top_k: int
    metrics: tuple[str, ...]
    mode: str
    overwrite: bool
    include_all_ranked: bool
    run_path: Path | None = None


ORGANIZE_STAGE = "organize"
ORGANIZE_COMMAND = "vs-organize"


def organize(request: OrganizeRequest) -> None:
    """Export generated, target, and source image files by metric ranking."""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    result = organize_by_metrics(
        csv_path=request.metrics_csv,
        output_dir=request.output_dir,
        top_n=request.top_k,
        metrics=list(request.metrics),
        mode=request.mode,
        overwrite=request.overwrite,
        include_all_ranked=request.include_all_ranked,
    )

    if request.run_path is None:
        print_info(
            "Artifact manifest",
            "Skipped; no run root was inferred for this ranked export.",
        )
        return

    if result.summary_csv is None:
        print_info("Artifact manifest", "Skipped; no ranked export artifacts were written.")
        return

    artifact_manifest_path = _register_organize_artifacts(
        request=request,
        summary_csv=result.summary_csv,
        summary_rows=result.summary_rows,
    )
    print_info("Artifact manifest", str(artifact_manifest_path))


def _register_organize_artifacts(
    *,
    request: OrganizeRequest,
    summary_csv: Path,
    summary_rows: list[dict[str, Any]],
) -> Path:
    assert request.run_path is not None
    metadata_base: dict[str, object] = {
        "command": ORGANIZE_COMMAND,
        "source_run": request.run_path.name,
        "artifact_format": "csv",
        "selected_metrics": list(request.metrics),
        "top_k": request.top_k,
        "export_mode": request.mode,
        "include_all_ranked": request.include_all_ranked,
    }
    artifacts = [
        EvaluationArtifact(
            stage=ORGANIZE_STAGE,
            artifact_type="organization_summary",
            path=summary_csv,
            description="Summary CSV for ranked sample file exports.",
            metadata=metadata_base,
        )
    ]

    for row in summary_rows:
        metric = str(row["metric"])
        kind = str(row["kind"])
        artifacts.append(
            EvaluationArtifact(
                stage=ORGANIZE_STAGE,
                artifact_type="ranked_sample_export",
                path=Path(str(row["output_dir"])),
                metric=metric,
                description="Ranked sample export directory.",
                metadata={
                    **metadata_base,
                    "artifact_format": "directory",
                    "kind": kind,
                    "rank_count": row["rank_count"],
                    "selected_file_roles": row["selected_file_roles"],
                    "files_exported": row["files_exported"],
                },
            )
        )

    return append_artifacts_to_manifest(
        artifacts,
        request.run_path / "evaluation" / "artifacts.json",
        run_root=request.run_path,
        replace_stages=(ORGANIZE_STAGE,),
    )
