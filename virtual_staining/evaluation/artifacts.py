from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationArtifact:
    stage: str
    artifact_type: str
    path: Path
    description: str
    metric: str | None = None
    sample_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _manifest_path(path: Path, run_root: Path) -> tuple[str, str]:
    resolved_path = path.resolve()
    resolved_root = run_root.resolve()

    if resolved_path.is_relative_to(resolved_root):
        return resolved_path.relative_to(resolved_root).as_posix(), "run_relative"

    return resolved_path.as_posix(), "absolute"


def _artifact_to_dict(artifact: EvaluationArtifact, run_root: Path) -> dict[str, object]:
    if not artifact.path.exists():
        raise FileNotFoundError(f"Evaluation artifact does not exist: {artifact.path}")

    path, path_type = _manifest_path(artifact.path, run_root)
    return {
        "stage": artifact.stage,
        "artifact_type": artifact.artifact_type,
        "path": path,
        "path_type": path_type,
        "metric": artifact.metric,
        "sample_id": artifact.sample_id,
        "description": artifact.description,
        "metadata": dict(artifact.metadata),
    }


def write_artifact_manifest(
    artifacts: Sequence[EvaluationArtifact],
    output_path: str | Path,
    *,
    run_root: str | Path,
    created_at: str,
) -> Path:
    """Write the canonical evaluation artifact manifest as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root_path = Path(run_root)
    payload = {
        "schema_version": 1,
        "created_at": created_at,
        "path_policy": (
            "Artifact paths are relative to the run root when possible; "
            "artifacts outside the run root use absolute paths."
        ),
        "artifacts": [_artifact_to_dict(artifact, run_root_path) for artifact in artifacts],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def residual_heatmap_artifacts_from_csv(
    csv_path: str | Path,
    *,
    stage: str = "evaluate",
) -> list[EvaluationArtifact]:
    """Build per-heatmap artifact entries from residual_heatmaps.csv."""
    artifacts: list[EvaluationArtifact] = []
    path = Path(csv_path)

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            heatmap_path = row.get("heatmap_path")
            if not heatmap_path:
                continue

            metadata = {
                "rank": row.get("rank", ""),
                "metric_value": row.get("metric_value", ""),
                "target_path": row.get("target_path", ""),
                "generated_path": row.get("generated_path", ""),
            }
            artifacts.append(
                EvaluationArtifact(
                    stage=stage,
                    artifact_type="residual_heatmap_png",
                    path=Path(heatmap_path),
                    metric=row.get("metric") or None,
                    sample_id=row.get("sample_id") or None,
                    description="Standalone absolute-error residual heatmap PNG.",
                    metadata=metadata,
                )
            )

    return artifacts
