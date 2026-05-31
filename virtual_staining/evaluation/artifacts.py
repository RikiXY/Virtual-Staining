from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
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


def _path_policy() -> str:
    return (
        "Artifact paths are relative to the run root when possible; "
        "artifacts outside the run root use absolute paths."
    )


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
        "path_policy": _path_policy(),
        "artifacts": [_artifact_to_dict(artifact, run_root_path) for artifact in artifacts],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _artifact_record_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("stage", "")),
        str(record.get("artifact_type", "")),
        str(record.get("path", "")),
        str(record.get("metric", "")),
        str(record.get("sample_id", "")),
    )


def _deduplicate_artifact_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        record_dict = dict(record)
        deduplicated[_artifact_record_key(record_dict)] = record_dict
    return list(deduplicated.values())


def _read_manifest_payload(path: Path, created_at: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": 1,
            "created_at": created_at,
            "path_policy": _path_policy(),
            "artifacts": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact manifest is not a JSON object: {path}")

    if not isinstance(payload.get("artifacts"), list):
        raise ValueError(f"Artifact manifest has no artifacts list: {path}")

    payload["schema_version"] = 1
    payload["created_at"] = payload.get("created_at") or created_at
    payload["path_policy"] = payload.get("path_policy") or _path_policy()
    return payload


def append_artifacts_to_manifest(
    artifacts: Sequence[EvaluationArtifact],
    output_path: str | Path,
    *,
    run_root: str | Path,
    replace_stages: Sequence[str] = (),
    updated_at: str | None = None,
) -> Path:
    """Append artifacts to a manifest, replacing prior entries for selected stages."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = updated_at or datetime.now(UTC).isoformat()
    payload = _read_manifest_payload(output, created_at=timestamp)
    replace_stage_names = set(replace_stages)
    existing_records = [
        dict(record)
        for record in payload["artifacts"]
        if isinstance(record, dict) and str(record.get("stage", "")) not in replace_stage_names
    ]
    appended_records = [_artifact_to_dict(artifact, Path(run_root)) for artifact in artifacts]

    payload["updated_at"] = timestamp
    payload["artifacts"] = _deduplicate_artifact_records([*existing_records, *appended_records])
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
