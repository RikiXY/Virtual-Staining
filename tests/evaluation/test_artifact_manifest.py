from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from virtual_staining.evaluation.artifacts import (
    EvaluationArtifact,
    append_artifacts_to_manifest,
    write_artifact_manifest,
)


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_append_artifacts_preserves_core_entries_and_replaces_stage(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "manifest_run"
    evaluation_dir = run_root / "evaluation"
    summary_csv = evaluation_dir / "summary.csv"
    summary_csv.parent.mkdir(parents=True)
    summary_csv.write_text("metric,mean\nssim,0.9\n", encoding="utf-8")
    manifest_path = evaluation_dir / "artifacts.json"

    write_artifact_manifest(
        [
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="summary_csv",
                path=summary_csv,
                description="Aggregate evaluation metric summary.",
            )
        ],
        manifest_path,
        run_root=run_root,
        created_at="2026-05-31T10:00:00+00:00",
    )

    ssim_summary = run_root / "comparisons" / "metrics" / "ssim" / "selection_summary.csv"
    ssim_summary.parent.mkdir(parents=True)
    ssim_summary.write_text("metric,kind\nssim,best\n", encoding="utf-8")
    append_artifacts_to_manifest(
        [
            EvaluationArtifact(
                stage="render_panels",
                artifact_type="selection_summary",
                path=ssim_summary,
                metric="ssim",
                description="Per-metric render-panels selection summary CSV.",
                metadata={"command": "vs-render-panels from-metrics"},
            )
        ],
        manifest_path,
        run_root=run_root,
        replace_stages=("render_panels",),
        updated_at="2026-05-31T10:01:00+00:00",
    )

    mae_summary = run_root / "comparisons" / "metrics" / "mae" / "selection_summary.csv"
    mae_summary.parent.mkdir(parents=True)
    mae_summary.write_text("metric,kind\nmae,best\n", encoding="utf-8")
    append_artifacts_to_manifest(
        [
            EvaluationArtifact(
                stage="render_panels",
                artifact_type="selection_summary",
                path=mae_summary,
                metric="mae",
                description="Per-metric render-panels selection summary CSV.",
                metadata={"command": "vs-render-panels from-metrics"},
            )
        ],
        manifest_path,
        run_root=run_root,
        replace_stages=("render_panels",),
        updated_at="2026-05-31T10:02:00+00:00",
    )

    manifest = _read_manifest(manifest_path)
    artifacts = manifest["artifacts"]

    assert manifest["created_at"] == "2026-05-31T10:00:00+00:00"
    assert manifest["updated_at"] == "2026-05-31T10:02:00+00:00"
    assert isinstance(artifacts, list)
    assert [artifact["stage"] for artifact in artifacts] == ["evaluate", "render_panels"]
    assert [artifact["path"] for artifact in artifacts] == [
        "evaluation/summary.csv",
        "comparisons/metrics/mae/selection_summary.csv",
    ]
    assert all(artifact["path_type"] == "run_relative" for artifact in artifacts)
