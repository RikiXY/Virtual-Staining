from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.evaluation.artifacts import (
    EvaluationArtifact,
    residual_heatmap_artifacts_from_csv,
    write_artifact_manifest,
)
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.panels import write_residual_heatmap_artifacts
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.reports import write_skipped_csv
from virtual_staining.evaluation.summaries import write_summary_csv, write_weak_tail_csv
from virtual_staining.experiment.metadata import (
    append_run_event,
    ensure_run_metadata,
    save_stage_metadata,
)
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.inference.outputs import generated_path_for_record

logger = logging.getLogger(__name__)

_METRIC_CONFIG: dict[str, bool] = {
    "mae": True,
    "mse": True,
    "rmse": True,
    "psnr": True,
    "ssim": True,
    "pcc_gray": True,
    "pcc_r": True,
    "pcc_g": True,
    "pcc_b": True,
    "pcc_rgb_mean": True,
}


def _write_evaluation_stage_metadata(paths: RunPaths, payload: dict[str, object]) -> None:
    stage_path = save_stage_metadata("evaluate", payload, paths.metadata_dir)
    if stage_path is not None:
        logger.info("Evaluation metadata written -> %s", stage_path)


def _plot_artifact(path: Path) -> EvaluationArtifact:
    if path.name == "metrics_boxplot.png":
        return EvaluationArtifact(
            stage="evaluate",
            artifact_type="metrics_boxplot",
            path=path,
            description="Boxplot summary of finite evaluation metrics.",
        )

    if path.name.endswith("_histogram.png"):
        metric = path.name.removesuffix("_histogram.png")
        return EvaluationArtifact(
            stage="evaluate",
            artifact_type="metric_histogram",
            path=path,
            metric=metric,
            description="Histogram of finite per-image metric values.",
        )

    return EvaluationArtifact(
        stage="evaluate",
        artifact_type="evaluation_plot",
        path=path,
        description="Evaluation diagnostic plot.",
    )


def _build_evaluation_artifacts(
    *,
    result_metrics_csv: Path | None,
    summary_csv: Path | None,
    weak_tail_csv: Path | None,
    skipped_csv: Path | None,
    plot_paths: list[Path],
    residual_heatmaps_csv: Path | None,
) -> list[EvaluationArtifact]:
    artifacts: list[EvaluationArtifact] = []

    if result_metrics_csv is not None:
        artifacts.append(
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="per_image_metrics_csv",
                path=result_metrics_csv,
                description="Per-image evaluation metrics for evaluated test samples.",
            )
        )

    if summary_csv is not None:
        artifacts.append(
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="summary_csv",
                path=summary_csv,
                description="Aggregate evaluation metric summary.",
            )
        )

    if weak_tail_csv is not None:
        artifacts.append(
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="weak_tail_csv",
                path=weak_tail_csv,
                description="Weak-tail threshold counts and percentiles.",
            )
        )

    if skipped_csv is not None:
        artifacts.append(
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="skipped_csv",
                path=skipped_csv,
                description="Samples that could not be evaluated.",
            )
        )

    artifacts.extend(_plot_artifact(path) for path in plot_paths)

    if residual_heatmaps_csv is not None:
        artifacts.append(
            EvaluationArtifact(
                stage="evaluate",
                artifact_type="residual_heatmaps_csv",
                path=residual_heatmaps_csv,
                description="Manifest of standalone residual heatmap PNGs.",
            )
        )
        artifacts.extend(residual_heatmap_artifacts_from_csv(residual_heatmaps_csv))

    return artifacts


def evaluate(config: RunConfig, config_path: Path) -> None:
    """
    Evaluate generated images against ground-truth targets.

    Reads target and generated image directories from RunConfig, writes
    per_image_metrics.csv and summary.csv to the evaluation output directory.
    Optionally writes plots if config.evaluation.save_graphs is True.
    """
    project = config.project
    run_root = project.results_path / project.run_name
    paths = RunPaths(run_root)
    paths.create_directories()
    snapshot_paths = resolve_run_snapshot_paths(stage="evaluation", run_paths=paths)
    save_stage_config_snapshots(
        config,
        config_path,
        input_dest=snapshot_paths.input_config,
        resolved_dest=snapshot_paths.resolved_config,
        hash_dest=snapshot_paths.config_hash,
    )
    save_environment_snapshot(snapshot_paths.environment)
    eval_cfg = config.evaluation

    generated_dir = (
        eval_cfg.generated_dir if eval_cfg and eval_cfg.generated_dir else paths.output_test_dir
    )
    output_dir = (
        eval_cfg.output_dir if eval_cfg and eval_cfg.output_dir else run_root / "evaluation"
    )
    save_graphs = eval_cfg.save_graphs if eval_cfg else False
    save_residual_heatmaps = eval_cfg.save_residual_heatmaps if eval_cfg else False
    residual_heatmap_metric = eval_cfg.residual_heatmap_metric if eval_cfg else "ssim"
    residual_heatmap_top_k = eval_cfg.residual_heatmap_top_k if eval_cfg else 25

    manifest = load_manifest_or_raise(project)
    manifest.validate(check_files_exist=True, require_splits={"test"})
    manifest_path = project.manifest_path
    manifest_hash = compute_manifest_hash(manifest_path)
    config_hash = snapshot_paths.config_hash.read_text(encoding="utf-8")
    ensure_run_metadata(
        paths.run_metadata,
        run_name=project.run_name,
        entrypoint="vs-evaluate",
        config_hash=config_hash,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_hash,
    )
    started_at = datetime.now(UTC).isoformat()
    test_records = manifest.filter_split("test").records
    pairs: list[tuple[Path, Path, str]] = []
    pairing_skipped: list[dict[str, str]] = []
    evaluation_details: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "generated_dir": str(generated_dir),
        "output_dir": str(output_dir),
        "metric_config": _METRIC_CONFIG,
        "residual_heatmap_config": {
            "enabled": save_residual_heatmaps,
            "metric": residual_heatmap_metric,
            "top_k": residual_heatmap_top_k,
        },
    }
    _write_evaluation_stage_metadata(
        paths,
        {
            "stage": "evaluate",
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "config_hash": config_hash,
            **evaluation_details,
        },
    )
    append_run_event(
        {
            "timestamp": started_at,
            "run_name": project.run_name,
            "stage": "evaluate",
            "event_type": "stage_started",
            "status": "running",
            "config_hash": config_hash,
            "details": evaluation_details,
        },
        paths.metadata_dir,
    )
    for record in test_records:
        target_path = project.dataset_root / record.target_path
        generated_path = generated_path_for_record(record, generated_dir)
        if target_path.exists() and generated_path.exists():
            pairs.append((target_path, generated_path, record.sample_id))
            continue

        reason = "missing_target" if not target_path.exists() else "missing_generated"
        pairing_skipped.append(
            {
                "sample_id": record.sample_id,
                "reason": reason,
                "target_path": str(target_path),
                "generated_path": str(generated_path),
            }
        )

    try:
        result = evaluate_pairs(pairs, output_dir)
    except Exception as exc:
        completed_at = datetime.now(UTC).isoformat()
        _write_evaluation_stage_metadata(
            paths,
            {
                "stage": "evaluate",
                "status": "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "config_hash": config_hash,
                **evaluation_details,
                "error": str(exc),
            },
        )
        append_run_event(
            {
                "timestamp": completed_at,
                "run_name": project.run_name,
                "stage": "evaluate",
                "event_type": "stage_failed",
                "status": "failed",
                "config_hash": config_hash,
                "details": {**evaluation_details, "error": str(exc)},
            },
            paths.metadata_dir,
        )
        raise

    all_skipped = pairing_skipped + result.skipped_rows
    skipped_path: Path | None = None
    if all_skipped:
        skipped_path = write_skipped_csv(all_skipped, output_dir / "skipped.csv")
        logger.info("Skipped samples written to %s", skipped_path)

    if result.rows:
        result.summary_csv = write_summary_csv(result.rows, output_dir)
        result.weak_tail_csv = write_weak_tail_csv(result.rows, output_dir)
        if save_residual_heatmaps:
            result.residual_heatmaps_csv = write_residual_heatmap_artifacts(
                result.rows,
                output_dir,
                metric_name=residual_heatmap_metric,
                top_k=residual_heatmap_top_k,
            )

    plot_paths: list[Path] = []
    if save_graphs and result.rows:
        plot_paths = save_dataset_plots(result.rows, output_dir)

    completed_at = datetime.now(UTC).isoformat()
    artifact_manifest_path = write_artifact_manifest(
        _build_evaluation_artifacts(
            result_metrics_csv=result.metrics_csv,
            summary_csv=result.summary_csv,
            weak_tail_csv=result.weak_tail_csv,
            skipped_csv=skipped_path,
            plot_paths=plot_paths,
            residual_heatmaps_csv=result.residual_heatmaps_csv,
        ),
        output_dir / "artifacts.json",
        run_root=run_root,
        created_at=completed_at,
    )
    _write_evaluation_stage_metadata(
        paths,
        {
            "stage": "evaluate",
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "config_hash": config_hash,
            **evaluation_details,
            "evaluated_count": result.num_evaluated,
            "skipped_count": len(all_skipped),
            "metrics_csv_path": str(output_dir / "per_image_metrics.csv"),
            "summary_csv_path": (
                str(result.summary_csv) if result.summary_csv is not None else None
            ),
            "weak_tail_csv_path": (
                str(result.weak_tail_csv) if result.weak_tail_csv is not None else None
            ),
            "residual_heatmaps_csv_path": (
                str(result.residual_heatmaps_csv)
                if result.residual_heatmaps_csv is not None
                else None
            ),
            "artifact_manifest_path": str(artifact_manifest_path),
        },
    )
    append_run_event(
        {
            "timestamp": completed_at,
            "run_name": project.run_name,
            "stage": "evaluate",
            "event_type": "stage_completed",
            "status": "completed",
            "config_hash": config_hash,
            "details": {
                **evaluation_details,
                "evaluated_count": result.num_evaluated,
                "skipped_count": len(all_skipped),
                "metrics_csv_path": str(output_dir / "per_image_metrics.csv"),
                "summary_csv_path": (
                    str(result.summary_csv) if result.summary_csv is not None else None
                ),
                "weak_tail_csv_path": (
                    str(result.weak_tail_csv) if result.weak_tail_csv is not None else None
                ),
                "residual_heatmaps_csv_path": (
                    str(result.residual_heatmaps_csv)
                    if result.residual_heatmaps_csv is not None
                    else None
                ),
                "artifact_manifest_path": str(artifact_manifest_path),
            },
        },
        paths.metadata_dir,
    )
    logger.info(
        "Evaluation complete: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        len(all_skipped),
        output_dir,
    )
