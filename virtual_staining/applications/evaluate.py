from __future__ import annotations

import logging
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.reports import write_skipped_csv
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.inference.outputs import generated_path_for_record

logger = logging.getLogger(__name__)


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

    manifest = load_manifest_or_raise(project)
    manifest.validate(check_files_exist=True, require_splits={"test"})
    test_records = manifest.filter_split("test").records
    pairs: list[tuple[Path, Path, str]] = []
    pairing_skipped: list[dict[str, str]] = []
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

    result = evaluate_pairs(pairs, output_dir)
    all_skipped = pairing_skipped + result.skipped_rows
    if all_skipped:
        skipped_path = write_skipped_csv(all_skipped, output_dir / "skipped.csv")
        logger.info("Skipped samples written to %s", skipped_path)

    if result.rows:
        result.summary_csv = write_summary_csv(result.rows, output_dir)

    if save_graphs and result.rows:
        save_dataset_plots(result.rows, output_dir)

    logger.info(
        "Evaluation complete: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        len(all_skipped),
        output_dir,
    )
