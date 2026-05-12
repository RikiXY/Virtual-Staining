from __future__ import annotations

import logging
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.io import build_evaluation_pairs
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)

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

    target_dir = (
        eval_cfg.target_dir if eval_cfg and eval_cfg.target_dir else project.dataset_test_dir
    )
    generated_dir = (
        eval_cfg.generated_dir if eval_cfg and eval_cfg.generated_dir else paths.output_test_dir
    )
    output_dir = (
        eval_cfg.output_dir if eval_cfg and eval_cfg.output_dir else run_root / "evaluation"
    )
    save_graphs = eval_cfg.save_graphs if eval_cfg else False

    pairs, skipped_ids = build_evaluation_pairs(target_dir, generated_dir)
    result = evaluate_pairs(pairs, output_dir)

    if result.rows:
        result.summary_csv = write_summary_csv(result.rows, output_dir)

    if save_graphs and result.rows:
        save_dataset_plots(result.rows, output_dir)

    total_skipped = result.num_skipped + len(skipped_ids)
    logger.info(
        "Evaluation complete: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        total_skipped,
        output_dir,
    )
