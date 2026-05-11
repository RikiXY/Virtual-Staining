from __future__ import annotations

import logging
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.io import collect_image_files
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.experiment.run_paths import RunPaths

logger = logging.getLogger(__name__)


def evaluate(config: RunConfig) -> None:
    """
    Evaluate generated images against ground-truth targets.

    Reads target and generated image directories from RunConfig, writes
    per_image_metrics.csv and summary.csv to the evaluation output directory.
    Optionally writes plots if config.evaluation.save_graphs is True.
    """
    project = config.project
    run_root = project.results_path / project.run_name
    paths = RunPaths(run_root)
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

    pairs = _build_pairs(target_dir, generated_dir)
    result = evaluate_pairs(pairs, output_dir)

    if result.rows:
        result.summary_csv = write_summary_csv(result.rows, output_dir)

    if save_graphs and result.rows:
        save_dataset_plots(result.rows, output_dir)

    total_skipped = result.num_skipped + _count_unmatched_samples(target_dir, generated_dir)
    logger.info(
        "Evaluation complete: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        total_skipped,
        output_dir,
    )


def _build_pairs(
    target_dir: Path,
    generated_dir: Path,
) -> list[tuple[Path, Path, str]]:
    """
    Pair target images with generated images by sample ID.

    Target files match `*_target.<ext>`; generated files match
    `*_target_generated.<ext>`. The sample_id is the prefix before `_target`.
    """
    target_files = collect_image_files(target_dir, "_target", "Target")
    generated_files = collect_image_files(generated_dir, "_target_generated", "Generated")
    all_sample_ids = sorted(set(target_files) | set(generated_files))
    pairs: list[tuple[Path, Path, str]] = []

    for sample_id in all_sample_ids:
        target_path = target_files.get(sample_id)
        generated_path = generated_files.get(sample_id)

        if target_path is None:
            logger.warning("Missing target for sample %s in %s", sample_id, target_dir)
            continue

        if generated_path is None:
            logger.warning("Missing generated image for sample %s in %s", sample_id, generated_dir)
            continue

        pairs.append((target_path, generated_path, sample_id))

    return pairs


def _count_unmatched_samples(target_dir: Path, generated_dir: Path) -> int:
    target_files = collect_image_files(target_dir, "_target", "Target")
    generated_files = collect_image_files(generated_dir, "_target_generated", "Generated")
    return len(set(target_files) ^ set(generated_files))
