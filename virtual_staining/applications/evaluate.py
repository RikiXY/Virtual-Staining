from __future__ import annotations

import csv
import logging
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.evaluation.evaluator import EvaluationSample, evaluate_samples
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.summaries import write_grouped_summaries
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.metrics import METRIC_SPECS

logger = logging.getLogger(__name__)


def evaluate(config: RunConfig, config_path: Path) -> None:
    """Evaluate generated images against ground-truth targets."""
    project = config.project
    with ExperimentSession.open(
        config=config, config_path=config_path, stage="evaluate"
    ) as session:
        eval_cfg = config.evaluation
        generated_dir = (
            eval_cfg.generated_dir
            if eval_cfg and eval_cfg.generated_dir
            else session.paths.output_test_dir
        )
        output_dir = (
            eval_cfg.output_dir
            if eval_cfg and eval_cfg.output_dir
            else session.paths.evaluation_dir
        )
        save_graphs = eval_cfg.save_graphs if eval_cfg else False

        dataset_layout = DatasetLayout.from_project(project)
        manifest = load_manifest_or_raise(project)
        manifest.validate(check_files_exist=True, require_splits={"test"})
        test_records = manifest.filter_split("test").records
        evaluation_details: dict[str, object] = {
            "generated_dir": str(generated_dir),
            "output_dir": str(output_dir),
            "metric_config": {name: True for name in METRIC_SPECS},
        }
        session.result(**evaluation_details)

        samples = tuple(
            EvaluationSample(
                sample_id=record.sample_id,
                set_id=record.set_id,
                target_path=project.dataset_root / record.target_path,
                generated_path=generated_path_for_record(record, generated_dir),
            )
            for record in test_records
        )
        result = evaluate_samples(samples, output_dir)
        if result.rows:
            with dataset_layout.slide_sets_path.open(newline="", encoding="utf-8") as handle:
                set_rows = {row["set_id"]: row for row in csv.DictReader(handle)}
            grouped_paths = write_grouped_summaries(
                list(result.rows),
                set_rows,
                output_dir,
                bootstrap_iterations=eval_cfg.bootstrap_iterations if eval_cfg else 10_000,
                bootstrap_seed=eval_cfg.bootstrap_seed if eval_cfg else 0,
            )
            session.result(grouped_summary_paths=[str(path) for path in grouped_paths])

        if save_graphs and result.rows:
            save_dataset_plots(list(result.rows), output_dir)

        session.result(
            evaluated_count=result.num_evaluated,
            skipped_count=result.num_skipped,
            metrics_csv_path=str(result.metrics_csv),
            summary_csv_path=str(result.summary_csv) if result.summary_csv is not None else None,
        )
    logger.info(
        "Evaluation complete: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        result.num_skipped,
        output_dir,
    )
