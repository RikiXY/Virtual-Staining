from __future__ import annotations

import csv
import logging
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import resolve_domain_images
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord, load_manifest_or_raise
from virtual_staining.evaluation.evaluator import (
    EvaluationResult,
    EvaluationSample,
    evaluate_samples,
)
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.summaries import write_grouped_summaries
from virtual_staining.evaluation.unpaired import evaluate_unpaired_distributions
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.metrics import DEFAULT_METRICS, METRIC_SPECS
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

_UNPAIRED_REPORT_NAMES = (
    "unpaired_evaluation.json",
    "unpaired_image_statistics.csv",
    "unpaired_feature_distributions.png",
    "unpaired_feature_wasserstein.png",
)
_PAIRED_REPORT_NAMES = (
    "per_image_metrics.csv",
    "summary.csv",
    "skipped.csv",
    "metrics_boxplot.png",
    *(f"{metric}_histogram.png" for metric in DEFAULT_METRICS),
    *(f"{unit}_metrics.csv" for unit in ("set", "specimen", "patient")),
    *(f"summary_{unit}.csv" for unit in ("set", "specimen", "patient")),
)


def _protocol(config: RunConfig) -> str:
    configured = config.evaluation.protocol if config.evaluation else "auto"
    return config.data.pairing if configured == "auto" else configured


def _remove_stale_reports(output_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (output_dir / name).unlink(missing_ok=True)


def _direction(config: RunConfig) -> str:
    if config.inference and config.inference.direction:
        return config.inference.direction
    return "A_to_B"


def _manifest_paths_for_direction(config: RunConfig, record: ManifestRecord) -> tuple[Path, Path]:
    """Return input and reference paths for the configured inference direction."""
    source_name = config.model.inputs[0]
    try:
        source_path = record.input_paths[source_name]
    except KeyError:
        raise ValueError(
            f"Manifest record {record.sample_id!r} has no input modality {source_name!r}"
        ) from None
    if _direction(config) == "B_to_A":
        return record.target_path, source_path
    return source_path, record.target_path


def _unpaired_generated_path(input_path: Path, generated_dir: Path) -> Path:
    return generated_dir / f"{input_path.stem}_generated{input_path.suffix}"


def _load_paired_test_manifest(config: RunConfig) -> DatasetManifest:
    manifest = load_manifest_or_raise(config.project)
    manifest.validate(check_files_exist=True, require_splits={"test"})
    return manifest.filter_split("test")


def _paired_samples(
    config: RunConfig, manifest: DatasetManifest, generated_dir: Path
) -> tuple[EvaluationSample, ...]:
    samples: list[EvaluationSample] = []
    for record in manifest.records:
        input_relative, reference_relative = _manifest_paths_for_direction(config, record)
        generated_path = (
            _unpaired_generated_path(input_relative, generated_dir)
            if config.data.pairing == "unpaired"
            else generated_path_for_record(record, generated_dir)
        )
        samples.append(
            EvaluationSample(
                sample_id=record.sample_id,
                set_id=record.set_id,
                target_path=config.project.dataset_root / reference_relative,
                generated_path=generated_path,
            )
        )
    return tuple(samples)


def _write_paired_reports(
    config: RunConfig,
    manifest: DatasetManifest,
    generated_dir: Path,
    output_dir: Path,
) -> tuple[EvaluationResult, list[Path]]:
    result = evaluate_samples(_paired_samples(config, manifest, generated_dir), output_dir)
    grouped_paths: list[Path] = []
    slide_sets_path = DatasetLayout.from_project(config.project).slide_sets_path
    if result.rows and slide_sets_path.is_file():
        with slide_sets_path.open(newline="", encoding="utf-8") as handle:
            set_rows = {row["set_id"]: row for row in csv.DictReader(handle)}
        grouped_paths = write_grouped_summaries(
            list(result.rows),
            set_rows,
            output_dir,
            bootstrap_iterations=(
                config.evaluation.bootstrap_iterations if config.evaluation else 10_000
            ),
            bootstrap_seed=config.evaluation.bootstrap_seed if config.evaluation else 0,
        )
    graph_paths = (
        save_dataset_plots(list(result.rows), output_dir)
        if config.evaluation and config.evaluation.save_graphs and result.rows
        else []
    )
    return result, [*grouped_paths, *graph_paths]


def _generated_images(generated_dir: Path) -> tuple[Path, ...]:
    if not generated_dir.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in generated_dir.glob("**/*")
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
    )


def _real_target_images(config: RunConfig) -> tuple[Path, ...]:
    eval_cfg = config.evaluation
    if eval_cfg and eval_cfg.real_target is not None:
        return resolve_domain_images(config.project.dataset_root, eval_cfg.real_target, "test")

    target_name = config.model.inputs[0] if _direction(config) == "B_to_A" else config.model.target
    domain_spec = config.data.domains.get(target_name)
    if domain_spec is not None:
        return resolve_domain_images(config.project.dataset_root, domain_spec, "test")

    manifest = _load_paired_test_manifest(config)
    return tuple(
        config.project.dataset_root / _manifest_paths_for_direction(config, record)[1]
        for record in manifest.records
    )


def evaluate(config: RunConfig, config_path: Path) -> None:
    """Evaluate generated images using a protocol independent from training pairing."""
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
        protocol = _protocol(config)

        if protocol == "unpaired":
            result = evaluate_unpaired_distributions(
                _generated_images(generated_dir),
                _real_target_images(config),
                output_dir,
                method=config.method.name,
                direction=_direction(config),
                save_graphs=eval_cfg.save_graphs if eval_cfg else False,
            )
            _remove_stale_reports(output_dir, _PAIRED_REPORT_NAMES)
            session.result(
                output_dir=str(output_dir),
                method=config.method.name,
                training_pairing=config.data.pairing,
                evaluation_protocol="unpaired",
                paired_metrics_available=False,
                generated_count=result.generated_count,
                real_target_count=result.real_target_count,
                statistics_csv_path=str(result.statistics_csv),
                graph_paths=[str(path) for path in result.graph_paths],
                metadata_path=str(result.metadata_path),
            )
            logger.info(
                "Unpaired evaluation compared %s generated and %s real-target images -> %s",
                result.generated_count,
                result.real_target_count,
                output_dir,
            )
            return

        manifest = _load_paired_test_manifest(config)
        evaluation_details: dict[str, object] = {
            "generated_dir": str(generated_dir),
            "output_dir": str(output_dir),
            "training_pairing": config.data.pairing,
            "evaluation_protocol": "paired",
            "direction": _direction(config),
            "paired_metrics_available": True,
            "manifest_path": str(DatasetLayout.from_project(project).manifest_path),
            "metric_config": {name: True for name in METRIC_SPECS},
        }
        session.result(**evaluation_details)
        result, report_paths = _write_paired_reports(config, manifest, generated_dir, output_dir)
        _remove_stale_reports(output_dir, _UNPAIRED_REPORT_NAMES)
        session.result(
            evaluated_count=result.num_evaluated,
            skipped_count=result.num_skipped,
            metrics_csv_path=str(result.metrics_csv),
            summary_csv_path=str(result.summary_csv) if result.summary_csv is not None else None,
            report_paths=[str(path) for path in report_paths],
        )
    logger.info(
        "Paired evaluation complete: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        result.num_skipped,
        output_dir,
    )
