from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from virtual_staining.config.evaluation import EvaluationProtocol
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    DatasetManifest,
    ManifestRecord,
    load_manifest_or_raise,
)
from virtual_staining.data.unpaired import resolve_domain_images
from virtual_staining.evaluation.evaluator import EvaluationSample, evaluate_samples
from virtual_staining.evaluation.plotting import METRIC_NAMES, save_dataset_plots
from virtual_staining.evaluation.summaries import write_grouped_summaries
from virtual_staining.evaluation.unpaired import (
    FEATURE_DEFINITIONS,
    UNPAIRED_FEATURE_COMPARISON_CSV,
    UNPAIRED_FEATURE_PLOT,
    UNPAIRED_IMAGE_STATISTICS_CSV,
    UNPAIRED_LIMITATIONS,
    evaluate_unpaired_collections,
)
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.inference.runner import inference_direction, inference_input_names
from virtual_staining.metrics import METRIC_SPECS
from virtual_staining.split_contract import TEST_SPLIT
from virtual_staining.utils.artifacts import collect_generated_artifacts

logger = logging.getLogger(__name__)

EVALUATION_METADATA_JSON = "evaluation_metadata.json"
EVALUATION_METADATA_SCHEMA_VERSION = 1
_GROUP_UNITS = ("set", "specimen", "patient")
# Every file the evaluate stage may write; nothing else in output_dir is ever removed.
EVALUATION_OWNED_OUTPUTS: tuple[str, ...] = (
    "per_image_metrics.csv",
    "summary.csv",
    "skipped.csv",
    *(f"{unit}_metrics.csv" for unit in _GROUP_UNITS),
    *(f"summary_{unit}.csv" for unit in _GROUP_UNITS),
    *(f"{metric}_histogram.png" for metric in METRIC_NAMES),
    "metrics_boxplot.png",
    UNPAIRED_IMAGE_STATISTICS_CSV,
    UNPAIRED_FEATURE_COMPARISON_CSV,
    UNPAIRED_FEATURE_PLOT,
    EVALUATION_METADATA_JSON,
)


def evaluation_protocol(config: RunConfig) -> EvaluationProtocol:
    """Return the configured protocol, defaulting to the method's training pairing."""
    configured = config.evaluation.protocol if config.evaluation is not None else None
    return configured or ("unpaired" if config.method.name == "cyclegan" else "paired")


def reference_domain(config: RunConfig) -> str:
    """Return the real domain the generated images are compared against."""
    return (
        config.model.inputs[0] if inference_direction(config) == "B_to_A" else config.model.target
    )


def paired_sample(
    config: RunConfig, record: ManifestRecord, generated_dir: Path
) -> EvaluationSample:
    """Map one aligned manifest record to its reference and direction-aware generated image.

    Pix2Pix and CycleGAN A_to_B: real target (B) <- generated B.
    CycleGAN B_to_A: real ``model.inputs[0]`` (A) <- generated A.
    """
    direction = inference_direction(config)
    reference = (
        record.input_paths[config.model.inputs[0]] if direction == "B_to_A" else record.target_path
    )
    return EvaluationSample(
        sample_id=record.sample_id,
        set_id=record.set_id,
        target_path=config.project.dataset_root / reference,
        generated_path=generated_path_for_record(record, generated_dir, direction),
    )


def _load_paired_manifest(config: RunConfig) -> DatasetManifest:
    try:
        manifest = load_manifest_or_raise(config.project)
        if config.method.name == "cyclegan" and (
            config.model.inputs[0] not in manifest.metadata.input_modalities
            or config.model.target != manifest.metadata.target_modality
        ):
            raise ValueError(
                f"manifest modalities {list(manifest.metadata.input_modalities)} -> "
                f"{manifest.metadata.target_modality!r} do not match domains "
                f"{config.model.inputs[0]!r} -> {config.model.target!r}"
            )
        manifest.validate(check_files_exist=True, require_splits={"test"})
    except (FileNotFoundError, ValueError) as exc:
        if config.method.name != "cyclegan":
            raise
        raise type(exc)(
            "CycleGAN evaluation.protocol='paired' requires an aligned held-out test manifest; "
            f"data.domains collections are never treated as pairs. {exc}"
        ) from exc
    return manifest


def collect_unpaired_collections(
    config: RunConfig, generated_dir: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return (generated, reference) image collections with no assumed correspondence."""
    direction = inference_direction(config)
    generated = (
        collect_generated_artifacts(generated_dir, direction) if generated_dir.is_dir() else ()
    )
    if not generated:
        raise ValueError(
            f"No {direction} generated images found under {generated_dir}; run inference "
            f"with inference.direction={direction} or set evaluation.generated_dir."
        )
    domain = reference_domain(config)
    reference = resolve_domain_images(
        config.data.domains[domain], TEST_SPLIT, config.project.dataset_root
    )
    return generated, reference


def remove_evaluation_outputs(output_dir: Path) -> None:
    """Delete known evaluate-stage reports so none from an earlier run appears current."""
    for name in EVALUATION_OWNED_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)


def _write_metadata(
    config: RunConfig,
    protocol: EvaluationProtocol,
    output_dir: Path,
    generated_dir: Path,
    counts: dict[str, int],
    artifacts: dict[str, object],
) -> Path:
    metadata: dict[str, object] = {
        "schema_version": EVALUATION_METADATA_SCHEMA_VERSION,
        "method": config.method.name,
        "training_pairing": config.data.pairing,
        "evaluation_protocol": protocol,
        "inference_direction": inference_direction(config),
        "source_domains": list(inference_input_names(config)),
        "reference_domain": reference_domain(config),
        "pairwise_metrics_available": protocol == "paired",
        "generated_dir": str(generated_dir),
        "counts": counts,
        "artifacts": artifacts,
    }
    if protocol == "unpaired":
        metadata["features"] = FEATURE_DEFINITIONS
        metadata["limitations"] = list(UNPAIRED_LIMITATIONS)
    path = output_dir / EVALUATION_METADATA_JSON
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def _evaluate_paired(
    config: RunConfig, session: ExperimentSession, generated_dir: Path, output_dir: Path
) -> tuple[dict[str, int], dict[str, object]]:
    eval_cfg = config.evaluation
    manifest = _load_paired_manifest(config)
    session.result(metric_config={name: True for name in METRIC_SPECS})
    samples = tuple(
        paired_sample(config, record, generated_dir)
        for record in manifest.filter_split("test").records
    )
    result = evaluate_samples(samples, output_dir)
    grouped_paths: list[Path] = []
    if result.rows:
        slide_sets_path = DatasetLayout.from_project(config.project).slide_sets_path
        with slide_sets_path.open(newline="", encoding="utf-8") as handle:
            set_rows = {row["set_id"]: row for row in csv.DictReader(handle)}
        grouped_paths = write_grouped_summaries(
            list(result.rows),
            set_rows,
            output_dir,
            bootstrap_iterations=eval_cfg.bootstrap_iterations if eval_cfg else 10_000,
            bootstrap_seed=eval_cfg.bootstrap_seed if eval_cfg else 0,
        )
        session.result(grouped_summary_paths=[str(path) for path in grouped_paths])

    if eval_cfg is not None and eval_cfg.save_graphs and result.rows:
        save_dataset_plots(list(result.rows), output_dir)

    summary_csv = str(result.summary_csv) if result.summary_csv is not None else None
    session.result(
        evaluated_count=result.num_evaluated,
        skipped_count=result.num_skipped,
        metrics_csv_path=str(result.metrics_csv),
        summary_csv_path=summary_csv,
    )
    logger.info(
        "Paired evaluation: %s evaluated, %s skipped -> %s",
        result.num_evaluated,
        result.num_skipped,
        output_dir,
    )
    counts = {
        "requested_count": len(samples),
        "evaluated_count": result.num_evaluated,
        "skipped_count": result.num_skipped,
    }
    artifacts = {
        "per_image_metrics_csv": str(result.metrics_csv),
        "summary_csv": summary_csv,
        "skipped_csv": str(result.skipped_csv) if result.skipped_csv is not None else None,
        "grouped_summaries": [str(path) for path in grouped_paths],
    }
    return counts, artifacts


def _evaluate_unpaired(
    config: RunConfig, session: ExperimentSession, generated_dir: Path, output_dir: Path
) -> tuple[dict[str, int], dict[str, object]]:
    generated, reference = collect_unpaired_collections(config, generated_dir)
    result = evaluate_unpaired_collections(
        generated,
        reference,
        output_dir,
        save_graphs=config.evaluation is not None and config.evaluation.save_graphs,
    )
    artifacts = {
        "unpaired_image_statistics_csv": str(result.image_statistics_csv),
        "unpaired_feature_comparison_csv": str(result.feature_comparison_csv),
        "unpaired_feature_plot": str(result.graph_path) if result.graph_path else None,
    }
    session.result(
        generated_count=result.generated_count,
        reference_count=result.reference_count,
        unpaired_image_statistics_path=artifacts["unpaired_image_statistics_csv"],
        unpaired_feature_comparison_path=artifacts["unpaired_feature_comparison_csv"],
        unpaired_feature_plot_path=artifacts["unpaired_feature_plot"],
    )
    logger.info(
        "Unpaired evaluation: %s generated vs %s reference images -> %s",
        result.generated_count,
        result.reference_count,
        output_dir,
    )
    counts = {
        "generated_count": result.generated_count,
        "reference_count": result.reference_count,
    }
    return counts, artifacts


def evaluate(config: RunConfig, config_path: Path) -> None:
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
        protocol = evaluation_protocol(config)
        session.result(
            generated_dir=str(generated_dir),
            output_dir=str(output_dir),
            evaluation_protocol=protocol,
            method=config.method.name,
            inference_direction=inference_direction(config),
        )
        remove_evaluation_outputs(output_dir)
        run = _evaluate_paired if protocol == "paired" else _evaluate_unpaired
        counts, artifacts = run(config, session, generated_dir, output_dir)
        metadata_path = _write_metadata(
            config, protocol, output_dir, generated_dir, counts, artifacts
        )
        session.result(evaluation_metadata_path=str(metadata_path))
