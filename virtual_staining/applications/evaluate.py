from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from virtual_staining.config.evaluation import EvaluationProtocol
from virtual_staining.config.run import RunConfig
from virtual_staining.data.consumption import (
    AssetRow,
    DataSnapshot,
    build_snapshot,
    compare_generated,
    enrich_with_groups,
    load_group_metadata,
    load_snapshot,
    relative_locator,
)
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    DatasetManifest,
    ManifestRecord,
    load_manifest_or_raise,
    load_set_groups,
    manifest_sources,
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
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.inference.runner import inference_direction, inference_input_names
from virtual_staining.metrics import METRIC_SPECS
from virtual_staining.split_contract import TEST_SPLIT
from virtual_staining.utils.artifacts import collect_generated_artifacts

logger = logging.getLogger(__name__)

EVALUATION_METADATA_JSON = "evaluation_metadata.json"
EVALUATION_METADATA_SCHEMA_VERSION = 1
PAIRED_EVALUATION_ADAPTER = "paired_evaluation/1"
UNPAIRED_EVALUATION_ADAPTER = "unpaired_evaluation/1"
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
    return configured or cast(EvaluationProtocol, config.data.pairing)


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
        if config.data.pairing == "unpaired" and (
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
        if config.data.pairing != "unpaired":
            raise
        raise type(exc)(
            "evaluation.protocol='paired' for a data.pairing='unpaired' run requires an aligned "
            "held-out test manifest; data.domains collections are never treated as pairs. "
            f"{exc}"
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


def _evaluation_context(config: RunConfig, protocol: EvaluationProtocol) -> dict[str, object]:
    return {
        "protocol": protocol,
        "method": config.method.name,
        "direction": inference_direction(config),
        "reference_domain": reference_domain(config),
    }


def paired_evaluation_snapshot(
    config: RunConfig,
    samples: Sequence[EvaluationSample],
    records: Sequence[ManifestRecord],
    generated_dir: Path,
) -> DataSnapshot:
    """Snapshot the exact generated/reference correspondence handed to the evaluator.

    Each sample contributes one ``reference`` and one ``generated`` row sharing its
    ``sample_id``; that shared ID is the explicit correspondence. A generated file that is
    requested but absent is recorded with ``status=missing``, never as read.
    """
    groups = load_set_groups(config.project)
    domain = reference_domain(config)
    rows: list[AssetRow] = []
    for sample, record in zip(samples, records, strict=True):
        specimen, patient = groups.get(record.set_id, ("", ""))
        common: dict[str, Any] = {
            "domain": domain,
            "split": record.split,
            "sample_id": sample.sample_id,
            "set_id": sample.set_id,
            "specimen_id": specimen,
            "patient_id": patient,
        }
        rows.append(
            AssetRow(
                root="dataset",
                locator=relative_locator(config.project.dataset_root, sample.target_path),
                role="reference",
                **common,
            )
        )
        rows.append(
            AssetRow(
                root="generated",
                locator=relative_locator(generated_dir, sample.generated_path),
                role="generated",
                **common,
            )
        )
    return build_snapshot(
        rows,
        kind="consumed",
        adapter=PAIRED_EVALUATION_ADAPTER,
        roots={"dataset": config.project.dataset_root, "generated": generated_dir},
        hash_policy=config.data.hash_policy,
        group_validation=config.data.group_validation,
        selection={"split": TEST_SPLIT, "correspondence": "manifest_sample_id"},
        context=_evaluation_context(config, "paired"),
        sources=manifest_sources(config.project),
        allow_missing=True,
    )


def unpaired_evaluation_snapshot(
    config: RunConfig,
    generated: Sequence[Path],
    reference: Sequence[Path],
    generated_dir: Path,
) -> DataSnapshot:
    """Snapshot two independent collections; no per-image correspondence is created."""
    domain = reference_domain(config)
    root = config.project.dataset_root
    reference_rows = [
        AssetRow(
            root="dataset",
            locator=relative_locator(root, path),
            role="reference",
            domain=domain,
            split=TEST_SPLIT,
        )
        for path in reference
    ]
    if config.data.group_metadata is not None:
        sidecar = config.data.group_metadata
        reference_rows = enrich_with_groups(
            reference_rows,
            load_group_metadata(sidecar if sidecar.is_absolute() else root / sidecar),
        )
    generated_rows = [
        AssetRow(
            root="generated",
            locator=relative_locator(generated_dir, path),
            role="generated",
            domain=domain,
            split=TEST_SPLIT,
        )
        for path in generated
    ]
    return build_snapshot(
        (*generated_rows, *reference_rows),
        kind="consumed",
        adapter=UNPAIRED_EVALUATION_ADAPTER,
        roots={"dataset": root, "generated": generated_dir},
        hash_policy=config.data.hash_policy,
        group_validation=config.data.group_validation,
        selection={
            "split": TEST_SPLIT,
            "correspondence": None,
            "reference_spec": config.data.domains[domain],
        },
        context=_evaluation_context(config, "unpaired"),
    )


def generated_producer(
    paths: RunLayout, snapshot: DataSnapshot, direction: str | None
) -> dict[str, object]:
    """Link consumed generated files to this run's tracked inference outputs, if they match.

    Linkage requires the recorded inference output snapshot to match the consumed generated
    files by locator, size, and (when both were hashed) content; a matching directory path
    alone never establishes it. Anything else is ``external`` or ``unlinked``.
    """
    record_path = paths.stage_record("infer")
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    produced = record.get("produced_data") if record.get("status") == "completed" else None
    if not isinstance(produced, dict):
        return {"status": "external", "reason": "no completed tracked inference in this run"}
    try:
        outputs = load_snapshot(paths.produced_data("infer"))
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "external", "reason": f"inference output snapshot unreadable: {exc}"}
    if outputs.snapshot_id != produced.get("snapshot_id"):
        return {"status": "external", "reason": "inference output snapshot does not match record"}
    diff = compare_generated(snapshot.rows, outputs)
    link: dict[str, object] = {
        "inference_output_snapshot_id": outputs.snapshot_id,
        "inference_consumed_snapshot_id": outputs.context.get("consumed_snapshot_id"),
        "checkpoint_sha256": outputs.context.get("checkpoint_sha256"),
        "direction": outputs.context.get("direction"),
        **{key: value for key, value in diff.items() if key != "content_verified"},
        "content_verified": diff["content_verified"],
    }
    if outputs.context.get("direction") != direction:
        return {"status": "unlinked", "reason": "inference direction differs", **link}
    if diff["missing"] or diff["extra"] or diff["changed"]:
        return {
            "status": "unlinked",
            "reason": "generated files differ from the recorded inference outputs",
            **link,
        }
    return {"status": "linked", **link}


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
    session: ExperimentSession,
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
        "consumed_data": session.consumed_data,
        "generated_producer": session.details.get("generated_producer"),
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
    direction = inference_direction(config)
    manifest = _load_paired_manifest(config)
    session.result(metric_config={name: True for name in METRIC_SPECS})
    records = manifest.filter_split("test").records
    samples = tuple(paired_sample(config, record, generated_dir) for record in records)
    # The evaluator receives exactly the samples the snapshot describes.
    snapshot = paired_evaluation_snapshot(config, samples, records, generated_dir)
    session.bind_inputs(snapshot)
    session.result(generated_producer=generated_producer(session.paths, snapshot, direction))
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
    direction = inference_direction(config)
    generated, reference = collect_unpaired_collections(config, generated_dir)
    snapshot = unpaired_evaluation_snapshot(config, generated, reference, generated_dir)
    session.bind_inputs(snapshot)
    session.result(generated_producer=generated_producer(session.paths, snapshot, direction))
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
            config, protocol, output_dir, generated_dir, counts, artifacts, session
        )
        session.result(evaluation_metadata_path=str(metadata_path))
