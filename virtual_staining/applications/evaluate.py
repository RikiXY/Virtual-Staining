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
    require_model_modalities,
)
from virtual_staining.data.unpaired import resolve_domain_images
from virtual_staining.evaluation.evaluator import EvaluationSample, evaluate_samples
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.reports import (
    COVERAGE_CSV,
    EVALUATION_RESULT_JSON,
    PER_IMAGE_METRICS_CSV,
)
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
from virtual_staining.inference.runner import (
    inference_direction,
    inference_input_names,
    inference_output_names,
)
from virtual_staining.metrics import ResolvedMetric, resolve_metrics
from virtual_staining.split_contract import TEST_SPLIT
from virtual_staining.utils.artifacts import collect_generated_artifacts

logger = logging.getLogger(__name__)

EVALUATION_METADATA_JSON = "evaluation_metadata.json"
EVALUATION_METADATA_SCHEMA_VERSION = 3
PAIRED_EVALUATION_ADAPTER = "paired_evaluation/1"
UNPAIRED_EVALUATION_ADAPTER = "unpaired_evaluation/1"
_GROUP_UNITS = ("set", "specimen", "patient")
# Every file the evaluate stage may write, plus ``*_histogram.png`` (one per output and
# metric); nothing else in output_dir is ever removed.
EVALUATION_OWNED_OUTPUTS: tuple[str, ...] = (
    PER_IMAGE_METRICS_CSV,
    "summary.csv",
    COVERAGE_CSV,
    EVALUATION_RESULT_JSON,
    *(f"{unit}_metrics.csv" for unit in _GROUP_UNITS),
    *(f"summary_{unit}.csv" for unit in _GROUP_UNITS),
    "metrics_boxplot.png",
    UNPAIRED_IMAGE_STATISTICS_CSV,
    UNPAIRED_FEATURE_COMPARISON_CSV,
    UNPAIRED_FEATURE_PLOT,
    EVALUATION_METADATA_JSON,
)


def requested_metrics(config: RunConfig) -> tuple[ResolvedMetric, ...]:
    """The configured metric request, or the built-in default set, resolved once."""
    if config.evaluation is not None and config.evaluation.metrics is not None:
        return config.evaluation.metrics
    return resolve_metrics(None, config.definitions.metrics)


def evaluation_protocol(config: RunConfig) -> EvaluationProtocol:
    """Return the configured protocol, defaulting to the method's training pairing."""
    configured = config.evaluation.protocol if config.evaluation is not None else None
    return configured or cast(EvaluationProtocol, config.data.pairing)


def evaluation_generated_dir(config: RunConfig, paths: RunLayout) -> Path:
    """Return ``evaluation.generated_dir``, defaulting to the run's inference test outputs."""
    configured = config.evaluation.generated_dir if config.evaluation is not None else None
    return configured or paths.output_test_dir


def reference_domains(config: RunConfig) -> tuple[str, ...]:
    """The real domains generated images are compared against: the predicted outputs.

    Pix2Pix and CycleGAN A_to_B compare each output with its real target; CycleGAN B_to_A
    compares generated A with the real ``model.inputs[0]``.
    """
    return inference_output_names(config)


def paired_samples(
    config: RunConfig, record: ManifestRecord, generated_dir: Path
) -> tuple[EvaluationSample, ...]:
    """One explicit pair per predicted output of an aligned manifest record.

    Each pair is ``(sample_id, output_name)``: the record's real file of that domain and
    the generated artifact of that output. Outputs are never pooled.
    """
    root = config.project.dataset_root
    return tuple(
        EvaluationSample(
            sample_id=record.sample_id,
            output_name=name,
            set_id=record.set_id,
            target_path=root / record.domain_path(name),
            generated_path=generated_path_for_record(record, generated_dir, name),
        )
        for name in reference_domains(config)
    )


def load_paired_evaluation_manifest(config: RunConfig) -> DatasetManifest:
    try:
        manifest = load_manifest_or_raise(config.project)
        require_model_modalities(manifest, config.model.inputs, config.model.outputs)
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


def unpaired_output(config: RunConfig) -> str:
    """The one predicted output an unpaired collection diagnostic describes."""
    outputs = reference_domains(config)
    if len(outputs) != 1:
        raise ValueError(
            f"Unpaired evaluation needs exactly one predicted output, got {list(outputs)}"
        )
    return outputs[0]


def unpaired_generated_collection(config: RunConfig, generated_dir: Path) -> tuple[Path, ...]:
    """Return the generated images of the predicted output under ``generated_dir``."""
    output = unpaired_output(config)
    generated = collect_generated_artifacts(generated_dir, output) if generated_dir.is_dir() else ()
    if not generated:
        direction = inference_direction(config)
        hint = f" with inference.direction={direction}" if direction is not None else ""
        raise ValueError(
            f"No generated {output!r} images found under {generated_dir / output}; run "
            f"inference{hint} or set evaluation.generated_dir."
        )
    return generated


def unpaired_reference_collection(config: RunConfig) -> tuple[Path, ...]:
    """Return the independent real reference images of the held-out test split."""
    return resolve_domain_images(
        unpaired_reference_spec(config), TEST_SPLIT, config.project.dataset_root
    )


def collect_unpaired_collections(
    config: RunConfig, generated_dir: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return (generated, reference) image collections with no assumed correspondence."""
    return unpaired_generated_collection(config, generated_dir), unpaired_reference_collection(
        config
    )


def unpaired_reference_spec(config: RunConfig) -> str:
    """Return the independent real reference collection spec for unpaired evaluation.

    ``evaluation.reference_collection`` wins; otherwise the reference domain's
    ``data.domains`` entry. A paired manifest is never used as a collection.
    """
    if config.evaluation is not None and config.evaluation.reference_collection is not None:
        return config.evaluation.reference_collection
    domain = unpaired_output(config)
    if domain not in config.data.domains:
        raise ValueError(
            f"Unpaired evaluation requires an independent real reference collection for "
            f"domain {domain!r}: set evaluation.reference_collection (a directory holding "
            "test/ or a path/glob containing {split})."
        )
    return config.data.domains[domain]


def _evaluation_context(config: RunConfig, protocol: EvaluationProtocol) -> dict[str, object]:
    return {
        "protocol": protocol,
        "method": config.method.name,
        "direction": inference_direction(config),
        "reference_domains": list(reference_domains(config)),
    }


def paired_evaluation_snapshot(
    config: RunConfig,
    samples: Sequence[EvaluationSample],
    records: Sequence[ManifestRecord],
    generated_dir: Path,
) -> DataSnapshot:
    """Snapshot the exact generated/reference correspondence handed to the evaluator.

    Each ``(sample_id, output_name)`` pair contributes one ``reference`` and one
    ``generated`` row sharing that ``sample_id`` and naming the output as ``domain``;
    that shared identity is the explicit correspondence. A generated file that is
    requested but absent is recorded with ``status=missing``, never as read.
    """
    groups = load_set_groups(config.project)
    by_sample = {record.sample_id: record for record in records}
    rows: list[AssetRow] = []
    for sample in samples:
        record = by_sample[sample.sample_id]
        specimen, patient = groups.get(record.set_id, ("", ""))
        common: dict[str, Any] = {
            "domain": sample.output_name,
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
        selection={"split": TEST_SPLIT, "correspondence": "manifest_sample_id_output_name"},
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
    domain = unpaired_output(config)
    root = config.project.dataset_root
    reference_spec = unpaired_reference_spec(config)
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
    # Training-domain group metadata never describes an explicit evaluation collection.
    from_domains = config.evaluation is None or config.evaluation.reference_collection is None
    if from_domains and config.data.group_metadata is not None:
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
            "reference_spec": reference_spec,
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
    for path in output_dir.glob("*_histogram.png"):
        path.unlink()


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
        "reference_domains": list(reference_domains(config)),
        "pairwise_metrics_available": protocol == "paired",
        "generated_dir": str(generated_dir),
        "counts": counts,
        "artifacts": artifacts,
        # Metric identities, statuses and coverage live in the evaluation result record.
        "consumed_data": session.consumed_data,
        "generated_producer": session.details.get("generated_producer"),
    }
    if protocol == "unpaired":
        metadata["features"] = FEATURE_DEFINITIONS
        metadata["limitations"] = list(UNPAIRED_LIMITATIONS)
    path = output_dir / EVALUATION_METADATA_JSON
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _evaluate_paired(
    config: RunConfig, session: ExperimentSession, generated_dir: Path, output_dir: Path
) -> tuple[dict[str, int], dict[str, object]]:
    eval_cfg = config.evaluation
    direction = inference_direction(config)
    metrics = requested_metrics(config)
    manifest = load_paired_evaluation_manifest(config)
    session.result(requested_metrics=[metric.name for metric in metrics])
    records = manifest.filter_split("test").records
    samples = tuple(
        sample for record in records for sample in paired_samples(config, record, generated_dir)
    )
    # The evaluator receives exactly the samples the snapshot describes.
    snapshot = paired_evaluation_snapshot(config, samples, records, generated_dir)
    session.bind_inputs(snapshot)
    session.result(generated_producer=generated_producer(session.paths, snapshot, direction))
    result = evaluate_samples(
        samples,
        output_dir,
        metrics=metrics,
        input_failures=eval_cfg.input_failures if eval_cfg else "strict",
    )
    slide_sets_path = DatasetLayout.from_project(config.project).slide_sets_path
    with slide_sets_path.open(newline="", encoding="utf-8") as handle:
        set_rows = {row["set_id"]: row for row in csv.DictReader(handle)}
    grouped_paths = write_grouped_summaries(
        result.rows,
        [metric.name for metric in metrics],
        set_rows,
        output_dir,
        bootstrap_iterations=eval_cfg.bootstrap_iterations if eval_cfg else 10_000,
        bootstrap_seed=eval_cfg.bootstrap_seed if eval_cfg else 0,
    )
    session.result(grouped_summary_paths=[str(path) for path in grouped_paths])

    if eval_cfg is not None and eval_cfg.save_graphs:
        save_dataset_plots(result.rows, metrics, output_dir)

    session.result(
        evaluated_count=result.num_evaluated,
        excluded_count=result.num_excluded,
        metrics_csv_path=str(result.metrics_csv),
        summary_csv_path=str(result.summary_csv),
    )
    logger.info(
        "Paired evaluation: %s evaluated, %s excluded -> %s",
        result.num_evaluated,
        result.num_excluded,
        output_dir,
    )
    counts = {
        "requested_count": result.num_requested,
        "evaluated_count": result.num_evaluated,
        "excluded_count": result.num_excluded,
    }
    artifacts = {
        "evaluation_result": str(result.result_json),
        "per_image_metrics_csv": str(result.metrics_csv),
        "summary_csv": str(result.summary_csv),
        "coverage_csv": str(result.coverage_csv),
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
        generated_dir = evaluation_generated_dir(config, session.paths)
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
