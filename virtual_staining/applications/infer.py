from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from torchvision.utils import save_image

from virtual_staining.config.run import RunConfig
from virtual_staining.data.consumption import AssetRow, build_snapshot, relative_locator
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    ManifestRecord,
    load_manifest_or_raise,
    load_set_groups,
    manifest_sources,
)
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.inference.runner import (
    InferenceResult,
    build_inference_transform,
    inference_direction,
    inference_input_names,
    load_inference_generator,
    predict_batch,
    resolve_inference_checkpoint,
    resolve_inference_device,
)
from virtual_staining.utils.hashing import sha256_file_verified, sha256_json

logger = logging.getLogger(__name__)

PAIRED_INFER_ADAPTER = "paired_manifest_infer/1"
INFER_OUTPUT_ADAPTER = "inference_outputs/1"


def _prediction_sources(
    record: ManifestRecord, source_names: tuple[str, ...], direction: str | None
) -> dict[str, Path]:
    """Root-relative files a record feeds to the predictor; nothing else is opened."""
    if direction == "B_to_A":
        return {name: record.target_path for name in source_names}
    return {name: record.input_paths[name] for name in source_names}


def infer(config: RunConfig, config_path: Path) -> InferenceResult:
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    with ExperimentSession.open(config=config, config_path=config_path, stage="infer") as session:
        output_dir = config.inference.output_dir or session.paths.output_test_dir
        direction = inference_direction(config)
        manifest = load_manifest_or_raise(config.project)
        if not set(config.model.inputs).issubset(manifest.metadata.input_modalities):
            raise ValueError("model.inputs must be a subset of manifest input modalities")
        if config.model.target != manifest.metadata.target_modality:
            raise ValueError("model.target must equal manifest target modality")
        manifest.validate(check_files_exist=True, require_splits={"test"})
        test_manifest = manifest.filter_split("test")
        checkpoint_path = resolve_inference_checkpoint(config, session.paths)
        checkpoint_sha256, _ = sha256_file_verified(checkpoint_path)

        # The snapshot and the prediction loop read the same _prediction_sources, so exactly
        # the files fed to the predictor are consumed; for B_to_A the held-out aligned target
        # is the domain-B source and the domain-A references are never opened.
        groups = load_set_groups(config.project)
        source_names = inference_input_names(config)
        rows = [
            AssetRow(
                root="dataset",
                locator=path.as_posix(),
                role="input",
                domain=name,
                split=record.split,
                sample_id=record.sample_id,
                set_id=record.set_id,
                specimen_id=groups.get(record.set_id, ("", ""))[0],
                patient_id=groups.get(record.set_id, ("", ""))[1],
            )
            for record in test_manifest.records
            for name, path in _prediction_sources(record, source_names, direction).items()
        ]
        resolved = config.to_dict()
        generation = {
            "method": resolved["method"],
            "model": resolved["model"],
            "image_size": list(config.project.image_size),
            "direction": direction,
        }
        snapshot = build_snapshot(
            rows,
            kind="consumed",
            adapter=PAIRED_INFER_ADAPTER,
            roots={"dataset": config.project.dataset_root},
            hash_policy=config.data.hash_policy,
            group_validation=config.data.group_validation,
            selection={"split": "test", "prediction_inputs": list(source_names)},
            context={
                "method": config.method.name,
                "direction": direction,
                "checkpoint_sha256": checkpoint_sha256,
                "generation_config_sha256": sha256_json(generation),
            },
            sources={**manifest_sources(config.project), "checkpoint_path": str(checkpoint_path)},
        )
        session.bind_inputs(snapshot)

        device = resolve_inference_device()
        logger.info("Inference device: %s", device)
        generator, checkpoint_path = load_inference_generator(
            config, session.paths, device, checkpoint_path
        )
        transform = build_inference_transform(config.project.image_size)
        logger.info("Loaded manifest: %s test samples", len(test_manifest))
        session.result(
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            output_dir=str(output_dir),
            test_sample_count=len(test_manifest.records),
            device=str(device),
            inferred_count=0,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        result = InferenceResult(output_dir=output_dir)
        if len(test_manifest) == 0:
            logger.warning(
                "No test pairs found in manifest: %s",
                DatasetLayout.from_project(config.project).manifest_path,
            )
        output_domain = config.model.inputs[0] if direction == "B_to_A" else config.model.target
        produced: list[AssetRow] = []
        for record in test_manifest.records:
            inputs = {
                name: transform(
                    Image.open(config.project.dataset_root / path).convert("RGB")
                ).unsqueeze(0)
                for name, path in _prediction_sources(record, source_names, direction).items()
            }
            output = predict_batch(generator, inputs, device)[0]
            out_path = generated_path_for_record(record, output_dir, direction)
            save_image(output, out_path)
            specimen, patient = groups.get(record.set_id, ("", ""))
            produced.append(
                AssetRow(
                    root="output",
                    locator=relative_locator(output_dir, out_path),
                    role="generated",
                    domain=output_domain,
                    split=record.split,
                    sample_id=record.sample_id,
                    set_id=record.set_id,
                    specimen_id=specimen,
                    patient_id=patient,
                )
            )
            result.generated_paths.append(out_path)
            result.num_samples += 1
            session.result(inferred_count=result.num_samples)
        outputs = session.record_outputs(
            build_snapshot(
                produced,
                kind="produced",
                adapter=INFER_OUTPUT_ADAPTER,
                roots={"output": output_dir},
                hash_policy=config.data.hash_policy,
                selection={"direction": direction, "output_domain": output_domain},
                context={"consumed_snapshot_id": snapshot.snapshot_id, **snapshot.context},
            )
        )
        session.result(produced_snapshot_id=outputs["snapshot_id"])
    logger.info("Inference complete: %s samples -> %s", result.num_samples, output_dir)
    return result
