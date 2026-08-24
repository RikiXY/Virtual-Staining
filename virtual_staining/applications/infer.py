from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
from torchvision.utils import save_image

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.metadata import RunProvenance, ensure_run_metadata
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.inference.outputs import generated_filename_for_sample
from virtual_staining.inference.runner import (
    InferenceResult,
    build_inference_transform,
    load_inference_generator,
    predict_batch,
    resolve_inference_device,
)

logger = logging.getLogger(__name__)


def infer(config: RunConfig, config_path: Path) -> InferenceResult:
    """Load a checkpoint, run the generator on the test split, and write outputs."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    device = resolve_inference_device()
    logger.info("Inference device: %s", device)

    paths = RunPaths(config.project.run_root)
    paths.create_directories()
    snapshot_paths = resolve_run_snapshot_paths(stage="inference", run_paths=paths)
    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=snapshot_paths.input_config,
        resolved_dest=snapshot_paths.resolved_config,
        hash_dest=snapshot_paths.config_hash,
    )
    save_environment_snapshot(snapshot_paths.environment)
    generator, checkpoint_path = load_inference_generator(config, paths, device)
    transform = build_inference_transform(config.project.image_size)

    output_dir = config.inference.output_dir or paths.output_test_dir
    manifest = load_manifest_or_raise(config.project)
    manifest.validate(check_files_exist=True, require_splits={"test"})
    manifest_path = config.project.manifest_path
    manifest_hash = compute_manifest_hash(manifest_path)
    ensure_run_metadata(
        paths.run_metadata,
        run_name=config.project.run_name,
        entrypoint="vs infer",
        config_hash=config_hash,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_hash,
        device=str(device),
    )
    test_manifest = manifest.filter_split("test")
    dataset = PairedManifestDataset(test_manifest, transform=transform)
    logger.info("Loaded manifest: %s test samples", len(dataset))
    infer_details: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "output_dir": str(output_dir),
        "test_sample_count": len(test_manifest.records),
        "device": str(device),
    }
    run = RunProvenance(paths.metadata_dir, config.project.run_name, config_hash)
    with run.stage("infer", details=infer_details) as stage:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = InferenceResult(output_dir=output_dir)
        stage.result(inferred_count=0)
        if len(dataset) == 0:
            logger.warning("No test pairs found in manifest: %s", config.project.manifest_path)
            return result

        for idx in range(len(dataset)):
            source_tensor, _ = dataset[idx]
            source_tensor = cast(torch.Tensor, source_tensor)
            record = test_manifest.records[idx]
            output = predict_batch(generator, source_tensor.unsqueeze(0), device)[0]
            out_path = output_dir / generated_filename_for_sample(
                record.sample_id, record.input_path.suffix
            )
            save_image(output, out_path)
            result.generated_paths.append(out_path)
            result.num_samples += 1
            stage.result(inferred_count=result.num_samples)
    logger.info("Inference complete: %s samples -> %s", result.num_samples, output_dir)
    return result
