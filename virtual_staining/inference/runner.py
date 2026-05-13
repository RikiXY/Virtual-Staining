from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torchvision import transforms

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.metadata import (
    append_run_event,
    ensure_run_metadata,
    save_stage_metadata,
)
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.inference.outputs import InferenceOutputWriter
from virtual_staining.inference.predictor import Predictor
from virtual_staining.inference.results import InferenceResult
from virtual_staining.models.factory import build_generator
from virtual_staining.training.checkpoints import (
    _check_generator_arch,
    _validate_checkpoint_metadata,
    resolve_best_checkpoint_path,
)
from virtual_staining.utils.dimensions import to_torchvision_hw

logger = logging.getLogger(__name__)


def _is_amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def _resolve_checkpoint(config: RunConfig, paths: RunPaths) -> Path:
    """Resolve the inference checkpoint path from RunConfig."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    if config.inference.checkpoint_path is not None:
        checkpoint_path = config.inference.checkpoint_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.root / checkpoint_path
        return checkpoint_path

    if config.inference.checkpoint_policy == "latest":
        candidates = sorted(paths.checkpoints_dir.glob("ep*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"checkpoint_policy='latest' but no checkpoints found in {paths.checkpoints_dir}"
            )
        return candidates[-1]

    if config.inference.checkpoint_policy == "best_val_loss":
        return resolve_best_checkpoint_path(
            paths.checkpoints_dir,
            policy=config.inference.checkpoint_policy,
        )

    raise ValueError(
        "inference.checkpoint_path or inference.checkpoint_policy must be set in the config."
    )


def _write_inference_stage_metadata(paths: RunPaths, payload: dict[str, object]) -> None:
    stage_path = save_stage_metadata("infer", payload, paths.metadata_dir)
    if stage_path is not None:
        logger.info("Inference metadata written -> %s", stage_path)


def run_inference(config: RunConfig, config_path: Path) -> InferenceResult:
    """Load a checkpoint, run the generator on the test split, and write outputs."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    checkpoint_path = _resolve_checkpoint(config, paths)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    stored_size = checkpoint.get("image_size")
    if stored_size is not None and tuple(stored_size) != tuple(config.project.image_size):
        raise ValueError(
            "Image size mismatch between checkpoint and inference config. "
            f"Checkpoint image_size={tuple(stored_size)}, "
            f"config image_size={tuple(config.project.image_size)}."
        )

    checkpoint_arch = _validate_checkpoint_metadata(checkpoint, checkpoint_path)

    generator = build_generator(config.model.generator).to(device)
    _check_generator_arch(checkpoint_arch, generator)
    generator.load_state_dict(checkpoint["generator_state_dict"])

    transform = transforms.Compose(
        [
            transforms.Resize(to_torchvision_hw(config.project.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    output_dir = config.inference.output_dir or paths.output_test_dir
    manifest = load_manifest_or_raise(config.project)
    manifest.validate(check_files_exist=True, require_splits={"test"})
    manifest_path = config.project.manifest_path
    manifest_hash = compute_manifest_hash(manifest_path)
    ensure_run_metadata(
        paths.run_metadata,
        run_name=config.project.run_name,
        entrypoint="vs-infer",
        config_hash=config_hash,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_hash,
        device=str(device),
    )
    started_at = datetime.now(UTC).isoformat()
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
    _write_inference_stage_metadata(
        paths,
        {
            "stage": "infer",
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "config_hash": config_hash,
            **infer_details,
        },
    )
    append_run_event(
        {
            "timestamp": started_at,
            "run_name": config.project.run_name,
            "stage": "infer",
            "event_type": "stage_started",
            "status": "running",
            "config_hash": config_hash,
            "details": infer_details,
        },
        paths.metadata_dir,
    )

    writer = InferenceOutputWriter(output_dir)
    result = InferenceResult(output_dir=output_dir)
    if len(dataset) == 0:
        logger.warning("No test pairs found in manifest: %s", config.project.manifest_path)
        completed_at = datetime.now(UTC).isoformat()
        _write_inference_stage_metadata(
            paths,
            {
                "stage": "infer",
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "config_hash": config_hash,
                **infer_details,
                "inferred_count": result.num_samples,
            },
        )
        append_run_event(
            {
                "timestamp": completed_at,
                "run_name": config.project.run_name,
                "stage": "infer",
                "event_type": "stage_completed",
                "status": "completed",
                "config_hash": config_hash,
                "details": {**infer_details, "inferred_count": result.num_samples},
            },
            paths.metadata_dir,
        )
        return result

    predictor = Predictor(generator, device, _is_amp_enabled(device))
    try:
        for idx in range(len(dataset)):
            source_tensor, _ = dataset[idx]
            source_tensor = cast(torch.Tensor, source_tensor)
            record = test_manifest.records[idx]
            batch = source_tensor.unsqueeze(0)
            output = predictor.predict_batch(batch)[0]
            out_path = writer.write(record.sample_id, record.input_path.suffix, output)
            result.generated_paths.append(out_path)
            result.num_samples += 1
    except Exception as exc:
        completed_at = datetime.now(UTC).isoformat()
        _write_inference_stage_metadata(
            paths,
            {
                "stage": "infer",
                "status": "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "config_hash": config_hash,
                **infer_details,
                "inferred_count": result.num_samples,
                "error": str(exc),
            },
        )
        append_run_event(
            {
                "timestamp": completed_at,
                "run_name": config.project.run_name,
                "stage": "infer",
                "event_type": "stage_failed",
                "status": "failed",
                "config_hash": config_hash,
                "details": {
                    **infer_details,
                    "inferred_count": result.num_samples,
                    "error": str(exc),
                },
            },
            paths.metadata_dir,
        )
        raise

    completed_at = datetime.now(UTC).isoformat()
    _write_inference_stage_metadata(
        paths,
        {
            "stage": "infer",
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "config_hash": config_hash,
            **infer_details,
            "inferred_count": result.num_samples,
        },
    )
    append_run_event(
        {
            "timestamp": completed_at,
            "run_name": config.project.run_name,
            "stage": "infer",
            "event_type": "stage_completed",
            "status": "completed",
            "config_hash": config_hash,
            "details": {**infer_details, "inferred_count": result.num_samples},
        },
        paths.metadata_dir,
    )
    logger.info("Inference complete: %s samples -> %s", result.num_samples, output_dir)
    return result
