from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms
from torchvision.utils import save_image

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.metadata import (
    RunProvenance,
    ensure_run_metadata,
)
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.inference.outputs import generated_filename_for_sample
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.training.checkpoints import (
    _check_generator_arch,
    _validate_checkpoint_metadata,
    resolve_best_checkpoint_path,
)
from virtual_staining.utils.dimensions import to_torchvision_hw

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    output_dir: Path
    generated_paths: list[Path] = field(default_factory=list)
    num_samples: int = 0


@torch.no_grad()
def _predict_batch(
    generator: nn.Module,
    source: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = generator(source.to(device))
    return (output * 0.5 + 0.5).clamp(0, 1)


def resolve_inference_device() -> torch.device:
    """Return the device used by inference entry points."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inference_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Build the image transform expected by the generator."""
    return transforms.Compose(
        [
            transforms.Resize(to_torchvision_hw(image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


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

    if config.inference.checkpoint_policy in {"best", "top_k"}:
        return resolve_best_checkpoint_path(
            paths.checkpoints_dir,
            policy=config.inference.checkpoint_policy,
            metric=config.inference.checkpoint_metric,
            rank=config.inference.checkpoint_rank or 1,
        )

    raise ValueError(
        "inference.checkpoint_path or inference.checkpoint_policy must be set in the config."
    )


def load_inference_generator(
    config: RunConfig,
    paths: RunPaths,
    device: torch.device,
) -> tuple[nn.Module, Path]:
    """Load and validate the configured generator checkpoint."""
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

    generator_config = config.model.generator
    generator = UNetGenerator(
        in_channels=generator_config.in_channels,
        out_channels=generator_config.out_channels,
        base_channels=generator_config.base_channels,
        norm=generator_config.norm,
        dropout=generator_config.dropout,
        bilinear=generator_config.bilinear,
    ).to(device)
    _check_generator_arch(checkpoint_arch, generator)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, checkpoint_path


def run_inference(config: RunConfig, config_path: Path) -> InferenceResult:
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
            batch = source_tensor.unsqueeze(0)
            output = _predict_batch(generator, batch, device)[0]
            out_path = output_dir / generated_filename_for_sample(
                record.sample_id, record.input_path.suffix
            )
            save_image(output, out_path)
            result.generated_paths.append(out_path)
            result.num_samples += 1
            stage.result(inferred_count=result.num_samples)
    logger.info("Inference complete: %s samples -> %s", result.num_samples, output_dir)
    return result
