from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
from torchvision import transforms

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedHistologyDataset, PairedManifestDataset
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.inference.outputs import InferenceOutputWriter
from virtual_staining.inference.predictor import Predictor
from virtual_staining.inference.results import InferenceResult
from virtual_staining.models.factory import build_generator
from virtual_staining.training.checkpoints import _check_generator_arch
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

    raise ValueError(
        "inference.checkpoint_path or inference.checkpoint_policy must be set in the config."
    )


def run_inference(config: RunConfig, config_path: Path) -> InferenceResult:
    """Load a checkpoint, run the generator on the test split, and write outputs."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    paths = RunPaths(config.project.run_root)
    paths.create_directories()
    snapshot_paths = resolve_run_snapshot_paths(stage="inference", run_paths=paths)
    save_stage_config_snapshots(
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

    checkpoint_arch = checkpoint.get("architecture")
    if checkpoint_arch is None:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has no architecture metadata. "
            "Only checkpoints saved with the current version are supported."
        )
    if not isinstance(checkpoint_arch, dict):
        raise ValueError("Checkpoint architecture metadata must be a mapping.")

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
    test_dir = config.inference.test_dir or config.project.dataset_test_dir

    use_manifest = False
    test_manifest = None
    legacy_dataset: PairedHistologyDataset | None = None
    try:
        manifest = load_manifest_or_raise(config.project)
        test_manifest = manifest.filter_split("test")
        dataset = PairedManifestDataset(test_manifest, transform=transform)
        use_manifest = True
        logger.info("Loaded manifest: %s test samples", len(dataset))
    except OSError as exc:
        if "Manifest not found at " not in str(exc):
            raise
        logger.warning(
            "Manifest not found at %s; falling back to directory scanning.",
            config.project.manifest_path,
        )
        legacy_dataset = PairedHistologyDataset(test_dir, transform=transform)
        dataset = legacy_dataset

    writer = InferenceOutputWriter(output_dir)
    result = InferenceResult(output_dir=output_dir)
    if len(dataset) == 0:
        if use_manifest:
            logger.warning("No test pairs found in manifest: %s", config.project.manifest_path)
        else:
            logger.warning("No test pairs found in: %s", test_dir)
        return result

    predictor = Predictor(generator, device, _is_amp_enabled(device))

    for idx in range(len(dataset)):
        source_tensor, _ = dataset[idx]
        source_tensor = cast(torch.Tensor, source_tensor)
        if use_manifest:
            assert test_manifest is not None
            record = test_manifest.records[idx]
            source_path = config.project.dataset_root / record.input_path
        else:
            assert legacy_dataset is not None
            source_path = legacy_dataset.pairs[idx][0]
        batch = source_tensor.unsqueeze(0)
        output = predictor.predict_batch(batch)[0]
        out_path = writer.write(source_path.stem, source_path.suffix, output)
        result.generated_paths.append(out_path)
        result.num_samples += 1

    logger.info("Inference complete: %s samples -> %s", result.num_samples, output_dir)
    return result
