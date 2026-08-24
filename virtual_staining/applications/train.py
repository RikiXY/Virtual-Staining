from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.metadata import (
    RunMetadata,
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
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.augmentation import build_training_paired_transform
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.trainer import Trainer
from virtual_staining.utils.dimensions import to_torchvision_hw

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _requires_foreground_masks(config: RunConfig) -> bool:
    if config.training is None:
        return False
    return any(term.requires_mask for term in config.training.losses.generator)


def train(
    config: RunConfig,
    config_path: Path,
) -> TrainingResult:
    """Build all training components, persist provenance, and execute training."""
    if config.training is None:
        raise ValueError("RunConfig.training must be present for train().")
    training = config.training

    seed = training.seed if training.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)

    run_root = config.project.results_path / config.project.run_name
    paths = RunPaths(run_root)
    paths.create_directories()
    snapshot_paths = resolve_run_snapshot_paths(stage="training", run_paths=paths)

    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=snapshot_paths.input_config,
        resolved_dest=snapshot_paths.resolved_config,
        hash_dest=snapshot_paths.config_hash,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    manifest = load_manifest_or_raise(config.project)
    if not set(config.model.inputs).issubset(manifest.metadata.input_modalities):
        raise ValueError("model.inputs must be a subset of manifest input modalities")
    if config.model.target != manifest.metadata.target_modality:
        raise ValueError("model.target must equal manifest target modality")
    train_manifest = manifest.filter_split("train")
    val_manifest = manifest.filter_split("val")
    manifest_path = config.project.manifest_path
    manifest_hash = compute_manifest_hash(manifest_path)

    run_metadata = RunMetadata.create(
        run_name=config.project.run_name,
        entrypoint="vs train",
        seed=seed,
        config_hash=config_hash,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_hash,
        device=str(device),
        cuda_device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    ensure_run_metadata(
        paths.run_metadata,
        run_name=run_metadata.run_name,
        entrypoint=run_metadata.entrypoint,
        config_hash=run_metadata.config_hash,
        manifest_path=run_metadata.manifest_path,
        manifest_sha256=run_metadata.manifest_sha256,
        seed=run_metadata.seed,
        device=run_metadata.device,
        cuda_device_name=run_metadata.cuda_device_name,
        git_commit=run_metadata.git_commit,
        git_dirty=run_metadata.git_dirty,
        package_version=run_metadata.package_version,
    )

    save_environment_snapshot(snapshot_paths.environment)

    effective_train_sample_count = (
        len(train_manifest) * training.augmentation.effective_expansion_factor
    )
    train_details = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "seed": seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_sample_count": len(train_manifest),
        "effective_train_sample_count": effective_train_sample_count,
        "augmentation_enabled": training.augmentation.enabled,
        "augmentation_intensity": training.augmentation.intensity,
        "augmentation_expansion_factor": training.augmentation.effective_expansion_factor,
        "val_sample_count": len(val_manifest),
    }

    transform = transforms.Compose(
        [
            transforms.Resize(to_torchvision_hw(config.project.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_dir = config.project.split_dir("train")
    val_dir = config.project.split_dir("val")
    include_foreground_mask = _requires_foreground_masks(config)
    train_paired_transform = build_training_paired_transform(
        training.augmentation,
        image_size=config.project.image_size,
        seed=seed,
        input_names=config.model.inputs,
        reference_modality=config.preprocessing.inputs.reference
        if config.preprocessing
        else config.model.inputs[0],
    )
    train_dataset = PairedManifestDataset(
        train_manifest,
        input_names=config.model.inputs,
        transform=None if train_paired_transform is not None else transform,
        paired_transform=train_paired_transform,
        include_foreground_mask=include_foreground_mask,
        virtual_expansion_factor=training.augmentation.effective_expansion_factor,
    )
    val_dataset = PairedManifestDataset(
        val_manifest,
        input_names=config.model.inputs,
        transform=transform,
        include_foreground_mask=include_foreground_mask,
    )
    logger.info(
        "Loaded manifest: %s train samples (%s effective), %s val samples",
        len(train_manifest),
        len(train_dataset),
        len(val_dataset),
    )

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(seed)
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=val_loader_generator,
    )

    generator_config = config.model.generator
    generator = ConcatUNetGenerator(
        config.model.inputs,
        base_channels=generator_config.base_channels,
        norm=generator_config.norm,
        dropout=generator_config.dropout,
        bilinear=generator_config.bilinear,
    ).to(device)
    discriminator_config = config.model.discriminator
    discriminator = PatchGANDiscriminator(
        in_channels=(3 * len(config.model.inputs)) + 3,
        ndf=discriminator_config.ndf,
        norm=discriminator_config.norm,
        use_sigmoid=discriminator_config.use_sigmoid,
    ).to(device)

    trainer = Trainer(
        config=training,
        run_paths=paths,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        image_size=config.project.image_size,
        train_dir=train_dir,
        val_dir=val_dir,
        losses=training.losses,
        target_modality=config.model.target,
    )

    start_epoch = 0
    if training.resume is not None:
        start_epoch = trainer.resume(training.resume)

    run = RunProvenance(paths.metadata_dir, config.project.run_name, config_hash)
    with run.stage("train", details=train_details) as stage:
        result = trainer.train(seed=seed, start_epoch=start_epoch)
        stage.result(
            final_epoch=result.final_epoch,
            stopped_early=result.stopped_early,
            stop_epoch=result.stop_epoch,
            stop_reason=result.stop_reason,
            early_stopping_monitor=result.early_stopping_monitor,
            early_stopping_mode=result.early_stopping_mode,
            early_stopping_best_epoch=result.early_stopping_best_epoch,
            early_stopping_best_value=result.early_stopping_best_value,
            best_checkpoint_path=(
                str(result.best_checkpoint_path)
                if result.best_checkpoint_path is not None
                else None
            ),
        )
    return result
