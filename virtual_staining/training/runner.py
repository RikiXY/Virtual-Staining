from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
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
    append_run_event,
    ensure_run_metadata,
    save_stage_metadata,
)
from virtual_staining.experiment.run_context import RunContext
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.models.factory import build_discriminator, build_generator
from virtual_staining.reporting.base import TrainingReporter
from virtual_staining.reporting.null import NullReporter
from virtual_staining.training.augmentation import build_training_paired_transform
from virtual_staining.training.checkpoints import CheckpointManager
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
    if config.losses is None:
        return False
    return any(term.requires_mask for term in config.losses.generator)


def run_training(
    config: RunConfig,
    config_path: Path,
    reporter: TrainingReporter | None = None,
) -> TrainingResult:
    """Build all training components, persist provenance, and execute training."""
    if config.training is None:
        raise ValueError("RunConfig.training must be present for run_training().")
    if config.losses is None:
        raise ValueError("RunConfig.losses must be present for run_training().")

    if reporter is None:
        reporter = NullReporter()

    seed = (
        config.training.seed if config.training.seed is not None else random.randint(0, 2**32 - 1)
    )
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
    manifest.validate(check_files_exist=True, require_splits={"train", "val"})
    train_manifest = manifest.filter_split("train")
    val_manifest = manifest.filter_split("val")
    manifest_path = config.project.manifest_path
    manifest_hash = compute_manifest_hash(manifest_path)

    run_metadata = RunMetadata.create(
        run_name=config.project.run_name,
        entrypoint="vs-train",
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

    started_at = datetime.now(UTC).isoformat()
    effective_train_sample_count = (
        len(train_manifest) * config.augmentation.effective_expansion_factor
    )
    train_details = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "seed": seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_sample_count": len(train_manifest),
        "effective_train_sample_count": effective_train_sample_count,
        "augmentation_enabled": config.augmentation.enabled,
        "augmentation_intensity": config.augmentation.intensity,
        "augmentation_expansion_factor": config.augmentation.effective_expansion_factor,
        "val_sample_count": len(val_manifest),
    }
    save_stage_metadata(
        "train",
        {
            "stage": "train",
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "config_hash": config_hash,
            **train_details,
        },
        paths.metadata_dir,
    )
    append_run_event(
        {
            "timestamp": started_at,
            "run_name": config.project.run_name,
            "stage": "train",
            "event_type": "stage_started",
            "status": "running",
            "config_hash": config_hash,
            "details": train_details,
        },
        paths.metadata_dir,
    )

    context = RunContext(
        name=config.project.run_name,
        paths=paths,
        seed=seed,
        device=str(device),
        config_hash=config_hash,
    )
    reporter.on_training_started(context)

    transform = transforms.Compose(
        [
            transforms.Resize(to_torchvision_hw(config.project.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                to_torchvision_hw(config.project.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
    )

    train_dir = config.project.split_dir("train")
    val_dir = config.project.split_dir("val")
    include_foreground_mask = _requires_foreground_masks(config)
    train_paired_transform = build_training_paired_transform(
        config.augmentation,
        image_size=config.project.image_size,
        seed=seed,
    )

    train_dataset = PairedManifestDataset(
        train_manifest,
        transform=None if train_paired_transform is not None else transform,
        mask_transform=None if train_paired_transform is not None else mask_transform,
        paired_transform=train_paired_transform,
        include_foreground_mask=include_foreground_mask,
        virtual_expansion_factor=config.augmentation.effective_expansion_factor,
    )
    val_dataset = PairedManifestDataset(
        val_manifest,
        transform=transform,
        mask_transform=mask_transform,
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
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=val_loader_generator,
    )

    generator = build_generator(config.model.generator).to(device)
    discriminator = build_discriminator(config.model.discriminator).to(device)

    trainer = Trainer(
        config=config.training,
        model_config=config.model,
        run_paths=paths,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        image_size=config.project.image_size,
        train_dir=train_dir,
        val_dir=val_dir,
        losses=config.losses,
    )

    start_epoch = 0
    if config.training.resume is not None:
        start_epoch = _resume_from_checkpoint(
            config.training.resume,
            trainer,
            paths,
            config.project.image_size,
            device,
        )

    try:
        result = trainer.train(seed=seed, start_epoch=start_epoch, reporter=reporter)
    except Exception as exc:
        completed_at = datetime.now(UTC).isoformat()
        save_stage_metadata(
            "train",
            {
                "stage": "train",
                "status": "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "config_hash": config_hash,
                **train_details,
                "error": str(exc),
            },
            paths.metadata_dir,
        )
        append_run_event(
            {
                "timestamp": completed_at,
                "run_name": config.project.run_name,
                "stage": "train",
                "event_type": "stage_failed",
                "status": "failed",
                "config_hash": config_hash,
                "details": {**train_details, "error": str(exc)},
            },
            paths.metadata_dir,
        )
        raise

    completed_at = datetime.now(UTC).isoformat()
    early_stopping_details = {
        "stopped_early": result.stopped_early,
        "stop_epoch": result.stop_epoch,
        "stop_reason": result.stop_reason,
        "early_stopping_monitor": result.early_stopping_monitor,
        "early_stopping_mode": result.early_stopping_mode,
        "early_stopping_best_epoch": result.early_stopping_best_epoch,
        "early_stopping_best_value": result.early_stopping_best_value,
    }
    save_stage_metadata(
        "train",
        {
            "stage": "train",
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "config_hash": config_hash,
            **train_details,
            "final_epoch": result.final_epoch,
            **early_stopping_details,
            "best_checkpoint_path": (
                str(result.best_checkpoint_path)
                if result.best_checkpoint_path is not None
                else None
            ),
        },
        paths.metadata_dir,
    )
    append_run_event(
        {
            "timestamp": completed_at,
            "run_name": config.project.run_name,
            "stage": "train",
            "event_type": "stage_completed",
            "status": "completed",
            "config_hash": config_hash,
            "details": {
                **train_details,
                "final_epoch": result.final_epoch,
                **early_stopping_details,
                "best_checkpoint_path": (
                    str(result.best_checkpoint_path)
                    if result.best_checkpoint_path is not None
                    else None
                ),
            },
        },
        paths.metadata_dir,
    )
    reporter.on_training_completed(result)
    return result


def _resume_from_checkpoint(
    resume: str,
    trainer: Trainer,
    paths: RunPaths,
    image_size: tuple[int, int],
    device: torch.device,
) -> int:
    """Resolve a resume target and return the next epoch to train."""
    ckpt_manager = CheckpointManager(
        checkpoints_dir=paths.checkpoints_dir,
        generator=trainer.generator,
        discriminator=trainer.discriminator,
        opt_G=trainer._opt_G,
        opt_D=trainer._opt_D,
        scaler_G=trainer._scaler_G,
        scaler_D=trainer._scaler_D,
        image_size=image_size,
        device=device,
        lr_g=trainer.config.lr_g,
        lr_d=trainer.config.lr_d,
        beta1=trainer.config.beta1,
        beta2=trainer.config.beta2,
        batch_size=trainer.config.batch_size,
        num_workers=trainer.config.num_workers,
    )

    if resume == "latest":
        checkpoint_path = ckpt_manager.latest()
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"resume='latest' but no checkpoints found in {paths.checkpoints_dir}"
            )
    else:
        checkpoint_path = _resolve_resume_checkpoint_path(resume, paths)

    return ckpt_manager.load(checkpoint_path)


def _resolve_resume_checkpoint_path(resume: str, paths: RunPaths) -> Path:
    """Resolve and preflight an explicit resume checkpoint path."""
    checkpoint_path = Path(resume)
    if not checkpoint_path.is_absolute():
        checkpoint_path = paths.checkpoints_dir / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    if checkpoint_path.suffix != ".pth":
        raise ValueError(f"resume checkpoint path must end with '.pth'; got {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    return checkpoint_path
