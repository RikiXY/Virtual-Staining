from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedHistologyDataset, PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.experiment.metadata import RunMetadata
from virtual_staining.experiment.run_context import RunContext
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.models.factory import build_discriminator, build_generator
from virtual_staining.reporting.base import TrainingReporter
from virtual_staining.reporting.null import NullReporter
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


def run_training(
    config: RunConfig,
    config_path: Path,
    reporter: TrainingReporter | None = None,
) -> TrainingResult:
    """Build all training components, persist provenance, and execute training."""
    if config.training is None:
        raise ValueError("RunConfig.training must be present for run_training().")

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

    metadata = RunMetadata.create(
        run_name=config.project.run_name,
        entrypoint="vs-train",
        seed=seed,
        config_hash=config_hash,
        device=str(device),
        cuda_device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    metadata.save(paths.run_metadata)

    save_environment_snapshot(snapshot_paths.environment)

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

    manifest_path = config.project.manifest_path
    train_dir = config.training.train_dir or config.project.dataset_train_dir
    val_dir = config.training.val_dir or config.project.dataset_val_dir

    if manifest_path.exists():
        manifest = DatasetManifest.from_csv(
            manifest_path,
            dataset_root=config.project.dataset_root,
        )
        train_dataset = PairedManifestDataset(
            manifest.filter_split("train"),
            transform=transform,
        )
        val_dataset = PairedManifestDataset(
            manifest.filter_split("val"),
            transform=transform,
        )
        logger.info(
            "Loaded manifest: %s train, %s val samples",
            len(train_dataset),
            len(val_dataset),
        )
    else:
        logger.warning(
            "Manifest not found at %s; falling back to directory scanning.",
            manifest_path,
        )
        train_dataset = PairedHistologyDataset(train_dir, transform=transform)
        val_dataset = PairedHistologyDataset(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    generator = build_generator(config.model.generator).to(device)
    discriminator = build_discriminator(config.model.discriminator).to(device)

    trainer = Trainer(
        config=config.training,
        run_paths=paths,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        image_size=config.project.image_size,
        train_dir=train_dir,
        val_dir=val_dir,
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
    except Exception:
        metadata.mark_failed()
        metadata.save(paths.run_metadata)
        raise

    metadata.mark_completed()
    metadata.save(paths.run_metadata)
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
        l1_weight=trainer.config.l1_weight,
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
        checkpoint_path = Path(resume)
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.checkpoints_dir / checkpoint_path

    return ckpt_manager.load(checkpoint_path)
