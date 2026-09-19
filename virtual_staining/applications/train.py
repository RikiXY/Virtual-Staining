from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import (
    PairedManifestDataset,
    UnpairedImageDataset,
    resolve_domain_images,
)
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.methods.registry import resolve_training_method
from virtual_staining.models.io_contract import build_model_input_transform
from virtual_staining.training.augmentation import build_training_paired_transform
from virtual_staining.training.progress import ProgressReporter, ProgressUpdate, format_progress_log
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = ["ProgressReporter", "ProgressUpdate", "format_progress_log", "train"]


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
    *,
    progress_reporter: ProgressReporter | None = None,
) -> TrainingResult:
    """Build training components, persist provenance, and execute training."""
    if config.training is None:
        raise ValueError("RunConfig.training must be present for train().")
    training = config.training

    with ExperimentSession.open(config=config, config_path=config_path, stage="train") as session:
        seed = training.seed if training.seed is not None else random.randint(0, 2**32 - 1)
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)
        dataset_layout = DatasetLayout.from_project(config.project)
        transform = build_model_input_transform(config.project.image_size)
        train_manifest = val_manifest = None
        if config.data.pairing == "paired":
            manifest = load_manifest_or_raise(config.project)
            if not set(config.model.inputs).issubset(manifest.metadata.input_modalities):
                raise ValueError("model.inputs must be a subset of manifest input modalities")
            if config.model.target != manifest.metadata.target_modality:
                raise ValueError("model.target must equal manifest target modality")
            manifest.validate(check_files_exist=True, require_splits={"train", "val"})
            train_manifest = manifest.filter_split("train")
            val_manifest = manifest.filter_split("val")
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
                include_foreground_mask=_requires_foreground_masks(config),
                virtual_expansion_factor=training.augmentation.effective_expansion_factor,
            )
            val_dataset = PairedManifestDataset(
                val_manifest,
                input_names=config.model.inputs,
                transform=transform,
                include_foreground_mask=_requires_foreground_masks(config),
            )
        else:
            source_name = config.model.inputs[0]
            train_dataset = UnpairedImageDataset(
                resolve_domain_images(
                    config.project.dataset_root,
                    config.data.domains[source_name],
                    "train",
                ),
                resolve_domain_images(
                    config.project.dataset_root,
                    config.data.domains[config.model.target],
                    "train",
                ),
                transform,
            )
            val_dataset = UnpairedImageDataset(
                resolve_domain_images(
                    config.project.dataset_root,
                    config.data.domains[source_name],
                    "val",
                ),
                resolve_domain_images(
                    config.project.dataset_root,
                    config.data.domains[config.model.target],
                    "val",
                ),
                transform,
                random_pairing=False,
            )

        effective_train_sample_count = len(train_dataset)
        train_details = {
            "seed": seed,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "method": config.method.name,
            "pairing": config.data.pairing,
            "train_sample_count": len(train_dataset),
            "effective_train_sample_count": effective_train_sample_count,
            "augmentation_enabled": training.augmentation.enabled,
            "augmentation_intensity": training.augmentation.intensity,
            "augmentation_expansion_factor": training.augmentation.effective_expansion_factor,
            "val_sample_count": len(val_dataset),
        }
        session.result(**train_details)

        logger.info(
            "Loaded %s data: %s train samples, %s val samples",
            config.data.pairing,
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
            pin_memory=device.type == "cuda",
            generator=train_loader_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            pin_memory=device.type == "cuda",
            generator=val_loader_generator,
        )

        method = resolve_training_method(config, device)
        generator = getattr(method, "generator", getattr(method, "G_A_to_B", torch.nn.Identity()))
        discriminator = getattr(
            method, "discriminator", getattr(method, "D_A", torch.nn.Identity())
        )

        trainer = Trainer(
            config=training,
            run_paths=session.paths,
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            image_size=config.project.image_size,
            train_dir=dataset_layout.split_dir("train"),
            progress_reporter=progress_reporter,
            val_dir=dataset_layout.split_dir("val"),
            losses=training.losses,
            target_modality=config.model.target,
            experiment_session=session,
            config_hash=session.config_hash,
            method_runtime=method,
            resolved_config=config.to_dict(),
        )
        start_epoch = trainer.resume(training.resume) if training.resume is not None else 0
        result = trainer.train(seed=seed, start_epoch=start_epoch)
        session.result(
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
