from __future__ import annotations

import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from virtual_staining.config.run import RunConfig
from virtual_staining.data.consumption import (
    DataSnapshot,
    build_snapshot,
)
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    load_manifest_or_raise,
    load_set_groups,
    manifest_sources,
    paired_record_rows,
)
from virtual_staining.data.unpaired import UnpairedImageDataset, resolve_domain_collections
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.methods.registry import resolve_training_method
from virtual_staining.models.io_contract import build_model_input_transform
from virtual_staining.split_contract import TRAIN_SPLIT, VAL_SPLIT, DatasetSplit
from virtual_staining.training.augmentation import build_training_paired_transform
from virtual_staining.training.preview import ValidationPreviewWriter
from virtual_staining.training.progress import ProgressReporter, ProgressUpdate, format_progress_log
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.trainer import Trainer

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder

logger = logging.getLogger(__name__)

__all__ = ["ProgressReporter", "ProgressUpdate", "format_progress_log", "train"]


def _set_seed(seed: int) -> None:
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


PAIRED_TRAIN_ADAPTER = "paired_manifest_train/1"
UNPAIRED_TRAIN_ADAPTER = "unpaired_domains_train/1"


def _paired_datasets(
    config: RunConfig,
    transform: Callable[[Any], Any],
    seed: int,
) -> tuple[PairedManifestDataset, PairedManifestDataset, dict[str, object], DataSnapshot]:
    assert config.training is not None
    training = config.training
    manifest = load_manifest_or_raise(config.project)
    if not set(config.model.inputs).issubset(manifest.metadata.input_modalities):
        raise ValueError("model.inputs must be a subset of manifest input modalities")
    if config.model.target != manifest.metadata.target_modality:
        raise ValueError("model.target must equal manifest target modality")
    manifest.validate(check_files_exist=True, require_splits={"train", "val"})
    train_manifest = manifest.filter_split("train")
    val_manifest = manifest.filter_split("val")
    include_mask = _requires_foreground_masks(config)

    # The snapshot and the datasets are built from these same filtered manifest objects.
    groups = load_set_groups(config.project)
    rows = paired_record_rows(
        (*train_manifest.records, *val_manifest.records),
        input_names=config.model.inputs,
        target=config.model.target,
        include_mask=include_mask,
        groups=groups,
    )
    # Held-out test records are not consumed but share the split partition for leakage.
    test_context = paired_record_rows(
        manifest.filter_split("test").records,
        input_names=(),
        target=config.model.target,
        include_mask=False,
        groups=groups,
    )
    snapshot = build_snapshot(
        rows,
        kind="consumed",
        adapter=PAIRED_TRAIN_ADAPTER,
        roots={"dataset": config.project.dataset_root},
        hash_policy=config.data.hash_policy,
        group_validation=config.data.group_validation,
        group_context=test_context,
        selection={
            "pairing": "paired",
            "splits": ["train", "val"],
            "inputs": list(config.model.inputs),
            "target": config.model.target,
            "foreground_mask": include_mask,
        },
        sources=manifest_sources(config.project),
    )

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
        include_foreground_mask=include_mask,
        virtual_expansion_factor=training.augmentation.effective_expansion_factor,
    )
    val_dataset = PairedManifestDataset(
        val_manifest,
        input_names=config.model.inputs,
        transform=transform,
        include_foreground_mask=include_mask,
    )
    logger.info(
        "Loaded manifest: %s train samples (%s effective), %s val samples",
        len(train_manifest),
        len(train_dataset),
        len(val_dataset),
    )
    details: dict[str, object] = {
        "train_sample_count": len(train_manifest),
        "effective_train_sample_count": (
            len(train_manifest) * training.augmentation.effective_expansion_factor
        ),
    }
    return train_dataset, val_dataset, details, snapshot


def _unpaired_datasets(
    config: RunConfig,
    transform: Callable[[Any], Any],
    seed: int,
) -> tuple[UnpairedImageDataset, UnpairedImageDataset, DataSnapshot]:
    domain_a, domain_b = config.model.inputs[0], config.model.target
    splits: tuple[tuple[DatasetSplit, int | None], ...] = ((TRAIN_SPLIT, seed), (VAL_SPLIT, None))
    paths, rows, groups = resolve_domain_collections(
        config.data.domains,
        config.project.dataset_root,
        splits=[split for split, _ in splits],
        roles={domain_a: "input", domain_b: "target"},
        group_metadata=config.data.group_metadata,
    )
    # Domain membership only: epoch pairings are a seeded sampling operation, not
    # correspondence, so no A/B pair is ever recorded.
    snapshot = build_snapshot(
        rows,
        kind="consumed",
        adapter=UNPAIRED_TRAIN_ADAPTER,
        roots={"dataset": config.project.dataset_root},
        hash_policy=config.data.hash_policy,
        group_validation=config.data.group_validation,
        group_context=groups,
        selection={
            "pairing": "unpaired",
            "splits": [split for split, _ in splits],
            "domains": {"A": domain_a, "B": domain_b},
            "domain_specs": {name: config.data.domains[name] for name in (domain_a, domain_b)},
            "group_metadata": str(config.data.group_metadata)
            if config.data.group_metadata
            else None,
        },
    )
    datasets = []
    for split, pairing_seed in splits:
        paths_a, paths_b = paths[split, domain_a], paths[split, domain_b]
        logger.info(
            "Unpaired %s split: %s %s images, %s %s images",
            split,
            len(paths_a),
            domain_a,
            len(paths_b),
            domain_b,
        )
        datasets.append(
            UnpairedImageDataset(paths_a, paths_b, transform=transform, pairing_seed=pairing_seed)
        )
    return datasets[0], datasets[1], snapshot


def train(
    config: RunConfig,
    config_path: Path,
    *,
    progress_reporter: ProgressReporter | None = None,
    benchmark_recorder: TrainingBenchmarkRecorder | None = None,
) -> TrainingResult:
    if config.training is None:
        raise ValueError("RunConfig.training must be present for train().")
    training = config.training

    with ExperimentSession.open(config=config, config_path=config_path, stage="train") as session:
        seed = training.seed if training.seed is not None else random.randint(0, 2**32 - 1)
        _set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)
        dataset_layout = DatasetLayout.from_project(config.project)
        transform = build_model_input_transform(config.project.image_size)
        if config.data.pairing == "unpaired":
            train_dataset, val_dataset, snapshot = _unpaired_datasets(config, transform, seed)
            dataset_details: dict[str, object] = {
                "train_sample_count": len(train_dataset),
                "effective_train_sample_count": len(train_dataset),
                "pairing_policy": "independent_domains_seeded_draw",
            }
        else:
            train_dataset, val_dataset, dataset_details, snapshot = _paired_datasets(
                config, transform, seed
            )
        session.bind_inputs(snapshot)
        train_details = {
            "seed": seed,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            **dataset_details,
            "augmentation_enabled": training.augmentation.enabled,
            "augmentation_intensity": training.augmentation.intensity,
            "augmentation_expansion_factor": training.augmentation.effective_expansion_factor,
            "val_sample_count": len(val_dataset),
        }
        session.result(**train_details)
        if benchmark_recorder is not None:
            benchmark_recorder.set_workload(**train_details, batch_size=training.batch_size)

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

        method = resolve_training_method(
            config,
            device,
            seed=seed,
            benchmark_recorder=benchmark_recorder,
        )
        trainer = Trainer(
            config=training,
            run_paths=session.paths,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            train_dir=dataset_layout.split_dir("train"),
            progress_reporter=progress_reporter,
            val_dir=dataset_layout.split_dir("val"),
            experiment_session=session,
            config_hash=session.config_hash or "",
            image_size=config.project.image_size,
            benchmark_recorder=benchmark_recorder,
            preview_sink=ValidationPreviewWriter(
                session.paths.output_val_dir, benchmark_recorder=benchmark_recorder
            ),
        )
        if benchmark_recorder is not None:
            benchmark_recorder.start_run()
        try:
            start_epoch = trainer.resume(training.resume) if training.resume is not None else 0
            result = trainer.train(seed=seed, start_epoch=start_epoch)
        finally:
            if benchmark_recorder is not None:
                benchmark_recorder.finish_run()
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
