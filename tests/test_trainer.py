from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from virtual_staining.data.dataset import PairedHistologyDataset
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.training.config import TrainingConfig
from virtual_staining.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Internal helper: build a second Trainer that can load a checkpoint
# ---------------------------------------------------------------------------


def _make_resume_trainer(
    config: TrainingConfig,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
) -> Trainer:
    device = torch.device("cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_dir = config.dataset_root / "dataset_train"
    val_dir = config.dataset_root / "dataset_val"
    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform), batch_size=1, num_workers=0
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform), batch_size=1, num_workers=0
    )
    return Trainer(
        config=config,
        generator=generator.to(device),
        discriminator=discriminator.to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_rgb_pair(directory: Path, prefix: str = "00000_00000") -> None:
    """Write a minimal 32x32 RGB source/target pair to *directory*."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(directory / f"{prefix}_source.png")
    Image.fromarray(arr).save(directory / f"{prefix}_target.png")


def _make_trainer(tmp_path: Path, checkpoint_rate: int) -> tuple[Trainer, TrainingConfig]:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "dataset_train"
    val_dir = dataset_root / "dataset_val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    _write_rgb_pair(train_dir)
    _write_rgb_pair(val_dir)

    config = TrainingConfig(
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="smoke_run",
        image_size=(32, 32),
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        l1_weight=25.0,
        seed=42,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=checkpoint_rate,
        log_rate=1,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    return Trainer(
        config=config,
        generator=UNetGenerator().to(device),
        discriminator=PatchGANDiscriminator().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    ), config


@pytest.fixture()
def smoke_trainer(tmp_path: Path) -> tuple[Trainer, TrainingConfig]:
    """Trainer that never saves a checkpoint (checkpoint_rate=2 > epochs=1)."""
    return _make_trainer(tmp_path, checkpoint_rate=2)


@pytest.fixture()
def checkpointing_trainer(tmp_path: Path) -> tuple[Trainer, TrainingConfig]:
    """Trainer that saves a checkpoint every epoch, for round-trip tests."""
    return _make_trainer(tmp_path, checkpoint_rate=1)


# ---------------------------------------------------------------------------
# Smoke tests (no checkpoint I/O)
# ---------------------------------------------------------------------------


def test_trainer_smoke_run_creates_expected_files(
    smoke_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    run_root = config.run_root
    assert (run_root / "run_config.json").exists()
    assert (run_root / "config.yaml").exists()
    assert (run_root / "environment.json").exists()
    assert (run_root / "metrics.csv").exists()
    assert any((run_root / "logs").glob("*.txt"))


def test_trainer_metrics_csv_structure(
    smoke_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    metrics_path = config.run_root / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    assert len(rows) == config.epochs
    assert set(rows[0].keys()) == {
        "epoch",
        "loss_G_train",
        "loss_D_train",
        "loss_G_val",
        "loss_D_val",
    }
    # validate_rate=1, so val columns are present on every epoch
    for row in rows:
        assert row["loss_G_val"] != ""
        assert row["loss_D_val"] != ""
        assert float(row["loss_G_train"]) > 0
        assert float(row["loss_D_train"]) > 0


# ---------------------------------------------------------------------------
# Multi-batch averaging
# ---------------------------------------------------------------------------


def test_trainer_train_losses_are_epoch_averages(tmp_path: Path) -> None:
    """Train losses in metrics.csv must be averages over all batches (not last-batch)."""
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "dataset_train"
    val_dir = dataset_root / "dataset_val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Two training samples → two batches with batch_size=1.
    _write_rgb_pair(train_dir, prefix="00000_00000")
    _write_rgb_pair(train_dir, prefix="00001_00001")
    _write_rgb_pair(val_dir)

    config = TrainingConfig(
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="avg_run",
        image_size=(32, 32),
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        l1_weight=25.0,
        seed=0,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=2,
        log_rate=1,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    device = torch.device("cpu")
    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    trainer = Trainer(
        config=config,
        generator=UNetGenerator().to(device),
        discriminator=PatchGANDiscriminator().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    trainer.train(seed=0)

    metrics_path = config.run_root / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert float(rows[0]["loss_G_train"]) > 0
    assert float(rows[0]["loss_D_train"]) > 0


# ---------------------------------------------------------------------------
# Checkpoint round-trip (writes one real checkpoint)
# ---------------------------------------------------------------------------


def test_trainer_checkpoint_round_trip(
    checkpointing_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    trainer, config = checkpointing_trainer

    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))

    device = torch.device("cpu")
    train_dir = config.dataset_root / "dataset_train"
    val_dir = config.dataset_root / "dataset_val"
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=1,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=1,
        num_workers=0,
    )

    trainer_2 = Trainer(
        config=config,
        generator=UNetGenerator().to(device),
        discriminator=PatchGANDiscriminator().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    start_epoch = trainer_2.load_checkpoint(checkpoint_path)
    assert start_epoch == 1


# ---------------------------------------------------------------------------
# run_config.json metadata accuracy
# ---------------------------------------------------------------------------


def test_run_config_records_default_dirs(
    smoke_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    with (config.run_root / "run_config.json").open(encoding="utf-8") as f:
        run_config = json.load(f)

    assert run_config["train_dir"] == str(config.dataset_root / "dataset_train")
    assert run_config["val_dir"] == str(config.dataset_root / "dataset_val")


def test_run_config_records_custom_dirs(tmp_path: Path) -> None:
    custom_train_dir = tmp_path / "custom_train"
    custom_val_dir = tmp_path / "custom_val"
    custom_train_dir.mkdir(parents=True)
    custom_val_dir.mkdir(parents=True)

    _write_rgb_pair(custom_train_dir)
    _write_rgb_pair(custom_val_dir)

    config = TrainingConfig(
        dataset_root=tmp_path / "dataset",
        results_path=tmp_path / "results",
        run_name="custom_dirs_run",
        image_size=(32, 32),
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        l1_weight=25.0,
        seed=42,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=2,
        log_rate=1,
        train_dir=custom_train_dir,
        val_dir=custom_val_dir,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    device = torch.device("cpu")
    train_loader = DataLoader(
        PairedHistologyDataset(custom_train_dir, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(custom_val_dir, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    trainer = Trainer(
        config=config,
        generator=UNetGenerator().to(device),
        discriminator=PatchGANDiscriminator().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    trainer.train(seed=42)

    with (config.run_root / "run_config.json").open(encoding="utf-8") as f:
        run_config = json.load(f)

    assert run_config["train_dir"] == str(custom_train_dir)
    assert run_config["val_dir"] == str(custom_val_dir)


# ---------------------------------------------------------------------------
# Architecture metadata: presence and validation
# ---------------------------------------------------------------------------


def test_checkpoint_architecture_metadata_present(
    checkpointing_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    """Saved checkpoint must include an 'architecture' key with correct model params."""
    trainer, config = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "architecture" in ck
    gen = ck["architecture"]["generator"]
    assert gen["class"] == "UNetGenerator"
    assert gen["in_channels"] == 3
    assert gen["out_channels"] == 3
    assert gen["base_channels"] == 64
    assert gen["bilinear"] is False
    disc = ck["architecture"]["discriminator"]
    assert disc["class"] == "PatchGANDiscriminator"
    assert disc["in_channels"] == 6
    assert disc["ndf"] == 64
    assert disc["use_sigmoid"] is False


def test_load_checkpoint_validates_matching_architecture(
    checkpointing_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    """load_checkpoint must succeed when architecture matches the checkpoint."""
    trainer, config = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))
    trainer_2 = _make_resume_trainer(config, UNetGenerator(), PatchGANDiscriminator())
    start_epoch = trainer_2.load_checkpoint(checkpoint_path)
    assert start_epoch == 1


def test_load_checkpoint_raises_on_architecture_mismatch(
    checkpointing_trainer: tuple[Trainer, TrainingConfig],
) -> None:
    """load_checkpoint must raise ValueError when generator architecture params don't match."""
    trainer, config = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))
    trainer_mismatch = _make_resume_trainer(
        config, UNetGenerator(base_channels=32), PatchGANDiscriminator()
    )
    with pytest.raises(ValueError, match="base_channels"):
        trainer_mismatch.load_checkpoint(checkpoint_path)


def test_load_checkpoint_raises_on_missing_architecture(
    checkpointing_trainer: tuple[Trainer, TrainingConfig],
    tmp_path: Path,
) -> None:
    """load_checkpoint must raise ValueError for checkpoints without architecture metadata."""
    trainer, config = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ck.pop("architecture")
    legacy_path = tmp_path / "no_arch.pth"
    torch.save(ck, legacy_path)

    trainer_2 = _make_resume_trainer(config, UNetGenerator(), PatchGANDiscriminator())
    with pytest.raises(ValueError, match="architecture metadata"):
        trainer_2.load_checkpoint(legacy_path)
