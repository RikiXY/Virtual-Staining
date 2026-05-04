from __future__ import annotations

import csv
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

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=1, shuffle=False, num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=1, shuffle=False, num_workers=0,
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
def smoke_trainer(tmp_path):
    """Trainer that never saves a checkpoint (checkpoint_rate=2 > epochs=1)."""
    return _make_trainer(tmp_path, checkpoint_rate=2)


@pytest.fixture()
def checkpointing_trainer(tmp_path):
    """Trainer that saves a checkpoint every epoch, for round-trip tests."""
    return _make_trainer(tmp_path, checkpoint_rate=1)


# ---------------------------------------------------------------------------
# Smoke tests (no checkpoint I/O)
# ---------------------------------------------------------------------------

def test_trainer_smoke_run_creates_expected_files(smoke_trainer):
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    run_root = config.run_root
    assert (run_root / "run_config.json").exists()
    assert (run_root / "config.yaml").exists()
    assert (run_root / "environment.json").exists()
    assert (run_root / "metrics.csv").exists()
    assert any((run_root / "logs").glob("*.txt"))


def test_trainer_metrics_csv_structure(smoke_trainer):
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    metrics_path = config.run_root / "metrics.csv"
    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == config.epochs
    assert set(rows[0].keys()) == {"epoch", "loss_G_train", "loss_D_train", "loss_G_val", "loss_D_val"}
    # validate_rate=1, so val columns are present on every epoch
    for row in rows:
        assert row["loss_G_val"] != ""
        assert row["loss_D_val"] != ""
        assert float(row["loss_G_train"]) > 0
        assert float(row["loss_D_train"]) > 0


# ---------------------------------------------------------------------------
# Checkpoint round-trip (writes one real checkpoint)
# ---------------------------------------------------------------------------

def test_trainer_checkpoint_round_trip(checkpointing_trainer):
    trainer, config = checkpointing_trainer

    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))

    device = torch.device("cpu")
    train_dir = config.dataset_root / "dataset_train"
    val_dir = config.dataset_root / "dataset_val"
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=1, num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=1, num_workers=0,
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
