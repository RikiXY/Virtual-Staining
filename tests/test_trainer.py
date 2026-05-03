from __future__ import annotations

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
# Fixtures
# ---------------------------------------------------------------------------

def _write_rgb_pair(directory: Path, prefix: str = "00000_00000") -> None:
    """Write a minimal 32x32 RGB source/target pair to *directory*."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(directory / f"{prefix}_source.png")
    Image.fromarray(arr).save(directory / f"{prefix}_target.png")


@pytest.fixture()
def smoke_trainer(tmp_path):
    """Return a fully wired Trainer ready for a 1-epoch smoke run."""
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
        checkpoint_rate=1,
        log_rate=1,
    )

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    trainer = Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    return trainer, config


# ---------------------------------------------------------------------------
# Smoke test: 1 epoch end-to-end
# ---------------------------------------------------------------------------

def test_trainer_smoke_run_creates_expected_files(smoke_trainer):
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    run_root = config.run_root
    assert (run_root / "run_config.json").exists()
    assert any((run_root / "logs").glob("*.txt"))
    assert any((run_root / "checkpoints").glob("*.pth"))


def test_trainer_checkpoint_round_trip(smoke_trainer):
    trainer, config = smoke_trainer

    trainer.train(seed=42)

    checkpoint_path = next((config.run_root / "checkpoints").glob("*.pth"))

    device = torch.device("cpu")
    generator_2 = UNetGenerator().to(device)
    discriminator_2 = PatchGANDiscriminator().to(device)

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
        generator=generator_2,
        discriminator=discriminator_2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    start_epoch = trainer_2.load_checkpoint(checkpoint_path)
    assert start_epoch == 1
