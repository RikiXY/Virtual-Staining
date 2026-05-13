from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.training.checkpoints import CheckpointManager


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.base_channels = 64
        self.norm = "batch"
        self.dropout = False
        self.bilinear = False
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 6
        self.ndf = 64
        self.norm = "instance"
        self.use_sigmoid = False
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


def _make_manager(tmp_path: Path, image_size: tuple[int, int] = (256, 256)) -> CheckpointManager:
    gen = _TinyGenerator()
    disc = _TinyDiscriminator()
    return CheckpointManager(
        checkpoints_dir=tmp_path / "checkpoints",
        generator=gen,
        discriminator=disc,
        opt_G=optim.Adam(gen.parameters()),
        opt_D=optim.Adam(disc.parameters()),
        scaler_G=GradScaler(enabled=False),
        scaler_D=GradScaler(enabled=False),
        image_size=image_size,
        device=torch.device("cpu"),
    )


def test_checkpoint_manager_save_creates_file(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=5)
    assert path.exists()
    assert path.name == "ep005.pth"


def test_checkpoint_manager_load_returns_correct_epoch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save(epoch=5)
    start_epoch = mgr.load(mgr.checkpoints_dir / "ep005.pth")
    assert start_epoch == 6


def test_checkpoint_manager_latest_returns_last(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save(epoch=0)
    mgr.save(epoch=9)
    latest = mgr.latest()
    assert latest is not None
    assert latest.name == "ep009.pth"


def test_checkpoint_manager_latest_none_when_empty(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.latest() is None


def test_checkpoint_manager_image_size_mismatch_raises(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, image_size=(256, 256))
    mgr.save(epoch=0)
    mgr_2 = _make_manager(tmp_path, image_size=(128, 128))
    with pytest.raises(ValueError, match="Image size mismatch"):
        mgr_2.load(tmp_path / "checkpoints" / "ep000.pth")


def test_checkpoint_manager_arch_mismatch_raises(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save(epoch=0)

    gen_different = _TinyGenerator()
    gen_different.in_channels = 1  # differs from saved checkpoint's 3
    disc = _TinyDiscriminator()
    mgr_bad = CheckpointManager(
        checkpoints_dir=tmp_path / "checkpoints",
        generator=gen_different,
        discriminator=disc,
        opt_G=optim.Adam(gen_different.parameters()),
        opt_D=optim.Adam(disc.parameters()),
        scaler_G=GradScaler(enabled=False),
        scaler_D=GradScaler(enabled=False),
        image_size=(256, 256),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="in_channels"):
        mgr_bad.load(tmp_path / "checkpoints" / "ep000.pth")
