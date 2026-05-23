from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.training.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    NORMALIZATION_CONTRACT,
    CheckpointManager,
    load_best_checkpoint_record,
    resolve_best_checkpoint_path,
)


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


def _make_manager(
    tmp_path: Path,
    image_size: tuple[int, int] = (256, 256),
    *,
    with_scheduler: bool = False,
) -> CheckpointManager:
    gen = _TinyGenerator()
    disc = _TinyDiscriminator()
    opt_g = optim.Adam(gen.parameters())
    opt_d = optim.Adam(disc.parameters())
    scheduler_g = (
        optim.lr_scheduler.StepLR(opt_g, step_size=1, gamma=0.5) if with_scheduler else None
    )
    scheduler_d = (
        optim.lr_scheduler.StepLR(opt_d, step_size=1, gamma=0.5) if with_scheduler else None
    )
    return CheckpointManager(
        checkpoints_dir=tmp_path / "checkpoints",
        generator=gen,
        discriminator=disc,
        opt_G=opt_g,
        opt_D=opt_d,
        scaler_G=GradScaler(enabled=False),
        scaler_D=GradScaler(enabled=False),
        image_size=image_size,
        device=torch.device("cpu"),
        scheduler_G=scheduler_g,
        scheduler_D=scheduler_d,
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


def test_checkpoint_manager_round_trips_scheduler_state(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, with_scheduler=True)
    assert isinstance(mgr.scheduler_G, optim.lr_scheduler.StepLR)
    assert isinstance(mgr.scheduler_D, optim.lr_scheduler.StepLR)
    mgr.opt_G.step()
    mgr.opt_D.step()
    mgr.scheduler_G.step()
    mgr.scheduler_D.step()
    path = mgr.save(epoch=0)

    mgr_2 = _make_manager(tmp_path, with_scheduler=True)
    assert isinstance(mgr_2.scheduler_G, optim.lr_scheduler.StepLR)
    assert isinstance(mgr_2.scheduler_D, optim.lr_scheduler.StepLR)
    mgr_2.load(path)

    assert mgr_2.scheduler_G.state_dict()["last_epoch"] == 1
    assert mgr_2.scheduler_D.state_dict()["last_epoch"] == 1


def test_checkpoint_stores_format_version(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=0)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_checkpoint_stores_output_activation(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=0)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["architecture"]["generator"]["output_activation"] == "tanh"


def test_checkpoint_stores_normalization_contract(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=0)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["normalization_contract"] == NORMALIZATION_CONTRACT


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


def test_resolve_best_checkpoint_path_raises_when_record_missing(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    with pytest.raises(FileNotFoundError, match="best.json"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy="best")


def test_resolve_best_checkpoint_path_raises_when_record_invalid(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.checkpoints_dir.mkdir(parents=True)
    (mgr.checkpoints_dir / "best.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy="best")


def test_update_selection_records_writes_per_metric_best_and_top_k(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_0 = mgr.save(epoch=0)
    checkpoint_1 = mgr.save(epoch=1)

    mgr.update_selection_records(
        metrics={"loss_G_val": 0.4, "val_ssim": 0.5, "val_mae": 0.3},
        modes={"loss_G_val": "min", "val_ssim": "max", "val_mae": "min"},
        top_k=2,
        epoch=0,
        checkpoint_path=checkpoint_0,
    )
    mgr.update_selection_records(
        metrics={"loss_G_val": 0.6, "val_ssim": 0.8, "val_mae": 0.2},
        modes={"loss_G_val": "min", "val_ssim": "max", "val_mae": "min"},
        top_k=2,
        epoch=1,
        checkpoint_path=checkpoint_1,
        config_hash="abc123",
        loss_config={"generator": [], "discriminator": []},
    )

    payload = json.loads(mgr.best_record_path.read_text(encoding="utf-8"))
    assert "default_metric" not in payload
    assert "default_checkpoint_path" not in payload
    assert payload["metrics"]["val_ssim"]["best"]["epoch"] == 1
    assert payload["metrics"]["val_ssim"]["records"][0]["metric_value"] == pytest.approx(0.8)
    assert payload["metrics"]["val_mae"]["best"]["epoch"] == 1
    assert payload["metrics"]["loss_G_val"]["best"]["epoch"] == 0
    assert payload["metrics"]["val_ssim"]["best"]["config_hash"] == "abc123"
    assert payload["metrics"]["val_ssim"]["best"]["loss_config"] == {
        "generator": [],
        "discriminator": [],
    }


def test_update_selection_records_replaces_duplicate_epoch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=0)

    mgr.update_selection_records(
        metrics={"loss_G_val": 0.5},
        modes={"loss_G_val": "min"},
        top_k=3,
        epoch=0,
        checkpoint_path=checkpoint_path,
    )
    mgr.update_selection_records(
        metrics={"loss_G_val": 0.1},
        modes={"loss_G_val": "min"},
        top_k=3,
        epoch=0,
        checkpoint_path=checkpoint_path,
    )

    payload = json.loads(mgr.best_record_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["loss_G_val"]["records"] == [
        {
            "epoch": 0,
            "checkpoint_path": "ep000.pth",
            "metric_value": 0.1,
            "rank": 1,
        }
    ]


def test_update_selection_records_rejects_missing_checkpoint(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        mgr.update_selection_records(
            metrics={"loss_G_val": 0.1},
            modes={"loss_G_val": "min"},
            top_k=1,
            epoch=0,
            checkpoint_path=mgr.checkpoints_dir / "ep000.pth",
        )


def test_resolve_best_checkpoint_path_reads_selection_by_metric_and_rank(
    tmp_path: Path,
) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_0 = mgr.save(epoch=0)
    checkpoint_1 = mgr.save(epoch=1)
    mgr.update_selection_records(
        metrics={"val_ssim": 0.5},
        modes={"val_ssim": "max"},
        top_k=2,
        epoch=0,
        checkpoint_path=checkpoint_0,
    )
    mgr.update_selection_records(
        metrics={"val_ssim": 0.8},
        modes={"val_ssim": "max"},
        top_k=2,
        epoch=1,
        checkpoint_path=checkpoint_1,
    )

    assert (
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy="best", metric="val_ssim")
        == checkpoint_1
    )
    assert (
        resolve_best_checkpoint_path(
            mgr.checkpoints_dir,
            policy="top_k",
            metric="val_ssim",
            rank=2,
        )
        == checkpoint_0
    )


def test_resolve_best_checkpoint_path_requires_metric(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=0)
    mgr.update_selection_records(
        metrics={"val_ssim": 0.5},
        modes={"val_ssim": "max"},
        top_k=1,
        epoch=0,
        checkpoint_path=checkpoint_path,
    )

    with pytest.raises(ValueError, match="checkpoint_metric"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy="best")


def test_load_best_checkpoint_record_returns_selection_metadata(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    mgr.update_selection_records(
        metrics={"loss_G_val": 0.1234},
        modes={"loss_G_val": "min"},
        top_k=1,
        epoch=3,
        checkpoint_path=checkpoint_path,
    )

    record = load_best_checkpoint_record(mgr.checkpoints_dir, policy="best", metric="loss_G_val")

    assert record.policy == "best"
    assert record.metric == "loss_G_val"
    assert record.mode == "min"
    assert record.epoch == 3
    assert record.checkpoint_path == checkpoint_path
    assert record.metric_value == pytest.approx(0.1234)


def test_resolve_best_checkpoint_path_raises_when_selection_checkpoint_missing(
    tmp_path: Path,
) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    mgr.update_selection_records(
        metrics={"loss_G_val": 0.1234},
        modes={"loss_G_val": "min"},
        top_k=1,
        epoch=3,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_path.unlink()

    with pytest.raises(FileNotFoundError, match="missing file"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy="best", metric="loss_G_val")


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


def test_checkpoint_manager_load_raises_on_version_mismatch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["format_version"] = 1
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="format version"):
        mgr.load(path)


def test_checkpoint_manager_load_raises_on_output_activation_mismatch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["architecture"]["generator"]["output_activation"] = "sigmoid"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="output_activation"):
        mgr.load(path)


def test_checkpoint_manager_load_raises_on_normalization_contract_mismatch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    path = mgr.save(epoch=0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["normalization_contract"] = {"input_range": "[0, 1]", "output_range": "[0, 1]"}
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="normalization_contract"):
        mgr.load(path)
