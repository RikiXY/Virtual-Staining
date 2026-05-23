from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.training.checkpoints import (
    BEST_CHECKPOINT_POLICY,
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


def test_checkpoint_manager_saves_best_record(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)

    best_record_path = mgr.save_best_record(
        policy=BEST_CHECKPOINT_POLICY,
        metric="loss_G_val",
        mode="min",
        epoch=3,
        checkpoint_path=checkpoint_path,
        metric_value=0.1234,
        config_hash="abc123",
        loss_config={"generator": [], "discriminator": []},
    )

    payload = json.loads(best_record_path.read_text(encoding="utf-8"))
    assert payload == {
        "policy": BEST_CHECKPOINT_POLICY,
        "metric": "loss_G_val",
        "mode": "min",
        "epoch": 3,
        "checkpoint_path": "ep003.pth",
        "metric_value": 0.1234,
        "config_hash": "abc123",
        "loss_config": {"generator": [], "discriminator": []},
    }


def test_resolve_best_checkpoint_path_returns_selected_checkpoint(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    mgr.save_best_record(
        policy=BEST_CHECKPOINT_POLICY,
        metric="loss_G_val",
        mode="min",
        epoch=3,
        checkpoint_path=checkpoint_path,
        metric_value=0.1234,
    )

    resolved = resolve_best_checkpoint_path(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)
    assert resolved == checkpoint_path


def test_load_best_checkpoint_record_returns_selected_checkpoint_metadata(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    mgr.save_best_record(
        policy=BEST_CHECKPOINT_POLICY,
        metric="loss_G_val",
        mode="min",
        epoch=3,
        checkpoint_path=checkpoint_path,
        metric_value=0.1234,
    )

    record = load_best_checkpoint_record(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)

    assert record.policy == BEST_CHECKPOINT_POLICY
    assert record.metric == "loss_G_val"
    assert record.mode == "min"
    assert record.epoch == 3
    assert record.checkpoint_path == checkpoint_path
    assert record.metric_value == pytest.approx(0.1234)


def test_resolve_best_checkpoint_path_raises_when_record_missing(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    with pytest.raises(FileNotFoundError, match="best.json"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)


def test_resolve_best_checkpoint_path_raises_when_record_invalid(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.checkpoints_dir.mkdir(parents=True)
    (mgr.checkpoints_dir / "best.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)


def test_resolve_best_checkpoint_path_raises_on_policy_mismatch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    mgr.save_best_record(
        policy="best_val_ssim",
        metric="val_ssim",
        mode="max",
        epoch=3,
        checkpoint_path=checkpoint_path,
        metric_value=0.9,
    )

    with pytest.raises(ValueError, match="not requested policy"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)


def test_resolve_best_checkpoint_path_raises_when_checkpoint_file_missing(
    tmp_path: Path,
) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    mgr.save_best_record(
        policy=BEST_CHECKPOINT_POLICY,
        metric="loss_G_val",
        mode="min",
        epoch=3,
        checkpoint_path=checkpoint_path,
        metric_value=0.1234,
    )
    checkpoint_path.unlink()

    with pytest.raises(FileNotFoundError, match="missing file"):
        resolve_best_checkpoint_path(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)


def test_load_best_checkpoint_record_accepts_legacy_record_without_mode(
    tmp_path: Path,
) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=3)
    (mgr.checkpoints_dir / "best.json").write_text(
        json.dumps(
            {
                "policy": BEST_CHECKPOINT_POLICY,
                "metric": "loss_G_val",
                "epoch": 3,
                "checkpoint_path": checkpoint_path.name,
                "metric_value": 0.1234,
            }
        ),
        encoding="utf-8",
    )

    record = load_best_checkpoint_record(mgr.checkpoints_dir, policy=BEST_CHECKPOINT_POLICY)

    assert record.mode is None
    assert record.checkpoint_path == checkpoint_path


def test_update_top_k_records_sorts_max_mode_and_trims(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoints = [mgr.save(epoch=epoch) for epoch in range(3)]

    for epoch, checkpoint_path, metric_value in [
        (0, checkpoints[0], 0.5),
        (1, checkpoints[1], 0.9),
        (2, checkpoints[2], 0.7),
    ]:
        mgr.update_top_k_records(
            policy="best_val_ssim",
            metric="val_ssim",
            mode="max",
            top_k=2,
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            metric_value=metric_value,
            config_hash="abc123",
            loss_config={"generator": [], "discriminator": []},
        )

    payload = json.loads(mgr.top_k_record_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "best_val_ssim"
    assert payload["metric"] == "val_ssim"
    assert payload["mode"] == "max"
    assert payload["top_k"] == 2
    ranked_values = [
        (record["rank"], record["epoch"], record["metric_value"]) for record in payload["records"]
    ]
    assert ranked_values == [(1, 1, 0.9), (2, 2, 0.7)]
    assert payload["records"][0]["config_hash"] == "abc123"
    assert payload["records"][0]["loss_config"] == {"generator": [], "discriminator": []}


def test_update_top_k_records_sorts_min_mode(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoints = [mgr.save(epoch=epoch) for epoch in range(3)]

    for epoch, checkpoint_path, metric_value in [
        (0, checkpoints[0], 0.5),
        (1, checkpoints[1], 0.2),
        (2, checkpoints[2], 0.4),
    ]:
        mgr.update_top_k_records(
            policy=BEST_CHECKPOINT_POLICY,
            metric="loss_G_val",
            mode="min",
            top_k=3,
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            metric_value=metric_value,
        )

    payload = json.loads(mgr.top_k_record_path.read_text(encoding="utf-8"))
    ranked_values = [
        (record["rank"], record["epoch"], record["metric_value"]) for record in payload["records"]
    ]
    assert ranked_values == [(1, 1, 0.2), (2, 2, 0.4), (3, 0, 0.5)]


def test_update_top_k_records_replaces_duplicate_epoch(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    checkpoint_path = mgr.save(epoch=0)

    mgr.update_top_k_records(
        policy=BEST_CHECKPOINT_POLICY,
        metric="loss_G_val",
        mode="min",
        top_k=3,
        epoch=0,
        checkpoint_path=checkpoint_path,
        metric_value=0.5,
    )
    mgr.update_top_k_records(
        policy=BEST_CHECKPOINT_POLICY,
        metric="loss_G_val",
        mode="min",
        top_k=3,
        epoch=0,
        checkpoint_path=checkpoint_path,
        metric_value=0.1,
    )

    payload = json.loads(mgr.top_k_record_path.read_text(encoding="utf-8"))
    assert payload["records"] == [
        {
            "epoch": 0,
            "checkpoint_path": "ep000.pth",
            "metric_value": 0.1,
            "rank": 1,
        }
    ]


def test_update_top_k_records_rejects_missing_checkpoint(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        mgr.update_top_k_records(
            policy=BEST_CHECKPOINT_POLICY,
            metric="loss_G_val",
            mode="min",
            top_k=1,
            epoch=0,
            checkpoint_path=mgr.checkpoints_dir / "ep000.pth",
            metric_value=0.1,
        )


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
