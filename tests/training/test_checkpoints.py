from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
import torch

from virtual_staining.checkpoint_selection import (
    RANKED_CHECKPOINT_POLICIES,
    SUPPORTED_CHECKPOINT_POLICIES,
    CheckpointMode,
    resolve_checkpoint_path,
    update_checkpoint_selection,
)
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.checkpoints import CheckpointManager


@pytest.mark.parametrize("policy", sorted(SUPPORTED_CHECKPOINT_POLICIES))
@pytest.mark.parametrize("mode", get_args(CheckpointMode))
def test_checkpoint_policy_and_mode_contracts(tmp_path: Path, policy: str, mode: str) -> None:
    config = InferenceConfig.from_mapping(
        {"checkpoint_policy": policy, "checkpoint_metric": "val_ssim"}
    )
    first, latest = tmp_path / "ep001.pth", tmp_path / "ep002.pth"
    for epoch, path, value in ((1, first, 0.1), (2, latest, 0.9)):
        path.touch()
        update_checkpoint_selection(
            tmp_path,
            metrics={"val_ssim": value},
            modes={"val_ssim": mode},
            top_k=2,
            epoch=epoch,
            checkpoint_path=path,
        )
    expected = latest if policy == "latest" or mode == "max" else first
    assert config.checkpoint_policy == policy
    assert resolve_checkpoint_path(tmp_path, policy=policy, metric="val_ssim") == expected


@pytest.mark.parametrize("policy", sorted(RANKED_CHECKPOINT_POLICIES))
def test_ranked_checkpoint_policies_require_metric_and_accept_rank(policy: str) -> None:
    with pytest.raises(ValueError, match="checkpoint_metric is required"):
        InferenceConfig.from_mapping({"checkpoint_policy": policy})
    config = InferenceConfig.from_mapping(
        {"checkpoint_policy": policy, "checkpoint_metric": "val_ssim", "checkpoint_rank": 2}
    )
    assert config.checkpoint_rank == 2


def test_checkpoint_policy_validation_remains_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown checkpoint_policy"):
        InferenceConfig.from_mapping({"checkpoint_policy": "unknown"})
    with pytest.raises(ValueError, match="Unsupported checkpoint policy"):
        resolve_checkpoint_path(tmp_path, policy="unknown")
    with pytest.raises(ValueError, match="checkpoint_rank is supported only"):
        InferenceConfig.from_mapping({"checkpoint_policy": "latest", "checkpoint_rank": 1})


def test_checkpoint_selection_rejects_unknown_mode_without_writing(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ep001.pth"
    checkpoint.touch()
    with pytest.raises(ValueError, match="mode must be one of"):
        update_checkpoint_selection(
            tmp_path,
            metrics={"val_ssim": 0.5},
            modes={"val_ssim": "unknown"},
            top_k=2,
            epoch=1,
            checkpoint_path=checkpoint,
        )
    assert not (tmp_path / "best.json").exists()


def _manager(root: Path, names=("LF", "AF"), target="stained") -> CheckpointManager:
    generator = ConcatUNetGenerator(names, base_channels=4)
    discriminator = PatchGANDiscriminator(in_channels=3 * len(names) + 3, ndf=4)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    return CheckpointManager(
        root,
        generator,
        discriminator,
        opt_g,
        opt_d,
        torch.amp.GradScaler("cpu", enabled=False),
        torch.amp.GradScaler("cpu", enabled=False),
        (16, 16),
        torch.device("cpu"),
        target_modality=target,
    )


def test_checkpoint_round_trip_contains_named_v3_metadata(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    path = manager.save(2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 3
    assert checkpoint["architecture"]["generator"]["input_names"] == ["LF", "AF"]
    assert checkpoint["architecture"]["generator"]["target_modality"] == "stained"
    assert manager.load(path) == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda checkpoint: checkpoint["architecture"]["generator"].update(input_names=["AF", "LF"]),
        lambda checkpoint: checkpoint["architecture"]["generator"].update(target_modality="other"),
        lambda checkpoint: checkpoint.update(format_version=2),
        lambda checkpoint: checkpoint.pop("architecture"),
    ],
)
def test_checkpoint_identity_mismatches_are_rejected(tmp_path: Path, mutate) -> None:
    manager = _manager(tmp_path)
    path = manager.save(0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mutate(checkpoint)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        manager.load(path)
