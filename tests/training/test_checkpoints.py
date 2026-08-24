from __future__ import annotations

from pathlib import Path

import pytest
import torch

from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.checkpoints import CheckpointManager


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
