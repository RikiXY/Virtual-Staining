from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

import virtual_staining.inference.runner as inference_runner
from virtual_staining.inference.runner import load_checkpoint_generator
from virtual_staining.inference.single import predict_single_patch, validate_patch_image
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.checkpoints import CheckpointManager


def _write_checkpoint(
    root: Path,
    *,
    input_names: tuple[str, ...] = ("label_free",),
    target_modality: str = "H&E",
) -> tuple[Path, ConcatUNetGenerator]:
    generator = ConcatUNetGenerator(input_names, base_channels=4)
    discriminator = PatchGANDiscriminator(in_channels=3 * len(input_names) + 3, ndf=4)
    manager = CheckpointManager(
        root,
        generator,
        discriminator,
        torch.optim.Adam(generator.parameters(), lr=1e-3),
        torch.optim.Adam(discriminator.parameters(), lr=1e-3),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.amp.GradScaler("cpu", enabled=False),
        (32, 32),
        torch.device("cpu"),
        target_modality=target_modality,
    )
    return manager.save(0), generator


def test_checkpoint_only_loader_reconstructs_training_checkpoint(tmp_path: Path) -> None:
    checkpoint_path, saved_generator = _write_checkpoint(tmp_path)

    loaded = load_checkpoint_generator(checkpoint_path, torch.device("cpu"))

    assert loaded.image_size == (32, 32)
    assert loaded.input_names == ("label_free",)
    assert loaded.target_modality == "H&E"
    assert loaded.channels_per_input == 3
    assert loaded.generator.training is False
    assert torch.equal(
        next(saved_generator.parameters()),
        next(loaded.generator.parameters()),
    )


def test_checkpoint_only_loader_uses_requested_runtime_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    runtime_device = torch.device("cpu")
    original_load = torch.load
    recorded_map_locations: list[object] = []

    def record_map_location(*args: Any, **kwargs: Any) -> Any:
        recorded_map_locations.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(inference_runner.torch, "load", record_map_location)

    loaded = load_checkpoint_generator(checkpoint_path, runtime_device)

    assert recorded_map_locations == [runtime_device]
    assert loaded.device == runtime_device
    assert {parameter.device for parameter in loaded.generator.parameters()} == {runtime_device}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda checkpoint: checkpoint.pop("image_size"), "image_size"),
        (lambda checkpoint: checkpoint.update(format_version=2), "format version"),
        (
            lambda checkpoint: checkpoint["architecture"]["generator"].update(in_channels=6),
            "weights do not match",
        ),
    ],
)
def test_checkpoint_only_loader_rejects_incompatible_metadata(
    tmp_path: Path, mutation, message: str
) -> None:
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mutation(checkpoint)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint_generator(checkpoint_path, torch.device("cpu"))


def test_patch_validation_rejects_incompatible_dimensions() -> None:
    image = Image.new("RGB", (64, 32))

    with pytest.raises(
        ValueError,
        match=r"Expected input size: 32 × 32 px.*Received: 64 × 32 px.*future version",
    ):
        validate_patch_image(image, (32, 32))


def test_valid_patch_passes_through_checkpoint_only_inference(tmp_path: Path) -> None:
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    runtime = load_checkpoint_generator(checkpoint_path, torch.device("cpu"))
    image = Image.new("RGB", runtime.image_size, color=(40, 80, 120))

    generated = predict_single_patch(runtime, image)

    assert generated.mode == "RGB"
    assert generated.size == runtime.image_size
