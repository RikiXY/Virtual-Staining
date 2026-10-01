from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def assert_nested_equal(actual: Any, expected: Any, path: str = "state") -> None:
    """Assert two nested state containers hold identical tensors (device-agnostic)."""
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and set(actual) == set(expected), path
        for key, value in expected.items():
            assert_nested_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_nested_equal(left, right, f"{path}[{index}]")
    elif isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert torch.equal(actual.cpu(), expected.cpu()), path
    else:
        assert actual == expected, path


def write_ui_checkpoint(
    path: Path,
    *,
    input_names: tuple[str, ...] = ("label_free",),
    target_modality: str = "HE",
    image_size: tuple[int, int] = (32, 32),
    output_names: tuple[str, ...] | None = None,
) -> Path:
    """A current checkpoint with the same method identity and model state as training."""
    from virtual_staining.checkpoint_contract import build_checkpoint_payload
    from virtual_staining.config.run import RunConfig

    config = RunConfig.from_mapping(
        {
            "dataset_root": ".",
            "results_path": ".",
            "run_name": "test",
            "image_size": list(image_size),
            "model": {
                "inputs": list(input_names),
                "outputs": list(output_names or (target_modality,)),
                "generator": {"base_channels": 4},
                "discriminator": {"ndf": 4},
            },
        }
    )
    generator = config.method.options.generator.build(
        input_names=config.model.inputs,
        output_names=config.model.outputs,
    )
    payload = build_checkpoint_payload(
        config.method.definition.checkpoint_identity(config),
        epoch=0,
        state={"models": {"generator": generator.state_dict()}},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
