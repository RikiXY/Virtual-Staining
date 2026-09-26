from __future__ import annotations

from collections.abc import Mapping
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
