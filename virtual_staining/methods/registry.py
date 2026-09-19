from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from virtual_staining.config.run import RunConfig
from virtual_staining.methods.cyclegan import CycleGANMethod
from virtual_staining.methods.pix2pix import Pix2PixMethod


def resolve_training_method(config: RunConfig, device: torch.device) -> Any:
    if config.method.name == "pix2pix":
        return Pix2PixMethod(config, device)
    if config.method.name == "cyclegan":
        return CycleGANMethod(config, device)
    assert config.method.class_path is not None
    method_class = _load_method_class(config.method.class_path)
    runtime = method_class(config=config, device=device, **config.method.params)
    _validate_runtime(runtime, config.method.class_path)
    return runtime


def _load_method_class(class_path: str) -> type[Any]:
    module_name, separator, class_name = class_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("Custom method class_path must use 'package.module:ClassName' syntax")
    try:
        candidate = getattr(import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import custom method '{class_path}': {exc}") from exc
    if not isinstance(candidate, type):
        raise TypeError(f"Custom method '{class_path}' must resolve to a class")
    return candidate


def _validate_runtime(runtime: Any, class_path: str) -> None:
    required_attributes = ("name", "pairing", "loss_names", "optimizers")
    required_methods = (
        "train_mode",
        "step",
        "validate",
        "component_metadata",
        "state_dict",
        "load_state_dict",
        "load_legacy_v3",
    )
    missing = [name for name in required_attributes if not hasattr(runtime, name)]
    missing.extend(name for name in required_methods if not callable(getattr(runtime, name, None)))
    optimizers = getattr(runtime, "optimizers", ())
    if hasattr(runtime, "optimizers") and len(optimizers) != 2:
        raise TypeError(f"Custom method '{class_path}' must expose exactly two optimizer groups")
    if missing:
        raise TypeError(
            f"Custom method '{class_path}' does not satisfy the training method contract; "
            f"missing: {', '.join(missing)}"
        )
