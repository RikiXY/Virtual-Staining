from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys
from virtual_staining.data.preprocessing import ALLOWED_MASK_STRATEGIES
from virtual_staining.utils.dimensions import parse_wh_size

_PREPROCESSING_KEYS: frozenset[str] = frozenset(
    {
        "patch_size",
        # section-specific
        "source_name",
        "target_name",
        "grid_movement",
        "margin",
        "seed",
        "save_masks",
        "save_discarded_patches",
        "mask_strategy",
        "source_mask_strategy",
        "target_mask_strategy",
        "mask_scale",
        "lowres_mask_filtering",
        "tiled_io",
        "max_memory_gb",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "min_foreground_ratio",
        "max_white_ratio",
        "white_threshold",
        "max_largest_white_component_ratio",
    }
)


def _pair(value: object, default: tuple[int, int]) -> tuple[int, int]:
    """Parse a generic two-integer tuple (e.g. grid_movement) from a config value."""
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Expected a two-value sequence, got {value!r}")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"Expected exactly two values, got {items}")
    return int(items[0]), int(items[1])


def _optional_strategy(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    strategy = value.strip()
    if strategy not in ALLOWED_MASK_STRATEGIES:
        allowed = ", ".join(ALLOWED_MASK_STRATEGIES)
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return strategy


@dataclass(frozen=True)
class PreprocessingConfig:
    dataset_root: Path
    source_name: str
    target_name: str
    image_size: tuple[int, int] = (256, 256)  # (width, height)
    grid_movement: tuple[int, int] = (256, 256)  # (x_step, y_step)
    margin: int = 200
    seed: int | None = None
    save_masks: bool = False
    save_discarded_patches: bool = False
    mask_strategy: str = "connected_components"
    source_mask_strategy: str | None = None
    target_mask_strategy: str | None = None
    mask_scale: float = 1.0
    lowres_mask_filtering: bool = False
    tiled_io: bool = False
    max_memory_gb: float | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.05
    test_ratio: float = 0.15
    min_foreground_ratio: float = 0.25
    max_white_ratio: float = 0.7
    white_threshold: int = 250
    max_largest_white_component_ratio: float = 0.20

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ValueError("source_name must be a non-empty string")
        if not isinstance(self.target_name, str) or not self.target_name.strip():
            raise ValueError("target_name must be a non-empty string")
        if self.source_name == self.target_name:
            raise ValueError("target_name must differ from source_name")

        for field_name, value in (
            ("image_size", self.image_size),
            ("grid_movement", self.grid_movement),
        ):
            if len(value) != 2 or any(dimension <= 0 for dimension in value):
                raise ValueError(f"{field_name} must contain two positive integers")

        if self.margin < 0:
            raise ValueError("margin must be greater than or equal to 0")

        if not (0.0 < self.mask_scale <= 1.0):
            raise ValueError(
                f"PreprocessingConfig.mask_scale must be in (0.0, 1.0], got {self.mask_scale}"
            )
        if self.max_memory_gb is not None and self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be greater than 0 when provided")

        split_ratios = {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }
        for field_name, value in split_ratios.items():
            if value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if not isclose(sum(split_ratios.values()), 1.0):
            raise ValueError("split ratios must sum to 1")

        for field_name, value in (
            ("min_foreground_ratio", self.min_foreground_ratio),
            ("max_white_ratio", self.max_white_ratio),
            (
                "max_largest_white_component_ratio",
                self.max_largest_white_component_ratio,
            ),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1")

        if self.white_threshold < 0 or self.white_threshold > 255:
            raise ValueError("white_threshold must be between 0 and 255")

        _optional_strategy(self.mask_strategy, "mask_strategy")
        _optional_strategy(self.source_mask_strategy, "source_mask_strategy")
        _optional_strategy(self.target_mask_strategy, "target_mask_strategy")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        dataset_root: Path,
        default_image_size: tuple[int, int],
    ) -> PreprocessingConfig:
        reject_unknown_keys(data, _PREPROCESSING_KEYS, "preprocessing")
        max_memory_gb = data.get("max_memory_gb")
        return cls(
            dataset_root=dataset_root,
            source_name=data["source_name"],
            target_name=data["target_name"],
            image_size=parse_wh_size(data.get("patch_size"), default_image_size),
            grid_movement=_pair(data.get("grid_movement"), default_image_size),
            margin=int(data.get("margin", 200)),
            seed=data.get("seed"),
            save_masks=parse_bool_strict(data.get("save_masks", False), "preprocessing.save_masks"),
            save_discarded_patches=parse_bool_strict(
                data.get("save_discarded_patches", False),
                "preprocessing.save_discarded_patches",
            ),
            mask_strategy=_optional_strategy(
                data.get("mask_strategy", "connected_components"),
                "preprocessing.mask_strategy",
            )
            or "connected_components",
            source_mask_strategy=_optional_strategy(
                data.get("source_mask_strategy"), "preprocessing.source_mask_strategy"
            ),
            target_mask_strategy=_optional_strategy(
                data.get("target_mask_strategy"), "preprocessing.target_mask_strategy"
            ),
            mask_scale=float(data.get("mask_scale", 1.0)),
            lowres_mask_filtering=parse_bool_strict(
                data.get("lowres_mask_filtering", False),
                "preprocessing.lowres_mask_filtering",
            ),
            tiled_io=parse_bool_strict(data.get("tiled_io", False), "preprocessing.tiled_io"),
            max_memory_gb=None if max_memory_gb is None else float(max_memory_gb),
            train_ratio=float(data.get("train_ratio", 0.8)),
            val_ratio=float(data.get("val_ratio", 0.05)),
            test_ratio=float(data.get("test_ratio", 0.15)),
            min_foreground_ratio=float(data.get("min_foreground_ratio", 0.25)),
            max_white_ratio=float(data.get("max_white_ratio", 0.7)),
            white_threshold=int(data.get("white_threshold", 250)),
            max_largest_white_component_ratio=float(
                data.get("max_largest_white_component_ratio", 0.20)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "patch_size": list(self.image_size),
            "grid_movement": list(self.grid_movement),
            "margin": self.margin,
            "seed": self.seed,
            "save_masks": self.save_masks,
            "save_discarded_patches": self.save_discarded_patches,
            "mask_strategy": self.mask_strategy,
            "source_mask_strategy": self.source_mask_strategy,
            "target_mask_strategy": self.target_mask_strategy,
            "mask_scale": self.mask_scale,
            "lowres_mask_filtering": self.lowres_mask_filtering,
            "tiled_io": self.tiled_io,
            "max_memory_gb": self.max_memory_gb,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "min_foreground_ratio": self.min_foreground_ratio,
            "max_white_ratio": self.max_white_ratio,
            "white_threshold": self.white_threshold,
            "max_largest_white_component_ratio": self.max_largest_white_component_ratio,
        }
