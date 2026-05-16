from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose
from pathlib import Path

from virtual_staining.config import (
    load_yaml_mapping,
    parse_bool_strict,
    reject_unknown_keys,
    section_with_shared_fields,
)
from virtual_staining.config.validation import _TOP_LEVEL_KEYS
from virtual_staining.data.preprocessing import ALLOWED_MASK_STRATEGIES
from virtual_staining.utils.dimensions import (
    parse_wh_size,
    parse_wh_size_from_aliases,
)

_PREPROCESSING_KEYS: frozenset[str] = frozenset(
    {
        # shared fields and size aliases
        "dataset_root",
        "image_size",
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
    def from_args(cls, args: argparse.Namespace) -> PreprocessingConfig:
        config = cls(
            dataset_root=Path(args.path),
            source_name=args.source_name,
            target_name=args.target_name,
            image_size=parse_wh_size(getattr(args, "image_size", (256, 256)), (256, 256)),
            grid_movement=_pair(getattr(args, "grid_movement", (256, 256)), (256, 256)),
            margin=getattr(args, "margin", 200),
            seed=getattr(args, "seed", None),
            save_masks=getattr(args, "save_masks", False),
            save_discarded_patches=getattr(args, "save_discarded_patches", False),
            mask_strategy=getattr(args, "mask_strategy", "connected_components"),
            source_mask_strategy=getattr(args, "source_mask_strategy", None),
            target_mask_strategy=getattr(args, "target_mask_strategy", None),
            mask_scale=getattr(args, "mask_scale", 1.0),
            lowres_mask_filtering=getattr(args, "lowres_mask_filtering", False),
            tiled_io=getattr(args, "tiled_io", False),
            max_memory_gb=getattr(args, "max_memory_gb", None),
            train_ratio=getattr(args, "train_ratio", 0.8),
            val_ratio=getattr(args, "val_ratio", 0.05),
            test_ratio=getattr(args, "test_ratio", 0.15),
            min_foreground_ratio=getattr(args, "min_foreground_ratio", 0.25),
            max_white_ratio=getattr(args, "max_white_ratio", 0.7),
            white_threshold=getattr(args, "white_threshold", 250),
            max_largest_white_component_ratio=getattr(
                args, "max_largest_white_component_ratio", 0.20
            ),
        )
        return config

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        data = {
            "dataset_root": str(self.dataset_root),
            "source_name": self.source_name,
            "target_name": self.target_name,
            "image_size": list(self.image_size),
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
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)


def load_preprocessing_config(path: str | Path) -> PreprocessingConfig:
    """Load a standalone preprocessing config file."""
    raw_data = load_yaml_mapping(path)
    if "preprocessing" in raw_data:
        reject_unknown_keys(raw_data, _TOP_LEVEL_KEYS, "top level")
    data = section_with_shared_fields(raw_data, "preprocessing", {"dataset_root", "image_size"})
    reject_unknown_keys(data, _PREPROCESSING_KEYS, "preprocessing")
    raw_max_memory_gb = data.get("max_memory_gb")

    config = PreprocessingConfig(
        dataset_root=Path(data["dataset_root"]),
        source_name=data["source_name"],
        target_name=data["target_name"],
        image_size=parse_wh_size_from_aliases(data, ("patch_size", "image_size"), (256, 256)),
        grid_movement=_pair(data.get("grid_movement"), (256, 256)),
        margin=int(data.get("margin", 200)),
        seed=data.get("seed"),
        save_masks=parse_bool_strict(data.get("save_masks", False), "save_masks"),
        save_discarded_patches=parse_bool_strict(
            data.get("save_discarded_patches", False),
            "save_discarded_patches",
        ),
        mask_strategy=_optional_strategy(
            data.get("mask_strategy", "connected_components"),
            "mask_strategy",
        )
        or "connected_components",
        source_mask_strategy=_optional_strategy(
            data.get("source_mask_strategy"),
            "source_mask_strategy",
        ),
        target_mask_strategy=_optional_strategy(
            data.get("target_mask_strategy"),
            "target_mask_strategy",
        ),
        mask_scale=float(data.get("mask_scale", 1.0)),
        lowres_mask_filtering=parse_bool_strict(
            data.get("lowres_mask_filtering", False),
            "lowres_mask_filtering",
        ),
        tiled_io=parse_bool_strict(data.get("tiled_io", False), "tiled_io"),
        max_memory_gb=None if raw_max_memory_gb is None else float(raw_max_memory_gb),
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
    return config
