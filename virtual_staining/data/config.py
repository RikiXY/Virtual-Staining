from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose
from pathlib import Path

from virtual_staining.run_config import load_yaml_mapping, section_with_shared_fields


def _pair(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Expected a two-value sequence, got {value!r}")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"Expected exactly two values, got {items}")
    return int(items[0]), int(items[1])


def _pair_from_aliases(
    data: dict[str, object], names: tuple[str, ...], default: tuple[int, int]
) -> tuple[int, int]:
    for name in names:
        if name in data:
            return _pair(data.get(name), default)
    return default


@dataclass(frozen=True)
class PreprocessingConfig:
    dataset_root: Path
    source_name: str
    target_name: str
    image_size: tuple[int, int] = (256, 256)
    grid_movement: tuple[int, int] = (256, 256)
    margin: int = 200
    seed: int | None = None
    save_masks: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.05
    test_ratio: float = 0.15
    min_foreground_ratio: float = 0.25
    max_white_ratio: float = 0.7
    white_threshold: int = 250
    max_largest_white_component_ratio: float = 0.20

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

    @classmethod
    def from_args(cls, args) -> PreprocessingConfig:
        config = cls(
            dataset_root=Path(args.path),
            source_name=args.source_name,
            target_name=args.target_name,
            image_size=_pair(getattr(args, "image_size", (256, 256)), (256, 256)),
            grid_movement=_pair(getattr(args, "grid_movement", (256, 256)), (256, 256)),
            margin=getattr(args, "margin", 200),
            seed=getattr(args, "seed", None),
            save_masks=getattr(args, "save_masks", False),
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
        config.validate()
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessingConfig:
        raw_data = load_yaml_mapping(path)
        data = section_with_shared_fields(raw_data, "preprocessing", {"dataset_root", "image_size"})

        config = cls(
            dataset_root=Path(data["dataset_root"]),
            source_name=data["source_name"],
            target_name=data["target_name"],
            image_size=_pair_from_aliases(data, ("patch_size", "image_size"), (256, 256)),
            grid_movement=_pair(data.get("grid_movement"), (256, 256)),
            margin=int(data.get("margin", 200)),
            seed=data.get("seed"),
            save_masks=bool(data.get("save_masks", False)),
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
        config.validate()
        return config
