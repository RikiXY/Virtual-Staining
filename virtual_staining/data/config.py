from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, cast

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys
from virtual_staining.data.preprocessing import ALLOWED_MASK_STRATEGIES
from virtual_staining.utils.dimensions import parse_wh_size

_LEGACY_KEYS = frozenset(
    {
        "patch_size",
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
_SECTION_KEYS = frozenset({"inputs", "patching", "masks", "alignment", "filtering", "split", "io"})


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"preprocessing.{name} must be a YAML mapping")
    return value


def _pair(value: object, default: tuple[int, int]) -> tuple[int, int]:
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
        raise ValueError(f"{field_name} must be one of: {', '.join(ALLOWED_MASK_STRATEGIES)}")
    return strategy


@dataclass(frozen=True)
class InputConfig:
    inventory: Path
    source_modality: str
    target_modality: str
    hash_verification: str = "cached"

    def __post_init__(self) -> None:
        if not self.source_modality.strip() or not self.target_modality.strip():
            raise ValueError("inputs source_modality and target_modality must not be empty")
        if self.hash_verification not in {"cached", "always"}:
            raise ValueError("inputs.hash_verification must be 'cached' or 'always'")


@dataclass(frozen=True)
class MaskConfig:
    provided_layout: str = "auto"
    generation: str = "if_missing"
    strategy: str = "connected_components"
    source_strategy: str | None = None
    target_strategy: str | None = None
    scale: float = 1.0
    lowres_filtering: bool = False
    save_resolved_masks: bool = False
    save_patch_masks: bool = False


@dataclass(frozen=True)
class AlignmentConfig:
    mode: str = "auto"
    method: str = "affine_sift"
    validate_declared: bool = True
    on_failure: str = "error"


@dataclass(frozen=True)
class SplitConfig:
    unit: str = "patch"
    train: float = 0.8
    val: float = 0.05
    test: float = 0.15
    seed: int = 0
    assignment_file: Path | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    # Legacy fields remain constructor-compatible for one transition period.
    dataset_root: Path
    source_name: str = ""
    target_name: str = ""
    patch_size: tuple[int, int] = (256, 256)
    grid_movement: tuple[int, int] = (256, 256)
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
    inputs: InputConfig | None = None
    masks: MaskConfig | None = None
    alignment: AlignmentConfig | None = None
    split: SplitConfig | None = None
    io_backend: str = "auto"
    foreground_enabled: bool = True
    foreground_policy: str = "both"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def effective_masks(self) -> MaskConfig:
        return self.masks or MaskConfig(
            strategy=self.mask_strategy,
            source_strategy=self.source_mask_strategy,
            target_strategy=self.target_mask_strategy,
            scale=self.mask_scale,
            lowres_filtering=self.lowres_mask_filtering,
            save_resolved_masks=self.save_masks,
            save_patch_masks=self.save_masks,
        )

    @property
    def effective_alignment(self) -> AlignmentConfig:
        return self.alignment or AlignmentConfig(mode="always", validate_declared=False)

    @property
    def effective_split(self) -> SplitConfig:
        return self.split or SplitConfig(
            train=self.train_ratio,
            val=self.val_ratio,
            test=self.test_ratio,
            seed=self.seed or 0,
        )

    @property
    def source_modality(self) -> str:
        return (
            self.inputs.source_modality if self.inputs else Path(self.source_name or "source").stem
        )

    @property
    def target_modality(self) -> str:
        return (
            self.inputs.target_modality if self.inputs else Path(self.target_name or "target").stem
        )

    def validate(self) -> None:
        if self.inputs is None:
            if not isinstance(self.source_name, str) or not self.source_name.strip():
                raise ValueError("source_name must be a non-empty string")
            if not isinstance(self.target_name, str) or not self.target_name.strip():
                raise ValueError("target_name must be a non-empty string")
            if self.source_name == self.target_name:
                raise ValueError("target_name must differ from source_name")
        elif self.source_name or self.target_name:
            raise ValueError("source_name/target_name cannot be combined with inputs.inventory")

        for field_name, value in (
            ("patch_size", self.patch_size),
            ("grid_movement", self.grid_movement),
        ):
            if len(value) != 2 or any(dimension <= 0 for dimension in value):
                raise ValueError(f"{field_name} must contain two positive integers")
        if self.margin < 0:
            raise ValueError("margin must be greater than or equal to 0")
        if self.max_memory_gb is not None and self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be greater than 0 when provided")
        if self.io_backend not in {"auto", "pillow", "openslide"}:
            raise ValueError("io.backend must be auto, pillow, or openslide")

        masks = self.effective_masks
        if masks.provided_layout not in {"auto", "none", "shared", "separate"}:
            raise ValueError("masks.provided_layout must be auto, none, shared, or separate")
        if masks.generation not in {"never", "if_missing", "always"}:
            raise ValueError("masks.generation must be never, if_missing, or always")
        if not (0.0 < masks.scale <= 1.0):
            name = "masks.scale" if self.masks is not None else "mask_scale"
            raise ValueError(f"{name} must be in (0.0, 1.0]")
        _optional_strategy(masks.strategy, "masks.strategy")
        _optional_strategy(masks.source_strategy, "masks.source_strategy")
        _optional_strategy(masks.target_strategy, "masks.target_strategy")

        alignment = self.effective_alignment
        if alignment.mode not in {"auto", "always", "never"}:
            raise ValueError("alignment.mode must be auto, always, or never")
        if alignment.method != "affine_sift":
            raise ValueError("alignment.method must be affine_sift")
        if alignment.on_failure not in {"error", "skip_pair"}:
            raise ValueError("alignment.on_failure must be error or skip_pair")

        split = self.effective_split
        if split.unit not in {"patch", "pair", "specimen", "patient"}:
            raise ValueError("split.unit must be patch, pair, specimen, or patient")
        ratios = (split.train, split.val, split.test)
        if any(value < 0 or value > 1 for value in ratios):
            raise ValueError("split ratios must be between 0 and 1")
        if not isclose(sum(ratios), 1.0):
            raise ValueError("split ratios must sum to 1")

        if self.foreground_policy not in {"both", "source", "target", "intersection", "union"}:
            raise ValueError("filtering.foreground.policy is invalid")
        for field_name, value in (
            ("min_foreground_ratio", self.min_foreground_ratio),
            ("max_white_ratio", self.max_white_ratio),
            ("max_largest_white_component_ratio", self.max_largest_white_component_ratio),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if not 0 <= self.white_threshold <= 255:
            raise ValueError("white_threshold must be between 0 and 255")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        dataset_root: Path,
        default_image_size: tuple[int, int],
    ) -> PreprocessingConfig:
        reject_unknown_keys(data, _LEGACY_KEYS | _SECTION_KEYS, "preprocessing")
        has_legacy = "source_name" in data or "target_name" in data
        has_inventory = "inputs" in data
        if has_legacy and has_inventory:
            raise ValueError("source_name/target_name cannot be combined with inputs.inventory")
        if not has_legacy and not has_inventory:
            raise ValueError("preprocessing requires source_name/target_name or inputs.inventory")
        if has_legacy:
            if any(key in data for key in _SECTION_KEYS):
                raise ValueError("legacy preprocessing fields cannot be mixed with nested sections")
            return cls._from_legacy(data, dataset_root, default_image_size)

        inputs_data = _mapping(data.get("inputs"), "inputs")
        reject_unknown_keys(
            inputs_data,
            frozenset({"inventory", "source_modality", "target_modality", "hash_verification"}),
            "preprocessing.inputs",
        )
        for required in ("inventory", "source_modality", "target_modality"):
            if required not in inputs_data:
                raise ValueError(f"preprocessing.inputs requires {required}")
        patching = _mapping(data.get("patching"), "patching")
        reject_unknown_keys(
            patching,
            frozenset({"patch_size", "grid_movement", "margin", "save_discarded_patches"}),
            "preprocessing.patching",
        )
        masks_data = _mapping(data.get("masks"), "masks")
        reject_unknown_keys(
            masks_data,
            frozenset(
                {
                    "provided_layout",
                    "generation",
                    "strategy",
                    "source_strategy",
                    "target_strategy",
                    "scale",
                    "lowres_filtering",
                    "save_resolved_masks",
                    "save_patch_masks",
                }
            ),
            "preprocessing.masks",
        )
        alignment_data = _mapping(data.get("alignment"), "alignment")
        reject_unknown_keys(
            alignment_data,
            frozenset({"mode", "method", "validate_declared", "on_failure"}),
            "preprocessing.alignment",
        )
        filtering = _mapping(data.get("filtering"), "filtering")
        reject_unknown_keys(
            filtering,
            frozenset(
                {
                    "foreground",
                    "max_white_ratio",
                    "white_threshold",
                    "max_largest_white_component_ratio",
                }
            ),
            "preprocessing.filtering",
        )
        foreground = _mapping(filtering.get("foreground"), "filtering.foreground")
        reject_unknown_keys(
            foreground,
            frozenset({"enabled", "policy", "min_ratio"}),
            "preprocessing.filtering.foreground",
        )
        split_data = _mapping(data.get("split"), "split")
        reject_unknown_keys(
            split_data,
            frozenset({"unit", "train", "val", "test", "seed", "assignment_file"}),
            "preprocessing.split",
        )
        for required in ("unit", "train", "val", "test"):
            if required not in split_data:
                raise ValueError(f"preprocessing.split requires {required}")
        io_data = _mapping(data.get("io"), "io")
        reject_unknown_keys(
            io_data, frozenset({"tiled", "backend", "max_memory_gb"}), "preprocessing.io"
        )
        max_memory = io_data.get("max_memory_gb")
        return cls(
            dataset_root=dataset_root,
            inputs=InputConfig(
                inventory=Path(inputs_data["inventory"]),
                source_modality=str(inputs_data["source_modality"]),
                target_modality=str(inputs_data["target_modality"]),
                hash_verification=str(inputs_data.get("hash_verification", "cached")),
            ),
            patch_size=parse_wh_size(patching.get("patch_size"), default_image_size),
            grid_movement=_pair(patching.get("grid_movement"), default_image_size),
            margin=int(patching.get("margin", 200)),
            save_discarded_patches=parse_bool_strict(
                patching.get("save_discarded_patches", False),
                "preprocessing.patching.save_discarded_patches",
            ),
            masks=MaskConfig(
                provided_layout=str(masks_data.get("provided_layout", "auto")),
                generation=str(masks_data.get("generation", "if_missing")),
                strategy=_optional_strategy(
                    masks_data.get("strategy", "connected_components"), "masks.strategy"
                )
                or "connected_components",
                source_strategy=_optional_strategy(
                    masks_data.get("source_strategy"), "masks.source_strategy"
                ),
                target_strategy=_optional_strategy(
                    masks_data.get("target_strategy"), "masks.target_strategy"
                ),
                scale=float(masks_data.get("scale", 1.0)),
                lowres_filtering=parse_bool_strict(
                    masks_data.get("lowres_filtering", False), "masks.lowres_filtering"
                ),
                save_resolved_masks=parse_bool_strict(
                    masks_data.get("save_resolved_masks", False), "masks.save_resolved_masks"
                ),
                save_patch_masks=parse_bool_strict(
                    masks_data.get("save_patch_masks", False), "masks.save_patch_masks"
                ),
            ),
            alignment=AlignmentConfig(
                mode=str(alignment_data.get("mode", "auto")),
                method=str(alignment_data.get("method", "affine_sift")),
                validate_declared=parse_bool_strict(
                    alignment_data.get("validate_declared", True), "alignment.validate_declared"
                ),
                on_failure=str(alignment_data.get("on_failure", "error")),
            ),
            split=SplitConfig(
                unit=str(split_data["unit"]),
                train=float(split_data["train"]),
                val=float(split_data["val"]),
                test=float(split_data["test"]),
                seed=int(split_data.get("seed", 0)),
                assignment_file=Path(split_data["assignment_file"])
                if split_data.get("assignment_file")
                else None,
            ),
            tiled_io=parse_bool_strict(io_data.get("tiled", True), "io.tiled"),
            io_backend=str(io_data.get("backend", "auto")),
            max_memory_gb=None if max_memory is None else float(max_memory),
            foreground_enabled=parse_bool_strict(
                foreground.get("enabled", True), "filtering.foreground.enabled"
            ),
            foreground_policy=str(foreground.get("policy", "both")),
            min_foreground_ratio=float(foreground.get("min_ratio", 0.25)),
            max_white_ratio=float(filtering.get("max_white_ratio", 0.7)),
            white_threshold=int(filtering.get("white_threshold", 250)),
            max_largest_white_component_ratio=float(
                filtering.get("max_largest_white_component_ratio", 0.2)
            ),
        )

    @classmethod
    def _from_legacy(
        cls, data: dict[str, Any], dataset_root: Path, default_image_size: tuple[int, int]
    ) -> PreprocessingConfig:
        max_memory = data.get("max_memory_gb")
        return cls(
            dataset_root=dataset_root,
            source_name=cast(str, data.get("source_name")),
            target_name=cast(str, data.get("target_name")),
            patch_size=parse_wh_size(data.get("patch_size"), default_image_size),
            grid_movement=_pair(data.get("grid_movement"), default_image_size),
            margin=int(data.get("margin", 200)),
            seed=data.get("seed"),
            save_masks=parse_bool_strict(data.get("save_masks", False), "preprocessing.save_masks"),
            save_discarded_patches=parse_bool_strict(
                data.get("save_discarded_patches", False), "preprocessing.save_discarded_patches"
            ),
            mask_strategy=_optional_strategy(
                data.get("mask_strategy", "connected_components"), "preprocessing.mask_strategy"
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
                data.get("lowres_mask_filtering", False), "preprocessing.lowres_mask_filtering"
            ),
            tiled_io=parse_bool_strict(data.get("tiled_io", False), "preprocessing.tiled_io"),
            max_memory_gb=None if max_memory is None else float(max_memory),
            train_ratio=float(data.get("train_ratio", 0.8)),
            val_ratio=float(data.get("val_ratio", 0.05)),
            test_ratio=float(data.get("test_ratio", 0.15)),
            min_foreground_ratio=float(data.get("min_foreground_ratio", 0.25)),
            max_white_ratio=float(data.get("max_white_ratio", 0.7)),
            white_threshold=int(data.get("white_threshold", 250)),
            max_largest_white_component_ratio=float(
                data.get("max_largest_white_component_ratio", 0.2)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.inputs is None:
            return {
                "source_name": self.source_name,
                "target_name": self.target_name,
                "patch_size": list(self.patch_size),
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
        masks, alignment, split = (
            self.effective_masks,
            self.effective_alignment,
            self.effective_split,
        )
        return {
            "inputs": {
                "inventory": str(self.inputs.inventory),
                "source_modality": self.inputs.source_modality,
                "target_modality": self.inputs.target_modality,
                "hash_verification": self.inputs.hash_verification,
            },
            "patching": {
                "patch_size": list(self.patch_size),
                "grid_movement": list(self.grid_movement),
                "margin": self.margin,
                "save_discarded_patches": self.save_discarded_patches,
            },
            "masks": {
                "provided_layout": masks.provided_layout,
                "generation": masks.generation,
                "strategy": masks.strategy,
                "source_strategy": masks.source_strategy,
                "target_strategy": masks.target_strategy,
                "scale": masks.scale,
                "lowres_filtering": masks.lowres_filtering,
                "save_resolved_masks": masks.save_resolved_masks,
                "save_patch_masks": masks.save_patch_masks,
            },
            "alignment": {
                "mode": alignment.mode,
                "method": alignment.method,
                "validate_declared": alignment.validate_declared,
                "on_failure": alignment.on_failure,
            },
            "filtering": {
                "foreground": {
                    "enabled": self.foreground_enabled,
                    "policy": self.foreground_policy,
                    "min_ratio": self.min_foreground_ratio,
                },
                "max_white_ratio": self.max_white_ratio,
                "white_threshold": self.white_threshold,
                "max_largest_white_component_ratio": self.max_largest_white_component_ratio,
            },
            "split": {
                "unit": split.unit,
                "train": split.train,
                "val": split.val,
                "test": split.test,
                "seed": split.seed,
                **(
                    {"assignment_file": str(split.assignment_file)} if split.assignment_file else {}
                ),
            },
            "io": {
                "tiled": self.tiled_io,
                "backend": self.io_backend,
                "max_memory_gb": self.max_memory_gb,
            },
        }
