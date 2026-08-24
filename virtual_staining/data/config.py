from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import isclose
from pathlib import Path
from typing import Any

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys
from virtual_staining.data.preprocessing import ALLOWED_MASK_STRATEGIES
from virtual_staining.utils.dimensions import parse_wh_size

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
class PatchingConfig:
    patch_size: tuple[int, int] = (256, 256)
    grid_movement: tuple[int, int] = (256, 256)
    margin: int = 200
    save_discarded_patches: bool = False


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
class ForegroundFilterConfig:
    enabled: bool = True
    policy: str = "both"
    min_ratio: float = 0.25


@dataclass(frozen=True)
class FilteringConfig:
    foreground: ForegroundFilterConfig = field(default_factory=ForegroundFilterConfig)
    max_white_ratio: float = 0.7
    white_threshold: int = 250
    max_largest_white_component_ratio: float = 0.20


@dataclass(frozen=True)
class SplitConfig:
    unit: str = "patch"
    train: float = 0.8
    val: float = 0.05
    test: float = 0.15
    seed: int = 0
    assignment_file: Path | None = None


@dataclass(frozen=True)
class IOConfig:
    tiled: bool = True
    backend: str = "auto"
    max_memory_gb: float | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    dataset_root: Path
    inputs: InputConfig
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    io: IOConfig = field(default_factory=IOConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        patching = self.patching
        for field_name, value in (
            ("patching.patch_size", patching.patch_size),
            ("patching.grid_movement", patching.grid_movement),
        ):
            if len(value) != 2 or any(dimension <= 0 for dimension in value):
                raise ValueError(f"{field_name} must contain two positive integers")
        if patching.margin < 0:
            raise ValueError("patching.margin must be greater than or equal to 0")

        io = self.io
        if io.max_memory_gb is not None and io.max_memory_gb <= 0:
            raise ValueError("io.max_memory_gb must be greater than 0 when provided")
        if io.backend not in {"auto", "pillow", "openslide"}:
            raise ValueError("io.backend must be auto, pillow, or openslide")

        masks = self.masks
        if masks.provided_layout not in {"auto", "none", "shared", "separate"}:
            raise ValueError("masks.provided_layout must be auto, none, shared, or separate")
        if masks.generation not in {"never", "if_missing", "always"}:
            raise ValueError("masks.generation must be never, if_missing, or always")
        if not (0.0 < masks.scale <= 1.0):
            raise ValueError("masks.scale must be in (0.0, 1.0]")
        _optional_strategy(masks.strategy, "masks.strategy")
        _optional_strategy(masks.source_strategy, "masks.source_strategy")
        _optional_strategy(masks.target_strategy, "masks.target_strategy")

        alignment = self.alignment
        if alignment.mode not in {"auto", "always", "never"}:
            raise ValueError("alignment.mode must be auto, always, or never")
        if alignment.method != "affine_sift":
            raise ValueError("alignment.method must be affine_sift")
        if alignment.on_failure not in {"error", "skip_pair"}:
            raise ValueError("alignment.on_failure must be error or skip_pair")

        split = self.split
        if split.unit not in {"patch", "pair", "specimen", "patient"}:
            raise ValueError("split.unit must be patch, pair, specimen, or patient")
        ratios = (split.train, split.val, split.test)
        if any(value < 0 or value > 1 for value in ratios):
            raise ValueError("split ratios must be between 0 and 1")
        if not isclose(sum(ratios), 1.0):
            raise ValueError("split ratios must sum to 1")

        filtering = self.filtering
        if filtering.foreground.policy not in {
            "both",
            "source",
            "target",
            "intersection",
            "union",
        }:
            raise ValueError("filtering.foreground.policy is invalid")
        for field_name, value in (
            ("filtering.foreground.min_ratio", filtering.foreground.min_ratio),
            ("filtering.max_white_ratio", filtering.max_white_ratio),
            (
                "filtering.max_largest_white_component_ratio",
                filtering.max_largest_white_component_ratio,
            ),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if not 0 <= filtering.white_threshold <= 255:
            raise ValueError("filtering.white_threshold must be between 0 and 255")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        dataset_root: Path,
        default_image_size: tuple[int, int],
    ) -> PreprocessingConfig:
        reject_unknown_keys(data, _SECTION_KEYS, "preprocessing")
        if "inputs" not in data:
            raise ValueError("preprocessing requires inputs")
        if "split" not in data:
            raise ValueError("preprocessing requires split")

        inputs_data = _mapping(data["inputs"], "inputs")
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
        split_data = _mapping(data["split"], "split")
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
            patching=PatchingConfig(
                patch_size=parse_wh_size(patching.get("patch_size"), default_image_size),
                grid_movement=_pair(patching.get("grid_movement"), default_image_size),
                margin=int(patching.get("margin", 200)),
                save_discarded_patches=parse_bool_strict(
                    patching.get("save_discarded_patches", False),
                    "preprocessing.patching.save_discarded_patches",
                ),
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
            filtering=FilteringConfig(
                foreground=ForegroundFilterConfig(
                    enabled=parse_bool_strict(
                        foreground.get("enabled", True), "filtering.foreground.enabled"
                    ),
                    policy=str(foreground.get("policy", "both")),
                    min_ratio=float(foreground.get("min_ratio", 0.25)),
                ),
                max_white_ratio=float(filtering.get("max_white_ratio", 0.7)),
                white_threshold=int(filtering.get("white_threshold", 250)),
                max_largest_white_component_ratio=float(
                    filtering.get("max_largest_white_component_ratio", 0.2)
                ),
            ),
            split=SplitConfig(
                unit=str(split_data["unit"]),
                train=float(split_data["train"]),
                val=float(split_data["val"]),
                test=float(split_data["test"]),
                seed=int(split_data.get("seed", 0)),
                assignment_file=(
                    Path(split_data["assignment_file"])
                    if split_data.get("assignment_file")
                    else None
                ),
            ),
            io=IOConfig(
                tiled=parse_bool_strict(io_data.get("tiled", True), "io.tiled"),
                backend=str(io_data.get("backend", "auto")),
                max_memory_gb=None if max_memory is None else float(max_memory),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("dataset_root")
        result["inputs"]["inventory"] = str(self.inputs.inventory)
        result["patching"]["patch_size"] = list(self.patching.patch_size)
        result["patching"]["grid_movement"] = list(self.patching.grid_movement)
        if self.split.assignment_file is not None:
            result["split"]["assignment_file"] = str(self.split.assignment_file)
        return result
