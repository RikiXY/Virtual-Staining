from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys

NormName = Literal["batch", "instance"]
_MODEL_KEYS = frozenset({"inputs", "target", "generator", "discriminator"})
_GENERATOR_KEYS = frozenset(
    {
        "architecture",
        "base_channels",
        "norm",
        "dropout",
        "bilinear",
        "blocks",
        "class_path",
        "params",
    }
)
_DISCRIMINATOR_KEYS = frozenset(
    {"architecture", "ndf", "norm", "use_sigmoid", "class_path", "params"}
)


def _choice(value: Any, field_name: str, choices: set[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string. Supported values: {sorted(choices)}.")
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}. Got {value!r}.")
    return value


@dataclass(frozen=True)
class GeneratorConfig:
    architecture: str = "concat_unet"
    base_channels: int = 64
    norm: NormName = "batch"
    dropout: bool = False
    bilinear: bool = False
    blocks: int = 6
    class_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscriminatorConfig:
    architecture: str = "patchgan"
    ndf: int = 64
    norm: NormName = "instance"
    use_sigmoid: bool = False
    class_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    inputs: tuple[str, ...]
    target: str
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()

    def __post_init__(self) -> None:
        if (
            not self.inputs
            or len(set(self.inputs)) != len(self.inputs)
            or any(not name.strip() for name in self.inputs)
        ):
            raise ValueError("model.inputs must be a non-empty tuple of unique names")
        if not self.target.strip():
            raise ValueError("model.target must not be blank")
        if self.generator.architecture not in {"concat_unet", "resnet", "custom"}:
            raise ValueError(
                "model.generator.architecture must be one of ['concat_unet', 'custom', 'resnet']"
            )
        if self.generator.architecture == "concat_unet" and self.generator.bilinear:
            raise ValueError("model.generator.bilinear=True is not supported; use false")
        if self.generator.blocks <= 0:
            raise ValueError("model.generator.blocks must be greater than 0")
        if self.generator.architecture == "custom" and not self.generator.class_path:
            raise ValueError("model.generator.class_path is required when architecture is 'custom'")
        if self.discriminator.architecture not in {"patchgan", "custom"}:
            raise ValueError("model.discriminator.architecture must be 'patchgan' or 'custom'")
        if self.discriminator.architecture == "custom" and not self.discriminator.class_path:
            raise ValueError(
                "model.discriminator.class_path is required when architecture is 'custom'"
            )
        if self.discriminator.use_sigmoid:
            raise ValueError(
                "model.discriminator.use_sigmoid=True cannot be used with "
                "BCEWithLogitsLoss; use false"
            )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelConfig:
        reject_unknown_keys(data, _MODEL_KEYS, "model")
        for required in ("inputs", "target"):
            if required not in data:
                raise ValueError(f"model requires {required}")
        raw_inputs = data["inputs"]
        if isinstance(raw_inputs, str) or not isinstance(raw_inputs, (list, tuple)):
            raise TypeError("model.inputs must be a sequence")
        generator_data = data.get("generator", {})
        discriminator_data = data.get("discriminator", {})
        if not isinstance(generator_data, dict) or not isinstance(discriminator_data, dict):
            raise TypeError("model.generator and model.discriminator must be YAML mappings")
        reject_unknown_keys(generator_data, _GENERATOR_KEYS, "model.generator")
        reject_unknown_keys(discriminator_data, _DISCRIMINATOR_KEYS, "model.discriminator")
        generator_params = generator_data.get("params", {})
        discriminator_params = discriminator_data.get("params", {})
        if not isinstance(generator_params, dict) or not isinstance(discriminator_params, dict):
            raise TypeError("model component params must be YAML mappings")
        return cls(
            inputs=tuple(str(value) for value in raw_inputs),
            target=str(data["target"]),
            generator=GeneratorConfig(
                architecture=_choice(
                    generator_data.get("architecture", "concat_unet"),
                    "model.generator.architecture",
                    {"concat_unet", "resnet", "custom"},
                ),
                base_channels=int(generator_data.get("base_channels", 64)),
                norm=cast(
                    NormName,
                    _choice(
                        generator_data.get("norm", "batch"),
                        "model.generator.norm",
                        {"batch", "instance"},
                    ),
                ),
                dropout=parse_bool_strict(
                    generator_data.get("dropout", False), "model.generator.dropout"
                ),
                bilinear=parse_bool_strict(
                    generator_data.get("bilinear", False), "model.generator.bilinear"
                ),
                blocks=int(generator_data.get("blocks", 6)),
                class_path=generator_data.get("class_path"),
                params=dict(generator_params),
            ),
            discriminator=DiscriminatorConfig(
                architecture=_choice(
                    discriminator_data.get("architecture", "patchgan"),
                    "model.discriminator.architecture",
                    {"patchgan", "custom"},
                ),
                ndf=int(discriminator_data.get("ndf", 64)),
                norm=cast(
                    NormName,
                    _choice(
                        discriminator_data.get("norm", "instance"),
                        "model.discriminator.norm",
                        {"batch", "instance"},
                    ),
                ),
                use_sigmoid=parse_bool_strict(
                    discriminator_data.get("use_sigmoid", False), "model.discriminator.use_sigmoid"
                ),
                class_path=discriminator_data.get("class_path"),
                params=dict(discriminator_params),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "target": self.target,
            "generator": {
                "architecture": self.generator.architecture,
                "base_channels": self.generator.base_channels,
                "norm": self.generator.norm,
                "dropout": self.generator.dropout,
                "bilinear": self.generator.bilinear,
                "blocks": self.generator.blocks,
                "class_path": self.generator.class_path,
                "params": dict(self.generator.params),
            },
            "discriminator": {
                "architecture": self.discriminator.architecture,
                "ndf": self.discriminator.ndf,
                "norm": self.discriminator.norm,
                "use_sigmoid": self.discriminator.use_sigmoid,
                "class_path": self.discriminator.class_path,
                "params": dict(self.discriminator.params),
            },
        }
