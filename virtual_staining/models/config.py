from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys

NormName = Literal["batch", "instance"]
_MODEL_KEYS = frozenset({"inputs", "target", "generator", "discriminator"})
_GENERATOR_KEYS = frozenset({"architecture", "base_channels", "norm", "dropout", "bilinear"})
_DISCRIMINATOR_KEYS = frozenset({"ndf", "norm", "use_sigmoid"})


def _choice(value: Any, field_name: str, choices: set[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string. Supported values: {sorted(choices)}.")
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}. Got {value!r}.")
    return value


@dataclass(frozen=True)
class GeneratorConfig:
    architecture: Literal["concat_unet"] = "concat_unet"
    base_channels: int = 64
    norm: NormName = "batch"
    dropout: bool = False
    bilinear: bool = False


@dataclass(frozen=True)
class DiscriminatorConfig:
    ndf: int = 64
    norm: NormName = "instance"
    use_sigmoid: bool = False


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
        if self.generator.architecture != "concat_unet":
            raise ValueError("model.generator.architecture must be concat_unet")
        if self.generator.bilinear:
            raise ValueError("model.generator.bilinear=True is not supported; use false")
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
        return cls(
            inputs=tuple(str(value) for value in raw_inputs),
            target=str(data["target"]),
            generator=GeneratorConfig(
                architecture=cast(
                    Literal["concat_unet"],
                    _choice(
                        generator_data.get("architecture", "concat_unet"),
                        "model.generator.architecture",
                        {"concat_unet"},
                    ),
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
            ),
            discriminator=DiscriminatorConfig(
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
            },
            "discriminator": {
                "ndf": self.discriminator.ndf,
                "norm": self.discriminator.norm,
                "use_sigmoid": self.discriminator.use_sigmoid,
            },
        }
