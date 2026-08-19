from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys

ModelName = Literal["pix2pix"]
NormName = Literal["batch", "instance"]

_MODEL_KEYS = frozenset({"name", "generator", "discriminator"})
_GENERATOR_KEYS = frozenset(
    {"name", "in_channels", "out_channels", "base_channels", "norm", "dropout", "bilinear"}
)
_DISCRIMINATOR_KEYS = frozenset({"name", "in_channels", "ndf", "norm", "use_sigmoid"})


def _choice(value: Any, field_name: str, choices: set[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string. Supported values: {sorted(choices)}.")
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}. Got {value!r}.")
    return value


@dataclass(frozen=True)
class GeneratorConfig:
    name: Literal["unet"] = "unet"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    norm: NormName = "batch"
    dropout: bool = False
    bilinear: bool = False


@dataclass(frozen=True)
class DiscriminatorConfig:
    name: Literal["patchgan"] = "patchgan"
    in_channels: int = 6
    ndf: int = 64
    norm: NormName = "instance"
    use_sigmoid: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName = "pix2pix"
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelConfig:
        reject_unknown_keys(data, _MODEL_KEYS, "model")
        generator_data = data.get("generator", {})
        discriminator_data = data.get("discriminator", {})
        if not isinstance(generator_data, dict):
            raise TypeError("model.generator must be a YAML mapping")
        if not isinstance(discriminator_data, dict):
            raise TypeError("model.discriminator must be a YAML mapping")
        reject_unknown_keys(generator_data, _GENERATOR_KEYS, "model.generator")
        reject_unknown_keys(discriminator_data, _DISCRIMINATOR_KEYS, "model.discriminator")
        return cls(
            name=cast(ModelName, _choice(data.get("name", "pix2pix"), "model.name", {"pix2pix"})),
            generator=GeneratorConfig(
                name=cast(
                    Literal["unet"],
                    _choice(generator_data.get("name", "unet"), "model.generator.name", {"unet"}),
                ),
                in_channels=int(generator_data.get("in_channels", 3)),
                out_channels=int(generator_data.get("out_channels", 3)),
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
                name=cast(
                    Literal["patchgan"],
                    _choice(
                        discriminator_data.get("name", "patchgan"),
                        "model.discriminator.name",
                        {"patchgan"},
                    ),
                ),
                in_channels=int(discriminator_data.get("in_channels", 6)),
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

    def validate(self) -> None:
        if self.generator.bilinear:
            raise ValueError("model.generator.bilinear=True is not supported; use false")
        if self.discriminator.use_sigmoid:
            raise ValueError(
                "model.discriminator.use_sigmoid=True cannot be used with "
                "BCEWithLogitsLoss; use false"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generator": {
                "name": self.generator.name,
                "in_channels": self.generator.in_channels,
                "out_channels": self.generator.out_channels,
                "base_channels": self.generator.base_channels,
                "norm": self.generator.norm,
                "dropout": self.generator.dropout,
                "bilinear": self.generator.bilinear,
            },
            "discriminator": {
                "name": self.discriminator.name,
                "in_channels": self.discriminator.in_channels,
                "ndf": self.discriminator.ndf,
                "norm": self.discriminator.norm,
                "use_sigmoid": self.discriminator.use_sigmoid,
            },
        }
