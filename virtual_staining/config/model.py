from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from virtual_staining.config.validation import parse_bool_strict, parse_choice, reject_unknown_keys

NormName = Literal["batch", "instance"]
GeneratorArchitecture = Literal["concat_unet", "resnet"]
_MODEL_KEYS = frozenset({"inputs", "target", "generator", "discriminator"})
_GENERATOR_KEYS: dict[str, frozenset[str]] = {
    "concat_unet": frozenset({"architecture", "base_channels", "norm", "dropout", "bilinear"}),
    "resnet": frozenset({"architecture", "base_channels", "norm", "blocks"}),
}
_DEFAULT_GENERATOR_NORM: dict[str, NormName] = {"concat_unet": "batch", "resnet": "instance"}
DEFAULT_RESNET_BLOCKS = 9
_DISCRIMINATOR_KEYS = frozenset({"ndf", "norm", "use_sigmoid"})


@dataclass(frozen=True)
class GeneratorConfig:
    architecture: GeneratorArchitecture = "concat_unet"
    base_channels: int = 64
    norm: NormName = "batch"
    dropout: bool = False
    bilinear: bool = False
    blocks: int | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.architecture == "resnet":
            return {
                "architecture": self.architecture,
                "base_channels": self.base_channels,
                "norm": self.norm,
                "blocks": self.blocks,
            }
        return {
            "architecture": self.architecture,
            "base_channels": self.base_channels,
            "norm": self.norm,
            "dropout": self.dropout,
            "bilinear": self.bilinear,
        }


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
        generator = self.generator
        if generator.architecture == "resnet":
            if generator.blocks is None or generator.blocks < 1:
                raise ValueError("model.generator.blocks must be an integer >= 1 for resnet")
            if generator.norm != "instance":
                raise ValueError(
                    "model.generator.norm must be 'instance' for architecture 'resnet'"
                )
            if generator.dropout or generator.bilinear:
                raise ValueError("model.generator.dropout/bilinear are concat_unet-only options")
        elif generator.architecture == "concat_unet":
            if generator.blocks is not None:
                raise ValueError(
                    "model.generator.blocks is supported only for architecture 'resnet'"
                )
        else:
            raise ValueError(
                "model.generator.architecture must be one of ['concat_unet', 'resnet']"
            )
        if generator.bilinear:
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
        architecture = cast(
            GeneratorArchitecture,
            parse_choice(
                generator_data.get("architecture", "concat_unet"),
                "model.generator.architecture",
                set(get_args(GeneratorArchitecture)),
            ),
        )
        reject_unknown_keys(
            generator_data,
            _GENERATOR_KEYS[architecture],
            f"model.generator (architecture {architecture!r})",
        )
        reject_unknown_keys(discriminator_data, _DISCRIMINATOR_KEYS, "model.discriminator")
        raw_blocks = generator_data.get(
            "blocks", DEFAULT_RESNET_BLOCKS if architecture == "resnet" else None
        )
        if raw_blocks is not None and (
            isinstance(raw_blocks, bool) or not isinstance(raw_blocks, int)
        ):
            raise TypeError("model.generator.blocks must be an integer")
        return cls(
            inputs=tuple(str(value) for value in raw_inputs),
            target=str(data["target"]),
            generator=GeneratorConfig(
                architecture=architecture,
                base_channels=int(generator_data.get("base_channels", 64)),
                norm=cast(
                    NormName,
                    parse_choice(
                        generator_data.get("norm", _DEFAULT_GENERATOR_NORM[architecture]),
                        "model.generator.norm",
                        set(get_args(NormName)),
                    ),
                ),
                dropout=parse_bool_strict(
                    generator_data.get("dropout", False), "model.generator.dropout"
                ),
                bilinear=parse_bool_strict(
                    generator_data.get("bilinear", False), "model.generator.bilinear"
                ),
                blocks=raw_blocks,
            ),
            discriminator=DiscriminatorConfig(
                ndf=int(discriminator_data.get("ndf", 64)),
                norm=cast(
                    NormName,
                    parse_choice(
                        discriminator_data.get("norm", "instance"),
                        "model.discriminator.norm",
                        set(get_args(NormName)),
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
            "generator": self.generator.to_dict(),
            "discriminator": {
                "ndf": self.discriminator.ndf,
                "norm": self.discriminator.norm,
                "use_sigmoid": self.discriminator.use_sigmoid,
            },
        }
