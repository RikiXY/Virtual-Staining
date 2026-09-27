"""Built-in network components registered through ``ComponentDefinition``.

Option parsing is torch-free; each factory imports its module only when building.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, get_args

from virtual_staining.config.validation import parse_bool_strict, parse_choice, reject_unknown_keys
from virtual_staining.definitions import ComponentContext, ComponentDefinition

if TYPE_CHECKING:
    from virtual_staining.models.discriminator import PatchGANDiscriminator
    from virtual_staining.models.generator import ConcatUNetGenerator, ResnetGenerator

NormName = Literal["batch", "instance"]
BUILTIN_SOURCE = "virtual_staining"
DEFAULT_RESNET_BLOCKS = 9
# Two stride-2 stages must round-trip exactly; residual reflection padding needs >= 2 px.
_RESNET_SIZE_MULTIPLE = 4
_RESNET_MIN_SIZE = 8


def _norm(raw: Mapping[str, Any], field: str, default: NormName) -> str:
    return parse_choice(raw.get("norm", default), f"{field}.norm", set(get_args(NormName)))


def _int(raw: Mapping[str, Any], key: str, field: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field}.{key} must be an integer")
    return int(value)


def _parse_concat_unet(raw: Mapping[str, Any], context: ComponentContext) -> dict[str, Any]:
    field = context.field
    reject_unknown_keys(
        raw, frozenset({"base_channels", "norm", "dropout", "bilinear"}), f"{field} (concat_unet)"
    )
    bilinear = parse_bool_strict(raw.get("bilinear", False), f"{field}.bilinear")
    if bilinear:
        raise ValueError(f"{field}.bilinear=True is not supported; use false")
    return {
        "base_channels": _int(raw, "base_channels", field, 64),
        "norm": _norm(raw, field, "batch"),
        "dropout": parse_bool_strict(raw.get("dropout", False), f"{field}.dropout"),
        "bilinear": bilinear,
    }


def _build_concat_unet(
    options: Mapping[str, Any], *, input_names: tuple[str, ...]
) -> ConcatUNetGenerator:
    from virtual_staining.models.generator import ConcatUNetGenerator

    return ConcatUNetGenerator(
        input_names,
        base_channels=options["base_channels"],
        norm=options["norm"],
        dropout=options["dropout"],
        bilinear=options["bilinear"],
    )


def _parse_resnet(raw: Mapping[str, Any], context: ComponentContext) -> dict[str, Any]:
    field = context.field
    reject_unknown_keys(raw, frozenset({"base_channels", "norm", "blocks"}), f"{field} (resnet)")
    norm = _norm(raw, field, "instance")
    if norm != "instance":
        raise ValueError(f"{field}.norm must be 'instance' for architecture 'resnet'")
    blocks = _int(raw, "blocks", field, DEFAULT_RESNET_BLOCKS)
    if blocks < 1:
        raise ValueError(f"{field}.blocks must be an integer >= 1 for resnet")
    width, height = context.image_size
    if (
        width % _RESNET_SIZE_MULTIPLE
        or height % _RESNET_SIZE_MULTIPLE
        or min(width, height) < _RESNET_MIN_SIZE
    ):
        raise ValueError(
            f"image_size {[width, height]} is invalid for the resnet generator: both "
            f"dimensions must be multiples of {_RESNET_SIZE_MULTIPLE} and at least "
            f"{_RESNET_MIN_SIZE}"
        )
    return {"base_channels": _int(raw, "base_channels", field, 64), "norm": norm, "blocks": blocks}


def _build_resnet(options: Mapping[str, Any]) -> ResnetGenerator:
    from virtual_staining.models.generator import ResnetGenerator

    return ResnetGenerator(base_channels=options["base_channels"], blocks=options["blocks"])


def _parse_patchgan(raw: Mapping[str, Any], context: ComponentContext) -> dict[str, Any]:
    field = context.field
    reject_unknown_keys(raw, frozenset({"ndf", "norm", "use_sigmoid"}), field)
    use_sigmoid = parse_bool_strict(raw.get("use_sigmoid", False), f"{field}.use_sigmoid")
    if use_sigmoid:
        raise ValueError(
            f"{field}.use_sigmoid=True cannot be used with BCEWithLogitsLoss; use false"
        )
    return {
        "ndf": _int(raw, "ndf", field, 64),
        "norm": _norm(raw, field, "instance"),
        "use_sigmoid": use_sigmoid,
    }


def _build_patchgan(options: Mapping[str, Any], *, in_channels: int) -> PatchGANDiscriminator:
    from virtual_staining.models.discriminator import PatchGANDiscriminator

    return PatchGANDiscriminator(
        in_channels=in_channels,
        ndf=options["ndf"],
        norm=options["norm"],
        use_sigmoid=options["use_sigmoid"],
    )


CONCAT_UNET = ComponentDefinition(
    name="concat_unet",
    version="1",
    source=BUILTIN_SOURCE,
    parse_options=_parse_concat_unet,
    factory=_build_concat_unet,
)
RESNET = ComponentDefinition(
    name="resnet",
    version="1",
    source=BUILTIN_SOURCE,
    parse_options=_parse_resnet,
    factory=_build_resnet,
)
PATCHGAN = ComponentDefinition(
    name="patchgan",
    version="1",
    source=BUILTIN_SOURCE,
    parse_options=_parse_patchgan,
    factory=_build_patchgan,
)
BUILTIN_COMPONENTS: tuple[ComponentDefinition, ...] = (CONCAT_UNET, RESNET, PATCHGAN)
