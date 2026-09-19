"""Image-size dimension-order helpers.

Convention throughout this codebase: sizes are stored as ``(width, height)``
tuples - the same order a human writes "640x480" or passes ``[W, H]`` in YAML.

torchvision ``transforms.Resize`` expects ``(height, width)``, so convert with
``to_torchvision_hw`` before passing any size to a transform.  Patch-extraction
in ``preprocessing.py`` reads ``size[0]`` as width and ``size[1]`` as height,
which is consistent with this convention.
"""

from __future__ import annotations

from collections.abc import Sequence


def parse_wh_size(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Expected a two-value sequence, got {value!r}")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"Expected exactly two values, got {items}")
    return int(items[0]), int(items[1])


def parse_wh_size_from_aliases(
    data: dict[str, object], names: tuple[str, ...], default: tuple[int, int]
) -> tuple[int, int]:
    for name in names:
        if name in data:
            return parse_wh_size(data.get(name), default)
    return default


def to_torchvision_hw(wh: tuple[int, int]) -> tuple[int, int]:
    return wh[1], wh[0]
