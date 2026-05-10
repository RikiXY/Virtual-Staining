"""Image-size dimension-order helpers.

Convention throughout this codebase: sizes are stored as ``(width, height)``
tuples — the same order a human writes "640x480" or passes ``[W, H]`` in YAML.

torchvision ``transforms.Resize`` expects ``(height, width)``, so convert with
``to_torchvision_hw`` before passing any size to a transform.  Patch-extraction
in ``preprocessing.py`` reads ``size[0]`` as width and ``size[1]`` as height,
which is consistent with this convention.
"""

from __future__ import annotations

from collections.abc import Sequence


def parse_wh_size(value: object, default: tuple[int, int]) -> tuple[int, int]:
    """Parse a ``(width, height)`` size tuple from a config value.

    Parameters
    ----------
    value:
        Raw config value — a two-element sequence of positive integers, or
        ``None`` to accept the default.
    default:
        Fallback ``(width, height)`` returned when *value* is ``None``.

    Raises
    ------
    ValueError
        If *value* is not a two-element sequence of integers.
    """
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
    """Parse a ``(width, height)`` size from the first matching key in *data*.

    Parameters
    ----------
    data:
        Mapping of config keys to raw values.
    names:
        Candidate key names tried in priority order.
    default:
        Fallback ``(width, height)`` when none of the keys are present.
    """
    for name in names:
        if name in data:
            return parse_wh_size(data.get(name), default)
    return default


def to_torchvision_hw(wh: tuple[int, int]) -> tuple[int, int]:
    """Convert a ``(width, height)`` pair to torchvision's ``(height, width)`` order.

    ``transforms.Resize((a, b))`` interprets ``a`` as height and ``b`` as
    width.  Pass every ``image_size`` through this function before handing it
    to a torchvision transform so that non-square sizes are not silently
    transposed.

    Parameters
    ----------
    wh:
        ``(width, height)`` size tuple as stored in config fields.

    Returns
    -------
    tuple[int, int]
        ``(height, width)`` tuple suitable for torchvision transforms.
    """
    return wh[1], wh[0]
