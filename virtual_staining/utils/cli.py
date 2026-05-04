from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TypeVar

T = TypeVar("T")

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "orange": "\033[38;5;208m",
}


def use_color() -> bool:
    """Returns True if using ANSI colours in the console makes sense."""
    return os.environ.get("NO_COLOR") is None and sys.stdout.isatty()


def style(text: str, *names: str) -> str:
    """Applies an ANSI style to the text, if colour output is enabled."""
    if not use_color():
        return text
    prefix = "".join(ANSI[name] for name in names if name in ANSI)
    return prefix + text + ANSI["reset"]


def print_section(title: str) -> None:
    """Prints a human-readable section header in the CLI."""
    print()
    print(style(f"=== {title} ===", "bold", "cyan"))


def print_info(label: str, value: str) -> None:
    """Prints a single label-value line."""
    print(f"{style(label + ':', 'bold', 'blue')} {value}")


def apply_namespace_overrides(
    config: T,
    args: object,
    fields: Mapping[str, str | tuple[str, Callable[[object], object]]],
) -> T:
    overrides: dict[str, object] = {}
    for arg_name, field in fields.items():
        if not hasattr(args, arg_name):
            continue

        value = getattr(args, arg_name)
        if isinstance(field, tuple):
            field_name, transform = field
            value = transform(value)
        else:
            field_name = field
        overrides[field_name] = value

    return replace(config, **overrides) if overrides else config
