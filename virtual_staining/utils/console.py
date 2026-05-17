from __future__ import annotations

import os
import sys
from typing import TextIO

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


def use_color(stream: TextIO = sys.stdout) -> bool:
    """Returns True if using ANSI colours in the console makes sense."""
    return os.environ.get("NO_COLOR") is None and stream.isatty()


def style(text: str, *names: str, stream: TextIO = sys.stdout) -> str:
    """Applies an ANSI style to the text, if colour output is enabled."""
    if not use_color(stream):
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
