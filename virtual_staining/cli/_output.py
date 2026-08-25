"""Shared terminal presentation helpers for CLI adapters."""

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
    "light_blue": "\033[94m",
    "light_magenta": "\033[95m",
    "orange": "\033[38;5;208m",
}


def use_color(stream: TextIO = sys.stdout) -> bool:
    """Return True if using ANSI colours in the console makes sense."""
    return os.environ.get("NO_COLOR") is None and stream.isatty()


def style(text: str, *names: str, stream: TextIO = sys.stdout) -> str:
    """Apply ANSI styles to text when colour output is enabled."""
    if not use_color(stream):
        return text
    prefix = "".join(ANSI[name] for name in names if name in ANSI)
    return prefix + text + ANSI["reset"]


def print_section(title: str) -> None:
    """Print a human-readable section header in the CLI."""
    print()
    print(style(f"=== {title} ===", "bold", "cyan"))


def print_info(label: str, value: str) -> None:
    """Print a single label-value line."""
    print(f"{style(label + ':', 'bold', 'blue')} {value}")


_COLORS = ("green", "yellow", "orange")
_FORMATS = {
    "ssim": ".6f",
    "psnr": ".4f",
    "mae": ".6f",
    "rmse": ".6f",
    "mse": ".6f",
    "pcc_gray": ".6f",
    "pcc_rgb_mean": ".6f",
    "pcc_r": ".6f",
    "pcc_g": ".6f",
    "pcc_b": ".6f",
}


def color_for_metric(metric_name: str, value: float) -> str:
    """Return the ANSI color name for a metric value. Fallback: 'cyan'."""
    from virtual_staining.metrics import METRIC_SPECS

    spec = METRIC_SPECS.get(metric_name)
    if spec is None:
        return "cyan"
    for threshold, color in zip(spec.thresholds, _COLORS, strict=True):
        if spec.higher_is_better and value >= threshold:
            return color
        if not spec.higher_is_better and value <= threshold:
            return color
    return "red"


def color_metric(metric_name: str, value: float) -> str:
    """Return a metric value as a colour-coded formatted string."""
    formatted = f"{value:{_FORMATS.get(metric_name, '.6f')}}"
    return style(formatted, color_for_metric(metric_name, value))


color_metric_value = color_metric

__all__ = [
    "ANSI",
    "color_for_metric",
    "color_metric",
    "color_metric_value",
    "print_info",
    "print_section",
    "style",
    "use_color",
]
