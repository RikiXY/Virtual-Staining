"""Shared terminal presentation helpers for CLI adapters."""

from virtual_staining.utils.console import print_info, print_section, style
from virtual_staining.utils.metrics import color_for_metric, color_metric, color_metric_value

__all__ = [
    "color_for_metric",
    "color_metric",
    "color_metric_value",
    "print_info",
    "print_section",
    "style",
]
