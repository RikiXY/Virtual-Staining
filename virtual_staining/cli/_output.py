"""Shared terminal presentation helpers for CLI adapters."""

from virtual_staining.metrics import METRIC_SPECS
from virtual_staining.utils.console import print_info, print_section, style

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
    """Returns the ANSI color name for a metric value. Fallback: 'cyan'."""
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
    """Returns the metric value as a colour-coded formatted string."""
    formatted = f"{value:{_FORMATS.get(metric_name, '.6f')}}"
    return style(formatted, color_for_metric(metric_name, value))


color_metric_value = color_metric

__all__ = [
    "color_for_metric",
    "color_metric",
    "color_metric_value",
    "print_info",
    "print_section",
    "style",
]
