from __future__ import annotations

from virtual_staining.utils.cli import style

# Threshold tables: (threshold, color) pairs, evaluated in order.
# For "higher is better" metrics (ssim, psnr), the first passing >= wins.
# For "lower is better" metrics (mae, rmse), the first passing <= wins.
_SSIM_THRESHOLDS = [(0.85, "green"), (0.75, "yellow"), (0.65, "orange")]
_PSNR_THRESHOLDS = [(25.0, "green"), (20.0, "yellow"), (15.0, "orange")]
_MAE_THRESHOLDS  = [(0.06, "green"), (0.10, "yellow"), (0.16, "orange")]
_RMSE_THRESHOLDS = [(0.08, "green"), (0.12, "yellow"), (0.20, "orange")]

_FORMAT: dict[str, str] = {
    "ssim": ".6f",
    "psnr": ".4f",
    "mae":  ".6f",
    "rmse": ".6f",
}


def color_for_metric(metric_name: str, value: float) -> str:
    """Returns the ANSI color name for a metric value. Fallback: 'cyan'."""
    if metric_name == "ssim":
        for threshold, color in _SSIM_THRESHOLDS:
            if value >= threshold:
                return color
    elif metric_name == "psnr":
        for threshold, color in _PSNR_THRESHOLDS:
            if value >= threshold:
                return color
    elif metric_name == "mae":
        for threshold, color in _MAE_THRESHOLDS:
            if value <= threshold:
                return color
    elif metric_name == "rmse":
        for threshold, color in _RMSE_THRESHOLDS:
            if value <= threshold:
                return color
    else:
        return "cyan"
    return "red"


def color_metric(metric_name: str, value: float) -> str:
    """Returns the metric value as a colour-coded formatted string."""
    fmt = _FORMAT.get(metric_name, ".6f")
    color = color_for_metric(metric_name, value)
    return style(f"{value:{fmt}}", color)


color_metric_value = color_metric
