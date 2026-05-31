from __future__ import annotations

from virtual_staining.utils.console import style

DEFAULT_METRICS = [
    "ssim",
    "psnr",
    "mae",
    "rmse",
    "mse",
    "pcc_rgb_mean",
    "pcc_gray",
]

METRIC_DIRECTIONS = {
    "ssim": True,
    "psnr": True,
    "mae": False,
    "rmse": False,
    "mse": False,
    "pcc_gray": True,
    "pcc_rgb_mean": True,
    "pcc_r": True,
    "pcc_g": True,
    "pcc_b": True,
}

METRIC_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "ssim": [(0.85, "green"), (0.75, "yellow"), (0.65, "orange")],
    "psnr": [(25.0, "green"), (20.0, "yellow"), (15.0, "orange")],
    "mae": [(0.06, "green"), (0.10, "yellow"), (0.16, "orange")],
    "rmse": [(0.08, "green"), (0.12, "yellow"), (0.20, "orange")],
    "mse": [(0.0036, "green"), (0.0100, "yellow"), (0.0256, "orange")],
    "pcc_gray": [(0.95, "green"), (0.90, "yellow"), (0.80, "orange")],
    "pcc_rgb_mean": [(0.95, "green"), (0.90, "yellow"), (0.80, "orange")],
    "pcc_r": [(0.95, "green"), (0.90, "yellow"), (0.80, "orange")],
    "pcc_g": [(0.95, "green"), (0.90, "yellow"), (0.80, "orange")],
    "pcc_b": [(0.95, "green"), (0.90, "yellow"), (0.80, "orange")],
}

DEFAULT_WEAK_TAIL_THRESHOLDS: dict[str, float] = {
    "ssim": 0.60,
    "psnr": 20.0,
    "mae": 0.08,
    "rmse": 0.12,
    "mse": 0.0100,
    "pcc_gray": 0.80,
    "pcc_rgb_mean": 0.80,
    "pcc_r": 0.80,
    "pcc_g": 0.80,
    "pcc_b": 0.80,
}

METRIC_PLOT_RANGES = {
    "mae": (0.0, 1.0),
    "mse": (0.0, 1.0),
    "rmse": (0.0, 1.0),
    "ssim": (0.0, 1.0),
    "pcc_gray": (-1.0, 1.0),
    "pcc_rgb_mean": (-1.0, 1.0),
    "pcc_r": (-1.0, 1.0),
    "pcc_g": (-1.0, 1.0),
    "pcc_b": (-1.0, 1.0),
    # PSNR has no finite theoretical maximum. For visual comparison we use
    # a fixed practical window, so that histograms remain comparable.
    "psnr": (0.0, 60.0),
}

_FORMAT: dict[str, str] = {
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


def is_higher_better_metric(metric_name: str) -> bool:
    """Returns True when larger values are better for a metric."""
    if metric_name not in METRIC_DIRECTIONS:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Supported metrics: {', '.join(METRIC_DIRECTIONS)}"
        )
    return METRIC_DIRECTIONS[metric_name]


def get_metric_thresholds(metric_name: str) -> list[float]:
    """Returns the default thresholds used by comparison summaries."""
    if metric_name not in METRIC_THRESHOLDS:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Supported metrics: {', '.join(METRIC_THRESHOLDS)}"
        )
    return sorted(threshold for threshold, _ in METRIC_THRESHOLDS[metric_name])


def get_metric_plot_range(metric_name: str) -> tuple[float, float]:
    """Returns the fixed plot range used for a metric."""
    if metric_name not in METRIC_PLOT_RANGES:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. "
            f"Supported metrics: {', '.join(METRIC_PLOT_RANGES)}"
        )
    return METRIC_PLOT_RANGES[metric_name]


def color_for_metric(metric_name: str, value: float) -> str:
    """Returns the ANSI color name for a metric value. Fallback: 'cyan'."""
    if metric_name not in METRIC_THRESHOLDS:
        return "cyan"

    higher_is_better = is_higher_better_metric(metric_name)
    for threshold, color in METRIC_THRESHOLDS[metric_name]:
        if higher_is_better and value >= threshold:
            return color
        if not higher_is_better and value <= threshold:
            return color

    return "red"


def color_metric(metric_name: str, value: float) -> str:
    """Returns the metric value as a colour-coded formatted string."""
    fmt = _FORMAT.get(metric_name, ".6f")
    color = color_for_metric(metric_name, value)
    return style(f"{value:{fmt}}", color)


color_metric_value = color_metric
