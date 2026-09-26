from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import TextIO

from virtual_staining.cli._output import style, use_color
from virtual_staining.training.progress import (
    ProgressUpdate,
    format_progress_log,
    format_progress_timing,
)


def render_training_progress(update: ProgressUpdate, stream: TextIO = sys.stderr) -> None:
    if not stream.isatty():
        stream.write(format_progress_log(update) + "\n")
        stream.flush()
        return

    color = use_color(stream)
    width = _terminal_width(stream)
    progress = min(max(update.progress, 0.0), 1.0)
    filled = int(40 * progress)
    if progress > 0 and filled == 0:
        filled = 1
    if progress >= 1:
        filled = 40
    bar = "▌" + "█" * filled + "░" * (40 - filled) + "▐"

    if color:
        bar = style(bar, "green", stream=stream)
        progress_text = style(
            f"{update.progress:.2%}", _progress_color(update.progress), stream=stream
        )
        last_checkpoint = _checkpoint_name(update.last_checkpoint_name, stream)
        best_checkpoint = _checkpoint_name(update.best_checkpoint_name, stream)
    else:
        progress_text = f"{update.progress:.2%}"
        last_checkpoint = update.last_checkpoint_name.strip()
        best_checkpoint = update.best_checkpoint_name.strip()

    step_text = _metric_text(update.step_metrics, stream, "cyan" if color else None)
    first_line = (
        f"{bar} ep {update.epoch + 1}/{update.total_epochs} ({progress_text}) | "
        f"b {update.batch_index + 1}/{update.total_batches} ({update.epoch_progress:.0%}) | "
        f"{step_text} | {format_progress_timing(update)} | last ckpt {last_checkpoint}"
    )
    if update.eval_metrics is None:
        eval_text = "eval --"
    else:
        epoch_text = f"ep {update.eval_epoch + 1} | " if update.eval_epoch is not None else ""
        eval_metrics = _metric_text(
            update.eval_metrics,
            stream,
            "light_blue" if color else None,
        )
        eval_text = f"eval {epoch_text}{eval_metrics}"

    best_value = (
        "--"
        if update.best_checkpoint_metric_value is None
        else f"{update.best_checkpoint_metric_value:.4f}"
    )
    if color and update.best_checkpoint_metric_value is not None:
        best_value = style(best_value, "light_blue", stream=stream)
    second_line = (
        f"{bar} {eval_text} | best ckpt {best_checkpoint} "
        f"({update.best_checkpoint_metric_name} {best_value})"
    )

    lines = [first_line, second_line]
    clean_lines = [line[: max(width - 1, 1)].ljust(max(width - 1, 1)) for line in lines]
    stream.write("\r\033[2K" + clean_lines[0])
    stream.write("\n\033[2K" + clean_lines[1])
    if update.progress >= 1.0:
        stream.write("\033[1E\n")
    else:
        stream.write("\033[1F")
    stream.flush()


def _metric_text(
    metrics: Mapping[str, float],
    stream: TextIO,
    color_name: str | None,
) -> str:
    parts: list[str] = []
    for name, value in metrics.items():
        rendered = f"{value:.4f}"
        if color_name is not None:
            rendered = style(rendered, color_name, stream=stream)
        parts.append(f"{name} {rendered}")
    return " | ".join(parts) or "metrics --"


def _checkpoint_name(name: str, stream: TextIO) -> str:
    value = name.strip()
    return style(value, "red", stream=stream) if value == "none" else value


def _progress_color(progress: float) -> str:
    if progress < 0.33:
        return "yellow"
    if progress < 0.66:
        return "cyan"
    if progress < 0.90:
        return "blue"
    return "green"


def _terminal_width(stream: TextIO) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except OSError:
        return 140
