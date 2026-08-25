from __future__ import annotations

import os
import sys
from typing import TextIO

from virtual_staining.applications.train import ProgressUpdate, format_progress_log
from virtual_staining.cli._output import style, use_color


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
        best_loss = style(
            "--"
            if update.best_checkpoint_loss_G_val is None
            else f"{update.best_checkpoint_loss_G_val:.4f}",
            "light_blue",
            stream=stream,
        )
        loss_g = style(f"{update.step_losses.loss_G:.4f}", "cyan", stream=stream)
        loss_d = style(f"{update.step_losses.loss_D:.4f}", "orange", stream=stream)
    else:
        progress_text = f"{update.progress:.2%}"
        last_checkpoint = update.last_checkpoint_name.strip()
        best_checkpoint = update.best_checkpoint_name.strip()
        best_loss = (
            "--"
            if update.best_checkpoint_loss_G_val is None
            else f"{update.best_checkpoint_loss_G_val:.4f}"
        )
        loss_g = f"{update.step_losses.loss_G:.4f}"
        loss_d = f"{update.step_losses.loss_D:.4f}"

    first_line = (
        f"{bar} ep {update.epoch + 1}/{update.total_epochs} ({progress_text}) | "
        f"b {update.batch_index + 1}/{update.total_batches} ({update.epoch_progress:.0%}) | "
        f"loss_G {loss_g} | loss_D {loss_d} | elapsed {update.elapsed_str} | "
        f"ETA {update.eta_str} | end {update.end_time_str} | last ckpt {last_checkpoint}"
    )
    if update.eval_losses is None:
        eval_text = "eval --"
    else:
        epoch_text = f"ep {update.eval_epoch + 1} | " if update.eval_epoch is not None else ""
        eval_g = f"{update.eval_losses.loss_G:.4f}"
        eval_d = f"{update.eval_losses.loss_D:.4f}"
        if color:
            eval_g = style(eval_g, "light_blue", stream=stream)
            eval_d = style(eval_d, "light_magenta", stream=stream)
        eval_text = f"eval {epoch_text}loss_G {eval_g} | loss_D {eval_d}"
    second_line = f"{bar} {eval_text} | best ckpt {best_checkpoint} ({best_loss})"

    lines = [first_line, second_line]
    clean_lines = [line[: max(width - 1, 1)].ljust(max(width - 1, 1)) for line in lines]
    stream.write("\r\033[2K" + clean_lines[0])
    stream.write("\n\033[2K" + clean_lines[1])
    if update.progress >= 1.0:
        stream.write("\033[1E\n")
    else:
        stream.write("\033[1F")
    stream.flush()


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
