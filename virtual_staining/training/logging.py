from __future__ import annotations

import io
import logging as stdlib_logging
import os
import sys
import time
from pathlib import Path
from typing import TextIO

from virtual_staining.training.steps import StepLosses
from virtual_staining.utils.console import style

logger = stdlib_logging.getLogger("virtual_staining.training.trainer")
_PLAIN_TEXT_STREAM = io.StringIO()


class TrainingLogSession:
    def __init__(self, log_file: Path, *loggers: stdlib_logging.Logger) -> None:
        self.log_file = log_file
        self.loggers = loggers
        self.handler: stdlib_logging.FileHandler | None = None
        self._old_states: list[tuple[stdlib_logging.Logger, int, bool]] = []

    def __enter__(self) -> None:
        handler = stdlib_logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        handler.setLevel(stdlib_logging.DEBUG)
        handler.setFormatter(stdlib_logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.handler = handler

        for log in self.loggers:
            self._old_states.append((log, log.level, log.propagate))
            log.setLevel(stdlib_logging.DEBUG)
            log.propagate = False
            log.addHandler(handler)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.handler is not None
        for log, old_level, old_propagate in reversed(self._old_states):
            log.removeHandler(self.handler)
            log.propagate = old_propagate
            log.setLevel(old_level)
        self.handler.close()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _style_console(text: str, *names: str, stream: TextIO, color: bool) -> str:
    if not color:
        return text
    return style(text, *names, stream=stream)


def _color_progress(progress: float, stream: TextIO = sys.stderr) -> str:
    text = f"{progress:.2%}"
    if progress < 0.33:
        return style(text, "yellow", stream=stream)
    if progress < 0.66:
        return style(text, "cyan", stream=stream)
    if progress < 0.90:
        return style(text, "blue", stream=stream)
    return style(text, "green", stream=stream)


def _render_progress_bar_rows(
    progress: float,
    width: int = 40,
    stream: TextIO = sys.stderr,
) -> tuple[str, str]:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(width * progress)
    if progress > 0 and filled == 0:
        filled = 1
    if progress >= 1:
        filled = width
    empty = width - filled
    bar = "█" * filled + "░" * empty
    colored_bar = style(bar, "green", stream=stream)
    return f"▌{colored_bar}▐", f"▌{colored_bar}▐"


def _format_checkpoint_name(name: str, *, stream: TextIO, color: bool) -> str:
    if name.strip() == "none":
        return _style_console("none", "red", stream=stream, color=color)
    return name


def _format_best_checkpoint_loss(
    loss_G_val: float | None,
    *,
    stream: TextIO,
    color: bool,
) -> str:
    value = "--" if loss_G_val is None else f"{loss_G_val:.4f}"
    return _style_console(value, "light_blue", stream=stream, color=color)


def _format_loss_parts(
    step_losses: StepLosses,
    *,
    stream: TextIO,
    color: bool,
) -> str:
    parts = [
        (
            f"{_style_console('loss_G', 'bold', stream=stream, color=color)} "
            f"{_style_console(f'{step_losses.loss_G:.4f}', 'cyan', stream=stream, color=color)}"
        )
    ]
    parts.append(
        f"{_style_console('loss_D', 'bold', stream=stream, color=color)} "
        f"{_style_console(f'{step_losses.loss_D:.4f}', 'orange', stream=stream, color=color)}"
    )
    return " | ".join(parts)


def _format_eval_parts(
    eval_losses: StepLosses | None,
    *,
    eval_epoch: int | None,
    stream: TextIO,
    color: bool,
) -> str:
    label = _style_console("eval", "bold", stream=stream, color=color)
    if eval_losses is None:
        return f"{label} --"

    epoch_text = f"ep {eval_epoch + 1} | " if eval_epoch is not None else ""
    eval_loss_g = _style_console(
        f"{eval_losses.loss_G:.4f}",
        "light_blue",
        stream=stream,
        color=color,
    )
    eval_loss_d = _style_console(
        f"{eval_losses.loss_D:.4f}",
        "light_magenta",
        stream=stream,
        color=color,
    )
    return (
        f"{label} {epoch_text}"
        f"{_style_console('loss_G', 'bold', stream=stream, color=color)} "
        f"{eval_loss_g}"
        f" | "
        f"{_style_console('loss_D', 'bold', stream=stream, color=color)} "
        f"{eval_loss_d}"
    )


def _format_progress_message(
    *,
    progress: float,
    epoch_progress: float,
    epoch: int,
    batch_index: int,
    progress_tracker: ProgressTracker,
    step_losses: StepLosses,
    eval_losses: StepLosses | None,
    eval_epoch: int | None,
    elapsed_str: str,
    eta_str: str,
    end_time_str: str,
    last_checkpoint_name: str,
    best_checkpoint_name: str,
    best_checkpoint_loss_G_val: float | None,
    stream: TextIO = sys.stderr,
    color: bool = True,
) -> str:
    bar_top, bar_bottom = (
        _render_progress_bar_rows(progress, stream=stream)
        if color
        else _render_progress_bar_rows(progress, stream=_PLAIN_TEXT_STREAM)
    )
    progress_text = _color_progress(progress, stream=stream) if color else f"{progress:.2%}"
    last_checkpoint = _format_checkpoint_name(
        last_checkpoint_name,
        stream=stream,
        color=color,
    )
    best_checkpoint = _format_checkpoint_name(
        best_checkpoint_name,
        stream=stream,
        color=color,
    )
    best_checkpoint_loss = _format_best_checkpoint_loss(
        best_checkpoint_loss_G_val,
        stream=stream,
        color=color,
    )
    first_line = (
        f"{bar_top} "
        f"ep {epoch + 1}/{progress_tracker.total_epochs} "
        f"({progress_text}) | "
        f"b {batch_index + 1}/{progress_tracker.total_batches} "
        f"({epoch_progress:.0%}) | "
        f"{_format_loss_parts(step_losses, stream=stream, color=color)} | "
        f"elapsed {elapsed_str} | "
        f"ETA {eta_str} | "
        f"end {end_time_str} | "
        f"last ckpt {last_checkpoint}"
    )
    second_line = (
        f"{bar_bottom} "
        f"{_format_eval_parts(eval_losses, eval_epoch=eval_epoch, stream=stream, color=color)} | "
        f"best ckpt {best_checkpoint} ({best_checkpoint_loss})"
    )
    return f"{first_line}\n{second_line}"


def _format_progress_log_message(
    *,
    progress: float,
    epoch_progress: float,
    epoch: int,
    batch_index: int,
    progress_tracker: ProgressTracker,
    step_losses: StepLosses,
    eval_losses: StepLosses | None,
    eval_epoch: int | None,
    elapsed_str: str,
    eta_str: str,
    end_time_str: str,
    last_checkpoint_name: str,
    best_checkpoint_name: str,
    best_checkpoint_loss_G_val: float | None,
) -> str:
    first_line = (
        f"ep {epoch + 1}/{progress_tracker.total_epochs} "
        f"({progress:.2%}) | "
        f"b {batch_index + 1}/{progress_tracker.total_batches} "
        f"({epoch_progress:.0%}) | "
        f"{_format_loss_parts(step_losses, stream=_PLAIN_TEXT_STREAM, color=False)} | "
        f"elapsed {elapsed_str} | "
        f"ETA {eta_str} | "
        f"end {end_time_str} | "
        f"last ckpt {last_checkpoint_name.strip()}"
    )
    eval_parts = _format_eval_parts(
        eval_losses,
        eval_epoch=eval_epoch,
        stream=_PLAIN_TEXT_STREAM,
        color=False,
    )
    best_checkpoint_loss = _format_best_checkpoint_loss(
        best_checkpoint_loss_G_val,
        stream=_PLAIN_TEXT_STREAM,
        color=False,
    )
    second_line = (
        f"{eval_parts} | best ckpt {best_checkpoint_name.strip()} ({best_checkpoint_loss})"
    )
    return f"{first_line}\n{second_line}"


def emit_progress_update(
    *,
    progress: float,
    epoch_progress: float,
    epoch: int,
    batch_index: int,
    progress_tracker: ProgressTracker,
    step_losses: StepLosses,
    elapsed_str: str,
    eta_str: str,
    end_time_str: str,
    last_checkpoint_name: str,
    best_checkpoint_name: str,
    best_checkpoint_loss_G_val: float | None = None,
    eval_losses: StepLosses | None = None,
    eval_epoch: int | None = None,
) -> None:
    console_message = _format_progress_message(
        progress=progress,
        epoch_progress=epoch_progress,
        epoch=epoch,
        batch_index=batch_index,
        progress_tracker=progress_tracker,
        step_losses=step_losses,
        eval_losses=eval_losses,
        eval_epoch=eval_epoch,
        elapsed_str=elapsed_str,
        eta_str=eta_str,
        end_time_str=end_time_str,
        last_checkpoint_name=last_checkpoint_name,
        best_checkpoint_name=best_checkpoint_name,
        best_checkpoint_loss_G_val=best_checkpoint_loss_G_val,
        stream=sys.stderr,
        color=True,
    )
    update_console_progress(console_message)

    log_message = _format_progress_log_message(
        progress=progress,
        epoch_progress=epoch_progress,
        epoch=epoch,
        batch_index=batch_index,
        progress_tracker=progress_tracker,
        step_losses=step_losses,
        eval_losses=eval_losses,
        eval_epoch=eval_epoch,
        elapsed_str=elapsed_str,
        eta_str=eta_str,
        end_time_str=end_time_str,
        last_checkpoint_name=last_checkpoint_name,
        best_checkpoint_name=best_checkpoint_name,
        best_checkpoint_loss_G_val=best_checkpoint_loss_G_val,
    )
    logger.debug("%s", log_message)


def _terminal_width(stream: TextIO = sys.stderr) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except OSError:
        return 140


def update_console_progress(message: str, stream: TextIO = sys.stderr) -> None:
    if not stream.isatty():
        stream.write(message + "\n")
        stream.flush()
        return

    terminal_width = _terminal_width(stream)
    lines = message.splitlines() or [""]
    clean_lines = [line[: terminal_width - 1].ljust(terminal_width - 1) for line in lines]
    stream.write("\r\033[2K" + clean_lines[0])
    for line in clean_lines[1:]:
        stream.write("\n\033[2K" + line)
    if len(clean_lines) > 1:
        stream.write(f"\033[{len(clean_lines) - 1}F")
    stream.flush()


def finish_console_progress(stream: TextIO = sys.stderr) -> None:
    if stream.isatty():
        stream.write("\033[1E\n")
    else:
        stream.write("\n")
    stream.flush()


class ProgressTracker:
    def __init__(
        self,
        total_epochs: int,
        total_batches: int,
        start_epoch: int = 0,
        max_history: int = 300,
        warmup_batches: int = 10,
        min_eta_batches: int = 5,
    ) -> None:
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_epoch = start_epoch
        self.max_history = max_history
        self.warmup_batches = warmup_batches
        self.min_eta_batches = min_eta_batches

        self.total_steps = total_epochs * total_batches
        self.start_step = start_epoch * total_batches

        self.start_time: float | None = None
        self.last_step_time: float | None = None
        self.step_durations: list[float] = []

    def start(self) -> None:
        now = time.time()
        self.start_time = now
        self.last_step_time = now
        self.step_durations = []

    def calculate_progress(
        self, epoch: int, batch: int
    ) -> tuple[float, float, float | None, float | None]:
        now = time.time()

        current_step = epoch * self.total_batches + batch + 1
        completed_since_start = current_step - self.start_step
        remaining_steps = max(self.total_steps - current_step, 0)

        assert self.last_step_time is not None, "call start() before calculate_progress()"
        step_duration = now - self.last_step_time
        self.last_step_time = now

        if completed_since_start > self.warmup_batches:
            self.step_durations.append(step_duration)
            if len(self.step_durations) > self.max_history:
                self.step_durations.pop(0)

        assert self.start_time is not None, "call start() before calculate_progress()"
        total_elapsed_time = now - self.start_time
        progress = current_step / self.total_steps if self.total_steps > 0 else 1.0

        if len(self.step_durations) < self.min_eta_batches:
            eta = None
            end_time = None
        else:
            avg_step_time = sum(self.step_durations) / len(self.step_durations)
            eta = avg_step_time * remaining_steps
            end_time = now + eta

        return progress, total_elapsed_time, eta, end_time
