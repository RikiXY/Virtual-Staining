from __future__ import annotations

import dataclasses
import datetime
import os
from io import StringIO

import pytest

from virtual_staining.cli._progress import render_training_progress
from virtual_staining.training import progress as progress_module
from virtual_staining.training.progress import ProgressTracker, ProgressUpdate, format_progress_log

_END = datetime.datetime(2030, 1, 2, 3, 4, 5)


def _update(
    *,
    progress: float = 0.5,
    eta_seconds: float | None = 125.0,
    estimated_end: datetime.datetime | None = _END,
) -> ProgressUpdate:
    return ProgressUpdate(
        progress=progress,
        epoch_progress=0.5,
        epoch=1,
        batch_index=2,
        total_epochs=4,
        total_batches=6,
        step_metrics={"loss_G": 1.25, "loss_D": 2.5},
        eval_metrics={"loss_G": 1.0, "loss_D": 2.0},
        eval_epoch=0,
        elapsed_seconds=3725.0,
        eta_seconds=eta_seconds,
        estimated_end=estimated_end,
        last_checkpoint_name="ep001.pth",
        best_checkpoint_name="ep001.pth",
        best_checkpoint_metric_name="loss_G_val",
        best_checkpoint_metric_value=1.0,
    )


class _TTY(StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 2


def _render_tty(update: ProgressUpdate, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(
        "virtual_staining.cli._progress.os.get_terminal_size",
        lambda fd: os.terminal_size((400, 24)),
    )
    monkeypatch.setenv("NO_COLOR", "1")
    stream = _TTY()
    render_training_progress(update, stream)
    return stream.getvalue()


class _Clock:
    def __init__(self) -> None:
        self.mono = 100.0
        self.wall = datetime.datetime(2030, 1, 1).timestamp()

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall


def test_progress_update_carries_raw_timing_not_presentation_strings() -> None:
    fields = {field.name: field.type for field in dataclasses.fields(ProgressUpdate)}
    assert fields["elapsed_seconds"] == "float"
    assert fields["eta_seconds"] == "float | None"
    assert fields["estimated_end"] == "datetime.datetime | None"
    assert not any(name.endswith("_str") for name in fields)


def test_plain_log_and_non_tty_renderer_format_raw_timing() -> None:
    update = _update()
    log = format_progress_log(update)
    assert "loss_G 1.2500" in log
    assert "elapsed 1h 02m | ETA 2m 05s | end 2030-01-02 03:04:05" in log

    stream = StringIO()
    render_training_progress(update, stream)
    output = stream.getvalue()
    assert "\033" not in output
    assert output == log + "\n"


def test_tty_renderer_formats_the_same_raw_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _render_tty(_update(), monkeypatch)
    assert "elapsed 1h 02m | ETA 2m 05s | end 2030-01-02 03:04:05" in output


def test_warming_up_eta_is_rendered_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    update = _update(eta_seconds=None, estimated_end=None)
    expected = "ETA -- | end warming up"
    assert expected in format_progress_log(update)
    assert expected in _render_tty(update, monkeypatch)


def test_completed_tty_progress_renders_zero_eta_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _render_tty(_update(progress=1.0, eta_seconds=0.0), monkeypatch)
    assert "ETA 0s | end 2030-01-02 03:04:05" in output
    assert output.endswith("\033[1E\n")


def test_tracker_durations_ignore_wall_clock_jumps(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(progress_module, "time", clock)
    tracker = ProgressTracker(total_epochs=1, total_batches=10, warmup_batches=0, min_eta_batches=1)
    tracker.start()

    clock.mono += 2.0
    clock.wall -= 86_400.0  # system clock steps back a day
    first = tracker.calculate_progress(0, 0)
    clock.mono += 2.0
    clock.wall += 172_800.0  # and then forward two days
    second = tracker.calculate_progress(0, 1)

    assert tracker.step_durations == [2.0, 2.0]
    assert (first.elapsed_seconds, second.elapsed_seconds) == (2.0, 4.0)
    assert second.eta_seconds == 16.0
    # Only the calendar estimate follows the (jumped) wall clock.
    assert second.estimated_end == datetime.datetime.fromtimestamp(clock.wall + 16.0)


def test_tracker_warms_up_then_reports_zero_eta_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(progress_module, "time", clock)
    tracker = ProgressTracker(total_epochs=2, total_batches=3, warmup_batches=1, min_eta_batches=2)
    tracker.start()
    estimates = []
    for step in range(6):
        clock.mono += 1.0
        estimates.append(tracker.calculate_progress(step // 3, step % 3))

    assert [e.eta_seconds for e in estimates] == [None, None, 3.0, 2.0, 1.0, 0.0]
    assert estimates[0].estimated_end is None
    assert estimates[-1].progress == 1.0
    assert tracker.estimate(tracker.total_steps).eta_seconds == 0.0
