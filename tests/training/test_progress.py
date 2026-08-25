from __future__ import annotations

import os
from io import StringIO

from virtual_staining.cli._progress import render_training_progress
from virtual_staining.training.losses import StepLosses
from virtual_staining.training.progress import ProgressUpdate, format_progress_log


def _update(*, progress: float = 0.5) -> ProgressUpdate:
    return ProgressUpdate(
        progress=progress,
        epoch_progress=0.5,
        epoch=1,
        batch_index=2,
        total_epochs=4,
        total_batches=6,
        step_losses=StepLosses(1.25, 2.5),
        eval_losses=StepLosses(1.0, 2.0),
        eval_epoch=0,
        elapsed_str="3s",
        eta_str="4s",
        end_time_str="now",
        last_checkpoint_name="ep001.pth",
        best_checkpoint_name="ep001.pth",
        best_checkpoint_loss_G_val=1.0,
    )


def test_progress_log_and_non_tty_renderer_are_plain() -> None:
    update = _update()
    assert "loss_G 1.2500" in format_progress_log(update)

    stream = StringIO()
    render_training_progress(update, stream)
    output = stream.getvalue()
    assert "\033" not in output
    assert "ep 2/4" in output


def test_completed_tty_progress_terminates_with_newline(monkeypatch) -> None:
    class TTY(StringIO):
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 2

    monkeypatch.setattr(
        "virtual_staining.cli._progress.os.get_terminal_size", lambda fd: os.terminal_size((80, 24))
    )
    stream = TTY()
    render_training_progress(_update(progress=1.0), stream)
    assert stream.getvalue().endswith("\033[1E\n")
