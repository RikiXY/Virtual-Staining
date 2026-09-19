"""Evaluation plotting runtime configuration.

Evaluation work is also executed from NiceGUI worker threads.  A non-interactive
backend avoids creating Tk objects in those threads, while the shared lock keeps
Matplotlib's process-global state from being mutated concurrently.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from threading import RLock
from typing import ParamSpec, TypeVar

import matplotlib

matplotlib.use("Agg", force=True)

P = ParamSpec("P")
R = TypeVar("R")

PLOT_LOCK = RLock()


def serialized_plot(function: Callable[P, R]) -> Callable[P, R]:
    """Serialize Matplotlib calls made by concurrent UI worker threads."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with PLOT_LOCK:
            return function(*args, **kwargs)

    return wrapped


__all__ = ["PLOT_LOCK", "serialized_plot"]
