"""NiceGUI presentation adapter for Virtual-Staining."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from virtual_staining.ui.app import run_ui

__all__ = ["run_ui"]


def __getattr__(name: str) -> Any:
    if name == "run_ui":
        from virtual_staining.ui.app import run_ui

        return run_ui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
