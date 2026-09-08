"""NiceGUI entry point for Virtual-Staining."""

__all__ = ["main"]


def main() -> None:
    from virtual_staining.ui.app import main as run

    run()
