from pathlib import Path

import pytest

import virtual_staining.cli.ui as ui_cli


def test_ui_cli_passes_explicit_portable_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_ui(
        checkpoint_directory: Path,
        output_directory: Path,
        results_directory: Path,
        *,
        host: str,
        port: int,
    ) -> None:
        captured.update(
            checkpoint_directory=checkpoint_directory,
            output_directory=output_directory,
            results_directory=results_directory,
            host=host,
            port=port,
        )

    monkeypatch.setattr(ui_cli, "run_ui", fake_run_ui)

    ui_cli.main(
        [
            "--checkpoint-dir",
            "elsewhere/models",
            "--output-dir",
            "elsewhere/results",
            "--results-dir",
            "elsewhere/runs",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]
    )

    assert captured == {
        "checkpoint_directory": Path("elsewhere/models"),
        "output_directory": Path("elsewhere/results"),
        "results_directory": Path("elsewhere/runs"),
        "host": "127.0.0.1",
        "port": 9000,
    }


def test_ui_cli_uses_environment_directory_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[Path, Path, Path]] = []
    monkeypatch.setenv(ui_cli.CHECKPOINT_DIRECTORY_ENV, "/portable/models")
    monkeypatch.setenv(ui_cli.OUTPUT_DIRECTORY_ENV, "/portable/results")
    monkeypatch.setenv(ui_cli.RESULTS_DIRECTORY_ENV, "/portable/runs")
    monkeypatch.setattr(
        ui_cli,
        "run_ui",
        lambda checkpoint_directory, output_directory, results_directory, **_kwargs: (
            captured.append((checkpoint_directory, output_directory, results_directory))
        ),
    )

    ui_cli.main([])

    assert captured == [
        (Path("/portable/models"), Path("/portable/results"), Path("/portable/runs"))
    ]
