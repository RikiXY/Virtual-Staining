from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.config_helpers import write_run_config
from virtual_staining import cli
from virtual_staining.applications.evaluate_single import SingleEvalResult
from virtual_staining.cli import compare, compare_panels, evaluate, infer_images, organize

COMMANDS = (
    "prepare",
    "run",
    "train",
    "infer",
    "infer-images",
    "evaluate",
    "compare",
    "convert",
    "panels",
    "organize",
    "queue",
    "status",
)


def _write_config(tmp_path: Path, section_yaml: str = "") -> Path:
    return write_run_config(tmp_path, section_yaml)


def test_pyproject_publishes_only_vs() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == {"vs": "virtual_staining.cli:main"}


def test_root_help_lists_the_public_commands() -> None:
    help_text = cli._build_parser().format_help()
    assert all(command in help_text for command in COMMANDS)


@pytest.mark.parametrize("command", ("prepare", "run", "train", "infer"))
def test_config_commands_require_config(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([command])
    assert exc.value.code != 0


@pytest.mark.parametrize("command", ("prepare", "train", "infer"))
def test_stage_commands_dispatch(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    captured: list[tuple[Path, str]] = []
    reporters: dict[str, object] = {}

    def fake_run_stage(path: Path, stage: str, **kwargs: object) -> None:
        captured.append((path, stage))
        reporters[stage] = kwargs["progress_reporter"]

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)

    cli.main([command, "--config", str(config_path)])

    assert captured == [(config_path.resolve(), command)]
    assert reporters[command] is (cli.render_training_progress if command == "train" else None)


def test_run_defaults_to_complete_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path)
    captured: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        cli,
        "run_stages",
        lambda path, **kwargs: captured.append((path, kwargs.get("stages"))),
    )

    cli.main(["run", "--config", str(config_path)])

    assert captured == [(config_path.resolve(), None)]


def test_run_passes_selected_stages_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    captured: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        cli,
        "run_stages",
        lambda path, stages, **kwargs: captured.append((path, tuple(stages))),
    )
    cli.main(["run", "--config", str(config_path), "--stages", "train", "infer", "evaluate"])

    assert captured == [(config_path.resolve(), ("train", "infer", "evaluate"))]


def test_run_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--config", str(_write_config(tmp_path)), "--stages", "publish"])
    assert exc.value.code != 0


def test_queue_dispatches_resolved_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue_path = tmp_path / "queue.yaml"
    captured: list[Path] = []

    def fake_run_queue(path: Path, **kwargs: object) -> object:
        captured.append(path)
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(cli, "run_queue", fake_run_queue)
    cli.main(["queue", "--queue", str(queue_path)])

    assert captured == [queue_path.resolve()]


def test_queue_returns_nonzero_for_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_queue", lambda path, **kwargs: SimpleNamespace(status="failed"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["queue", "--queue", str(tmp_path / "queue.yaml")])
    assert exc.value.code == 1


def test_evaluate_config_dispatches_pipeline_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    captured: list[tuple[Path, str]] = []
    monkeypatch.setattr(evaluate, "run_stage", lambda path, stage: captured.append((path, stage)))

    cli.main(["evaluate", "--config", str(config_path)])

    assert captured == [(config_path.resolve(), "evaluate")]


def test_evaluate_pair_dispatches_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "sample_target.png"
    generated = tmp_path / "sample_target_generated.png"
    output_dir = tmp_path / "evaluation"
    captured: list[tuple[Path, Path, Path | None]] = []

    def fake_evaluate_pair(a: Path, b: Path, output: Path | None) -> SingleEvalResult:
        captured.append((a, b, output))
        return SingleEvalResult(
            target=a,
            generated=b,
            metrics={
                "mae": 0.1,
                "mse": 0.01,
                "rmse": 0.1,
                "psnr": 20.0,
                "ssim": 0.9,
                "pcc_gray": 0.9,
                "pcc_rgb_mean": 0.9,
            },
            shape=(16, 16, 3),
            single_case_csv=output_dir / "sample.csv",
        )

    monkeypatch.setattr(evaluate, "evaluate_pair", fake_evaluate_pair)
    cli.main(
        [
            "evaluate",
            "--pair",
            str(target),
            str(generated),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert captured == [(target, generated, output_dir)]


def test_evaluate_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        cli.main(["evaluate"])
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--config", "run.yaml", "--pair", "a.png", "b.png"])


def test_infer_images_passes_repeated_named_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}

    def fake_infer_images(
        config: Path, input_specs: tuple[str, ...], output_path: Path | None, **kwargs: object
    ) -> object:
        captured.update(config=config, input=input_specs, output=output_path, **kwargs)
        return SimpleNamespace(
            input_paths={"LF": Path("lf.png"), "AF": Path("af.png")},
            checkpoint_path=Path("checkpoint.pth"),
            output_path=output_path,
            mode=kwargs["mode"],
        )

    monkeypatch.setattr(infer_images, "infer_images", fake_infer_images)
    cli.main(
        [
            "infer-images",
            "--config",
            str(config_path),
            "--input",
            "AF=af.png",
            "--input",
            "LF=lf.png",
            "--output",
            str(tmp_path / "output.png"),
            "--mode",
            "tile",
        ]
    )

    assert captured["config"] == config_path.resolve()
    assert captured["input"] == ("AF=af.png", "LF=lf.png")
    assert captured["mode"] == "tile"


@pytest.mark.parametrize(
    ("main", "argv"),
    [
        (compare.main, []),
        (compare_panels.main, []),
        (organize.main, []),
    ],
)
def test_utility_commands_without_args_fail(main: object, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)  # type: ignore[misc]
    assert exc.value.code != 0


def test_makefile_exposes_only_development_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert ".PHONY: sync format lint typecheck test qa clean" in makefile
    assert "vs-" not in makefile
    assert "CONFIG" not in makefile
