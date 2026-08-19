from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.config_helpers import write_run_config
from virtual_staining.applications import pipeline
from virtual_staining.applications.evaluate_single import DatasetEvalResult
from virtual_staining.cli import compare as compare_cli
from virtual_staining.cli import compare_panels as compare_panels_cli
from virtual_staining.cli import complete_run as complete_run_cli
from virtual_staining.cli import evaluate as evaluate_cli
from virtual_staining.cli import evaluate_single as evaluate_single_cli
from virtual_staining.cli import infer as infer_cli
from virtual_staining.cli import infer_images as infer_images_cli
from virtual_staining.cli import organize as organize_cli
from virtual_staining.cli import prepare_dataset as prepare_cli
from virtual_staining.cli import run_queue as run_queue_cli
from virtual_staining.cli import train as train_cli
from virtual_staining.config.run import RunConfig

CLI_SCRIPTS = {
    "vs-prepare": "virtual_staining.cli.prepare_dataset:main",
    "vs-complete-run": "virtual_staining.cli.complete_run:main",
    "vs-run-queue": "virtual_staining.cli.run_queue:main",
    "vs-train": "virtual_staining.cli.train:main",
    "vs-infer": "virtual_staining.cli.infer:main",
    "vs-infer-images": "virtual_staining.cli.infer_images:main",
    "vs-evaluate": "virtual_staining.cli.evaluate:main",
    "vs-compare": "virtual_staining.cli.compare:main",
    "vs-evaluate-single": "virtual_staining.cli.evaluate_single:main",
    "vs-compare-panels": "virtual_staining.cli.compare_panels:main",
    "vs-organize": "virtual_staining.cli.organize:main",
}
RUN_CONFIG_CLI_MODULES = (
    prepare_cli,
    complete_run_cli,
    train_cli,
    infer_cli,
    infer_images_cli,
    evaluate_cli,
)
MAKE_EXPERIMENT_TARGETS = ("dataset", "train", "infer", "evaluate", "complete-run")


def _write_metrics_csv(run_path: Path) -> Path:
    evaluation_dir = run_path / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evaluation_dir / "per_image_metrics.csv"
    csv_path.write_text("sample_id,ssim\nsample_001,0.9\n", encoding="utf-8")
    return csv_path


def _write_config(tmp_path: Path, section_yaml: str) -> Path:
    return write_run_config(tmp_path, section_yaml)


def _run_command(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, env=env, check=False)


def test_pyproject_console_scripts_point_to_cli_mains() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == CLI_SCRIPTS


@pytest.mark.parametrize("cli_module", RUN_CONFIG_CLI_MODULES)
def test_run_config_cli_help_includes_config_flag(cli_module: object) -> None:
    assert "--config" in cli_module._build_parser().format_help()  # type: ignore[attr-defined]


@pytest.mark.parametrize("cli_module", RUN_CONFIG_CLI_MODULES)
def test_run_config_clis_exit_nonzero_without_config(cli_module: object) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module.main([])  # type: ignore[attr-defined]

    assert exc.value.code != 0


@pytest.mark.parametrize("target", MAKE_EXPERIMENT_TARGETS)
def test_make_target_fails_without_config(target: str) -> None:
    env = {**os.environ, "CONFIG": ""}
    result = _run_command("make", target, env=env)
    assert result.returncode != 0, (
        f"make {target} succeeded without CONFIG; require-config guard is missing"
    )


def test_make_infer_images_fails_without_input_path(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        inference:
          checkpoint_path: /tmp/ep000.pth
        """,
    )
    env = {**os.environ, "CONFIG": str(config_path), "INPUT_PATH": ""}
    result = _run_command("make", "infer-images", env=env)

    assert result.returncode != 0
    assert "INPUT_PATH is required" in result.stdout + result.stderr


def test_makefile_has_infer_images_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "infer-images: require-config require-input-path" in makefile
    assert "$(UV) run vs-infer-images --config $(CONFIG) --input $(INPUT_PATH)" in makefile


def test_makefile_has_no_args_variable() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "$(ARGS)" not in makefile, "Makefile still references $(ARGS)"


def test_make_complete_run_delegates_to_complete_run_cli() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "$(UV) run vs-complete-run --config $(CONFIG)" in makefile
    assert "$(MAKE) dataset CONFIG=$(CONFIG)" not in makefile


def test_example_yaml_is_valid_mapping() -> None:
    config_path = Path("config/runs/example.yaml")
    assert config_path.exists(), "config/runs/example.yaml is missing"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "config/runs/example.yaml is not a YAML mapping"


def test_example_queue_yaml_is_valid_mapping() -> None:
    queue_path = Path("config/queues/example.yaml")
    assert queue_path.exists(), "config/queues/example.yaml is missing"
    data = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "config/queues/example.yaml is not a YAML mapping"


def test_readme_examples_use_local_run_configs() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "config/runs/local/" in readme
    assert "ARGS=" not in readme
    assert "CONFIG=config/runs/example.yaml" not in readme
    assert "QUEUE=config/queues/example.yaml" in readme


def test_run_queue_help_includes_queue_flag() -> None:
    assert "--queue" in run_queue_cli._build_parser().format_help()


def test_run_queue_exits_nonzero_without_queue() -> None:
    with pytest.raises(SystemExit) as exc:
        run_queue_cli.main([])

    assert exc.value.code != 0


def test_make_run_queue_fails_without_queue() -> None:
    env = {**os.environ, "QUEUE": ""}
    result = _run_command("make", "run-queue", env=env)
    assert result.returncode != 0


@pytest.mark.parametrize(
    ("main_func", "argv"),
    [
        (compare_cli.main, []),
        (compare_panels_cli.main, []),
        (evaluate_single_cli.main, []),
        (organize_cli.main, []),
    ],
)
def test_utility_clis_without_args_exit_non_zero(main_func: object, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main_func(argv)  # type: ignore[misc]
    assert exc.value.code != 0


def test_utility_help_omits_config() -> None:
    for module in (compare_cli, compare_panels_cli, organize_cli):
        assert "--config" not in module._build_parser().format_help()


def test_compare_panels_help_uses_artifacts_output_test_path() -> None:
    help_text = compare_panels_cli._build_parser().format_help()
    assert "artifacts/output_test" in help_text
    assert "results/your_run/output_test" not in help_text


def test_evaluate_single_help_includes_config() -> None:
    assert "--config" in evaluate_single_cli._build_parser().format_help()


def test_evaluate_single_help_uses_artifacts_output_test_path() -> None:
    help_text = evaluate_single_cli._build_parser().format_help()
    assert "artifacts/output_test" in help_text
    assert "results/your_run/output_test" not in help_text


def test_evaluate_single_main_with_config_invokes_dataset_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        evaluation:
          save_graphs: true
        """,
    )

    captured: list[Path] = []

    def _fake_evaluate_dataset(path: Path) -> object:
        captured.append(path)
        output_dir = tmp_path / "results" / "current_run" / "evaluation"
        return DatasetEvalResult(
            target_files={},
            generated_files={},
            per_image_rows=[],
            skipped_rows=[],
            output_dir=output_dir,
            per_image_csv=output_dir / "per_image_metrics.csv",
            summary_csv=output_dir / "summary.csv",
            skipped_csv=output_dir / "skipped.csv",
            plot_paths=[],
        )

    monkeypatch.setattr(evaluate_single_cli, "evaluate_dataset", _fake_evaluate_dataset)

    evaluate_single_cli.main(["--config", str(config_path)])

    assert captured == [config_path]


def test_train_main_calls_train_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        training:
          epochs: 1
          losses: {}
        """,
    )
    captured: list[tuple[Path, str]] = []
    monkeypatch.setattr(train_cli, "run_stage", lambda path, stage: captured.append((path, stage)))

    train_cli.main(["--config", str(config_path)])

    assert captured == [(config_path.resolve(), "train")]


def test_infer_main_calls_infer_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        inference:
          checkpoint_path: /tmp/ep000.pth
        """,
    )
    captured: list[tuple[Path, str]] = []
    monkeypatch.setattr(infer_cli, "run_stage", lambda path, stage: captured.append((path, stage)))

    infer_cli.main(["--config", str(config_path)])

    assert captured == [(config_path.resolve(), "infer")]


def test_infer_images_main_passes_path_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        inference:
          checkpoint_path: /tmp/ep000.pth
        """,
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    captured: dict[str, object] = {}

    def _fake_infer_images(
        incoming_config_path: Path,
        incoming_input: Path,
        incoming_output: Path | None = None,
        *,
        recursive: bool = False,
        mode: str = "auto",
        tile_overlap: int = 16,
        output_format: str = "same",
    ) -> object:
        captured["config_path"] = incoming_config_path
        captured["input_path"] = incoming_input
        captured["output_path"] = incoming_output
        captured["recursive"] = recursive
        captured["mode"] = mode
        captured["tile_overlap"] = tile_overlap
        captured["output_format"] = output_format
        return SimpleNamespace(
            input_path=incoming_input,
            checkpoint_path=Path("/tmp/ep000.pth"),
            output_path=incoming_output,
            mode=mode,
        )

    monkeypatch.setattr(infer_images_cli, "infer_images", _fake_infer_images)

    infer_images_cli.main(
        [
            "--config",
            str(config_path),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--recursive",
            "--mode",
            "tile",
            "--tile-overlap",
            "8",
            "--output-format",
            "png",
        ]
    )

    assert captured["config_path"] == config_path.resolve()
    assert captured["input_path"] == input_dir
    assert captured["output_path"] == output_dir
    assert captured["recursive"] is True
    assert captured["mode"] == "tile"
    assert captured["tile_overlap"] == 8
    assert captured["output_format"] == "png"


def test_evaluate_main_calls_evaluate_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        evaluation:
          save_graphs: false
        """,
    )
    captured: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        evaluate_cli, "run_stage", lambda path, stage: captured.append((path, stage))
    )

    evaluate_cli.main(["--config", str(config_path)])

    assert captured == [(config_path.resolve(), "evaluate")]


def test_prepare_main_calls_prepare_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        preprocessing:
          source_name: source.png
          target_name: target.png
        """,
    )
    captured: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        prepare_cli, "run_stage", lambda path, stage: captured.append((path, stage))
    )

    prepare_cli.main(["--config", str(config_path)])

    assert captured == [(config_path.resolve(), "prepare")]


def test_complete_run_main_calls_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        preprocessing:
          source_name: source.png
          target_name: target.png
        training:
          epochs: 1
          losses: {}
        inference:
          checkpoint_path: /tmp/ep000.pth
        evaluation:
          save_graphs: false
        """,
    )
    captured: list[Path] = []
    monkeypatch.setattr(complete_run_cli, "run_stages", lambda path: captured.append(path))

    complete_run_cli.main(["--config", str(config_path)])

    assert captured == [config_path.resolve()]


def test_complete_run_executes_stages_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        preprocessing:
          source_name: source.png
          target_name: target.png
        training:
          epochs: 1
        inference:
          checkpoint_path: /tmp/ep000.pth
        evaluation:
          save_graphs: false
        """,
    )
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "prepare", lambda config, path: calls.append("prepare"))
    monkeypatch.setattr(pipeline, "run_training", lambda config, path: calls.append("train"))
    monkeypatch.setattr(pipeline, "run_inference", lambda config, path: calls.append("infer"))
    monkeypatch.setattr(pipeline, "evaluate", lambda config, path: calls.append("evaluate"))

    pipeline.run_stages(config_path)

    assert calls == ["prepare", "train", "infer", "evaluate"]


def test_complete_run_stops_on_first_failing_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        preprocessing:
          source_name: source.png
          target_name: target.png
        training:
          epochs: 1
        inference:
          checkpoint_path: /tmp/ep000.pth
        evaluation:
          save_graphs: false
        """,
    )
    calls: list[str] = []

    def _fail(config: RunConfig, path: Path) -> None:
        del config, path
        calls.append("prepare")
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(pipeline, "prepare", _fail)
    monkeypatch.setattr(pipeline, "run_training", lambda config, path: calls.append("train"))

    with pytest.raises(RuntimeError, match="prepare failed"):
        pipeline.run_stages(config_path)

    assert calls == ["prepare"]
