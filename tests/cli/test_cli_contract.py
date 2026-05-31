from __future__ import annotations

import argparse
import os
import subprocess
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.config_helpers import write_run_config
from virtual_staining.applications.complete_run import complete_run
from virtual_staining.applications.evaluate_single import SingleEvalResult
from virtual_staining.applications.render_panels import FromMetricsResult
from virtual_staining.cli import compare as compare_cli
from virtual_staining.cli import complete_run as complete_run_cli
from virtual_staining.cli import evaluate as evaluate_cli
from virtual_staining.cli import evaluate_single as evaluate_single_cli
from virtual_staining.cli import infer as infer_cli
from virtual_staining.cli import infer_images as infer_images_cli
from virtual_staining.cli import organize as organize_cli
from virtual_staining.cli import prepare_dataset as prepare_cli
from virtual_staining.cli import render_panels as render_panels_cli
from virtual_staining.cli import run_queue as run_queue_cli
from virtual_staining.cli import train as train_cli
from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.statistics import PairedSummary

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
    "vs-render-panels": "virtual_staining.cli.render_panels:main",
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


def test_make_render_panels_delegates_to_render_panels_cli() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "render-panels: require-config" in makefile
    assert "$(UV) run vs-render-panels --config $(CONFIG)" in makefile
    assert "-".join(("compare", "panels:")) not in makefile
    assert "vs-" + "-".join(("compare", "panels")) not in makefile


def test_make_evaluate_single_fails_without_target_image() -> None:
    env = {**os.environ, "TARGET_IMAGE": "", "GENERATED_IMAGE": ""}
    result = _run_command("make", "evaluate-single", env=env)

    assert result.returncode != 0
    assert "TARGET_IMAGE is required" in result.stdout + result.stderr


def test_make_evaluate_single_fails_without_generated_image(tmp_path: Path) -> None:
    target_path = tmp_path / "00001_target.png"
    target_path.write_text("not an actual image", encoding="utf-8")
    env = {**os.environ, "TARGET_IMAGE": str(target_path), "GENERATED_IMAGE": ""}
    result = _run_command("make", "evaluate-single", env=env)

    assert result.returncode != 0
    assert "GENERATED_IMAGE is required" in result.stdout + result.stderr


def test_make_evaluate_single_delegates_to_flat_cli_with_image_vars(tmp_path: Path) -> None:
    target_path = tmp_path / "00001_target.png"
    generated_path = tmp_path / "00001_target_generated.png"
    output_dir = tmp_path / "evaluation"
    target_path.write_text("not an actual image", encoding="utf-8")
    generated_path.write_text("not an actual image", encoding="utf-8")
    env = {
        **os.environ,
        "UV": "true",
        "TARGET_IMAGE": str(target_path),
        "GENERATED_IMAGE": str(generated_path),
        "OUTPUT_DIR": str(output_dir),
    }
    result = _run_command("make", "evaluate-single", env=env)

    assert result.returncode == 0
    assert "true run vs-evaluate-single" in result.stdout
    assert f'--target-image "{target_path}"' in result.stdout
    assert f'--generated-image "{generated_path}"' in result.stdout
    assert f'--output-dir "{output_dir}"' in result.stdout
    assert "--config" not in result.stdout


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
        (render_panels_cli.main, []),
        (evaluate_single_cli.main, []),
        (organize_cli.main, []),
    ],
)
def test_utility_clis_without_args_exit_non_zero(main_func: object, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main_func(argv)  # type: ignore[misc]
    assert exc.value.code != 0


def test_compare_help_includes_config() -> None:
    assert "--config" in compare_cli._build_parser().format_help()


def test_compare_main_with_config_invokes_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    current_run = tmp_path / "results" / "current_run"
    baseline_run = tmp_path / "results" / "baseline_run"
    _write_metrics_csv(current_run)
    _write_metrics_csv(baseline_run)
    config_path = _write_config(
        tmp_path,
        f"""\
        compare:
          mode: paired
          run_b: {baseline_run}
          column: ssim
        """,
    )

    captured: dict[str, object] = {}

    def _fake_compare(request: object) -> object:
        captured["request"] = request
        return SimpleNamespace(
            mode="paired",
            output_dir=request.output_dir,  # type: ignore[attr-defined]
            paired_summary=PairedSummary(
                label_a="current_run",
                label_b="baseline_run",
                n_pairs=1,
                tolerance=0.0,
                mean_signed_delta=0.0,
                median_signed_delta=0.0,
                share_b_better=0.0,
                share_a_better=0.0,
                share_equal=1.0,
                wilcoxon_statistic=0.0,
                wilcoxon_pvalue=1.0,
                better_label="tie",
            ),
        )

    monkeypatch.setattr(compare_cli, "compare", _fake_compare)

    compare_cli.main(["--config", str(config_path)])
    output = capsys.readouterr().out

    request = captured["request"]
    assert request.csv_a == current_run / "evaluation" / "per_image_metrics.csv"  # type: ignore[attr-defined]
    assert request.csv_b == baseline_run / "evaluation" / "per_image_metrics.csv"  # type: ignore[attr-defined]
    assert request.mode == "paired"  # type: ignore[attr-defined]
    assert "Score-based suggestion" in output


def test_compare_main_with_config_passes_multi_metric_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_run = tmp_path / "results" / "current_run"
    baseline_run = tmp_path / "results" / "baseline_run"
    _write_metrics_csv(current_run)
    _write_metrics_csv(baseline_run)
    config_path = _write_config(
        tmp_path,
        f"""\
        compare:
          mode: paired
          run_b: {baseline_run}
          column: all
          metrics: [ssim, mae]
          tolerance: 0.001
        """,
    )

    captured: dict[str, object] = {}

    def _fake_compare(request: object) -> object:
        captured["request"] = request
        return SimpleNamespace(
            mode="paired",
            output_dir=request.output_dir,  # type: ignore[attr-defined]
            paired_summary=None,
            paired_sample_deltas_all_metrics_csv=(
                request.output_dir / "paired_sample_deltas_all_metrics.csv"  # type: ignore[attr-defined]
            ),
            paired_metric_delta_summary_csv=(
                request.output_dir / "paired_metric_delta_summary.csv"  # type: ignore[attr-defined]
            ),
        )

    monkeypatch.setattr(compare_cli, "compare", _fake_compare)

    compare_cli.main(["--config", str(config_path)])

    request = captured["request"]
    assert request.mode == "paired"  # type: ignore[attr-defined]
    assert request.column == "all"  # type: ignore[attr-defined]
    assert request.metrics == ("ssim", "mae")  # type: ignore[attr-defined]
    assert request.tolerance == pytest.approx(0.001)  # type: ignore[attr-defined]
    assert request.output_dir.name == "paired_all_metrics"  # type: ignore[attr-defined]


def test_render_panels_help_includes_config() -> None:
    assert "--config" in render_panels_cli._build_parser().format_help()


def test_render_panels_help_uses_artifacts_output_test_path() -> None:
    help_text = render_panels_cli._build_parser().format_help()
    assert "artifacts/output_test" in help_text
    assert "results/your_run/output_test" not in help_text


def test_render_panels_main_with_config_invokes_render_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        render_panels:
          mode: from_metrics
          hide_graphs_path: true
          top_k: 2
          metrics: [ssim, mae]
          kinds: [best, worst]
        """,
    )

    captured: dict[str, object] = {}

    def _fake_render_panels(request: object) -> object:
        captured["request"] = request
        return FromMetricsResult(
            run_path=request.run_path,  # type: ignore[attr-defined]
            available_metrics=["ssim"],
            per_metric_representative_rows={
                "ssim": {
                    "best": {"sample_id": "best", "ssim": "0.95"},
                    "median": {"sample_id": "median", "ssim": "0.50"},
                    "worst": {"sample_id": "worst", "ssim": "0.10"},
                }
            },
            saved_aggregated_paths=[],
            metrics_dir=(request.run_path / "comparisons" / "metrics"),  # type: ignore[attr-defined]
        )

    monkeypatch.setattr(render_panels_cli, "render_panels", _fake_render_panels)

    render_panels_cli.main(["--config", str(config_path)])

    request = captured["request"]
    assert request.mode == "from_metrics"  # type: ignore[attr-defined]
    assert request.run_path == tmp_path / "results" / "current_run"  # type: ignore[attr-defined]
    assert request.top_k == 2  # type: ignore[attr-defined]
    assert request.metrics == ("ssim", "mae")  # type: ignore[attr-defined]
    assert request.kinds == ("best", "worst")  # type: ignore[attr-defined]


def test_old_panel_config_section_is_rejected(tmp_path: Path) -> None:
    old_key = "_".join(("compare", "panels"))
    config_path = _write_config(
        tmp_path,
        f"""\
        {old_key}:
          mode: from_metrics
        """,
    )

    with pytest.raises(ValueError, match=f"{old_key}.*render_panels"):
        RunConfig.from_yaml(config_path)


def test_evaluate_single_help_is_single_pair_only() -> None:
    help_text = evaluate_single_cli._build_parser().format_help()

    assert "--config" not in help_text
    assert "{single}" not in help_text
    assert "--target-image" in help_text
    assert "--generated-image" in help_text
    assert "vs-evaluate-single dataset" not in help_text
    assert "vs-evaluate-single single" not in help_text


def test_evaluate_single_help_uses_artifacts_output_test_path() -> None:
    help_text = evaluate_single_cli._build_parser().format_help()
    assert "artifacts/output_test" in help_text
    assert "results/your_run/output_test" not in help_text


def test_evaluate_single_main_invokes_single_pair_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target_path = tmp_path / "datasets" / "run" / "splits" / "test" / "00512_09216_target.tif"
    generated_path = (
        tmp_path
        / "results"
        / "current_run"
        / "artifacts"
        / "output_test"
        / "00512_09216_target_generated.tif"
    )

    captured: dict[str, object] = {}

    def _fake_evaluate_single(request: object) -> object:
        captured["request"] = request
        return SingleEvalResult(
            target=target_path,
            generated=generated_path,
            metrics={
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "psnr": float("inf"),
                "ssim": 1.0,
                "pcc_gray": 1.0,
                "pcc_rgb_mean": 1.0,
            },
            shape=(8, 8, 3),
            single_case_csv=(
                request.output_dir  # type: ignore[attr-defined]
                / "individual_cases"
                / "00512_09216_evaluation.csv"
            ),
        )

    monkeypatch.setattr(evaluate_single_cli, "evaluate_single", _fake_evaluate_single)

    evaluate_single_cli.main(
        [
            "--target-image",
            str(target_path),
            "--generated-image",
            str(generated_path),
        ]
    )
    output = capsys.readouterr().out

    request = captured["request"]
    assert request.sample_id == "00512_09216"  # type: ignore[attr-defined]
    assert request.target_dir == target_path.parent  # type: ignore[attr-defined]
    assert request.generated_dir == generated_path.parent  # type: ignore[attr-defined]
    assert request.output_dir == tmp_path / "results" / "current_run" / "evaluation"  # type: ignore[attr-defined]
    assert "Single evaluation CSV" in output


def test_evaluate_single_build_request_rejects_mismatched_sample_ids(tmp_path: Path) -> None:
    args = argparse.Namespace(
        target=str(tmp_path / "00001_target.png"),
        generated=str(tmp_path / "00002_target_generated.png"),
        output_dir=str(tmp_path / "evaluation"),
    )

    with pytest.raises(ValueError, match="different sample ids"):
        evaluate_single_cli._build_request(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "config/runs/example.yaml"],
        ["dataset", "--config", "config/runs/example.yaml"],
    ],
)
def test_evaluate_single_removed_dataset_invocations_fail_clearly(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        evaluate_single_cli.main(argv)

    err = capsys.readouterr().err
    assert exc.value.code != 0
    assert "dataset/config mode was removed" in err
    assert "vs-evaluate --config" in err


def test_evaluate_single_removed_single_subcommand_fails_clearly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        evaluate_single_cli.main(["single", "--target-image", "a_target.png"])

    err = capsys.readouterr().err
    assert exc.value.code != 0
    assert "'single' subcommand was removed" in err
    assert "vs-evaluate-single --target-image" in err


def test_organize_help_includes_config() -> None:
    help_text = organize_cli._build_parser().format_help()

    assert "--config" in help_text
    assert "Export ranked generated, target, and source image files by metric" in help_text
    assert "This command places files only" in help_text
    assert "vs-render-panels" in help_text
    assert "RUN/evaluation/sorted_by_metrics/" in help_text


def test_organize_main_with_config_invokes_organize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_run = tmp_path / "results" / "current_run"
    _write_metrics_csv(current_run)
    config_path = _write_config(
        tmp_path,
        """\
        organize:
          top_k: 5
          mode: copy
          include_all_ranked: true
        """,
    )

    captured: dict[str, object] = {}

    def _fake_organize(request: object) -> None:
        captured["request"] = request

    monkeypatch.setattr(organize_cli, "organize", _fake_organize)

    organize_cli.main(["--config", str(config_path)])

    request = captured["request"]
    assert request.metrics_csv == current_run / "evaluation" / "per_image_metrics.csv"  # type: ignore[attr-defined]
    assert request.output_dir == current_run / "evaluation" / "sorted_by_metrics"  # type: ignore[attr-defined]
    assert request.run_path == current_run  # type: ignore[attr-defined]
    assert request.top_k == 5  # type: ignore[attr-defined]
    assert request.mode == "copy"  # type: ignore[attr-defined]
    assert request.include_all_ranked is True  # type: ignore[attr-defined]


def test_train_main_passes_full_run_config_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        training:
          epochs: 1
        """,
    )
    captured: dict[str, object] = {}

    def _fake_train(config: RunConfig, incoming_path: Path) -> None:
        captured["config"] = config
        captured["config_path"] = incoming_path

    monkeypatch.setattr(train_cli, "train", _fake_train)

    train_cli.main(["--config", str(config_path)])

    assert isinstance(captured["config"], RunConfig)
    assert captured["config_path"] == config_path.resolve()


def test_infer_main_passes_full_run_config_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        inference:
          checkpoint_path: /tmp/ep000.pth
        """,
    )
    captured: dict[str, object] = {}

    def _fake_infer(config: RunConfig, incoming_path: Path) -> None:
        captured["config"] = config
        captured["config_path"] = incoming_path

    monkeypatch.setattr(infer_cli, "infer", _fake_infer)

    infer_cli.main(["--config", str(config_path)])

    assert isinstance(captured["config"], RunConfig)
    assert captured["config_path"] == config_path.resolve()


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
        config: RunConfig,
        incoming_input: Path,
        incoming_output: Path | None = None,
        *,
        recursive: bool = False,
        mode: str = "auto",
        tile_overlap: int = 16,
        output_format: str = "same",
    ) -> object:
        captured["config"] = config
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

    assert isinstance(captured["config"], RunConfig)
    assert captured["input_path"] == input_dir
    assert captured["output_path"] == output_dir
    assert captured["recursive"] is True
    assert captured["mode"] == "tile"
    assert captured["tile_overlap"] == 8
    assert captured["output_format"] == "png"


def test_evaluate_main_passes_full_run_config_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        evaluation:
          save_graphs: false
        """,
    )
    captured: dict[str, object] = {}

    def _fake_evaluate(config: RunConfig, incoming_path: Path) -> None:
        captured["config"] = config
        captured["config_path"] = incoming_path

    monkeypatch.setattr(evaluate_cli, "evaluate", _fake_evaluate)

    evaluate_cli.main(["--config", str(config_path)])

    assert isinstance(captured["config"], RunConfig)
    assert captured["config_path"] == config_path.resolve()


def test_prepare_main_passes_full_run_config_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        preprocessing:
          source_name: source.png
          target_name: target.png
        """,
    )
    captured: dict[str, object] = {}

    def _fake_prepare(config: RunConfig, incoming_path: Path) -> None:
        captured["config"] = config
        captured["config_path"] = incoming_path

    monkeypatch.setattr(prepare_cli, "prepare", _fake_prepare)

    prepare_cli.main(["--config", str(config_path)])

    assert isinstance(captured["config"], RunConfig)
    assert captured["config_path"] == config_path.resolve()


def test_complete_run_main_passes_full_run_config_and_path(
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
    captured: dict[str, object] = {}

    def _fake_complete_run(config: RunConfig, incoming_path: Path) -> None:
        captured["config"] = config
        captured["config_path"] = incoming_path

    monkeypatch.setattr(complete_run_cli, "complete_run", _fake_complete_run)

    complete_run_cli.main(["--config", str(config_path)])

    assert isinstance(captured["config"], RunConfig)
    assert captured["config_path"] == config_path.resolve()


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
    config = RunConfig.from_yaml(config_path)
    calls: list[str] = []

    def _fake_run_stages(
        config_arg: RunConfig,
        config_path_arg: Path,
        stages: Sequence[str],
    ) -> None:
        assert config_arg is config
        assert config_path_arg == config_path
        calls.extend(stages)

    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.run_stages",
        _fake_run_stages,
    )

    complete_run(config, config_path)

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
    config = RunConfig.from_yaml(config_path)
    calls: list[str] = []

    def _fake_run_stages(
        config_arg: RunConfig,
        config_path_arg: Path,
        stages: Sequence[str],
    ) -> None:
        assert config_arg is config
        assert config_path_arg == config_path
        assert list(stages) == ["prepare", "train", "infer", "evaluate"]
        calls.append("prepare")
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.run_stages",
        _fake_run_stages,
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        complete_run(config, config_path)

    assert calls == ["prepare"]
