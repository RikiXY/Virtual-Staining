from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from virtual_staining.applications.compare_panels import FromMetricsResult
from virtual_staining.applications.complete_run import complete_run
from virtual_staining.applications.evaluate_single import DatasetEvalResult
from virtual_staining.applications.run_queue import load_local_run_queue, run_queue
from virtual_staining.cli import compare as compare_cli
from virtual_staining.cli import compare_panels as compare_panels_cli
from virtual_staining.cli import complete_run as complete_run_cli
from virtual_staining.cli import evaluate as evaluate_cli
from virtual_staining.cli import evaluate_single as evaluate_single_cli
from virtual_staining.cli import infer as infer_cli
from virtual_staining.cli import organize as organize_cli
from virtual_staining.cli import prepare_dataset as prepare_cli
from virtual_staining.cli import run_queue as run_queue_cli
from virtual_staining.cli import train as train_cli
from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.statistics import PairedSummary

UV = ("uv", "run")
CLI_ENTRYPOINTS = (
    "vs-prepare",
    "vs-complete-run",
    "vs-train",
    "vs-infer",
    "vs-evaluate",
)
MAKE_EXPERIMENT_TARGETS = ("dataset", "train", "infer", "evaluate", "complete-run")


def _write_metrics_csv(run_path: Path) -> Path:
    evaluation_dir = run_path / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evaluation_dir / "per_image_metrics.csv"
    csv_path.write_text("sample_id,ssim\nsample_001,0.9\n", encoding="utf-8")
    return csv_path


def _write_config(tmp_path: Path, section_yaml: str) -> Path:
    config_path = tmp_path / "run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    section = textwrap.dedent(section_yaml).strip()
    config_path.write_text(
        (
            f"dataset_root: {tmp_path / 'data'}\n"
            f"results_path: {tmp_path / 'results'}\n"
            "run_name: current_run\n"
            f"{section}\n"
        ),
        encoding="utf-8",
    )
    return config_path


def _run_subprocess(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, env=env, check=False)


def _write_queue(tmp_path: Path, jobs_yaml: str, *, continue_on_failure: bool = False) -> Path:
    queue_path = tmp_path / "config" / "queues" / "nightly.yaml"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        (
            "name: nightly\n"
            f"continue_on_failure: {'true' if continue_on_failure else 'false'}\n"
            "jobs:\n"
            f"{textwrap.dedent(jobs_yaml)}"
        ),
        encoding="utf-8",
    )
    return queue_path


@pytest.mark.parametrize("cmd", CLI_ENTRYPOINTS)
def test_cli_help_includes_config_flag(cmd: str) -> None:
    result = _run_subprocess(*UV, cmd, "--help")
    assert result.returncode == 0, f"{cmd} --help failed: {result.stderr}"
    assert "--config" in result.stdout, f"{cmd} --help does not mention --config"


@pytest.mark.parametrize("cmd", CLI_ENTRYPOINTS)
def test_cli_exits_nonzero_without_config(cmd: str) -> None:
    result = _run_subprocess(*UV, cmd)
    assert result.returncode != 0, f"{cmd} succeeded without --config"


@pytest.mark.parametrize("target", MAKE_EXPERIMENT_TARGETS)
def test_make_target_fails_without_config(target: str) -> None:
    env = {**os.environ, "CONFIG": ""}
    result = _run_subprocess("make", target, env=env)
    assert result.returncode != 0, (
        f"make {target} succeeded without CONFIG; require-config guard is missing"
    )


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
    result = _run_subprocess(*UV, "vs-run-queue", "--help")
    assert result.returncode == 0, f"vs-run-queue --help failed: {result.stderr}"
    assert "--queue" in result.stdout


def test_run_queue_exits_nonzero_without_queue() -> None:
    result = _run_subprocess(*UV, "vs-run-queue")
    assert result.returncode != 0


def test_make_run_queue_fails_without_queue() -> None:
    env = {**os.environ, "QUEUE": ""}
    result = _run_subprocess("make", "run-queue", env=env)
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


def test_compare_help_includes_config() -> None:
    assert "--config" in compare_cli._build_parser().format_help()


def test_compare_main_with_config_invokes_compare(
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

    request = captured["request"]
    assert request.csv_a == current_run / "evaluation" / "per_image_metrics.csv"  # type: ignore[attr-defined]
    assert request.csv_b == baseline_run / "evaluation" / "per_image_metrics.csv"  # type: ignore[attr-defined]
    assert request.mode == "paired"  # type: ignore[attr-defined]


def test_compare_panels_help_includes_config() -> None:
    assert "--config" in compare_panels_cli._build_parser().format_help()


def test_compare_panels_help_uses_artifacts_output_test_path() -> None:
    help_text = compare_panels_cli._build_parser().format_help()
    assert "artifacts/output_test" in help_text
    assert "results/your_run/output_test" not in help_text


def test_compare_panels_main_with_config_invokes_compare_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        compare_panels:
          mode: from_metrics
          hide_graphs_path: true
        """,
    )

    captured: dict[str, object] = {}

    def _fake_compare_panels(request: object) -> object:
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

    monkeypatch.setattr(compare_panels_cli, "compare_panels", _fake_compare_panels)

    compare_panels_cli.main(["--config", str(config_path)])

    request = captured["request"]
    assert request.mode == "from_metrics"  # type: ignore[attr-defined]
    assert request.run_path == tmp_path / "results" / "current_run"  # type: ignore[attr-defined]


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

    captured: dict[str, object] = {}

    def _fake_evaluate_single(request: object) -> object:
        captured["request"] = request
        output_dir = request.output_dir  # type: ignore[attr-defined]
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

    monkeypatch.setattr(evaluate_single_cli, "evaluate_single", _fake_evaluate_single)

    evaluate_single_cli.main(["--config", str(config_path)])

    request = captured["request"]
    assert request.sample_id is None  # type: ignore[attr-defined]
    assert request.save_graphs is True  # type: ignore[attr-defined]
    assert request.output_dir == tmp_path / "results" / "current_run" / "evaluation"  # type: ignore[attr-defined]


def test_organize_help_includes_config() -> None:
    assert "--config" in organize_cli._build_parser().format_help()


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

    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.prepare",
        lambda *_: calls.append("prepare"),
    )
    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.train",
        lambda *_: calls.append("train"),
    )
    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.infer",
        lambda *_: calls.append("infer"),
    )
    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.evaluate",
        lambda *_: calls.append("evaluate"),
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

    def _fail_prepare(*_args: object) -> None:
        calls.append("prepare")
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.prepare",
        _fail_prepare,
    )
    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.train",
        lambda *_: calls.append("train"),
    )
    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.infer",
        lambda *_: calls.append("infer"),
    )
    monkeypatch.setattr(
        "virtual_staining.applications.complete_run.evaluate",
        lambda *_: calls.append("evaluate"),
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        complete_run(config, config_path)

    assert calls == ["prepare"]


def test_run_queue_main_passes_queue_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        training:
          epochs: 1
        """,
    )
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_path}
        """,
    )
    captured: dict[str, object] = {}

    def _fake_run_queue(incoming_path: Path) -> object:
        captured["queue_path"] = incoming_path
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(run_queue_cli, "run_queue", _fake_run_queue)

    run_queue_cli.main(["--queue", str(queue_path)])

    assert captured["queue_path"] == queue_path.resolve()


def test_load_local_run_queue_resolves_relative_job_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "job.yaml"
    config_path.write_text(
        "dataset_root: /tmp/data\nresults_path: /tmp/results\nrun_name: demo\n",
        encoding="utf-8",
    )
    queue_path = _write_queue(
        tmp_path,
        """\
          - config_path: ../../configs/job.yaml
            label: baseline
            notes: first run
        """,
    )

    queue = load_local_run_queue(queue_path)

    assert queue.name == "nightly"
    assert queue.continue_on_failure is False
    assert queue.jobs[0].config_path == config_path.resolve()
    assert queue.jobs[0].label == "baseline"
    assert queue.jobs[0].notes == "first run"


def test_load_local_run_queue_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    queue_path = tmp_path / "config" / "queues" / "nightly.yaml"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        (
            "name: nightly\n"
            "continue_on_failure: false\n"
            "unexpected: true\n"
            "jobs:\n"
            "  - config_path: ../runs/local/run_a.yaml\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown key\(s\) in queue: unexpected"):
        load_local_run_queue(queue_path)


def test_load_local_run_queue_rejects_unknown_job_keys(tmp_path: Path) -> None:
    queue_path = tmp_path / "config" / "queues" / "nightly.yaml"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        (
            "name: nightly\n"
            "continue_on_failure: false\n"
            "jobs:\n"
            "  - config_path: ../runs/local/run_a.yaml\n"
            "    unexpected: true\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown key\(s\) in queue\.jobs\[0\]: unexpected"):
        load_local_run_queue(queue_path)


def test_load_local_run_queue_requires_yaml_boolean_for_continue_on_failure(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "config" / "queues" / "nightly.yaml"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        (
            "name: nightly\n"
            'continue_on_failure: "false"\n'
            "jobs:\n"
            "  - config_path: ../runs/local/run_a.yaml\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="continue_on_failure"):
        load_local_run_queue(queue_path)


def test_run_queue_executes_jobs_in_order_and_persists_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    config_b = _write_config(tmp_path / "b", "training:\n  epochs: 1\n")
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
            label: first
          - config_path: {config_b}
            label: second
        """,
        continue_on_failure=False,
    )
    calls: list[Path] = []

    def _fake_complete_run(config: RunConfig, config_path: Path) -> None:
        del config
        calls.append(config_path)

    monkeypatch.setattr("virtual_staining.applications.run_queue.complete_run", _fake_complete_run)

    state = run_queue(queue_path)
    state_path = tmp_path / "local_workspace" / "queues" / "nightly.state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))

    assert calls == [config_a.resolve(), config_b.resolve()]
    assert state.status == "completed"
    assert state_data["status"] == "completed"
    assert [job["status"] for job in state_data["jobs"]] == ["completed", "completed"]
    assert state_path.exists()


def test_run_queue_stops_on_failure_when_continue_on_failure_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    config_b = _write_config(tmp_path / "b", "training:\n  epochs: 1\n")
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
          - config_path: {config_b}
        """,
        continue_on_failure=False,
    )
    calls: list[Path] = []

    def _fake_complete_run(config: RunConfig, config_path: Path) -> None:
        del config
        calls.append(config_path)
        raise RuntimeError("boom")

    monkeypatch.setattr("virtual_staining.applications.run_queue.complete_run", _fake_complete_run)

    state = run_queue(queue_path)
    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "nightly.state.json").read_text(encoding="utf-8")
    )

    assert calls == [config_a.resolve()]
    assert state.status == "failed"
    assert [job["status"] for job in state_data["jobs"]] == ["failed", "pending"]
    assert state_data["jobs"][0]["error"] == "boom"


def test_run_queue_continues_after_failure_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    config_b = _write_config(tmp_path / "b", "training:\n  epochs: 1\n")
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
          - config_path: {config_b}
        """,
        continue_on_failure=True,
    )
    calls: list[Path] = []

    def _fake_complete_run(config: RunConfig, config_path: Path) -> None:
        del config
        calls.append(config_path)
        if config_path == config_a.resolve():
            raise RuntimeError("boom")

    monkeypatch.setattr("virtual_staining.applications.run_queue.complete_run", _fake_complete_run)

    state = run_queue(queue_path)
    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "nightly.state.json").read_text(encoding="utf-8")
    )

    assert calls == [config_a.resolve(), config_b.resolve()]
    assert state.status == "failed"
    assert [job["status"] for job in state_data["jobs"]] == ["failed", "completed"]
    assert state_data["continue_on_failure"] is True
