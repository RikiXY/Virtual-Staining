from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from virtual_staining.applications.compare_panels import FromMetricsResult
from virtual_staining.applications.evaluate_single import DatasetEvalResult
from virtual_staining.cli import compare as compare_cli
from virtual_staining.cli import compare_panels as compare_panels_cli
from virtual_staining.cli import evaluate as evaluate_cli
from virtual_staining.cli import evaluate_single as evaluate_single_cli
from virtual_staining.cli import infer as infer_cli
from virtual_staining.cli import organize as organize_cli
from virtual_staining.cli import prepare_dataset as prepare_cli
from virtual_staining.cli import train as train_cli
from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.statistics import PairedSummary


def _write_metrics_csv(run_path: Path) -> Path:
    evaluation_dir = run_path / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evaluation_dir / "per_image_metrics.csv"
    csv_path.write_text("sample_id,ssim\nsample_001,0.9\n", encoding="utf-8")
    return csv_path


def _write_config(tmp_path: Path, section_yaml: str) -> Path:
    config_path = tmp_path / "run.yaml"
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
