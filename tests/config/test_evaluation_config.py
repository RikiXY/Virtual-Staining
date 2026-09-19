from __future__ import annotations

from pathlib import Path

import pytest

from tests.config_helpers import write_run_config
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.experiment.run_layout import RunLayout


def test_evaluation_config_defaults_to_run_dirs(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        "model:\n  inputs: [label_free]\n  target: stained\nevaluation:\n  save_graphs: false",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.evaluation is not None
    dataset_layout = DatasetLayout.from_project(run_config.project)
    run_layout = RunLayout.from_project(run_config.project)

    assert dataset_layout.split_dir("test") == tmp_path / "data" / "splits" / "test"
    assert run_layout.output_test_dir == (
        tmp_path / "results" / "section_run" / "artifacts" / "output_test"
    )
    assert run_layout.evaluation_dir == tmp_path / "results" / "section_run" / "evaluation"
    assert run_config.evaluation.save_graphs is False
    assert run_config.evaluation.generated_dir is None
    assert run_config.evaluation.output_dir is None


def test_evaluation_config_accepts_explicit_dirs(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        """\
        model:
          inputs: [label_free]
          target: stained
        evaluation:
          generated_dir: /custom/generated
          output_dir: /custom/evaluation
        """,
        dataset_root=Path("/data"),
        results_path=Path("/results"),
        run_name="section_run",
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.evaluation is not None

    assert run_config.evaluation.generated_dir == Path("/custom/generated")
    assert run_config.evaluation.output_dir == Path("/custom/evaluation")


def test_evaluation_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        "model:\n  inputs: [label_free]\n  target: stained\n"
        "evaluation:\n  target_dir: /custom/targets",
        filename="typo.yaml",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )
    with pytest.raises(ValueError, match="target_dir"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_unknown_top_level_key_raises(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        """
        model:
          inputs: [label_free]
          target: stained
        typo_field: oops
        evaluation:
          save_graphs: false
        """,
        filename="typo_top.yaml",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )
    with pytest.raises(ValueError, match="typo_field"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_string_bool_save_graphs_raises(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        'model:\n  inputs: [label_free]\n  target: stained\nevaluation:\n  save_graphs: "false"',
        filename="str_bool.yaml",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )
    with pytest.raises(TypeError, match="save_graphs"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_string_bool_hide_graphs_path_raises(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        "model:\n  inputs: [label_free]\n  target: stained\n"
        'evaluation:\n  hide_graphs_path: "false"',
        filename="str_bool2.yaml",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )
    with pytest.raises(ValueError, match="hide_graphs_path"):
        RunConfig.from_yaml(yaml_file)
