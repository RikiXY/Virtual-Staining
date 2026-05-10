from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths


def test_evaluation_config_defaults_to_run_dirs(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent(f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        evaluation:
          save_graphs: false
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.evaluation is not None
    paths = RunPaths(run_config.project.run_root)

    assert run_config.project.dataset_test_dir == tmp_path / "data" / "dataset_test"
    assert paths.output_test_dir == (
        tmp_path / "results" / "section_run" / "artifacts" / "output_test"
    )
    assert run_config.project.run_root / "evaluation" == (
        tmp_path / "results" / "section_run" / "evaluation"
    )
    assert run_config.evaluation.save_graphs is False
    assert run_config.evaluation.target_dir is None
    assert run_config.evaluation.generated_dir is None
    assert run_config.evaluation.output_dir is None


def test_evaluation_config_accepts_explicit_dirs(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        evaluation:
          target_dir: /custom/targets
          generated_dir: /custom/generated
          output_dir: /custom/evaluation
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.evaluation is not None

    assert run_config.evaluation.target_dir == Path("/custom/targets")
    assert run_config.evaluation.generated_dir == Path("/custom/generated")
    assert run_config.evaluation.output_dir == Path("/custom/evaluation")


def test_evaluation_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent(f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        evaluation:
          save_graph: false
    """)
    yaml_file = tmp_path / "typo.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="save_graph"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_unknown_top_level_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent(f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        typo_field: oops
        evaluation:
          save_graphs: false
    """)
    yaml_file = tmp_path / "typo_top.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="typo_field"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_string_bool_save_graphs_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent(f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        evaluation:
          save_graphs: "false"
    """)
    yaml_file = tmp_path / "str_bool.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(TypeError, match="save_graphs"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_string_bool_hide_graphs_path_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent(f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        evaluation:
          hide_graphs_path: "false"
    """)
    yaml_file = tmp_path / "str_bool2.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="hide_graphs_path"):
        RunConfig.from_yaml(yaml_file)
