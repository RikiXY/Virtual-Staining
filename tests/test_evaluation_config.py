from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

from tools.evaluate_generation import apply_dataset_config


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

    args = apply_dataset_config(argparse.Namespace(config=str(yaml_file)))

    assert args.target_dir == str(tmp_path / "data" / "dataset_test")
    assert args.generated_dir == str(tmp_path / "results" / "section_run" / "output_test")
    assert args.output_dir == str(tmp_path / "results" / "section_run" / "evaluation")
    assert args.save_graphs is False
    assert args.hide_graphs_path is False


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

    args = apply_dataset_config(argparse.Namespace(config=str(yaml_file)))

    assert args.target_dir == "/custom/targets"
    assert args.generated_dir == "/custom/generated"
    assert args.output_dir == "/custom/evaluation"
    assert args.hide_graphs_path is False


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
        apply_dataset_config(argparse.Namespace(config=str(yaml_file)))


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
        apply_dataset_config(argparse.Namespace(config=str(yaml_file)))


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
        apply_dataset_config(argparse.Namespace(config=str(yaml_file)))


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
    with pytest.raises(TypeError, match="hide_graphs_path"):
        apply_dataset_config(argparse.Namespace(config=str(yaml_file)))
