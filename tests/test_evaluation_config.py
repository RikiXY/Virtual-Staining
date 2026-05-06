from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

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
    assert args.generated_dir == str(
        tmp_path / "results" / "section_run" / "output_test"
    )
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
