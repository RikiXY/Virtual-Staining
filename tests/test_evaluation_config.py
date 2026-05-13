from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from PIL import Image

from virtual_staining.applications.evaluate import evaluate
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


def test_evaluate_writes_stage_scoped_snapshot_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "dataset_test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()

    Image.new("RGB", (16, 16)).save(target_dir / "00000_00000_target.png")
    Image.new("RGB", (16, 16)).save(generated_dir / "00000_00000_target_generated.png")

    yaml_file = tmp_path / "evaluate.yaml"
    yaml_file.write_text(
        textwrap.dedent(f"""\
            dataset_root: {dataset_root}
            results_path: {tmp_path / "results"}
            run_name: eval_run
            evaluation:
              target_dir: {target_dir}
              generated_dir: {generated_dir}
              output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """),
        encoding="utf-8",
    )

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    run_root = tmp_path / "results" / "eval_run"
    assert (run_root / "config" / "evaluation.input.yaml").exists()
    assert (run_root / "config" / "evaluation.resolved.yaml").exists()
    assert (run_root / "metadata" / "evaluation_config_hash.txt").exists()
    assert (run_root / "metadata" / "evaluation_environment.json").exists()
    assert not (run_root / "config" / "input.yaml").exists()
    assert not (run_root / "config" / "resolved.yaml").exists()
    assert not (run_root / "metadata" / "config_hash.txt").exists()


def test_evaluate_preserves_existing_training_snapshot_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "dataset_test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()

    Image.new("RGB", (16, 16)).save(target_dir / "00000_00000_target.png")
    Image.new("RGB", (16, 16)).save(generated_dir / "00000_00000_target_generated.png")

    yaml_file = tmp_path / "evaluate.yaml"
    yaml_file.write_text(
        textwrap.dedent(f"""\
            dataset_root: {dataset_root}
            results_path: {tmp_path / "results"}
            run_name: eval_run
            evaluation:
              target_dir: {target_dir}
              generated_dir: {generated_dir}
              output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """),
        encoding="utf-8",
    )
    run_root = tmp_path / "results" / "eval_run"
    config_dir = run_root / "config"
    metadata_dir = run_root / "metadata"
    config_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (config_dir / "input.yaml").write_text("train input\n", encoding="utf-8")
    (config_dir / "resolved.yaml").write_text("train resolved\n", encoding="utf-8")
    (metadata_dir / "config_hash.txt").write_text("sha256:train\n", encoding="utf-8")

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    assert (config_dir / "input.yaml").read_text(encoding="utf-8") == "train input\n"
    assert (config_dir / "resolved.yaml").read_text(encoding="utf-8") == "train resolved\n"
    assert (metadata_dir / "config_hash.txt").read_text(encoding="utf-8") == "sha256:train\n"
