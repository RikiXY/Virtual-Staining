from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tests.config_helpers import write_run_config, yaml_section
from tests.image_helpers import write_rgb_image, write_rgb_pair
from tests.manifest_helpers import make_manifest_record, write_manifest_csv
from virtual_staining.applications.evaluate import evaluate
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.inference.outputs import (
    generated_filename_for_sample,
    generated_path_for_record,
)


def _write_test_manifest(dataset_root: Path, sample_ids: list[str]) -> None:
    records = tuple(make_manifest_record(sample_id, "test", ext=".png") for sample_id in sample_ids)
    write_manifest_csv(dataset_root, records)


def _write_evaluate_config(
    tmp_path: Path,
    dataset_root: Path,
    section_yaml: str,
    *,
    filename: str = "evaluate.yaml",
) -> Path:
    return write_run_config(
        tmp_path,
        yaml_section("evaluation", section_yaml),
        filename=filename,
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="eval_run",
    )


def test_evaluation_config_defaults_to_run_dirs(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        "evaluation:\n  save_graphs: false",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.evaluation is not None
    paths = RunPaths(run_config.project.run_root)

    assert run_config.project.split_dir("test") == tmp_path / "data" / "splits" / "test"
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
    yaml_file = write_run_config(
        tmp_path,
        """\
        evaluation:
          target_dir: /custom/targets
          generated_dir: /custom/generated
          output_dir: /custom/evaluation
        """,
        dataset_root=Path("/data"),
        results_path=Path("/results"),
        run_name="section_run",
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.evaluation is not None

    assert run_config.evaluation.target_dir == Path("/custom/targets")
    assert run_config.evaluation.generated_dir == Path("/custom/generated")
    assert run_config.evaluation.output_dir == Path("/custom/evaluation")


def test_evaluation_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        "evaluation:\n  save_graph: false",
        filename="typo.yaml",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )
    with pytest.raises(ValueError, match="save_graph"):
        RunConfig.from_yaml(yaml_file)


def test_evaluation_from_yaml_unknown_top_level_key_raises(tmp_path: Path) -> None:
    yaml_file = write_run_config(
        tmp_path,
        """
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
        'evaluation:\n  save_graphs: "false"',
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
        'evaluation:\n  hide_graphs_path: "false"',
        filename="str_bool2.yaml",
        dataset_root=tmp_path / "data",
        results_path=tmp_path / "results",
        run_name="section_run",
    )
    with pytest.raises(ValueError, match="hide_graphs_path"):
        RunConfig.from_yaml(yaml_file)


def test_evaluate_writes_stage_scoped_snapshot_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000"])

    write_rgb_pair(target_dir, "00000_00000")
    write_rgb_image(generated_dir / "00000_00000_target_generated.png")

    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          target_dir: {target_dir}
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
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
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000"])

    write_rgb_pair(target_dir, "00000_00000")
    write_rgb_image(generated_dir / "00000_00000_target_generated.png")

    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          target_dir: {target_dir}
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
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


def test_evaluate_raises_if_manifest_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()

    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          target_dir: {target_dir}
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        evaluate(run_config, yaml_file)


def test_evaluate_raises_if_required_test_split_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    write_rgb_pair(target_dir, "00000_00000")
    write_manifest_csv(
        dataset_root,
        (
            make_manifest_record(
                "00000_00000",
                "val",
                ext=".png",
                input_path=Path("splits/test/00000_00000_source.png"),
                target_path=Path("splits/test/00000_00000_target.png"),
            ),
        ),
    )

    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          target_dir: {target_dir}
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    with pytest.raises(ValueError, match="test"):
        evaluate(run_config, yaml_file)


def test_evaluate_pairs_from_manifest_test_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000", "00256_00000"])

    for sample_id in ["00000_00000", "00256_00000"]:
        write_rgb_pair(target_dir, sample_id)
        write_rgb_image(generated_dir / generated_filename_for_sample(sample_id, ".PNG"))
    write_rgb_image(generated_dir / "99999_99999_target_generated.png")

    output_dir = tmp_path / "results" / "eval_run" / "evaluation"
    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          generated_dir: {generated_dir}
          output_dir: {output_dir}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    per_image_metrics = output_dir / "per_image_metrics.csv"
    rows = per_image_metrics.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "99999_99999" not in per_image_metrics.read_text(encoding="utf-8")
    assert not (output_dir / "skipped.csv").exists()


def test_evaluate_writes_stage_metadata_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000"])
    write_rgb_pair(target_dir, "00000_00000")
    write_rgb_image(generated_dir / "00000_00000_target_generated.png")

    output_dir = tmp_path / "results" / "eval_run" / "evaluation"
    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          generated_dir: {generated_dir}
          output_dir: {output_dir}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    metadata_path = tmp_path / "results" / "eval_run" / "metadata" / "stages" / "evaluate.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = run_config.project.manifest_path
    expected_manifest_hash = f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"

    assert metadata["stage"] == "evaluate"
    assert metadata["status"] == "completed"
    assert metadata["completed_at"]
    assert metadata["started_at"]
    assert metadata["manifest_path"] == str(manifest_path)
    assert metadata["manifest_sha256"] == expected_manifest_hash
    assert metadata["evaluated_count"] == 1
    assert metadata["skipped_count"] == 0
    assert metadata["metrics_csv_path"] == str(output_dir / "per_image_metrics.csv")
    assert metadata["summary_csv_path"] == str(output_dir / "summary.csv")
    assert metadata["weak_tail_csv_path"] == str(output_dir / "weak_tail.csv")
    assert metadata["metric_config"]["ssim"] is True
    assert (output_dir / "weak_tail.csv").exists()
    with (output_dir / "weak_tail.csv").open(encoding="utf-8", newline="") as handle:
        weak_tail_reader = csv.DictReader(handle)
        assert weak_tail_reader.fieldnames is not None
        assert "metric" in weak_tail_reader.fieldnames
        assert "threshold" in weak_tail_reader.fieldnames
        assert "direction" in weak_tail_reader.fieldnames
        assert "weak_count" in weak_tail_reader.fieldnames
        assert "weak_share" in weak_tail_reader.fieldnames

    events = [
        json.loads(line)
        for line in (tmp_path / "results" / "eval_run" / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]
    assert all(event["stage"] == "evaluate" for event in events)
    assert events[-1]["details"]["weak_tail_csv_path"] == str(output_dir / "weak_tail.csv")


def test_evaluate_writes_skipped_csv_for_missing_generated(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000"])
    write_rgb_pair(target_dir, "00000_00000")

    output_dir = tmp_path / "results" / "eval_run" / "evaluation"
    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          generated_dir: {generated_dir}
          output_dir: {output_dir}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    skipped_csv = output_dir / "skipped.csv"
    assert skipped_csv.exists()
    assert "missing_generated" in skipped_csv.read_text(encoding="utf-8")


def test_evaluate_skipped_csv_has_correct_columns(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000"])
    write_rgb_pair(target_dir, "00000_00000")

    output_dir = tmp_path / "results" / "eval_run" / "evaluation"
    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          generated_dir: {generated_dir}
          output_dir: {output_dir}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    with (output_dir / "skipped.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["sample_id", "reason", "target_path", "generated_path"]


def test_generated_filename_for_sample() -> None:
    assert (
        generated_filename_for_sample("00512_09216", ".tif") == "00512_09216_target_generated.tif"
    )
    assert generated_filename_for_sample("patch_001", ".PNG") == "patch_001_target_generated.png"


def test_generated_path_for_record_uses_sample_id(tmp_path: Path) -> None:
    record = make_manifest_record("00512_09216", "test")

    result = generated_path_for_record(record, tmp_path)

    assert result == tmp_path / "00512_09216_target_generated.tif"
