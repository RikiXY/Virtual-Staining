from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tests.config_helpers import write_run_config, yaml_section
from tests.image_helpers import write_rgb_image, write_rgb_pair
from tests.manifest_helpers import make_manifest_record, manifest_metadata, write_manifest_csv
from virtual_staining.applications.evaluate import evaluate
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.metrics import METRIC_SPECS
from virtual_staining.utils.artifacts import generated_filename


def _write_test_manifest(dataset_root: Path, sample_ids: list[str]) -> None:
    records = tuple(
        make_manifest_record(
            sample_id,
            "test",
            input_paths={"label_free": Path(f"splits/test/{sample_id}_source.png")},
            ext=".png",
            target_path=Path(f"splits/test/{sample_id}_target.png"),
        )
        for sample_id in sample_ids
    )
    write_manifest_csv(dataset_root, records)
    with (dataset_root / "manifests" / "slide_sets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "set_id",
                "split",
                "patient_id",
                "specimen_id",
                "status",
                "label_free__alignment_method",
                "target__alignment_method",
                "label_free__alignment_metadata",
                "target__alignment_metadata",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "set_id": "P1",
                "split": "test",
                "patient_id": "patient-1",
                "specimen_id": "specimen-1",
                "status": "processed",
                "label_free__alignment_method": "identity",
                "target__alignment_method": "identity",
                "label_free__alignment_metadata": "{}",
                "target__alignment_metadata": "{}",
            }
        )
    (dataset_root / "manifests" / "manifest_metadata.json").write_text(
        json.dumps(manifest_metadata().to_dict()), encoding="utf-8"
    )


def _write_evaluate_config(
    tmp_path: Path,
    dataset_root: Path,
    section_yaml: str,
    *,
    filename: str = "evaluate.yaml",
) -> Path:
    return write_run_config(
        tmp_path,
        "model:\n  inputs: [label_free]\n  target: stained\n"
        + yaml_section("evaluation", section_yaml),
        filename=filename,
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="eval_run",
    )


def _write_manifest_metadata(dataset_root: Path) -> None:
    (dataset_root / "manifests" / "manifest_metadata.json").write_text(
        json.dumps(manifest_metadata().to_dict()), encoding="utf-8"
    )


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
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    run_root = tmp_path / "results" / "eval_run"
    assert (run_root / "config" / "evaluate" / "input.yaml").exists()
    assert (run_root / "config" / "evaluate" / "resolved.yaml").exists()
    assert (run_root / "metadata" / "environments" / "evaluate.json").exists()
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
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
    )
    run_root = tmp_path / "results" / "eval_run"
    config_dir = run_root / "config" / "train"
    metadata_dir = run_root / "metadata"
    config_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (config_dir / "input.yaml").write_text("train input\n", encoding="utf-8")
    (config_dir / "resolved.yaml").write_text("train resolved\n", encoding="utf-8")

    run_config = RunConfig.from_yaml(yaml_file)
    evaluate(run_config, yaml_file)

    assert (config_dir / "input.yaml").read_text(encoding="utf-8") == "train input\n"
    assert (config_dir / "resolved.yaml").read_text(encoding="utf-8") == "train resolved\n"


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
                input_paths={"label_free": Path("splits/test/00000_00000_source.png")},
                target_path=Path("splits/test/00000_00000_target.png"),
            ),
        ),
    )
    _write_manifest_metadata(dataset_root)

    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"""\
          generated_dir: {generated_dir}
          output_dir: {tmp_path / "results" / "eval_run" / "evaluation"}
        """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    with pytest.raises(ValueError, match="test"):
        evaluate(run_config, yaml_file)


def test_evaluate_records_from_manifest_test_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    generated_dir = tmp_path / "generated"
    target_dir.mkdir(parents=True)
    generated_dir.mkdir()
    _write_test_manifest(dataset_root, ["00000_00000", "00256_00000"])

    for sample_id in ["00000_00000", "00256_00000"]:
        write_rgb_pair(target_dir, sample_id)
        write_rgb_image(generated_dir / generated_filename(sample_id, ".PNG"))
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
    manifest_path = DatasetLayout.from_project(run_config.project).manifest_path
    expected_manifest_hash = f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"

    assert metadata["stage"] == "evaluate"
    assert metadata["status"] == "completed"
    assert metadata["completed_at"]
    assert metadata["started_at"]
    assert metadata["dataset"]["manifest_path"] == str(manifest_path)
    assert metadata["dataset"]["manifest_sha256"] == expected_manifest_hash
    assert metadata["details"]["evaluated_count"] == 1
    assert metadata["details"]["skipped_count"] == 0
    assert metadata["details"]["metrics_csv_path"] == str(output_dir / "per_image_metrics.csv")
    assert metadata["details"]["summary_csv_path"] == str(output_dir / "summary.csv")
    assert metadata["details"]["metric_config"] == {name: True for name in METRIC_SPECS}

    events = [
        json.loads(line)
        for line in (tmp_path / "results" / "eval_run" / "metadata" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]
    assert all(event["stage"] == "evaluate" for event in events)


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
