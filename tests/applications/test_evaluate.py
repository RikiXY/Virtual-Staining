from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.config_helpers import (
    cyclegan_config_data,
    write_config_data,
    write_run_config,
    yaml_section,
)
from tests.image_helpers import write_rgb_image, write_rgb_pair
from tests.manifest_helpers import make_manifest_record, manifest_metadata, write_manifest_csv
from virtual_staining.applications.evaluate import (
    EVALUATION_METADATA_JSON,
    evaluate,
    paired_sample,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.evaluation.unpaired import (
    UNPAIRED_FEATURE_COMPARISON_CSV,
    UNPAIRED_FEATURE_PLOT,
    UNPAIRED_IMAGE_STATISTICS_CSV,
)
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


_SOURCE_COLOR = (10, 20, 30)
_TARGET_COLOR = (200, 150, 100)
_PAIRED_OUTPUTS = ("per_image_metrics.csv", "summary.csv", "summary_set.csv", "set_metrics.csv")


def _cyclegan_eval_config(
    tmp_path: Path, direction: str | None = None, **evaluation: object
) -> tuple[RunConfig, Path]:
    data = cyclegan_config_data(tmp_path)
    if direction is not None:
        data["inference"]["direction"] = direction
    data["evaluation"] = {
        "generated_dir": str(tmp_path / "generated"),
        "output_dir": str(tmp_path / "evaluation"),
        "bootstrap_iterations": 10,
        **evaluation,
    }
    path = write_config_data(tmp_path / f"cyclegan_{direction}.yaml", data)
    return RunConfig.from_yaml(path), path


def _write_aligned_cyclegan_dataset(tmp_path: Path, sample_ids: list[str]) -> Path:
    dataset_root = tmp_path / "dataset"
    _write_test_manifest(dataset_root, sample_ids)
    for sample_id in sample_ids:
        test_dir = dataset_root / "splits" / "test"
        write_rgb_image(test_dir / f"{sample_id}_source.png", color=_SOURCE_COLOR)
        write_rgb_image(test_dir / f"{sample_id}_target.png", color=_TARGET_COLOR)
    return dataset_root


def _write_domains(tmp_path: Path, counts: dict[str, int]) -> None:
    for domain, count in counts.items():
        for i in range(count):
            write_rgb_image(
                tmp_path / "dataset" / "domains" / domain / "test" / f"real_{domain}_{i}.png",
                size=(16 + 4 * i, 16),
                color=(i * 40, 100, 50),
            )


def _metadata(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / EVALUATION_METADATA_JSON).read_text(encoding="utf-8"))


def test_paired_sample_maps_method_and_direction_to_reference(tmp_path: Path) -> None:
    record = make_manifest_record(
        "00000_00000",
        "test",
        ext=".png",
        input_paths={"label_free": Path("a/x_source.png")},
        target_path=Path("b/x_target.png"),
    )
    pix2pix = RunConfig.from_yaml(
        write_run_config(tmp_path, "model:\n  inputs: [label_free]\n  target: stained")
    )
    root = pix2pix.project.dataset_root
    generated = tmp_path / "generated"

    sample = paired_sample(pix2pix, record, generated)
    assert sample.target_path == root / "b/x_target.png"
    assert sample.generated_path == generated / "00000_00000_target_generated.png"
    assert (sample.sample_id, sample.set_id) == ("00000_00000", "P1")

    a_to_b = paired_sample(_cyclegan_eval_config(tmp_path, "A_to_B")[0], record, generated)
    assert a_to_b.target_path.relative_to(tmp_path / "dataset") == Path("b/x_target.png")
    assert a_to_b.generated_path == generated / "00000_00000_A_to_B_generated.png"

    b_to_a = paired_sample(_cyclegan_eval_config(tmp_path, "B_to_A")[0], record, generated)
    assert b_to_a.target_path.relative_to(tmp_path / "dataset") == Path("a/x_source.png")
    assert b_to_a.generated_path == generated / "00000_00000_B_to_A_generated.png"
    assert (b_to_a.sample_id, b_to_a.set_id) == ("00000_00000", "P1")


@pytest.mark.parametrize(
    ("direction", "reference_color", "reference_domain"),
    [("A_to_B", _TARGET_COLOR, "stained"), ("B_to_A", _SOURCE_COLOR, "label_free")],
)
def test_cyclegan_paired_evaluation_uses_direction_reference(
    tmp_path: Path, direction: str, reference_color: tuple[int, int, int], reference_domain: str
) -> None:
    sample_ids = ["00000_00000", "00256_00000"]
    _write_aligned_cyclegan_dataset(tmp_path, sample_ids)
    wrong_color = _SOURCE_COLOR if direction == "A_to_B" else _TARGET_COLOR
    opposite = "B_to_A" if direction == "A_to_B" else "A_to_B"
    for sample_id in sample_ids:
        write_rgb_image(
            tmp_path / "generated" / generated_filename(sample_id, ".png", direction),
            color=reference_color,
        )
        write_rgb_image(
            tmp_path / "generated" / generated_filename(sample_id, ".png", opposite),
            color=wrong_color,
        )
    config, path = _cyclegan_eval_config(tmp_path, direction, protocol="paired")

    evaluate(config, path)

    output_dir = tmp_path / "evaluation"
    with (output_dir / "per_image_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == sample_ids
    assert all(float(row["mae"]) == 0.0 for row in rows)
    assert all(f"_{direction}_generated" in row["generated_path"] for row in rows)
    assert (output_dir / "summary_set.csv").exists()
    assert (output_dir / "summary_patient.csv").exists()
    metadata = _metadata(output_dir)
    assert metadata["schema_version"] == 1
    assert metadata["method"] == "cyclegan"
    assert metadata["training_pairing"] == "unpaired"
    assert metadata["evaluation_protocol"] == "paired"
    assert metadata["inference_direction"] == direction
    assert metadata["reference_domain"] == reference_domain
    assert metadata["pairwise_metrics_available"] is True
    assert metadata["counts"] == {"requested_count": 2, "evaluated_count": 2, "skipped_count": 0}
    assert "limitations" not in metadata


def test_cyclegan_paired_evaluation_requires_aligned_manifest(tmp_path: Path) -> None:
    _write_domains(tmp_path, {"label_free": 2, "stained": 2})
    write_rgb_image(tmp_path / "generated" / "x_A_to_B_generated.png")
    config, path = _cyclegan_eval_config(tmp_path, protocol="paired")

    with pytest.raises(FileNotFoundError, match="aligned held-out test manifest"):
        evaluate(config, path)
    assert not (tmp_path / "evaluation" / UNPAIRED_IMAGE_STATISTICS_CSV).exists()


def test_cyclegan_paired_evaluation_rejects_missing_reference_file(tmp_path: Path) -> None:
    dataset_root = _write_aligned_cyclegan_dataset(tmp_path, ["00000_00000"])
    (dataset_root / "splits" / "test" / "00000_00000_source.png").unlink()
    config, path = _cyclegan_eval_config(tmp_path, "B_to_A", protocol="paired")

    with pytest.raises(FileNotFoundError, match="aligned held-out test manifest"):
        evaluate(config, path)


@pytest.mark.parametrize(
    ("direction", "reference_domain", "source_domain"),
    [("A_to_B", "stained", "label_free"), ("B_to_A", "label_free", "stained")],
)
def test_cyclegan_unpaired_evaluation_compares_collections(
    tmp_path: Path, direction: str, reference_domain: str, source_domain: str
) -> None:
    _write_domains(tmp_path, {reference_domain: 3, source_domain: 1})
    opposite = "B_to_A" if direction == "A_to_B" else "A_to_B"
    generated_dir = tmp_path / "generated"
    expected_generated = [
        write_rgb_image(generated_dir / "case1" / f"x_{direction}_generated.png", size=(8, 8)),
        write_rgb_image(generated_dir / "case2" / f"y_{direction}_generated.png", size=(12, 4)),
    ]
    write_rgb_image(generated_dir / "case1" / f"x_{opposite}_generated.png")
    write_rgb_image(generated_dir / "case1" / "unrelated.png")
    config, path = _cyclegan_eval_config(tmp_path, direction)

    evaluate(config, path)

    output_dir = tmp_path / "evaluation"
    with (output_dir / UNPAIRED_IMAGE_STATISTICS_CSV).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["path"] for row in rows if row["collection"] == "generated"] == [
        str(p) for p in expected_generated
    ]
    references = [row["path"] for row in rows if row["collection"] == "reference"]
    assert len(references) == 3
    assert all(f"/domains/{reference_domain}/test/" in ref for ref in references)
    assert (output_dir / UNPAIRED_FEATURE_COMPARISON_CSV).exists()
    assert not any((output_dir / name).exists() for name in _PAIRED_OUTPUTS)
    assert not (output_dir / UNPAIRED_FEATURE_PLOT).exists()

    metadata = _metadata(output_dir)
    assert metadata["evaluation_protocol"] == "unpaired"
    assert metadata["training_pairing"] == "unpaired"
    assert metadata["inference_direction"] == direction
    assert metadata["source_domains"] == [source_domain]
    assert metadata["reference_domain"] == reference_domain
    assert metadata["pairwise_metrics_available"] is False
    assert metadata["counts"] == {"generated_count": 2, "reference_count": 3}
    assert metadata["artifacts"]["unpaired_feature_plot"] is None
    assert metadata["limitations"]
    assert "mean_luminance" in metadata["features"]

    stage = json.loads(
        (tmp_path / "results" / "cyclegan_run" / "metadata" / "stages" / "evaluate.json").read_text(
            encoding="utf-8"
        )
    )["details"]
    assert stage["evaluation_protocol"] == "unpaired"
    assert stage["generated_count"] == 2
    assert stage["reference_count"] == 3
    assert stage["evaluation_metadata_path"] == str(output_dir / EVALUATION_METADATA_JSON)
    assert "metric_config" not in stage


def test_cyclegan_unpaired_evaluation_rejects_empty_generated(tmp_path: Path) -> None:
    _write_domains(tmp_path, {"label_free": 1, "stained": 1})
    write_rgb_image(tmp_path / "generated" / "x_B_to_A_generated.png")
    config, path = _cyclegan_eval_config(tmp_path, "A_to_B")

    with pytest.raises(ValueError, match="No A_to_B generated images"):
        evaluate(config, path)


def test_cyclegan_unpaired_evaluation_rejects_missing_or_empty_reference(tmp_path: Path) -> None:
    write_rgb_image(tmp_path / "generated" / "x_A_to_B_generated.png")
    config, path = _cyclegan_eval_config(tmp_path, "A_to_B")
    with pytest.raises(FileNotFoundError, match="no 'test' split"):
        evaluate(config, path)

    (tmp_path / "dataset" / "domains" / "stained" / "test").mkdir(parents=True)
    with pytest.raises(ValueError, match="matched no supported images"):
        evaluate(config, path)


def test_protocol_switches_remove_stale_reports(tmp_path: Path) -> None:
    _write_aligned_cyclegan_dataset(tmp_path, ["00000_00000"])
    _write_domains(tmp_path, {"label_free": 2, "stained": 2})
    write_rgb_image(tmp_path / "generated" / "00000_00000_A_to_B_generated.png")
    output_dir = tmp_path / "evaluation"
    unrelated = output_dir / "notes.txt"
    output_dir.mkdir()
    unrelated.write_text("keep")
    (output_dir / "skipped.csv").write_text("stale")

    evaluate(*_cyclegan_eval_config(tmp_path, protocol="paired", save_graphs=True))
    assert (output_dir / "per_image_metrics.csv").exists()
    assert (output_dir / "metrics_boxplot.png").exists()
    assert (output_dir / "mae_histogram.png").exists()
    assert not (output_dir / "skipped.csv").exists()

    evaluate(*_cyclegan_eval_config(tmp_path, protocol="unpaired", save_graphs=True))
    assert not any((output_dir / name).exists() for name in _PAIRED_OUTPUTS)
    assert not (output_dir / "metrics_boxplot.png").exists()
    assert not list(output_dir.glob("*_histogram.png"))
    assert (output_dir / UNPAIRED_FEATURE_PLOT).exists()
    assert _metadata(output_dir)["evaluation_protocol"] == "unpaired"

    evaluate(*_cyclegan_eval_config(tmp_path, protocol="unpaired", save_graphs=False))
    assert not (output_dir / UNPAIRED_FEATURE_PLOT).exists()
    assert (output_dir / UNPAIRED_IMAGE_STATISTICS_CSV).exists()

    evaluate(*_cyclegan_eval_config(tmp_path, protocol="paired"))
    assert not (output_dir / UNPAIRED_IMAGE_STATISTICS_CSV).exists()
    assert not (output_dir / UNPAIRED_FEATURE_COMPARISON_CSV).exists()
    assert not (output_dir / "metrics_boxplot.png").exists()
    assert (output_dir / "per_image_metrics.csv").exists()
    assert _metadata(output_dir)["evaluation_protocol"] == "paired"
    assert unrelated.read_text() == "keep"


def test_pix2pix_evaluation_writes_paired_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    target_dir = dataset_root / "splits" / "test"
    target_dir.mkdir(parents=True)
    _write_test_manifest(dataset_root, ["00000_00000", "00256_00000"])
    write_rgb_pair(target_dir, "00000_00000")
    write_rgb_pair(target_dir, "00256_00000")
    write_rgb_image(tmp_path / "generated" / "00000_00000_target_generated.png")
    output_dir = tmp_path / "evaluation"
    yaml_file = _write_evaluate_config(
        tmp_path,
        dataset_root,
        f"generated_dir: {tmp_path / 'generated'}\noutput_dir: {output_dir}",
    )

    evaluate(RunConfig.from_yaml(yaml_file), yaml_file)

    metadata = _metadata(output_dir)
    assert metadata["method"] == "pix2pix"
    assert metadata["training_pairing"] == "paired"
    assert metadata["evaluation_protocol"] == "paired"
    assert metadata["inference_direction"] is None
    assert metadata["source_domains"] == ["label_free"]
    assert metadata["reference_domain"] == "stained"
    assert metadata["pairwise_metrics_available"] is True
    assert metadata["counts"] == {"requested_count": 2, "evaluated_count": 1, "skipped_count": 1}
    assert metadata["artifacts"]["skipped_csv"] == str(output_dir / "skipped.csv")
