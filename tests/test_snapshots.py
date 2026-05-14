from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    build_dataset_fingerprint_metadata,
    build_file_provenance,
    compute_manifest_hash,
    resolve_prepare_snapshot_paths,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)


def _write_run_config(path: Path, *, run_name: str = "snapshots_run") -> RunConfig:
    path.write_text(
        f"""
dataset_root: {path.parent / "dataset"}
results_path: {path.parent / "results"}
run_name: {run_name}
image_size: [32, 32]
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return RunConfig.from_yaml(path)


def test_save_stage_config_snapshots_writes_input_resolved_and_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config = _write_run_config(config_path)

    input_dest = tmp_path / "artifacts" / "config" / "input.yaml"
    resolved_dest = tmp_path / "artifacts" / "config" / "resolved.yaml"
    hash_dest = tmp_path / "artifacts" / "metadata" / "config_hash.txt"

    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=input_dest,
        resolved_dest=resolved_dest,
        hash_dest=hash_dest,
    )

    assert input_dest.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
    assert yaml.safe_load(resolved_dest.read_text(encoding="utf-8")) == config.to_yaml_dict()
    expected = f"sha256:{hashlib.sha256(resolved_dest.read_bytes()).hexdigest()}"
    assert config_hash == expected
    assert hash_dest.read_text(encoding="utf-8") == expected


def test_save_stage_config_snapshots_is_hash_stable_for_same_effective_config(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first = _write_run_config(first_path, run_name="stable_run")
    second = _write_run_config(second_path, run_name="stable_run")

    first_hash = save_stage_config_snapshots(
        first,
        first_path,
        input_dest=tmp_path / "first" / "config" / "input.yaml",
        resolved_dest=tmp_path / "first" / "config" / "resolved.yaml",
        hash_dest=tmp_path / "first" / "metadata" / "config_hash.txt",
    )
    second_hash = save_stage_config_snapshots(
        second,
        second_path,
        input_dest=tmp_path / "second" / "config" / "input.yaml",
        resolved_dest=tmp_path / "second" / "config" / "resolved.yaml",
        hash_dest=tmp_path / "second" / "metadata" / "config_hash.txt",
    )

    assert first_hash == second_hash


def test_compute_manifest_hash_returns_sha256_prefix(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("sample_id,split\nabc,train\n", encoding="utf-8")

    manifest_hash = compute_manifest_hash(manifest_path)

    assert manifest_hash.startswith("sha256:")
    assert len(manifest_hash) > 10


def test_compute_manifest_hash_changes_with_file_contents(tmp_path: Path) -> None:
    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"
    first_manifest.write_text("sample_id,split\nabc,train\n", encoding="utf-8")
    second_manifest.write_text("sample_id,split\nxyz,val\n", encoding="utf-8")

    assert compute_manifest_hash(first_manifest) != compute_manifest_hash(second_manifest)


def test_save_environment_snapshot_writes_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "virtual_staining.experiment.snapshots.collect_environment",
        lambda: {"python_version": "3.11.0", "cuda_available": False},
    )

    dest = tmp_path / "metadata" / "environment.json"
    save_environment_snapshot(dest)

    assert json.loads(dest.read_text(encoding="utf-8")) == {
        "python_version": "3.11.0",
        "cuda_available": False,
    }


def test_resolve_run_snapshot_paths_training_uses_canonical_run_paths(tmp_path: Path) -> None:
    run_paths = RunPaths(tmp_path / "results" / "demo")

    resolved = resolve_run_snapshot_paths(stage="training", run_paths=run_paths)

    assert resolved.input_config == run_paths.input_config
    assert resolved.resolved_config == run_paths.resolved_config
    assert resolved.config_hash == run_paths.config_hash
    assert resolved.environment == run_paths.environment_metadata


def test_resolve_run_snapshot_paths_stage_scoped_names_do_not_collide(tmp_path: Path) -> None:
    run_paths = RunPaths(tmp_path / "results" / "demo")

    training = resolve_run_snapshot_paths(stage="training", run_paths=run_paths)
    inference = resolve_run_snapshot_paths(stage="inference", run_paths=run_paths)
    evaluation = resolve_run_snapshot_paths(stage="evaluation", run_paths=run_paths)

    assert inference != training
    assert evaluation != training
    assert inference != evaluation
    assert inference.input_config.name == "inference.input.yaml"
    assert inference.resolved_config.name == "inference.resolved.yaml"
    assert inference.config_hash.name == "inference_config_hash.txt"
    assert evaluation.input_config.name == "evaluation.input.yaml"
    assert evaluation.resolved_config.name == "evaluation.resolved.yaml"
    assert evaluation.config_hash.name == "evaluation_config_hash.txt"


def test_resolve_prepare_snapshot_paths_uses_dataset_root_canonical_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"

    resolved = resolve_prepare_snapshot_paths(dataset_root)

    assert resolved.input_config == dataset_root / "config" / "input.yaml"
    assert resolved.resolved_config == dataset_root / "config" / "resolved.yaml"
    assert resolved.config_hash == dataset_root / "metadata" / "config_hash.txt"
    assert resolved.environment == dataset_root / "metadata" / "environment.json"


def test_build_file_provenance_records_required_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"source-image-bytes")

    provenance = build_file_provenance(image_path)

    assert provenance["path"] == str(image_path.resolve())
    assert provenance["size"] == len(b"source-image-bytes")
    assert isinstance(provenance["mtime_ns"], int)
    assert provenance["sha256"].startswith("sha256:")


def test_dataset_fingerprint_changes_with_preprocessing_fields(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source_path = dataset_root / "source.png"
    target_path = dataset_root / "target.png"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"target")

    first = build_dataset_fingerprint_metadata(
        dataset_root=dataset_root,
        preprocessing_config={
            "dataset_root": str(dataset_root.resolve()),
            "source_name": "source.png",
            "target_name": "target.png",
            "image_size": [64, 64],
        },
        source_path=source_path,
        target_path=target_path,
        prepared_at="2026-01-01T00:00:00+00:00",
    )
    second = build_dataset_fingerprint_metadata(
        dataset_root=dataset_root,
        preprocessing_config={
            "dataset_root": str(dataset_root.resolve()),
            "source_name": "source.png",
            "target_name": "target.png",
            "image_size": [128, 64],
        },
        source_path=source_path,
        target_path=target_path,
        prepared_at="2026-01-01T00:00:00+00:00",
    )

    assert first["fingerprint"] != second["fingerprint"]


def test_dataset_fingerprint_changes_with_input_file_content(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source_path = dataset_root / "source.png"
    target_path = dataset_root / "target.png"
    source_path.write_bytes(b"source-v1")
    target_path.write_bytes(b"target")

    kwargs = {
        "dataset_root": dataset_root,
        "preprocessing_config": {
            "dataset_root": str(dataset_root.resolve()),
            "source_name": "source.png",
            "target_name": "target.png",
            "image_size": [64, 64],
        },
        "source_path": source_path,
        "target_path": target_path,
        "prepared_at": "2026-01-01T00:00:00+00:00",
    }
    first = build_dataset_fingerprint_metadata(**kwargs)
    source_path.write_bytes(b"source-v2")
    second = build_dataset_fingerprint_metadata(**kwargs)

    assert first["source"]["sha256"] != second["source"]["sha256"]
    assert first["fingerprint"] != second["fingerprint"]
