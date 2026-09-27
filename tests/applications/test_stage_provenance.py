"""Stage adapters freeze exactly the files they consume and link produced outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from tests.config_helpers import cyclegan_config_data, write_config_data
from tests.image_helpers import write_rgb_image
from tests.manifest_helpers import (
    make_manifest_record,
    manifest_metadata,
    write_aligned_test_manifest,
)
from virtual_staining.applications import prepare as prepare_app
from virtual_staining.applications import train as train_app
from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuildResult
from virtual_staining.data.consumption import DataLeakageError, load_snapshot
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.data.provenance import save_dataset_fingerprint
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.methods.cyclegan import CycleGANMethod
from virtual_staining.training.checkpoints import MethodCheckpointManager

_CPU = torch.device("cpu")
_SAMPLES = ("00000_00000", "00256_00000")


def _stage(layout: RunLayout, stage: Any) -> dict[str, Any]:
    return json.loads(layout.stage_record(stage).read_text(encoding="utf-8"))


def _write_slide_sets(root: Path, rows: list[dict[str, str]]) -> None:
    path = DatasetLayout(root).slide_sets_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["set_id", "split", "patient_id", "specimen_id"])
        writer.writeheader()
        writer.writerows(rows)


# Paired training


def _paired_manifest(root: Path, *, test_patient: str = "p4") -> DatasetManifest:
    records = []
    for index, (split, set_id) in enumerate(
        (("train", "S1"), ("train", "S2"), ("val", "S3"), ("test", "S4"))
    ):
        sample_id = f"{index * 256:05}_00000"
        record = make_manifest_record(
            sample_id,
            split,
            set_id=set_id,
            ext=".png",
            foreground_mask_path=Path(f"splits/{split}/{sample_id}__mask.png"),
        )
        records.append(record)
        for offset, path in enumerate(
            (*record.input_paths.values(), record.target_path, record.foreground_mask_path)
        ):
            assert path is not None
            write_rgb_image(root / path, color=(index, offset, 7))
    manifest = DatasetManifest(tuple(records), root, manifest_metadata())
    layout = DatasetLayout(root)
    manifest.to_csv(layout.manifest_path)
    layout.manifest_metadata_path.write_text(
        json.dumps(manifest_metadata().to_dict()), encoding="utf-8"
    )
    _write_slide_sets(
        root,
        [
            {"set_id": "S1", "split": "train", "patient_id": "p1", "specimen_id": "s1"},
            {"set_id": "S2", "split": "train", "patient_id": "p1", "specimen_id": "s2"},
            {"set_id": "S3", "split": "val", "patient_id": "p3", "specimen_id": "s3"},
            {"set_id": "S4", "split": "test", "patient_id": test_patient, "specimen_id": "s4"},
        ],
    )
    return manifest


def _pix2pix_config(
    tmp_path: Path,
    data_section: dict[str, Any] | None = None,
    image_size: int = 16,
    **training: Any,
) -> RunConfig:
    data: dict[str, Any] = {
        "dataset_root": str(tmp_path / "dataset"),
        "results_path": str(tmp_path / "results"),
        "run_name": "paired",
        "image_size": [image_size, image_size],
        "model": {"inputs": ["label_free"], "target": "stained"},
        "training": {
            "epochs": 1,
            "losses": {"generator": [{"name": "l1", "weight": 1.0}], "discriminator": []},
            **training,
        },
    }
    if data_section is not None:
        data["data"] = data_section
    return RunConfig.from_yaml(write_config_data(tmp_path / "paired.yaml", data))


@pytest.mark.parametrize("include_mask", [False, True])
def test_paired_training_snapshots_the_filtered_records_that_build_the_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_mask: bool
) -> None:
    _paired_manifest(tmp_path / "dataset")
    config = _pix2pix_config(
        tmp_path, augmentation={"enabled": True, "expansion_factor": 3, "intensity": "light"}
    )
    monkeypatch.setattr(train_app, "_requires_foreground_masks", lambda _config: include_mask)
    filtered: dict[str, DatasetManifest] = {}
    original = DatasetManifest.filter_split

    def recording_filter(self: DatasetManifest, split: Any) -> DatasetManifest:
        filtered[split] = original(self, split)
        return filtered[split]

    monkeypatch.setattr(DatasetManifest, "filter_split", recording_filter)

    train_ds, val_ds, _details, snapshot = train_app._paired_datasets(config, lambda x: x, 0)

    assert train_ds.manifest is filtered["train"]
    assert val_ds.manifest is filtered["val"]
    assert len(train_ds) == 6  # virtual expansion, not new assets
    roles = sorted({(row.split, row.role) for row in snapshot.rows})
    expected_roles = {"input", "target", *(("mask",) if include_mask else ())}
    assert roles == sorted((split, role) for split in ("train", "val") for role in expected_roles)
    assert len(snapshot.rows) == 3 * len(expected_roles)
    locators = {row.locator for row in snapshot.rows}
    for record in (*filtered["train"].records, *filtered["val"].records):
        assert record.target_path.as_posix() in locators
        assert (record.foreground_mask_path.as_posix() in locators) is include_mask  # type: ignore[union-attr]
    assert not any(row.split == "test" for row in snapshot.rows)
    # Held-out test inputs/target (and mask when used) are leakage-checked, not consumed.
    assert snapshot.validation_context["row_count"] == len(expected_roles)
    assert snapshot.validation_context["splits"] == ["test"]
    assert {row.patient_id for row in snapshot.rows} == {"p1", "p3"}
    assert snapshot.group_validation["unit"] == "patient"
    assert snapshot.sources["manifest_sha256"].startswith("sha256:")


@pytest.mark.parametrize("group_validation", ["auto", "unavailable"])
def test_paired_training_rejects_patient_shared_with_held_out_test(
    tmp_path: Path, group_validation: str
) -> None:
    _paired_manifest(tmp_path / "dataset", test_patient="p1")
    config = _pix2pix_config(tmp_path, {"group_validation": group_validation})
    with pytest.raises(DataLeakageError, match="patient_id"):
        train_app._paired_datasets(config, lambda x: x, 0)


def test_paired_training_on_declared_patch_split_records_shared_groups(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _paired_manifest(root, test_patient="p1")
    DatasetLayout(root).metadata_dir.mkdir(parents=True, exist_ok=True)
    DatasetLayout(root).split_assignment_path.write_text(
        "group_id,unit,split\n00000_00000,patch,train\n", encoding="utf-8"
    )
    config = _pix2pix_config(tmp_path, {"group_validation": "unavailable"})
    _, _, _, snapshot = train_app._paired_datasets(config, lambda x: x, 0)
    assert snapshot.group_validation["split_unit"] == "patch"
    assert snapshot.group_validation["groups_shared_across_splits"] == {"patient": 1}
    assert any("split unit is patch" in text for text in snapshot.limitations)
    with pytest.raises(DataLeakageError, match="patient_id"):
        train_app._paired_datasets(_pix2pix_config(tmp_path), lambda x: x, 0)


def test_tracked_training_accepts_externally_produced_current_schema_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hand-written manifest, metadata, files, and slide-set groups: no preparation run,
    # preparation config snapshot, or dataset fingerprint exists for this dataset.
    root = tmp_path / "dataset"
    _paired_manifest(root)
    layout = DatasetLayout(root)
    assert not layout.metadata_dir.exists() and not layout.config_dir.exists()
    config = _pix2pix_config(tmp_path, image_size=32, validate_rate=1, checkpoint_rate=1)
    logged: list[tuple[dict[str, float], int]] = []
    original_log_metrics = train_app.ExperimentSession.log_metrics

    def recording_log_metrics(self: Any, metrics: Any, *, step: int) -> None:
        logged.append((dict(metrics), step))
        original_log_metrics(self, metrics, step=step)

    monkeypatch.setattr(train_app.ExperimentSession, "log_metrics", recording_log_metrics)

    result = train_app.train(config, tmp_path / "paired.yaml")

    run = RunLayout.from_project(config.project)
    snapshot = load_snapshot(run.consumed_data("train"))
    record = _stage(run, "train")
    run_metadata = json.loads(run.run_metadata.read_text(encoding="utf-8"))
    # The consumed-data snapshot, not a preparation identity, is the training identity.
    assert run_metadata["training_data"]["snapshot_id"] == snapshot.snapshot_id
    assert record["consumed_data"]["snapshot_id"] == snapshot.snapshot_id
    assert snapshot.sources["dataset_fingerprint"] is None
    assert snapshot.group_validation["unit"] == "patient"
    assert {row.split for row in snapshot.rows} == {"train", "val"}
    # Tracked training still reports every epoch to its real session.
    assert [step for _metrics, step in logged] == [0]
    assert {"loss_G_train", "loss_G_val"} <= logged[0][0].keys()
    assert result.best_checkpoint_path is not None and result.best_checkpoint_path.is_file()


def _alias(link: Path, source: Path, kind: str) -> None:
    link.unlink()
    if kind == "copy":
        link.write_bytes(source.read_bytes())
    elif kind == "hardlink":
        os.link(source, link)
    else:
        link.symlink_to(source)


@pytest.mark.parametrize(
    ("kind", "hash_policy", "source_split", "match"),
    [
        ("copy", "content", "train", "same content"),
        ("copy", "content", "val", "same content"),
        ("hardlink", "membership", "train", "same file"),
        ("symlink", "membership", "val", "same file"),
        ("symlink", "content", "train", "same file"),
    ],
)
def test_paired_training_rejects_held_out_test_file_leakage(
    tmp_path: Path, kind: str, hash_policy: str, source_split: str, match: str
) -> None:
    root = tmp_path / "dataset"
    manifest = _paired_manifest(root)
    by_split = {record.split: record for record in manifest.records}
    source = root / by_split[source_split].input_paths["label_free"]
    # The leaking test asset is the held-out target, differently named from its source.
    _alias(root / by_split["test"].target_path, source, kind)
    config = _pix2pix_config(tmp_path, {"hash_policy": hash_policy})
    with pytest.raises(DataLeakageError, match=f"{match} appears in disjoint splits"):
        train_app._paired_datasets(config, lambda x: x, 0)


# Unpaired training


def _cyclegan_config(tmp_path: Path, name: str = "cyclegan", **data_fields: Any) -> RunConfig:
    data = cyclegan_config_data(tmp_path)
    data["data"].update(data_fields)
    return RunConfig.from_yaml(write_config_data(tmp_path / f"{name}.yaml", data))


def _write_domains(root: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> None:
    for offset, domain in enumerate(("label_free", "stained")):
        for split_index, split in enumerate(splits):
            for index in range(2):
                write_rgb_image(
                    root / "domains" / domain / split / f"{index}.png",
                    size=(32, 32),
                    color=(index, 10 * split_index, 100 * offset),
                )


def test_unpaired_training_snapshots_domain_membership_without_pairs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_domains(root)
    config = _cyclegan_config(tmp_path)

    train_ds, val_ds, snapshot = train_app._unpaired_datasets(config, lambda x: x, 1)
    # An unrelated prepared manifest at the dataset root never affects unpaired identity.
    DatasetLayout(root).manifest_path.parent.mkdir(parents=True)
    DatasetLayout(root).manifest_path.write_text("unrelated\n", encoding="utf-8")
    _, _, reseeded = train_app._unpaired_datasets(config, lambda x: x, 2)

    assert reseeded.snapshot_id == snapshot.snapshot_id
    assert snapshot.sources == {}
    assert {(row.split, row.domain, row.role) for row in snapshot.rows} == {
        (split, domain, role)
        for split in ("train", "val")
        for domain, role in (("label_free", "input"), ("stained", "target"))
    }
    assert all(row.sample_id == "" for row in snapshot.rows)
    consumed = {root / row.locator for row in snapshot.rows}
    assert (
        set(train_ds.paths_a) | set(train_ds.paths_b) | set(val_ds.paths_a) | set(val_ds.paths_b)
        == consumed
    )
    assert snapshot.group_validation["status"] == "unavailable"
    assert snapshot.validation_context["row_count"] == 4  # both domains' test collections
    assert snapshot.group_validation["splits"] == ["test", "train", "val"]


@pytest.mark.parametrize(
    ("source", "leaked"),
    [
        ("label_free/train/0.png", "label_free/test/1.png"),
        ("label_free/train/0.png", "stained/test/0.png"),
        ("stained/val/1.png", "label_free/test/0.png"),
    ],
)
def test_unpaired_training_rejects_content_shared_with_held_out_test(
    tmp_path: Path, source: str, leaked: str
) -> None:
    domains = tmp_path / "dataset" / "domains"
    _write_domains(tmp_path / "dataset")
    (domains / leaked).write_bytes((domains / source).read_bytes())
    with pytest.raises(DataLeakageError, match="same content appears in disjoint splits"):
        train_app._unpaired_datasets(_cyclegan_config(tmp_path), lambda x: x, 1)


def test_unpaired_training_group_context_is_the_selected_test_collections(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_domains(root)
    _write_group_sidecar(
        root,
        {
            ("label_free", "train"): "p1",
            ("stained", "train"): "p1",
            ("label_free", "val"): "p2",
            ("stained", "val"): "p2",
            ("label_free", "test"): "p3",
            ("stained", "test"): "p3",
        },
    )
    # A sidecar entry for an unselected collection is not part of this dataset.
    with (root / "groups.csv").open("a", encoding="utf-8") as handle:
        handle.write("other/test/0.png,other,test,p1-set,p1-sp,p1\n")
    config = _cyclegan_config(tmp_path, group_validation="auto", group_metadata="groups.csv")
    _, _, snapshot = train_app._unpaired_datasets(config, lambda x: x, 1)
    assert snapshot.group_validation["unit"] == "patient"
    assert not any(row.split == "test" for row in snapshot.rows)


def test_unpaired_training_requires_explicit_unavailable_without_group_metadata(
    tmp_path: Path,
) -> None:
    _write_domains(tmp_path / "dataset")
    config = _cyclegan_config(tmp_path, group_validation="auto")
    with pytest.raises(ValueError, match="data.group_validation: unavailable"):
        train_app._unpaired_datasets(config, lambda x: x, 1)


def _write_group_sidecar(root: Path, patient_of: dict[tuple[str, str], str]) -> None:
    lines = ["path,domain,split,set_id,specimen_id,patient_id"]
    for (domain, split), patient in patient_of.items():
        for index in range(2):
            locator = f"domains/{domain}/{split}/{index}.png"
            lines.append(f"{locator},{domain},{split},{patient}-set,{patient}-sp,{patient}")
    (root / "groups.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_unpaired_group_sidecar_validates_both_domains_across_splits(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_domains(root)
    _write_group_sidecar(
        root,
        {
            ("label_free", "train"): "p1",
            ("stained", "train"): "p1",  # same patient across domains in one split: allowed
            ("label_free", "val"): "p2",
            ("stained", "val"): "p2",
            ("label_free", "test"): "p3",
            ("stained", "test"): "p3",
        },
    )
    config = _cyclegan_config(tmp_path, group_validation="auto", group_metadata="groups.csv")
    _, _, snapshot = train_app._unpaired_datasets(config, lambda x: x, 1)
    assert snapshot.group_validation["unit"] == "patient"
    assert snapshot.group_validation["splits"] == ["test", "train", "val"]

    _write_group_sidecar(
        root,
        {
            ("label_free", "train"): "p1",
            ("stained", "train"): "p2",
            ("label_free", "val"): "p3",
            ("stained", "val"): "p4",
            ("label_free", "test"): "p5",
            ("stained", "test"): "p1",  # domain B test shares domain A train patient
        },
    )
    with pytest.raises(DataLeakageError, match="patient_id"):
        train_app._unpaired_datasets(config, lambda x: x, 1)


# Inference and evaluation lineage


def _aligned_dataset(root: Path) -> None:
    write_aligned_test_manifest(root, list(_SAMPLES))
    for index, sample_id in enumerate(_SAMPLES):
        test_dir = root / "splits" / "test"
        write_rgb_image(test_dir / f"{sample_id}_source.png", size=(32, 32), color=(index, 1, 2))
        write_rgb_image(test_dir / f"{sample_id}_target.png", size=(32, 32), color=(index, 3, 4))
    _write_domains(root, ("test",))


def _config_path(tmp_path: Path, direction: str, protocol: str = "paired", **extra: Any) -> Path:
    data = cyclegan_config_data(tmp_path)
    data["inference"]["direction"] = direction
    data["evaluation"] = {"protocol": protocol, "bootstrap_iterations": 10, **extra}
    return write_config_data(tmp_path / f"{direction}_{protocol}.yaml", data)


def _checkpoint(tmp_path: Path, seed: int) -> None:
    config = RunConfig.from_yaml(_config_path(tmp_path, "A_to_B"))
    torch.manual_seed(seed)
    method = CycleGANMethod(config, _CPU, seed=seed)
    batch = {"domain_a": torch.rand(2, 3, 32, 32) * 2 - 1, "domain_b": torch.rand(2, 3, 32, 32)}
    method.step(batch, epoch=0, global_step=0)
    layout = RunLayout.from_project(config.project)
    ensure_run_directories(layout)
    MethodCheckpointManager(method, layout.checkpoints_dir).save(0)


def _run(tmp_path: Path, stage: str, direction: str = "A_to_B", **kwargs: Any) -> RunLayout:
    path = _config_path(tmp_path, direction, **kwargs)
    config = RunConfig.from_yaml(path)
    (infer if stage == "infer" else evaluate)(config, path)
    return RunLayout.from_project(config.project)


@pytest.mark.parametrize(
    ("direction", "domain", "locator"),
    [
        ("A_to_B", "label_free", "splits/test/{sample}_source.png"),
        ("B_to_A", "stained", "splits/test/{sample}_target.png"),
    ],
)
def test_inference_snapshots_only_predictor_inputs_and_binds_outputs(
    tmp_path: Path, direction: str, domain: str, locator: str
) -> None:
    _aligned_dataset(tmp_path / "dataset")
    _checkpoint(tmp_path, 0)
    layout = _run(tmp_path, "infer", direction)

    record = _stage(layout, "infer")
    consumed = load_snapshot(layout.consumed_data("infer"))
    produced = load_snapshot(layout.produced_data("infer"))
    assert record["consumed_data"]["snapshot_id"] == consumed.snapshot_id
    assert record["produced_data"]["snapshot_id"] == produced.snapshot_id
    assert [(row.role, row.domain, row.locator) for row in consumed.rows] == [
        ("input", domain, locator.format(sample=sample)) for sample in _SAMPLES
    ]
    assert consumed.context["direction"] == direction
    assert consumed.context["checkpoint_sha256"] == record["details"]["checkpoint_sha256"]
    assert consumed.context["generation_config_sha256"].startswith("sha256:")
    assert produced.kind == "produced"
    assert produced.context["consumed_snapshot_id"] == consumed.snapshot_id
    assert [(row.role, row.sample_id, row.locator) for row in produced.rows] == [
        ("generated", sample, f"{sample}_{direction}_generated.png") for sample in _SAMPLES
    ]
    assert all(row.sha256 for row in produced.rows)


@pytest.mark.parametrize("direction", ["A_to_B", "B_to_A"])
def test_inference_opens_exactly_the_consumed_input_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    root = tmp_path / "dataset"
    _aligned_dataset(root)
    _checkpoint(tmp_path, 0)
    opened: list[Path] = []
    real_open = Image.open

    def recording_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(path).resolve())
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", recording_open)
    layout = _run(tmp_path, "infer", direction)

    consumed = load_snapshot(layout.consumed_data("infer"))
    dataset_reads = sorted(path for path in opened if path.is_relative_to(root.resolve()))
    assert dataset_reads == sorted((root / row.locator).resolve() for row in consumed.rows)
    unread = "_target.png" if direction == "A_to_B" else "_source.png"
    assert not any(path.name.endswith(unread) for path in dataset_reads)


def _producer(layout: RunLayout) -> dict[str, Any]:
    return _stage(layout, "evaluate")["details"]["generated_producer"]


def test_paired_evaluation_persists_correspondence_and_links_tracked_inference(
    tmp_path: Path,
) -> None:
    _aligned_dataset(tmp_path / "dataset")
    _checkpoint(tmp_path, 0)
    infer_layout = _run(tmp_path, "infer")
    layout = _run(tmp_path, "evaluate")

    snapshot = load_snapshot(layout.consumed_data("evaluate"))
    pairs = {
        row.sample_id: {r.role: r.locator for r in snapshot.rows if r.sample_id == row.sample_id}
        for row in snapshot.rows
    }
    assert pairs == {
        sample: {
            "reference": f"splits/test/{sample}_target.png",
            "generated": f"{sample}_A_to_B_generated.png",
        }
        for sample in _SAMPLES
    }
    producer = _producer(layout)
    assert producer["status"] == "linked"
    produced = _stage(infer_layout, "infer")["produced_data"]["snapshot_id"]
    assert producer["inference_output_snapshot_id"] == produced
    first_checkpoint = producer["checkpoint_sha256"]

    # Changed checkpoint identity is visible through the re-linked producer.
    _checkpoint(tmp_path, 1)
    _run(tmp_path, "infer")
    _run(tmp_path, "evaluate")
    assert _producer(layout)["status"] == "linked"
    assert _producer(layout)["checkpoint_sha256"] != first_checkpoint

    # Modified generated bytes.
    generated = infer_layout.output_test_dir / f"{_SAMPLES[0]}_A_to_B_generated.png"
    write_rgb_image(generated, size=(32, 32), color=(9, 9, 9))
    _run(tmp_path, "evaluate")
    assert _producer(layout)["status"] == "unlinked"
    assert _producer(layout)["changed"] == [generated.name]

    # Deleted generated file: requested, recorded missing, and skipped by the evaluator.
    generated.unlink()
    _run(tmp_path, "evaluate")
    evaluate_record = _stage(layout, "evaluate")
    missing = [
        row
        for row in load_snapshot(layout.consumed_data("evaluate")).rows
        if row.status == "missing"
    ]
    assert [(row.role, row.locator, row.sha256) for row in missing] == [
        ("generated", generated.name, None)
    ]
    assert evaluate_record["details"]["skipped_count"] == 1
    assert evaluate_record["details"]["evaluated_count"] == 1
    assert _producer(layout)["missing"] == [generated.name]

    # Wrong direction: the tracked inference produced A_to_B, evaluation asks for B_to_A.
    _run(tmp_path, "evaluate", "B_to_A")
    assert _producer(layout)["status"] == "unlinked"
    assert _producer(layout)["reason"] == "inference direction differs"


def test_unpaired_evaluation_persists_independent_collections(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _aligned_dataset(root)
    _checkpoint(tmp_path, 0)
    infer_layout = _run(tmp_path, "infer")
    layout = _run(tmp_path, "evaluate", protocol="unpaired")

    snapshot = load_snapshot(layout.consumed_data("evaluate"))
    assert snapshot.context["protocol"] == "unpaired"
    assert snapshot.selection["correspondence"] is None
    assert all(row.sample_id == "" for row in snapshot.rows)
    assert sorted(row.locator for row in snapshot.rows if row.role == "reference") == [
        "domains/stained/test/0.png",
        "domains/stained/test/1.png",
    ]
    assert _producer(layout)["status"] == "linked"

    # Stale extra output in the generated directory.
    extra = infer_layout.output_test_dir / "stale_A_to_B_generated.png"
    write_rgb_image(extra, size=(32, 32), color=(5, 5, 5))
    _run(tmp_path, "evaluate", protocol="unpaired")
    assert _producer(layout)["status"] == "unlinked"
    assert _producer(layout)["extra"] == [extra.name]
    extra.unlink()

    # Changed reference membership changes the stage identity.
    before = _stage(layout, "evaluate")["consumed_data"]["snapshot_id"]
    write_rgb_image(root / "domains" / "stained" / "test" / "2.png", size=(32, 32))
    _run(tmp_path, "evaluate", protocol="unpaired")
    assert _stage(layout, "evaluate")["consumed_data"]["snapshot_id"] != before


def test_external_generated_images_remain_valid_but_unlinked(tmp_path: Path) -> None:
    _aligned_dataset(tmp_path / "dataset")
    external = tmp_path / "external"
    for sample in _SAMPLES:
        write_rgb_image(external / f"{sample}_A_to_B_generated.png", size=(32, 32))
    layout = _run(tmp_path, "evaluate", generated_dir=str(external))

    record = _stage(layout, "evaluate")
    assert record["status"] == "completed"
    assert record["details"]["evaluated_count"] == 2
    assert _producer(layout) == {
        "status": "external",
        "reason": "no completed tracked inference in this run",
    }
    snapshot = load_snapshot(layout.consumed_data("evaluate"))
    assert snapshot.roots["generated"] == str(external.resolve())


def test_failed_input_resolution_records_attempt_without_snapshot(tmp_path: Path) -> None:
    (tmp_path / "dataset").mkdir()
    _checkpoint(tmp_path, 0)
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        _run(tmp_path, "infer")
    layout = RunLayout.from_project(RunConfig.from_yaml(_config_path(tmp_path, "A_to_B")).project)
    record = _stage(layout, "infer")
    assert record["status"] == "failed"
    assert record["error_type"] == "FileNotFoundError"
    assert record["consumed_data"] is None
    assert record["produced_data"] is None
    assert not layout.consumed_data("infer").metadata.exists()
    events = [json.loads(line) for line in layout.events.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["stage_failed"]


# Preparation


def _prepare_config(tmp_path: Path) -> tuple[RunConfig, Path]:
    root = tmp_path / "dataset"
    for name, content in (("lf.tif", b"lf-bytes"), ("st.tif", b"st-bytes"), ("m.png", b"mask")):
        (root / "raw").mkdir(parents=True, exist_ok=True)
        (root / "raw" / name).write_bytes(content)
    (root / "raw" / "unselected.tif").write_bytes(b"never selected")
    (root / "inventory.csv").write_text(
        "set_id,input__label_free_path,input__label_free_aligned,input__label_free_mask,"
        "target_path,target_aligned,patient_id,specimen_id\n"
        "S1,raw/lf.tif,true,raw/m.png,raw/st.tif,true,patient-1,specimen-1\n",
        encoding="utf-8",
    )
    data = {
        "dataset_root": str(root),
        "results_path": str(tmp_path / "results"),
        "run_name": "prepare",
        "model": {"inputs": ["label_free"], "target": "stained"},
        "preprocessing": {
            "inputs": {
                "inventory": "inventory.csv",
                "modalities": ["label_free"],
                "reference": "label_free",
                "target_modality": "stained",
            },
            "split": {"unit": "patient", "train": 0.8, "val": 0.1, "test": 0.1},
        },
    }
    path = write_config_data(tmp_path / "prepare.yaml", data)
    return RunConfig.from_yaml(path), path


def test_prepare_snapshots_selected_sources_and_reverifies_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path = _prepare_config(tmp_path)
    layout = DatasetLayout(config.project.dataset_root)
    built: list[tuple[Any, dict[str, Any]]] = []
    resolved: list[Any] = []
    real_resolve = prepare_app.resolve_slide_sets

    class RecordingBuilder:
        def __init__(self, _config: Any, *, slide_sets: Any, fingerprint_metadata: Any) -> None:
            self.slide_sets, self.fingerprint = slide_sets, fingerprint_metadata

        def run_all(self) -> DatasetBuildResult:
            built.append((self.slide_sets, self.fingerprint))
            save_dataset_fingerprint(self.fingerprint, layout.dataset_fingerprint_path)
            return DatasetBuildResult(1, 0, 0, 0, layout.root)

    def recording_resolve(cfg: Any) -> Any:
        resolved.append(real_resolve(cfg))
        return resolved[-1]

    monkeypatch.setattr(prepare_app, "resolve_slide_sets", recording_resolve)
    monkeypatch.setattr(prepare_app, "DatasetBuilder", RecordingBuilder)
    monkeypatch.setattr(prepare_app, "_dataset_outputs_are_complete", lambda _root: True)
    monkeypatch.setattr(
        prepare_app,
        "_build_reused_result",
        lambda root: DatasetBuildResult(1, 0, 0, 0, root, reused=True),
    )

    assert not prepare_app.prepare(config, path).reused
    snapshot = load_snapshot(layout.source_snapshot)
    assert built[0][0] is resolved[0]
    assert [(row.role, row.domain, row.locator) for row in snapshot.rows] == [
        ("input", "label_free", "raw/lf.tif"),
        ("mask", "label_free", "raw/m.png"),
        ("target", "stained", "raw/st.tif"),
    ]
    assert {(row.set_id, row.specimen_id, row.patient_id) for row in snapshot.rows} == {
        ("S1", "specimen-1", "patient-1")
    }
    fingerprint = json.loads(layout.dataset_fingerprint_path.read_text())
    assert fingerprint["fingerprint"].startswith("sha256:")
    assert fingerprint["source_snapshot_id"] == snapshot.snapshot_id

    assert prepare_app.prepare(config, path).reused
    assert len(built) == 1

    # Same size and mtime: the mtime hash cache alone would miss this edit.
    target = layout.root / "raw" / "st.tif"
    stat = target.stat()
    target.write_bytes(b"ST-BYTES")
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not prepare_app.prepare(config, path).reused
    assert len(built) == 2
    assert load_snapshot(layout.source_snapshot).snapshot_id != snapshot.snapshot_id
