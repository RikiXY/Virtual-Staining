from __future__ import annotations

from PIL import Image

from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.inference.outputs import generated_path_for_record


def test_manifest_dataset_fixture_passes_full_validation(manifest_dataset) -> None:
    manifest_dataset.manifest.validate(
        check_files_exist=True,
        require_splits={"train", "val", "test"},
    )


def test_manifest_training_views_load_only_train_and_val(manifest_dataset) -> None:
    manifest = DatasetManifest.from_csv(
        manifest_dataset.manifest_path,
        dataset_root=manifest_dataset.root,
    )
    train_dataset = PairedManifestDataset(manifest.filter_split("train"))
    val_dataset = PairedManifestDataset(manifest.filter_split("val"))

    assert train_dataset.sample_ids == [
        record.sample_id for record in manifest_dataset.train_records
    ]
    assert val_dataset.sample_ids == [record.sample_id for record in manifest_dataset.val_records]
    assert manifest_dataset.test_records[0].sample_id not in train_dataset.sample_ids
    assert manifest_dataset.test_records[0].sample_id not in val_dataset.sample_ids


def test_manifest_inference_view_loads_only_test(manifest_dataset) -> None:
    manifest = DatasetManifest.from_csv(
        manifest_dataset.manifest_path,
        dataset_root=manifest_dataset.root,
    )
    test_dataset = PairedManifestDataset(manifest.filter_split("test"))

    assert test_dataset.sample_ids == [record.sample_id for record in manifest_dataset.test_records]


def test_manifest_evaluation_view_uses_only_test_records(manifest_dataset) -> None:
    manifest = DatasetManifest.from_csv(
        manifest_dataset.manifest_path,
        dataset_root=manifest_dataset.root,
    )
    test_records = manifest.filter_split("test").records

    assert [record.sample_id for record in test_records] == [
        record.sample_id for record in manifest_dataset.test_records
    ]
    assert all(record.target_path.parts[:2] == ("splits", "test") for record in test_records)


def test_extra_files_in_directory_are_not_in_manifest(manifest_dataset) -> None:
    stray_source = manifest_dataset.root / "splits" / "test" / "stray_source.tif"
    stray_target = manifest_dataset.root / "splits" / "test" / "stray_target.tif"
    Image.new("RGB", (3, 3)).save(stray_source)
    Image.new("RGB", (3, 3)).save(stray_target)

    manifest = DatasetManifest.from_csv(
        manifest_dataset.manifest_path,
        dataset_root=manifest_dataset.root,
    )
    dataset = PairedManifestDataset(manifest.filter_split("test"))

    assert len(dataset) == len(manifest_dataset.test_records)
    assert "stray" not in dataset.sample_ids


def test_manifest_sample_ids_are_stable_across_reload(manifest_dataset) -> None:
    manifest_1 = DatasetManifest.from_csv(
        manifest_dataset.manifest_path,
        dataset_root=manifest_dataset.root,
    )
    manifest_2 = DatasetManifest.from_csv(
        manifest_dataset.manifest_path,
        dataset_root=manifest_dataset.root,
    )

    assert [record.sample_id for record in manifest_1.records] == [
        record.sample_id for record in manifest_2.records
    ]


def test_generated_path_for_test_record_is_manifest_driven(manifest_dataset) -> None:
    test_record = manifest_dataset.test_records[0]

    generated_path = generated_path_for_record(test_record, manifest_dataset.root / "generated")

    assert generated_path.name == f"{test_record.sample_id}_target_generated.tif"
