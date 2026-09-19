from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from virtual_staining.config.data import (
    AlignmentConfig,
    InputConfig,
    IOConfig,
    MaskConfig,
    PatchingConfig,
    PreprocessingConfig,
    SplitConfig,
)
from virtual_staining.data import builder as builder_module
from virtual_staining.data import slide_set_processor as processor_module
from virtual_staining.data.alignment import AlignmentResult, identity_alignment
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
)
from virtual_staining.data.provenance import build_dataset_fingerprint_metadata
from virtual_staining.data.slide_set_processor import SetBuildResult, SlideSetProcessor
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.utils.image_io import PillowRegionImageReader


def _config(root: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(root / "inputs.csv", ("LF", "AF"), "LF", "target"),
        patching=PatchingConfig(patch_size=(8, 8), grid_movement=(8, 8), margin=0),
        split=SplitConfig(unit="set", train=1.0, val=0.0, test=0.0),
    )


def _slide_set(root: Path, set_id: str = "set-1") -> SlideSet:
    directory = Path("raw") / set_id
    (root / directory).mkdir(parents=True)
    image = np.full((8, 16, 3), 100, dtype=np.uint8)
    image[:, 8:] = 255
    for name in ("lf.png", "af.png", "target.png"):
        assert cv2.imwrite(str(root / directory / name), image)
    mask_path = directory / "mask.png"
    assert cv2.imwrite(str(root / mask_path), np.full((8, 16), 255, dtype=np.uint8))
    return SlideSet(
        set_id,
        (
            SlideAsset("LF", directory / "lf.png", already_aligned=True, mask_path=mask_path),
            SlideAsset("AF", directory / "af.png", already_aligned=True, mask_path=mask_path),
        ),
        SlideAsset("target", directory / "target.png", already_aligned=True, mask_path=mask_path),
        "LF",
    )


def test_builder_emits_dynamic_manifest_and_set_metadata(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    slide_sets = (_slide_set(tmp_path, "set-2"), _slide_set(tmp_path))

    class FakeProcessor:
        # Only the result boundary exists; private writes and state reads must fail.
        __slots__ = ("result",)

        def __init__(self, config, slide_set, assigned_split=None):
            self.result = SetBuildResult(
                set_id=slide_set.set_id,
                split=assigned_split,
                valid_rows=(
                    {
                        "sample_id": f"{slide_set.set_id}__x00000000_y00000000",
                        "split": assigned_split,
                        "inputs": {"LF": "lf.png", "AF": "af.png"},
                        "target": "target.png",
                        "foreground_mask": None,
                        "x": 0,
                        "y": 0,
                    },
                ),
                discarded_rows=(),
                metadata={
                    f"{name}__alignment_method": "identity"
                    for name in (*config.inputs.modalities, "target")
                },
            )

        def process(self):
            return self.result

    monkeypatch.setattr(builder_module, "SlideSetProcessor", FakeProcessor)
    builder = DatasetBuilder(config, slide_sets, {"schema_version": MANIFEST_SCHEMA_VERSION})
    result = builder.run_all()
    assert result.train_count == 2
    assert not hasattr(builder, "_current_set_id")
    header = (tmp_path / "manifests" / "manifest.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:6] == [
        "sample_id",
        "set_id",
        "split",
        "input__LF",
        "input__AF",
        "target_path",
    ]
    layout = DatasetLayout(tmp_path)
    rows = _read_csv(layout.manifest_path)
    assert [row["set_id"] for row in rows] == ["set-1", "set-2"]
    assert [row["input__LF"] for row in rows] == [
        "splits/train/set-1/lf.png",
        "splits/train/set-2/lf.png",
    ]
    assert [row["LF__alignment_method"] for row in _read_csv(layout.slide_sets_path)] == [
        "identity",
        "identity",
    ]
    assert _read_csv(layout.metadata_dir / "excluded_sets.csv") == []
    assert json.loads(layout.dataset_fingerprint_path.read_text()) == {
        "schema_version": MANIFEST_SCHEMA_VERSION
    }


@pytest.mark.parametrize("discarded", [False, True])
def test_records_use_explicit_set_id_without_mutating_builder(tmp_path, discarded) -> None:
    config = _config(tmp_path)
    builder = DatasetBuilder(
        config, (_slide_set(tmp_path),), {"schema_version": MANIFEST_SCHEMA_VERSION}
    )
    rows = (
        {
            "sample_id": "x",
            "split": "train",
            "inputs": {"LF": "x.png", "AF": "y.png"},
            "target": "t.png",
            "foreground_mask": None,
            "x": 0,
            "y": 0,
        },
    )
    before = vars(builder).copy()
    for set_id in ("set-2", "set-1", "set-2"):
        (record,) = builder._records(set_id, rows, discarded=discarded)
        assert record.set_id == set_id
        base = Path("discarded_patches" if discarded else "splits/train") / set_id
        assert record.input_paths["LF"] == base / ("LF/x.png" if discarded else "x.png")
        assert record.target_path == base / ("target/t.png" if discarded else "t.png")
    assert vars(builder) == before


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("assigned_split", [None, "val"])
@pytest.mark.parametrize("seed", [0, 17])
def test_process_returns_rows_and_metadata_after_cleanup(
    tmp_path, monkeypatch, assigned_split, seed
):
    config = replace(
        _config(tmp_path),
        masks=MaskConfig(save_patch_masks=True),
        split=SplitConfig(unit="patch", seed=seed, train=0.5, val=0.0, test=0.5),
    )
    slide_set = _slide_set(tmp_path)
    close = Mock()
    monkeypatch.setattr(PillowRegionImageReader, "close", close)

    result = SlideSetProcessor(config, slide_set, assigned_split).process()

    assert close.call_count == 3
    assert result.set_id == slide_set.set_id
    assert result.split == assigned_split
    assert result.error is None
    (valid,) = result.valid_rows
    (discarded,) = result.discarded_rows
    assert (valid["x"], valid["y"]) == (0, 0)
    assert valid["split"] == (assigned_split or ("train" if seed == 0 else "test"))
    assert (discarded["x"], discarded["y"]) == (8, 0)
    assert discarded["reasons"]
    assert set(discarded["ratios"]) == {"LF", "AF", "target", "all", "intersection", "union"}
    assert result.metadata == {
        **{f"{name}__alignment_method": "identity" for name in ("LF", "AF", "target")},
        **{
            f"{name}__alignment_metadata": json.dumps(
                {
                    "method": "identity",
                    "reason": "reference" if name == "LF" else "declared_aligned",
                    "warp_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                },
                sort_keys=True,
            )
            for name in ("LF", "AF", "target")
        },
    }
    # Dataset-level record conversion needs only the completed result and config.
    builder = DatasetBuilder(config, (slide_set,))
    records = builder._records(result.set_id, result.valid_rows)
    manifest = DatasetManifest(
        records, tmp_path, ManifestMetadata(MANIFEST_SCHEMA_VERSION, ("LF", "AF"), "LF", "target")
    )
    manifest.validate(check_files_exist=True)


@pytest.mark.parametrize("on_failure", ["error", "skip_set"])
@pytest.mark.parametrize("stage", ["opening", "mask", "alignment", "patch"])
def test_process_closes_readers_on_failure(tmp_path, monkeypatch, on_failure, stage):
    config = replace(_config(tmp_path), alignment=AlignmentConfig(on_failure=on_failure))
    slide_set = _slide_set(tmp_path)
    processor = SlideSetProcessor(config, slide_set, "train")
    readers = []
    failure = ValueError("processing failed")

    def open_reader(path, *, backend):
        if stage == "opening" and readers:
            raise failure
        reader = PillowRegionImageReader(path)
        monkeypatch.setattr(reader, "close", Mock())
        readers.append(reader)
        return reader

    monkeypatch.setattr(processor_module, "open_image_reader", open_reader)
    if stage == "mask":
        # Fail after the first reader has opened, during mask generation.
        processor.inputs["LF"].asset = replace(slide_set.inputs[0], mask_path=None)
        monkeypatch.setattr(processor, "_calculate_mask", Mock(side_effect=failure))
    elif stage == "alignment":
        monkeypatch.setattr(processor_module, "resolve_alignment", Mock(side_effect=failure))
    elif stage == "patch":
        extract = processor.extract_asset_patch

        def fail_after_one_patch(state, **kwargs):
            if kwargs["x"] == 8:
                raise failure
            return extract(state, **kwargs)

        monkeypatch.setattr(processor, "extract_asset_patch", fail_after_one_patch)

    if on_failure == "error":
        with pytest.raises(ValueError) as caught:
            processor.process()
        assert caught.value is failure
    else:
        result = processor.process()
        assert result.error == "processing failed"
        assert result.set_id == "set-1"
        assert result.split == "train"
        assert result.valid_rows == result.discarded_rows == ()
        assert result.metadata["LF__alignment_method"] == (
            "identity" if stage in {"alignment", "patch"} else ""
        )
        assert result.metadata["AF__alignment_method"] == ("identity" if stage == "patch" else "")
    assert len(readers) == (1 if stage in {"opening", "mask"} else 3)
    for reader in readers:
        reader.close.assert_called_once()


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("unit", ["set", "patch"])
def test_build_outputs_are_stable_with_an_excluded_set(tmp_path, tiled, unit):
    config = replace(
        _config(tmp_path),
        io=IOConfig(tiled=tiled, backend="pillow"),
        masks=MaskConfig(save_patch_masks=True),
        patching=PatchingConfig(
            patch_size=(8, 8), grid_movement=(8, 8), margin=0, save_discarded_patches=True
        ),
        alignment=AlignmentConfig(on_failure="skip_set"),
        split=SplitConfig(unit=unit, seed=17, train=0.5, val=0.0, test=0.5),
    )
    sets = tuple(
        replace(_slide_set(tmp_path, f"set-{i}"), patient_id=f"P{i}", specimen_id=f"SP{i}")
        for i in range(1, 4)
    )
    assert cv2.imwrite(str(tmp_path / sets[1].target.path), np.full((8, 8, 3), 100, dtype=np.uint8))
    layout = DatasetLayout(tmp_path)
    builder = DatasetBuilder(config, tuple(reversed(sets)))
    result = builder.run_all()

    assert not hasattr(builder, "_current_set_id")
    assert result.train_count + result.val_count + result.test_count == 2
    assert result.skipped_count == 2
    metadata = ManifestMetadata(MANIFEST_SCHEMA_VERSION, ("LF", "AF"), "LF", "target")
    valid = DatasetManifest.from_csv(layout.manifest_path, tmp_path, metadata)
    discarded = DatasetManifest.from_csv(layout.discarded_manifest_path, tmp_path, metadata)
    valid.validate(check_files_exist=True)
    discarded.validate(check_files_exist=True)
    assert [record.set_id for record in valid.records] == ["set-1", "set-3"]
    assert [record.set_id for record in discarded.records] == ["set-1", "set-3"]
    for record in (*valid.records, *discarded.records):
        expected_x = 8 if record.split == "discarded" else 0
        assert record.sample_id == f"{record.set_id}__x{expected_x:08}_y00000000"
        assert (record.x, record.y, record.width, record.height) == (expected_x, 0, 8, 8)
        assert record.target_path.name == f"{record.sample_id}__target.png"
        assert record.input_paths["LF"].name == f"{record.sample_id}__input__LF.png"
    rows = _read_csv(layout.slide_sets_path)
    assert [row["status"] for row in rows] == ["processed", "excluded", "processed"]
    assert [row["patient_id"] for row in rows] == ["P1", "P2", "P3"]
    assert [row["specimen_id"] for row in rows] == ["SP1", "SP2", "SP3"]
    assert [row["target__alignment_method"] for row in rows] == ["identity", "", "identity"]
    assert rows[1]["LF__alignment_method"] == rows[1]["AF__alignment_method"] == "identity"
    assert rows[1]["target__alignment_metadata"] == ""
    assert _read_csv(layout.metadata_dir / "excluded_sets.csv") == [
        {
            "set_id": "set-2",
            "split": rows[1]["split"],
            "error": "identity alignment requires equal geometry for target",
        }
    ]
    assignments = _read_csv(layout.split_assignment_path)
    expected_assignments = (
        {record.sample_id: record.split for record in valid.records}
        if unit == "patch"
        else {row["set_id"]: row["split"] for row in rows}
    )
    assert {row["group_id"]: row["split"] for row in assignments} == expected_assignments
    assert all(row["unit"] == unit for row in assignments)
    build_metadata = json.loads(layout.dataset_build_path.read_text())
    assert build_metadata == {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "num_sets": 3,
        "num_sets_excluded": 1,
        "patches": {
            "train": result.train_count,
            "val": result.val_count,
            "test": result.test_count,
            "discarded": 2,
        },
    }
    manifest_metadata = json.loads(layout.manifest_metadata_path.read_text())
    assert manifest_metadata.pop("created_at")
    assert manifest_metadata == {
        **metadata.to_dict(),
        "record_count": 2,
        "splits": {
            "train": result.train_count,
            "val": result.val_count,
            "test": result.test_count,
        },
    }
    fingerprint = json.loads(layout.dataset_fingerprint_path.read_text())
    assert fingerprint["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert fingerprint == build_dataset_fingerprint_metadata(
        dataset_root=tmp_path,
        preprocessing_config=config.to_dict(),
        slide_sets=sets,
        prepared_at=fingerprint["prepared_at"],
    )

    def outputs():
        snapshot = {}
        for directory in (
            layout.manifests_dir,
            layout.metadata_dir,
            layout.splits_dir,
            layout.discarded_patches_dir,
        ):
            for path in directory.rglob("*"):
                if path.is_file():
                    value = path.read_bytes()
                    if path.suffix == ".json":
                        data = json.loads(value)
                        data.pop("created_at", None)
                        data.pop("prepared_at", None)
                        value = json.dumps(data, sort_keys=True).encode()
                    snapshot[path.relative_to(tmp_path)] = value
        return snapshot

    before = outputs()
    assert DatasetBuilder(config, sets).run_all() == result
    assert outputs() == before


def test_align_delegates_all_moving_assets_with_explicit_data(tmp_path, monkeypatch) -> None:
    processor = SlideSetProcessor(_config(tmp_path), _slide_set(tmp_path))
    processor.compute_masks()
    result = identity_alignment("delegated")
    resolve = Mock(return_value=result)
    monkeypatch.setattr(processor_module, "resolve_alignment", resolve)
    try:
        processor.align()
        assert processor.reference.alignment is not None
        assert processor.reference.alignment.reason == "reference"
        assert resolve.call_count == 2
        for call, state in zip(
            resolve.call_args_list, (processor.inputs["AF"], processor.target), strict=True
        ):
            reference, moving, policy = call.args
            assert reference.preview is processor.reference.preview
            assert reference.full_shape == processor.reference.shape
            assert moving.preview is state.preview and moving.mask is state.mask
            assert moving.full_shape == state.shape
            assert moving.name == state.asset.modality
            assert moving.mpp == (None, None)
            assert policy is processor.config.alignment
            assert call.kwargs == {"already_aligned": state.asset.already_aligned}
            assert state.alignment is result
    finally:
        processor.close()


@pytest.mark.parametrize("tiled", [False, True])
def test_affine_extraction_handles_downsampled_masks_in_both_io_paths(tmp_path, tiled) -> None:
    config = replace(_config(tmp_path), io=IOConfig(tiled=tiled, backend="pillow"))
    processor = SlideSetProcessor(config, _slide_set(tmp_path))
    try:
        processor.compute_masks()
        state = processor.target
        state.mask = np.zeros((4, 8), dtype=np.uint8)
        state.mask[:, :4] = 255
        state.alignment = AlignmentResult(
            "affine_sift", np.array([[1.0, 0.0, -2.0], [0.0, 1.0, 0.0]])
        )
        image, mask = processor.extract_asset_patch(state, x=0, y=0, width=8, height=8)
        expected_mask = cv2.warpAffine(
            state.mask,
            np.array([[2.0, 0.0, -2.0], [0.0, 2.0, 0.0]]),
            (8, 8),
            flags=cv2.INTER_NEAREST,
        )
        expected_image = np.full((8, 8, 3), 255, dtype=np.uint8)
        expected_image[:, :6] = 100
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(image, expected_image)
    finally:
        processor.close()
