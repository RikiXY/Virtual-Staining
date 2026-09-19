from __future__ import annotations

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
from virtual_staining.data import slide_set_processor as processor_module
from virtual_staining.data.alignment import AlignmentResult, identity_alignment
from virtual_staining.data.slide_set_processor import SlideSetProcessor
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
