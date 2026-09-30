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


def _config(root: Path, targets: tuple[str, ...] = ("HE",)) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(root / "inputs.csv", ("LF", "AF"), "LF", targets),
        patching=PatchingConfig(patch_size=(8, 8), grid_movement=(8, 8), margin=0),
        split=SplitConfig(unit="set", train=1.0, val=0.0, test=0.0),
    )


def _slide_set(root: Path, set_id: str = "set-1", targets: tuple[str, ...] = ("HE",)) -> SlideSet:
    directory = Path("raw") / set_id
    (root / directory).mkdir(parents=True)
    image = np.full((8, 16, 3), 100, dtype=np.uint8)
    image[:, 8:] = 255
    for name in ("LF", "AF", *targets):
        assert cv2.imwrite(str(root / directory / f"{name}.png"), image)
    mask_path = directory / "mask.png"
    assert cv2.imwrite(str(root / mask_path), np.full((8, 16), 255, dtype=np.uint8))
    return SlideSet(
        set_id,
        (
            SlideAsset("LF", directory / "LF.png", already_aligned=True, mask_path=mask_path),
            SlideAsset("AF", directory / "AF.png", already_aligned=True, mask_path=mask_path),
        ),
        tuple(
            SlideAsset(name, directory / f"{name}.png", already_aligned=True, mask_path=mask_path)
            for name in targets
        ),
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

    # Three source readers, plus one header read per verified written file (LF, AF, HE and
    # the HE mask of the one accepted sample); every reader is closed.
    assert close.call_count == 3 + 4
    assert result.set_id == slide_set.set_id
    assert result.split == assigned_split
    assert result.error is None
    (valid,) = result.valid_rows
    (discarded,) = result.discarded_rows
    assert (valid["x"], valid["y"]) == (0, 0)
    assert valid["split"] == (assigned_split or ("train" if seed == 0 else "test"))
    assert (discarded["x"], discarded["y"]) == (8, 0)
    assert discarded["reasons"]
    assert set(discarded["ratios"]) == {"LF", "AF", "HE", "all", "intersection", "union"}
    assert result.metadata == {
        **{f"{name}__alignment_method": "identity" for name in ("LF", "AF", "HE")},
        **{
            f"{name}__alignment_metadata": json.dumps(
                {
                    "method": "identity",
                    "reason": "reference" if name == "LF" else "declared_aligned",
                    "warp_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                },
                sort_keys=True,
            )
            for name in ("LF", "AF", "HE")
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
            resolve.call_args_list, (processor.inputs["AF"], processor.targets["HE"]), strict=True
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
        state = processor.targets["HE"]
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


def test_every_target_is_aligned_extracted_and_committed_on_one_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(tmp_path, ("PAS", "HE")),
        masks=MaskConfig(save_patch_masks=True),
    )
    slide_set = _slide_set(tmp_path, targets=("PAS", "HE"))
    resolve = Mock(return_value=identity_alignment("declared_aligned"))
    monkeypatch.setattr(processor_module, "resolve_alignment", resolve)

    result = SlideSetProcessor(config, slide_set, "train").process()

    # AF, PAS and HE are aligned to the reference frame; the reference is identity.
    assert [call.args[1].name for call in resolve.call_args_list] == ["AF", "PAS", "HE"]
    (valid,) = result.valid_rows
    sample = valid["sample_id"]
    assert valid["targets"] == {
        "PAS": f"{sample}__target__PAS.png",
        "HE": f"{sample}__target__HE.png",
    }
    assert valid["foreground_masks"] == {
        "PAS": f"{sample}__foreground_mask__PAS.png",
        "HE": f"{sample}__foreground_mask__HE.png",
    }
    written = tmp_path / "splits" / "train" / slide_set.set_id
    for name in (*valid["inputs"].values(), *valid["targets"].values()):
        image = cv2.imread(str(written / name))
        assert image is not None and image.shape == (8, 8, 3)
    for name in valid["foreground_masks"].values():
        assert (written / name).is_file()
    assert {"PAS__alignment_method", "HE__alignment_method"} <= set(result.metadata)


def test_target_policy_requires_every_target_foreground(tmp_path: Path) -> None:
    config = _config(tmp_path, ("HE", "PAS"))
    config = replace(
        config,
        filtering=replace(
            config.filtering, foreground=replace(config.filtering.foreground, policy="target")
        ),
    )
    slide_set = _slide_set(tmp_path, targets=("HE", "PAS"))
    empty = Path("raw/set-1/empty.png")
    assert cv2.imwrite(str(tmp_path / empty), np.zeros((8, 16), dtype=np.uint8))
    slide_set = replace(
        slide_set, targets=(slide_set.targets[0], replace(slide_set.targets[1], mask_path=empty))
    )

    result = SlideSetProcessor(config, slide_set, "train").process()

    assert result.valid_rows == ()
    assert all("foreground_target" in row["reasons"] for row in result.discarded_rows)


def test_a_sample_is_committed_only_when_every_image_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, ("HE", "PAS"))
    slide_set = _slide_set(tmp_path, targets=("HE", "PAS"))
    real_write = cv2.imwrite

    def fail_pas(path: str, image: np.ndarray) -> bool:
        return False if "__target__PAS" in path else real_write(path, image)

    monkeypatch.setattr(processor_module.cv2, "imwrite", fail_pas)

    with pytest.raises(OSError, match="Could not write patch"):
        SlideSetProcessor(config, slide_set, "train").process()

    written = sorted((tmp_path / "splits" / "train" / slide_set.set_id).iterdir())
    assert written == []


def _sample_images(root: Path) -> dict[Path, np.ndarray]:
    """One M=2 sample: two inputs, two targets and two target masks, in write order."""
    rgb = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    root.mkdir(parents=True, exist_ok=True)
    names = ("s__input__LF", "s__input__AF", "s__target__HE", "s__target__PAS")
    images: dict[Path, np.ndarray] = {root / f"{name}.png": rgb for name in names}
    images.update({root / f"s__foreground_mask__{t}.png": mask for t in ("HE", "PAS")})
    return images


def _processor(tmp_path: Path) -> SlideSetProcessor:
    return SlideSetProcessor(
        _config(tmp_path, ("HE", "PAS")), _slide_set(tmp_path, targets=("HE", "PAS"))
    )


def test_written_rgb_patches_and_masks_are_verified(tmp_path: Path) -> None:
    images = _sample_images(tmp_path / "out")

    _processor(tmp_path)._write_sample(images)

    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(
        path.name for path in images
    )


def _fail_on(monkeypatch: pytest.MonkeyPatch, marker: str, effect: str) -> None:
    real_write = cv2.imwrite

    def write(path: str, image: np.ndarray) -> bool:
        if marker not in path:
            return real_write(path, image)
        if effect == "false":
            return False
        if effect == "corrupt":
            Path(path).write_bytes(b"not an image")
            return True
        if effect == "vanish":
            return True
        assert effect == "resized"
        return real_write(path, image[:4, :6])

    monkeypatch.setattr(processor_module.cv2, "imwrite", write)


@pytest.mark.parametrize(
    ("marker", "effect", "match"),
    [
        ("__target__HE", "false", "Could not write patch"),
        ("__target__PAS", "false", "Could not write patch"),
        ("__target__PAS", "corrupt", "not a readable image"),
        ("__target__PAS", "vanish", "not a readable image"),
        ("__target__PAS", "resized", "is 6x4, expected 8x8"),
        ("__foreground_mask__PAS", "resized", "is 6x4, expected 8x8"),
        ("__foreground_mask__HE", "corrupt", "not a readable image"),
    ],
)
def test_a_failed_or_unverifiable_write_removes_every_file_of_the_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str, effect: str, match: str
) -> None:
    out = tmp_path / "out"
    images = _sample_images(out)
    unrelated = out / "unrelated.png"
    unrelated.write_bytes(b"user file")
    _fail_on(monkeypatch, marker, effect)

    with pytest.raises(OSError, match=match):
        _processor(tmp_path)._write_sample(images)

    # Earlier inputs/targets of the same sample and the failing file itself are gone.
    assert [path.name for path in out.iterdir()] == ["unrelated.png"]
    assert unrelated.read_bytes() == b"user file"


def test_a_failed_write_keeps_a_pre_existing_file_it_did_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    images = _sample_images(out)
    existing = out / "s__target__PAS.png"
    existing.write_bytes(b"previous content")
    _fail_on(monkeypatch, "__target__PAS", "false")

    with pytest.raises(OSError):
        _processor(tmp_path)._write_sample(images)

    assert [path.name for path in out.iterdir()] == ["s__target__PAS.png"]
    assert existing.read_bytes() == b"previous content"


@pytest.mark.parametrize("effect", ["corrupt", "resized"])
def test_no_manifest_row_is_accepted_for_an_unverified_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect: str
) -> None:
    config = replace(
        _config(tmp_path, ("HE", "PAS")),
        masks=MaskConfig(save_patch_masks=True),
        alignment=AlignmentConfig(on_failure="skip_set"),
    )
    slide_set = _slide_set(tmp_path, targets=("HE", "PAS"))
    _fail_on(monkeypatch, "__target__PAS", effect)

    result = SlideSetProcessor(config, slide_set, "train").process()

    assert result.valid_rows == ()
    assert result.error is not None and "Written patch" in result.error
    written = tmp_path / "splits" / "train" / slide_set.set_id
    assert not written.exists() or list(written.iterdir()) == []
