from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pyvips
import torch
from PIL import Image

from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    _run_single_image_inference,
)
from virtual_staining.utils.image_io import (
    ImageMetadata,
    OpenSlideRegionImageReader,
    open_image_reader,
    write_pyramidal_tiff_from_raw_rgb,
)


def test_pyramidal_tiff_round_trip_uses_required_native_libraries(tmp_path: Path) -> None:
    raw_path = tmp_path / "generated.rgb"
    output_path = tmp_path / "output.tif"
    pixels = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
    pixels.tofile(raw_path)
    write_pyramidal_tiff_from_raw_rgb(
        raw_path, output_path, ImageMetadata(512, 512, mpp_x=0.5, mpp_y=0.5)
    )
    reader = open_image_reader(output_path)
    try:
        assert isinstance(reader, OpenSlideRegionImageReader)
        assert reader.metadata.level_count > 1
        assert reader.metadata.mpp_x == pytest.approx(0.5)
        np.testing.assert_array_equal(reader.read_full()[:, :, ::-1], pixels)
    finally:
        reader.close()


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_pyvips_load_failure_is_not_an_optional_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "pyvips":
            raise error_type("broken pyvips runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    expected = ImportError if error_type is ImportError else RuntimeError
    message = (
        "broken pyvips runtime" if error_type is ImportError else "Could not load native libvips"
    )
    with pytest.raises(expected, match=message):
        write_pyramidal_tiff_from_raw_rgb(
            tmp_path / "raw.rgb", tmp_path / "output.tif", ImageMetadata(1, 1)
        )


def test_native_write_error_keeps_context_and_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output.tif"
    output_path.write_bytes(b"existing output")

    def fail(*args, **kwargs):
        raise pyvips.Error("write failed")

    monkeypatch.setattr(pyvips.Image, "rawload", fail)
    with pytest.raises(RuntimeError, match="Could not write pyramidal TIFF"):
        write_pyramidal_tiff_from_raw_rgb(tmp_path / "raw.rgb", output_path, ImageMetadata(1, 1))
    assert output_path.read_bytes() == b"existing output"


def test_large_tiled_inference_requires_a_compatible_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(image_path)
    runtime = InferenceRuntime(
        lambda inputs: pytest.fail("predictor must not run"),
        PredictionContract(("LF",), (4, 4)),
        torch.device("cpu"),
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    with pytest.raises(RuntimeError, match="OpenSlide-compatible and use the OpenSlide backend"):
        _run_single_image_inference(
            runtime, {"LF": image_path}, tmp_path / "output.tif", mode="tile"
        )


# Height stays a multiple of the 256-px TIFF tile: OpenSlide 4.0.0 with libtiff 4.7.1
# misreads the bottom partial tile row of generic TIFFs. The width still ends in a
# partial tile, so non-aligned edge geometry is exercised.
WSI_SIZE = (300, 256)


def _slide(path: Path, mpp: float | None, seed: int = 0) -> np.ndarray:
    width, height = WSI_SIZE
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    raw = path.with_suffix(".rgb")
    pixels.tofile(raw)
    write_pyramidal_tiff_from_raw_rgb(raw, path, ImageMetadata(width, height, mpp_x=mpp, mpp_y=mpp))
    raw.unlink()
    return pixels


def _identity_runtime(calls: list[int] | None = None) -> InferenceRuntime:
    def predictor(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if calls is not None:
            calls.append(1)
        return inputs["LF"]

    return InferenceRuntime(
        predictor, PredictionContract(("AF", "LF"), (64, 64)), torch.device("cpu")
    )


def _wsi_inputs(tmp_path: Path, af_mpp: float | None, lf_mpp: float | None) -> dict[str, Path]:
    inputs = {"AF": tmp_path / "in" / "af.tif", "LF": tmp_path / "in" / "lf.tif"}
    inputs["AF"].parent.mkdir()
    _slide(inputs["AF"], af_mpp, seed=1)
    _slide(inputs["LF"], lf_mpp, seed=2)
    return inputs


def _run_wsi(runtime: InferenceRuntime, inputs: dict[str, Path], output: Path) -> None:
    _run_single_image_inference(runtime, inputs, output, mode="tile", tile_overlap=8)


def _output_listing(output: Path) -> list[str]:
    return sorted(path.name for path in output.parent.iterdir())


def test_wsi_inference_reads_regions_and_publishes_same_grid_with_mpp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _wsi_inputs(tmp_path, 0.5, 0.5)
    regions: list[tuple[int, int]] = []
    original = OpenSlideRegionImageReader.read_region

    def recording_read_region(self, x, y, width, height):
        regions.append((width, height))
        return original(self, x, y, width, height)

    def forbidden(self):
        raise AssertionError("full-image decode")

    monkeypatch.setattr(OpenSlideRegionImageReader, "read_region", recording_read_region)
    monkeypatch.setattr(OpenSlideRegionImageReader, "read_full", forbidden)
    output = tmp_path / "out" / "generated.tif"
    output.parent.mkdir()
    _run_wsi(_identity_runtime(), inputs, output)
    monkeypatch.setattr(OpenSlideRegionImageReader, "read_region", original)

    assert regions and max(max(size) for size in regions) <= 64
    assert _output_listing(output) == ["generated.tif"]
    reader = open_image_reader(output)
    try:
        assert reader.size == WSI_SIZE
        assert reader.metadata.mpp_x == pytest.approx(0.5)
        expected = open_image_reader(inputs["LF"])
        try:
            np.testing.assert_allclose(
                reader.read_region(0, 0, *WSI_SIZE).astype(int),
                expected.read_region(0, 0, *WSI_SIZE).astype(int),
                atol=1,
            )
        finally:
            expected.close()
    finally:
        reader.close()


@pytest.mark.parametrize(("af_mpp", "lf_mpp", "expected"), [(None, None, None), (None, 0.25, 0.25)])
def test_unknown_wsi_mpp_stays_unknown(
    tmp_path: Path, af_mpp: float | None, lf_mpp: float | None, expected: float | None
) -> None:
    inputs = _wsi_inputs(tmp_path, af_mpp, lf_mpp)
    output = tmp_path / "generated.tif"
    _run_wsi(_identity_runtime(), inputs, output)
    reader = open_image_reader(output)
    try:
        assert reader.size == WSI_SIZE
        assert reader.metadata.mpp_x == (pytest.approx(expected) if expected else None)
        assert reader.metadata.mpp_y == (pytest.approx(expected) if expected else None)
    finally:
        reader.close()


def test_conflicting_wsi_mpp_fails_before_prediction(tmp_path: Path) -> None:
    inputs = _wsi_inputs(tmp_path, 0.5, 0.25)
    calls: list[int] = []
    with pytest.raises(ValueError, match="mpp_x values conflict"):
        _run_wsi(_identity_runtime(calls), inputs, tmp_path / "out" / "generated.tif")
    assert calls == []


def test_insufficient_wsi_scratch_fails_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import virtual_staining.inference.single as single

    inputs = _wsi_inputs(tmp_path, 0.5, 0.5)
    calls: list[int] = []
    output = tmp_path / "out" / "generated.tif"
    monkeypatch.setattr(single.shutil, "disk_usage", lambda path: SimpleNamespace(free=1000))
    required = WSI_SIZE[0] * WSI_SIZE[1] * 3 * 5
    with pytest.raises(OSError, match=rf"at least {required} bytes.*{output.parent}.*only 1000"):
        _run_wsi(_identity_runtime(calls), inputs, output)
    assert calls == []
    assert _output_listing(output) == []


def test_wsi_predictor_failure_cleans_scratch_and_keeps_existing_output(tmp_path: Path) -> None:
    inputs = _wsi_inputs(tmp_path, 0.5, 0.5)
    output = tmp_path / "out" / "generated.tif"
    output.parent.mkdir()
    output.write_bytes(b"previous result")

    def failing(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs["LF"][:, :, :-1]

    runtime = InferenceRuntime(
        failing, PredictionContract(("AF", "LF"), (64, 64)), torch.device("cpu")
    )
    with pytest.raises(ValueError, match="same-grid"):
        _run_wsi(runtime, inputs, output)
    assert _output_listing(output) == ["generated.tif"]
    assert output.read_bytes() == b"previous result"


def test_wsi_writer_failure_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import virtual_staining.inference.single as single

    inputs = _wsi_inputs(tmp_path, 0.5, 0.5)
    output = tmp_path / "out" / "generated.tif"
    output.parent.mkdir()
    output.write_bytes(b"previous result")

    def failing_writer(raw_path: Path, output_path: Path, metadata: ImageMetadata) -> None:
        raw_path.with_suffix(".tif").write_bytes(b"partial")
        raise RuntimeError("Generated dimensions differ")

    monkeypatch.setattr(single, "write_pyramidal_tiff_from_raw_rgb", failing_writer)
    with pytest.raises(RuntimeError, match="Generated dimensions differ"):
        _run_wsi(_identity_runtime(), inputs, output)
    assert _output_listing(output) == ["generated.tif"]
    assert output.read_bytes() == b"previous result"
