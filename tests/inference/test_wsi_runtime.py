from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import pyvips
from PIL import Image

from virtual_staining.inference.single import (
    InferenceRuntime,
    _save_pyramidal_tiff,
    run_single_image_inference,
)
from virtual_staining.utils.image_io import (
    ImageMetadata,
    OpenSlideRegionImageReader,
    open_image_reader,
)


def test_pyramidal_tiff_round_trip_uses_required_native_libraries(tmp_path: Path) -> None:
    raw_path = tmp_path / "generated.rgb"
    output_path = tmp_path / "output.tif"
    pixels = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
    pixels.tofile(raw_path)
    _save_pyramidal_tiff(raw_path, output_path, ImageMetadata(512, 512, mpp_x=0.5, mpp_y=0.5))
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
        _save_pyramidal_tiff(tmp_path / "raw.rgb", tmp_path / "output.tif", ImageMetadata(1, 1))


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
        _save_pyramidal_tiff(tmp_path / "raw.rgb", output_path, ImageMetadata(1, 1))
    assert output_path.read_bytes() == b"existing output"


def test_large_tiled_inference_requires_a_compatible_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(image_path)
    runtime = cast(
        InferenceRuntime,
        SimpleNamespace(
            generator=SimpleNamespace(input_names=("LF",)),
            image_size=(4, 4),
        ),
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    with pytest.raises(RuntimeError, match="OpenSlide-compatible and use the OpenSlide backend"):
        run_single_image_inference(
            runtime, {"LF": image_path}, tmp_path / "output.tif", mode="tile"
        )
