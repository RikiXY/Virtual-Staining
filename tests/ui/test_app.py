from pathlib import Path

import pytest
from PIL import Image

import virtual_staining.ui.app as ui_app


def test_save_generated_image_uses_relative_custom_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui_app, "_PROJECT_ROOT", tmp_path)
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))

    output_path = ui_app._save_generated_image(image, "sample_source.tif", "custom/results")

    assert output_path == tmp_path / "custom" / "results" / "sample_source_generated.png"
    with Image.open(output_path) as saved:
        assert saved.mode == "RGB"
        assert saved.size == (8, 8)


def test_save_generated_image_does_not_overwrite_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui_app, "_PROJECT_ROOT", tmp_path)
    image = Image.new("RGB", (8, 8))

    first = ui_app._save_generated_image(image, "sample.tif", "outputs")
    second = ui_app._save_generated_image(image, "sample.tif", "outputs")

    assert first.name == "sample_generated.png"
    assert second.name == "sample_generated_2.png"


def test_save_generated_image_rejects_empty_folder() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ui_app._save_generated_image(Image.new("RGB", (8, 8)), "sample.tif", "  ")
