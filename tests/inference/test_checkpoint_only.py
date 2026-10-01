from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.checkpoint_helpers import write_ui_checkpoint
from virtual_staining.applications.ui_inference import UIInferenceService
from virtual_staining.inference.single import validate_patch_image


def test_current_checkpoint_uses_shared_prediction_contract(tmp_path: Path) -> None:
    path = write_ui_checkpoint(tmp_path / "model.pth")
    service = UIInferenceService(tmp_path, tmp_path / "outputs")
    model = service.discover_models().models[0]
    result = service.run_inference(model.identifier, Image.new("RGB", (32, 32)), "source.png")
    assert result.generated_image.size == (32, 32)
    assert result.generated_image.mode == "RGB"
    payload = torch.load(path, weights_only=True)
    payload["method"]["components"]["generator"]["version"] = "unregistered"
    torch.save(payload, path)
    assert not service.discover_models().models


@pytest.mark.parametrize("mode", ["L", "RGBA", "CMYK", "P"])
def test_patch_rejects_mode_conversion(mode: str) -> None:
    with pytest.raises(ValueError, match="requires an RGB image"):
        validate_patch_image(Image.new(mode, (32, 32)), (32, 32))


def test_patch_rejects_implicit_resizing() -> None:
    with pytest.raises(ValueError, match="Expected input size: 32 × 32.*Received: 64 × 32"):
        validate_patch_image(Image.new("RGB", (64, 32)), (32, 32))


def test_multi_output_checkpoint_is_reported_without_dropping_outputs(tmp_path: Path) -> None:
    write_ui_checkpoint(tmp_path / "model.pth", output_names=("HE", "PAS"))
    catalog = UIInferenceService(tmp_path, tmp_path / "outputs").discover_models()
    assert not catalog.models
    assert "exactly one output" in catalog.issues[0].reason
