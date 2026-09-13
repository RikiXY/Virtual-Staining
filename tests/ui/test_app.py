from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import virtual_staining.ui.app as ui_app
from virtual_staining.applications.ui_inference import UIInferenceError, UIInferenceService
from virtual_staining.checkpoint_contract import (
    CHECKPOINT_FORMAT_VERSION,
    NORMALIZATION_CONTRACT,
    make_arch_metadata,
)
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator


def _write_checkpoint(
    path: Path,
    *,
    input_names: tuple[str, ...] = ("label_free",),
    target_modality: str = "H&E",
    image_size: tuple[int, int] = (32, 32),
) -> Path:
    generator = ConcatUNetGenerator(input_names, base_channels=4)
    discriminator = PatchGANDiscriminator(in_channels=3 * len(input_names) + 3, ndf=4)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": make_arch_metadata(
            generator, discriminator, target_modality=target_modality
        ),
        "normalization_contract": NORMALIZATION_CONTRACT,
        "generator_state_dict": generator.state_dict(),
        "image_size": image_size,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def _service(tmp_path: Path) -> UIInferenceService:
    return UIInferenceService(
        Path("portable-checkpoints"),
        Path("portable-outputs"),
        working_directory=tmp_path,
    )


def test_model_discovery_uses_configurable_directory_and_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(
        tmp_path / "portable-checkpoints" / "experiment-a" / "model.pth",
        input_names=("unstained",),
        target_modality="H&E",
    )
    service = _service(tmp_path)

    catalog = service.discover_models()

    assert catalog.checkpoint_directory_exists is True
    assert catalog.issues == ()
    assert len(catalog.models) == 1
    descriptor = catalog.models[0]
    assert descriptor.identifier == "experiment-a/model.pth"
    assert descriptor.checkpoint_filename == checkpoint.name
    assert descriptor.input_domains == ("unstained",)
    assert descriptor.target_domain == "H&E"
    assert descriptor.transformation == "unstained → H&E"
    assert descriptor.architecture_id == "concat_unet"
    assert descriptor.model_class == "ConcatUNetGenerator"
    assert descriptor.checkpoint_schema_version == 3
    assert descriptor.image_size == (32, 32)
    assert descriptor.channels_per_input == 3


def test_empty_checkpoint_catalog_is_valid_for_missing_directory(tmp_path: Path) -> None:
    catalog = _service(tmp_path).discover_models()

    assert catalog.models == ()
    assert catalog.issues == ()
    assert catalog.checkpoint_directory_exists is False


def test_incompatible_checkpoints_are_reported_and_not_selectable(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path / "portable-checkpoints" / "legacy.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["format_version"] = 2
    torch.save(checkpoint, checkpoint_path)

    catalog = _service(tmp_path).discover_models()

    assert catalog.models == ()
    assert len(catalog.issues) == 1
    assert catalog.issues[0].checkpoint == "legacy.pth"
    assert "requires the current v3 format" in catalog.issues[0].reason


def test_multi_input_checkpoint_is_incompatible_with_single_patch_ui(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path / "portable-checkpoints" / "multi.pth",
        input_names=("LF", "AF"),
    )

    catalog = _service(tmp_path).discover_models()

    assert catalog.models == ()
    assert "exactly one input modality" in catalog.issues[0].reason


@pytest.mark.parametrize("mode", ["L", "RGBA", "CMYK", "P"])
def test_ui_service_rejects_non_rgb_image_modes(tmp_path: Path, mode: str) -> None:
    _write_checkpoint(tmp_path / "portable-checkpoints" / "model.pth")
    service = _service(tmp_path)
    descriptor = service.discover_models().models[0]

    with pytest.raises(UIInferenceError, match=r"requires an RGB image.*Received image mode"):
        service.validate_input(descriptor.identifier, Image.new(mode, (32, 32)))


def test_ui_service_preserves_exact_image_size_validation(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "portable-checkpoints" / "model.pth")
    service = _service(tmp_path)
    descriptor = service.discover_models().models[0]

    with pytest.raises(UIInferenceError, match=r"Expected input size: 32 × 32.*64 × 32"):
        service.validate_input(descriptor.identifier, Image.new("RGB", (64, 32)))


def test_inference_result_contains_portable_provenance(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path / "portable-checkpoints" / "nested" / "model.pth",
        input_names=("label_free",),
        target_modality="H&E",
    )
    service = _service(tmp_path)
    descriptor = service.discover_models().models[0]

    result = service.run_inference(
        descriptor.identifier,
        Image.new("RGB", (32, 32), color=(40, 80, 120)),
        "uploads/sample_source.tif",
    )
    payload = result.provenance.to_dict()

    assert result.generated_image.mode == "RGB"
    assert payload["transformation"] == "label_free → H&E"
    assert payload["model"]["identifier"] == "nested/model.pth"
    assert payload["model"]["checkpoint_schema_version"] == 3
    assert payload["model"]["required_image_size"] == [32, 32]
    assert payload["input"] == {
        "filename": "sample_source.tif",
        "image_size": [32, 32],
        "image_mode": "RGB",
    }
    assert payload["output"]["filename"] == "sample_source_generated.png"
    assert payload["runtime"]["device"] in {"cpu", "cuda"}
    assert str(tmp_path) not in json.dumps(payload)


def test_save_result_writes_json_sidecar_and_does_not_overwrite(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "portable-checkpoints" / "model.pth")
    service = _service(tmp_path)
    descriptor = service.discover_models().models[0]
    result = service.run_inference(
        descriptor.identifier,
        Image.new("RGB", (32, 32)),
        "sample.tif",
    )

    first = service.save_result(result)
    second = service.save_result(result)

    assert first.image_path == tmp_path / "portable-outputs" / "sample_generated.png"
    assert first.sidecar_path == tmp_path / "portable-outputs" / "sample_generated.json"
    assert second.image_path.name == "sample_generated_2.png"
    assert second.sidecar_path.name == "sample_generated_2.json"
    with Image.open(first.image_path) as saved:
        assert saved.mode == "RGB"
        assert saved.size == (32, 32)
    sidecar = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == first.provenance.to_dict()
    assert sidecar["output"]["filename"] == first.image_path.name
    assert str(tmp_path) not in first.sidecar_path.read_text(encoding="utf-8")


def test_save_result_rejects_empty_folder(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "portable-checkpoints" / "model.pth")
    service = _service(tmp_path)
    descriptor = service.discover_models().models[0]
    result = service.run_inference(
        descriptor.identifier,
        Image.new("RGB", (32, 32)),
        "sample.tif",
    )

    with pytest.raises(UIInferenceError, match="must not be empty"):
        service.save_result(result, "  ")


def test_nicegui_page_has_no_repository_checkpoint_mapping() -> None:
    source = Path(ui_app.__file__).read_text(encoding="utf-8")

    assert "local_workspace" not in source
    assert "lf-to-he" not in source
    assert "TRANSFORMATIONS" not in source
    assert "models.generator" not in source
    assert "inference.runner" not in source
