from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from PIL import Image

from virtual_staining.config.run import RunConfig
from virtual_staining.data.manifest import ManifestMetadata, ManifestRecord
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.inference.runner import predict_batch
from virtual_staining.inference.single import (
    SingleInferenceResult,
    _predict_images,
    _run_tiled_prediction,
    run_image_directory_inference,
    run_image_path_inference,
)
from virtual_staining.models.generator import ConcatUNetGenerator


def test_manifest_inference_passes_named_inputs_in_order() -> None:
    generator = ConcatUNetGenerator(("AF", "LF", "TH"), base_channels=4)
    generator.eval()
    inputs = {name: torch.zeros(1, 3, 32, 32) for name in ("AF", "LF", "TH")}
    output = predict_batch(generator, inputs, torch.device("cpu"))
    assert output.shape == (1, 3, 32, 32)


def test_output_naming_uses_target_suffix(tmp_path: Path) -> None:
    metadata = ManifestMetadata("3.0", ("LF",), "LF", "target")
    record = ManifestRecord(
        "S1__x00000000_y00000000",
        "S1",
        "test",
        {"LF": Path("splits/test/input.png")},
        Path("splits/test/target.TIFF"),
        0,
        0,
        16,
        16,
    )
    assert (
        generated_path_for_record(record, tmp_path).name
        == "S1__x00000000_y00000000_target_generated.tiff"
    )
    assert metadata.target_modality == "target"


def test_predict_images_passes_all_modalities_in_generator_order() -> None:
    generator = ConcatUNetGenerator(("LF", "AF"), base_channels=4)
    generator.eval()
    images = {
        "AF": Image.new("RGB", (32, 32), color=(40, 50, 60)),
        "LF": Image.new("RGB", (32, 32), color=(10, 20, 30)),
    }
    seen: list[tuple[str, ...]] = []

    def transform(image: Image.Image) -> torch.Tensor:
        pixel = cast(tuple[int, int, int], image.getpixel((0, 0)))
        return torch.full((3, 32, 32), pixel[0] / 255)

    original_forward = generator.forward

    def recording_forward(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        seen.append(tuple(inputs))
        return original_forward(inputs)

    generator.forward = recording_forward  # type: ignore[method-assign]
    output = _predict_images(images, generator, torch.device("cpu"), transform)  # type: ignore[arg-type]

    assert output.shape == (3, 32, 32)
    assert seen == [("LF", "AF")]


def test_tiled_prediction_uses_shared_coordinates_for_all_modalities() -> None:
    generator = ConcatUNetGenerator(("LF", "AF"), base_channels=4)
    generator.eval()
    images = {
        "LF": Image.new("RGB", (8, 8), color=(10, 20, 30)),
        "AF": Image.new("RGB", (8, 8), color=(40, 50, 60)),
    }
    seen: list[tuple[tuple[int, int], ...]] = []

    def recording_predict(
        tiles: dict[str, Image.Image],
        model: torch.nn.Module,
        device: torch.device,
        transform: object,
    ) -> torch.Tensor:
        seen.append(tuple(tile.size for tile in tiles.values()))
        return torch.zeros(3, 4, 4)

    import virtual_staining.inference.single as single

    original = single._predict_images
    single._predict_images = recording_predict  # type: ignore[assignment]
    try:
        output = _run_tiled_prediction(
            images, generator, torch.device("cpu"), (4, 4), tile_overlap=0
        )
    finally:
        single._predict_images = original

    assert output.shape == (3, 8, 8)
    assert seen == [((4, 4), (4, 4))] * 4


def _write_image(path: Path, size: tuple[int, int] = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def test_directory_inputs_pair_exact_relative_paths_and_preserve_subdirectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lf_root = tmp_path / "lf"
    af_root = tmp_path / "af"
    for root in (lf_root, af_root):
        _write_image(root / "top.png")
        _write_image(root / "nested" / "sample.png")

    config = SimpleNamespace(
        model=SimpleNamespace(inputs=("LF", "AF")),
        inference=SimpleNamespace(output_dir=None),
    )
    runtime = SimpleNamespace(
        generator=SimpleNamespace(input_names=("LF", "AF")),
        paths=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"),
        checkpoint_path=tmp_path / "checkpoint.pth",
        image_size=(4, 4),
        device=torch.device("cpu"),
    )
    results: list[SingleInferenceResult] = []

    import virtual_staining.inference.single as single

    monkeypatch.setattr(single, "_build_runtime", lambda config: runtime)

    def fake_run_one(
        runtime: object,
        input_images: dict[str, Path],
        *,
        output_path: Path,
        mode: str,
        tile_overlap: int,
    ) -> SingleInferenceResult:
        result = SingleInferenceResult(
            input_paths=input_images,
            output_path=output_path,
            checkpoint_path=Path("checkpoint.pth"),
            image_size=(4, 4),
            mode=mode,
            device="cpu",
        )
        results.append(result)
        return result

    monkeypatch.setattr(single, "_run_one_image", fake_run_one)

    result = run_image_directory_inference(
        cast(RunConfig, config),
        {"AF": af_root, "LF": lf_root},
        tmp_path / "out",
        recursive=True,
    )

    assert result.input_dirs == {"LF": lf_root, "AF": af_root}
    assert [item.output_path.relative_to(tmp_path / "out").as_posix() for item in results] == [
        "nested/sample_target_generated.png",
        "top_target_generated.png",
    ]
    assert all(tuple(item.input_paths) == ("LF", "AF") for item in results)


def test_directory_input_set_mismatch_names_offending_modality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lf_root = tmp_path / "lf"
    af_root = tmp_path / "af"
    _write_image(lf_root / "sample.png")
    _write_image(af_root / "other.png")
    config = SimpleNamespace(
        model=SimpleNamespace(inputs=("LF", "AF")),
        inference=SimpleNamespace(output_dir=None),
    )

    import virtual_staining.inference.single as single

    monkeypatch.setattr(
        single,
        "_build_runtime",
        lambda config: pytest.fail("checkpoint must not load for path mismatch"),
    )
    with pytest.raises(ValueError, match=r"Input modality AF.*missing=.*sample.*extra=.*other"):
        run_image_directory_inference(cast(RunConfig, config), {"LF": lf_root, "AF": af_root})


def test_file_inputs_reject_unequal_dimensions_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lf_path = tmp_path / "lf.png"
    af_path = tmp_path / "af.png"
    _write_image(lf_path, (4, 4))
    _write_image(af_path, (5, 4))
    config = SimpleNamespace(
        model=SimpleNamespace(inputs=("LF", "AF")),
        inference=SimpleNamespace(output_dir=None),
    )
    runtime = SimpleNamespace(
        generator=SimpleNamespace(input_names=("LF", "AF")),
        paths=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"),
        checkpoint_path=tmp_path / "checkpoint.pth",
        image_size=(4, 4),
        device=torch.device("cpu"),
    )

    import virtual_staining.inference.single as single

    monkeypatch.setattr(single, "_build_runtime", lambda config: runtime)
    with pytest.raises(ValueError, match="dimensions must match"):
        run_image_path_inference(
            cast(RunConfig, config),
            {"AF": af_path, "LF": lf_path},
            tmp_path / "out.png",
        )


def test_file_and_directory_inputs_are_rejected_before_dispatch(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    directory = tmp_path / "images"
    _write_image(image)
    directory.mkdir()
    config = SimpleNamespace(model=SimpleNamespace(inputs=("LF", "AF")))
    with pytest.raises(ValueError, match="files or all input paths must be directories"):
        run_image_path_inference(cast(RunConfig, config), {"LF": image, "AF": directory})
