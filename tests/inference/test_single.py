from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from PIL import Image

from virtual_staining.data.manifest import ManifestRecord
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.inference.runner import predict_batch
from virtual_staining.inference.single import (
    DirectoryInferenceResult,
    InferenceRuntime,
    PredictionContract,
    SingleInferenceResult,
    _predict_images,
    _run_tiled_prediction,
    run_image_path_inference,
)
from virtual_staining.models.generator import ConcatUNetGenerator


def test_manifest_inference_passes_named_inputs_in_order() -> None:
    generator = ConcatUNetGenerator(("AF", "LF", "TH"), ("PAS", "HE"), base_channels=4)
    generator.eval()
    inputs = {name: torch.zeros(1, 3, 32, 32) for name in ("AF", "LF", "TH")}
    outputs = predict_batch(generator, inputs, torch.device("cpu"), ("PAS", "HE"))
    assert list(outputs) == ["PAS", "HE"]
    assert all(output.shape == (1, 3, 32, 32) for output in outputs.values())


def test_output_naming_uses_each_output_domain_suffix(tmp_path: Path) -> None:
    record = ManifestRecord(
        sample_id="S1__x00000000_y00000000",
        set_id="S1",
        split="test",
        input_paths={"LF": Path("splits/test/input.png")},
        target_paths={"HE": Path("splits/test/he.TIFF"), "PAS": Path("splits/test/pas.png")},
        foreground_mask_paths={"HE": None, "PAS": None},
        x=0,
        y=0,
        width=16,
        height=16,
    )
    assert generated_path_for_record(record, tmp_path, "HE") == (
        tmp_path / "HE" / "S1__x00000000_y00000000_generated.tiff"
    )
    assert generated_path_for_record(record, tmp_path, "PAS").suffix == ".png"


def test_predict_images_passes_all_modalities_in_generator_order() -> None:
    generator = ConcatUNetGenerator(("LF", "AF"), ("HE",), base_channels=4)
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

    def recording_forward(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seen.append(tuple(inputs))
        return original_forward(inputs)

    generator.forward = recording_forward  # type: ignore[method-assign]
    runtime = InferenceRuntime(
        generator, PredictionContract(("LF", "AF"), ("HE",), (32, 32)), torch.device("cpu")
    )
    output = _predict_images(images, runtime, transform)  # type: ignore[arg-type]

    assert output["HE"].shape == (3, 32, 32)
    assert seen == [("LF", "AF")]


def test_tiled_prediction_uses_shared_coordinates_for_all_modalities() -> None:
    generator = ConcatUNetGenerator(("LF", "AF"), ("HE", "PAS"), base_channels=4)
    generator.eval()
    images = {
        "LF": Image.new("RGB", (8, 8), color=(10, 20, 30)),
        "AF": Image.new("RGB", (8, 8), color=(40, 50, 60)),
    }
    seen: list[tuple[tuple[int, int], ...]] = []

    def recording_predict(
        tiles: dict[str, Image.Image], runtime: InferenceRuntime, transform: object
    ) -> dict[str, torch.Tensor]:
        seen.append(tuple(tile.size for tile in tiles.values()))
        return {"HE": torch.zeros(3, 4, 4), "PAS": torch.ones(3, 4, 4)}

    import virtual_staining.inference.single as single

    original = single._predict_images
    single._predict_images = recording_predict  # type: ignore[assignment]
    try:
        runtime = InferenceRuntime(
            generator, PredictionContract(("LF", "AF"), ("HE", "PAS"), (4, 4)), torch.device("cpu")
        )
        output = _run_tiled_prediction(images, runtime, tile_overlap=0)
    finally:
        single._predict_images = original

    # One prediction per tile feeds one accumulator per output.
    assert {name: tuple(value.shape) for name, value in output.items()} == {
        "HE": (3, 8, 8),
        "PAS": (3, 8, 8),
    }
    assert torch.equal(output["HE"], torch.zeros(3, 8, 8))
    assert torch.equal(output["PAS"], torch.ones(3, 8, 8))
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

    runtime = InferenceRuntime(
        predictor=lambda inputs: pytest.fail("predictor must not run"),
        contract=PredictionContract(("LF", "AF"), ("HE",), (4, 4)),
        device=torch.device("cpu"),
    )
    results: list[SingleInferenceResult] = []

    import virtual_staining.inference.single as single

    def runtime_factory() -> InferenceRuntime:
        return runtime

    def fake_run_one(
        runtime: object,
        input_images: dict[str, Path],
        *,
        output_paths: dict[str, Path],
        mode: str,
        tile_overlap: int,
    ) -> SingleInferenceResult:
        result = SingleInferenceResult(
            input_paths=input_images,
            output_paths=output_paths,
            image_size=(4, 4),
            mode=mode,
            device="cpu",
        )
        results.append(result)
        return result

    monkeypatch.setattr(single, "_run_one_image", fake_run_one)

    result = run_image_path_inference(
        runtime_factory,
        {"LF": lf_root, "AF": af_root},
        tmp_path / "out",
        recursive=True,
    )
    assert isinstance(result, DirectoryInferenceResult)

    assert result.input_dirs == {"LF": lf_root, "AF": af_root}
    assert [
        item.output_paths["HE"].relative_to(tmp_path / "out").as_posix() for item in results
    ] == ["nested/HE/sample_generated.png", "HE/top_generated.png"]
    assert all(tuple(item.input_paths) == ("LF", "AF") for item in results)


def test_directory_input_set_mismatch_names_offending_modality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lf_root = tmp_path / "lf"
    af_root = tmp_path / "af"
    _write_image(lf_root / "sample.png")
    _write_image(af_root / "other.png")

    def runtime_factory() -> InferenceRuntime:
        pytest.fail("checkpoint must not load for path mismatch")

    with pytest.raises(ValueError, match=r"Input modality AF.*missing=.*sample.*extra=.*other"):
        run_image_path_inference(runtime_factory, {"LF": lf_root, "AF": af_root})


def test_file_inputs_reject_unequal_dimensions_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lf_path = tmp_path / "lf.png"
    af_path = tmp_path / "af.png"
    _write_image(lf_path, (4, 4))
    _write_image(af_path, (5, 4))
    runtime = InferenceRuntime(
        predictor=lambda inputs: pytest.fail("predictor must not run"),
        contract=PredictionContract(("LF", "AF"), ("HE",), (4, 4)),
        device=torch.device("cpu"),
    )

    def runtime_factory() -> InferenceRuntime:
        return runtime

    with pytest.raises(ValueError, match="dimensions must match"):
        run_image_path_inference(
            runtime_factory,
            {"AF": af_path, "LF": lf_path},
            tmp_path / "out.png",
        )


def test_file_and_directory_inputs_are_rejected_before_dispatch(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    directory = tmp_path / "images"
    _write_image(image)
    directory.mkdir()
    with pytest.raises(ValueError, match="files or all input paths must be directories"):
        run_image_path_inference(
            lambda: pytest.fail("runtime must not load"),
            {"LF": image, "AF": directory},
        )
