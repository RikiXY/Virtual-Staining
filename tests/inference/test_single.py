from __future__ import annotations

from pathlib import Path
from typing import cast
from weakref import ref

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


@pytest.mark.parametrize("names", [("PAS",), ("PAS", "HE"), ("PAS", "HE", "DAB", "TRI")])
@pytest.mark.parametrize(
    ("size", "overlap", "xs", "ys", "x_weights", "y_weights"),
    [
        ((4, 4), 0, [0], [0], [1] * 4, [1] * 4),
        ((3, 2), 1, [0], [0], [1] * 3, [1] * 2),
        ((7, 5), 0, [0, 3], [0, 1], [1, 1, 1, 2, 1, 1, 1], [1, 2, 2, 2, 1]),
        ((8, 7), 1, [0, 3, 4], [0, 3], [1, 1, 1, 2, 2, 2, 2, 1], [1, 1, 1, 2, 1, 1, 1]),
        ((6, 6), 2, [0, 2], [0, 2], [1, 1, 2, 2, 1, 1], [1, 1, 2, 2, 1, 1]),
    ],
)
def test_tiled_finalization_matches_reference_and_owns_outputs(
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    size: tuple[int, int],
    overlap: int,
    xs: list[int],
    ys: list[int],
    x_weights: list[int],
    y_weights: list[int],
) -> None:
    import virtual_staining.inference.single as single

    width, height = size
    image = Image.new("RGB", size)
    image.putdata([(x, y, 37) for y in range(height) for x in range(width)])
    source_bytes = image.tobytes()
    coordinates = [(x, y) for y in ys for x in xs]
    ramp = torch.linspace(-0.1, 0.1, 16).reshape(4, 4)
    # Bypass the runner's earlier clamp to exercise both final clamp boundaries.
    predictions = [
        {
            name: torch.stack((-1 + ramp, 0.2 + ramp, 1.5 + ramp)) + i * 0.037 + j * 0.013
            for j, name in enumerate(names)
        }
        for i in range(len(coordinates))
    ]
    snapshots = [{name: value.clone() for name, value in tile.items()} for tile in predictions]
    prediction_refs = [ref(value) for tile in predictions for value in tile.values()]
    sums = {name: torch.zeros(3, height, width) for name in names}
    for i, (x, y) in enumerate(coordinates):
        w, h = min(4, width - x), min(4, height - y)
        for name in names:
            sums[name][:, y : y + h, x : x + w] += predictions[i][name][:, :h, :w]
    weights = (
        torch.tensor(y_weights, dtype=torch.float32)[:, None]
        * torch.tensor(x_weights, dtype=torch.float32)[None, :]
    ).unsqueeze(0)
    expected = {name: (value / weights.clamp_min(1.0)).clamp(0, 1) for name, value in sums.items()}
    calls = 0

    def predict(tiles: dict[str, Image.Image], runtime: object, transform: object):
        nonlocal calls
        x, y = coordinates[calls]
        expected_tile = Image.new("RGB", (4, 4), "white")
        expected_tile.paste(image.crop((x, y, min(x + 4, width), min(y + 4, height))), (0, 0))
        assert tiles["LF"].tobytes() == expected_tile.tobytes()
        calls += 1
        return predictions[calls - 1]

    monkeypatch.setattr(single, "_predict_images", predict)
    runtime = InferenceRuntime(
        lambda inputs: pytest.fail("use the local prediction seam"),
        PredictionContract(("LF",), names, (4, 4)),
        torch.device("cpu"),
    )
    outputs = _run_tiled_prediction({"LF": image}, runtime, overlap)

    assert calls == len(coordinates)
    assert tuple(outputs) == names
    assert len({value.untyped_storage().data_ptr() for value in outputs.values()}) == len(names)
    for name, output in outputs.items():
        assert output.dtype == torch.float32 and output.shape == (3, height, width)
        assert torch.equal(output, expected[name])
        assert (output[0] == 0).all() and (output[2] == 1).all()
        assert ((output[1] > 0) & (output[1] < 1)).all()
        assert all(
            output.untyped_storage().data_ptr() != tile[name].untyped_storage().data_ptr()
            for tile in predictions
        )
    outputs[names[0]].fill_(0.75)
    assert all(torch.equal(outputs[name], expected[name]) for name in names[1:])
    assert image.tobytes() == source_bytes
    assert all(
        torch.equal(tile[name], snapshot[name])
        for tile, snapshot in zip(predictions, snapshots, strict=True)
        for name in names
    )
    predictions.clear()
    assert all(value() is None for value in prediction_refs)


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
