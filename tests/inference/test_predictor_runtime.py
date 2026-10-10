"""Direct inference with a caller-constructed predictor: no checkpoint, config or run."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, PngImagePlugin

from virtual_staining.inference import outputs as output_writer
from virtual_staining.inference.outputs import save_rgb
from virtual_staining.inference.single import (
    DirectoryInferenceResult,
    InferenceRuntime,
    PredictionContract,
    SingleInferenceResult,
    run_image_path_inference,
)
from virtual_staining.models.io_contract import denormalize_model_output
from virtual_staining.utils.image_io import PillowRegionImageReader

CPU = torch.device("cpu")


class InMemoryPredictor(torch.nn.Module):
    """Predicts HE = LF and PAS = -LF, checking both modalities share coordinates."""

    def __init__(self, outputs: tuple[str, ...] = ("HE",)) -> None:
        super().__init__()
        self.outputs = outputs
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.calls: list[tuple[str, ...]] = []

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls.append(tuple(inputs))
        assert torch.equal(inputs["AF"], inputs["LF"]), "modalities read different regions"
        signs = {"HE": 1.0, "PAS": -1.0}
        return {name: inputs["LF"] * self.scale * signs[name] for name in self.outputs}


def _random_image(path: Path, size: tuple[int, int], seed: int = 0) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def _runtime(
    predictor: Callable[[dict[str, torch.Tensor]], object],
    names: tuple[str, ...] = ("AF", "LF"),
    size: tuple[int, int] = (16, 16),
    outputs: tuple[str, ...] = ("HE",),
) -> InferenceRuntime:
    return InferenceRuntime(predictor, PredictionContract(names, outputs, size), CPU)  # type: ignore[arg-type]


def _pair(root: Path, size: tuple[int, int], name: str = "sample.png") -> dict[str, Path]:
    for modality in ("AF", "LF"):
        _random_image(root / modality / name, size)
    return {"AF": root / "AF" / name, "LF": root / "LF" / name}


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_in_memory_predictor_runs_file_directory_and_tiled_inference(tmp_path: Path) -> None:
    predictor = InMemoryPredictor().eval()
    runtime = _runtime(predictor)
    exact = _pair(tmp_path / "exact", (16, 16))
    large = _pair(tmp_path / "large", (40, 23))
    for root in (tmp_path / "dirs" / "AF", tmp_path / "dirs" / "LF"):
        _random_image(root / "a.png", (16, 16))
        _random_image(root / "nested" / "b.png", (20, 16))

    single = run_image_path_inference(runtime, exact, tmp_path / "out" / "exact.png")
    tiled = run_image_path_inference(runtime, large, tmp_path / "out" / "large.png", tile_overlap=4)
    directory = run_image_path_inference(
        runtime,
        {"LF": tmp_path / "dirs" / "LF", "AF": tmp_path / "dirs" / "AF"},
        tmp_path / "out" / "dirs",
        recursive=True,
        tile_overlap=4,
    )

    assert isinstance(single, SingleInferenceResult)
    assert isinstance(tiled, SingleInferenceResult)
    assert isinstance(directory, DirectoryInferenceResult)
    assert (single.mode, tiled.mode) == ("resize", "tile")
    assert single.checkpoint_path is None and single.predictor_identity is None
    assert directory.checkpoint_path is None
    assert tuple(directory.input_dirs) == ("AF", "LF")
    assert sorted(
        r.output_paths["HE"].relative_to(directory.output_dir) for r in directory.results
    ) == [Path("HE/a_generated.png"), Path("nested/HE/b_generated.png")]
    assert set(predictor.calls) == {("AF", "LF")}
    assert single.output_paths == {"HE": tmp_path / "out" / "exact.png"}
    assert np.abs(_read(tiled.output_paths["HE"]).astype(int) - _read(large["LF"])).max() <= 1
    # The runtime stays usable and untouched by transport.
    assert not predictor.training
    assert predictor.scale.device == CPU and predictor.scale.requires_grad
    run_image_path_inference(runtime, exact, tmp_path / "out" / "again.png")


def test_runtime_factory_is_still_accepted(tmp_path: Path) -> None:
    paths = _pair(tmp_path, (16, 16))
    result = run_image_path_inference(
        lambda: _runtime(InMemoryPredictor()), paths, tmp_path / "out.png"
    )
    assert isinstance(result, SingleInferenceResult)


@pytest.mark.parametrize(
    ("size", "tile", "overlap"),
    [
        ((16, 16), (16, 16), 4),  # exact configured size, forced tile mode
        ((10, 30), (16, 16), 4),  # width smaller than a tile
        ((30, 10), (16, 16), 4),  # height smaller than a tile
        ((37, 21), (16, 16), 0),  # non-stride-aligned edge tiles, no overlap
        ((37, 21), (16, 12), 5),  # non-square tile, nonzero overlap
    ],
)
def test_tile_geometry_reproduces_identity_on_the_input_grid(
    tmp_path: Path, size: tuple[int, int], tile: tuple[int, int], overlap: int
) -> None:
    paths = _pair(tmp_path, size)
    output = tmp_path / "out.png"
    run_image_path_inference(
        _runtime(InMemoryPredictor(), size=tile), paths, output, mode="tile", tile_overlap=overlap
    )
    generated = _read(output)
    assert generated.shape == (size[1], size[0], 3)  # padding removed
    assert np.abs(generated.astype(int) - _read(paths["LF"])).max() <= 1


def test_tile_count_follows_stride_and_edge_anchoring(tmp_path: Path) -> None:
    predictor = InMemoryPredictor()
    paths = _pair(tmp_path, (37, 21))
    run_image_path_inference(
        _runtime(predictor), paths, tmp_path / "out.png", mode="tile", tile_overlap=4
    )
    # x starts 0, 12, 21; y starts 0, 5
    assert len(predictor.calls) == 6


@pytest.mark.parametrize("overlap", [16, 20])
def test_overlap_not_smaller_than_tile_is_rejected(tmp_path: Path, overlap: int) -> None:
    paths = _pair(tmp_path, (40, 40))
    with pytest.raises(ValueError, match="tile_overlap must be smaller"):
        run_image_path_inference(
            _runtime(InMemoryPredictor()), paths, tmp_path / "out.png", tile_overlap=overlap
        )


def _shift(transform: Callable[[torch.Tensor], object]) -> Callable[..., object]:
    return lambda inputs: {"HE": transform(inputs["LF"])}


MALFORMED = {
    "batch": _shift(lambda x: torch.cat([x, x])),
    "channels": _shift(lambda x: torch.cat([x, x[:, :1]], 1)),
    "smaller": _shift(lambda x: x[:, :, :-1, :-1]),
    "larger": _shift(lambda x: torch.nn.functional.pad(x, (0, 1, 0, 1))),
    "unbatched": _shift(lambda x: x[0]),
    "nan": _shift(lambda x: x * math.nan),
    "inf": _shift(lambda x: x + math.inf),
    "range": _shift(lambda x: x * 3),
    "integer": _shift(lambda x: x.long()),
    "not_a_tensor": _shift(lambda x: (x, x)),
    "bare_tensor": lambda inputs: inputs["LF"],
    "tuple": lambda inputs: (inputs["LF"],),
    "renamed": lambda inputs: {"PAS": inputs["LF"]},
    "extra": lambda inputs: {"HE": inputs["LF"], "PAS": inputs["LF"]},
    "empty": lambda inputs: {},
}


@pytest.mark.parametrize("mode", ["resize", "tile"])
@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_prediction_is_rejected_before_publication(
    tmp_path: Path, case: str, mode: str
) -> None:
    paths = _pair(tmp_path / "in", (16, 16) if mode == "resize" else (24, 20))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "prediction.png"
    existing.write_bytes(b"previous result")
    with pytest.raises((TypeError, ValueError), match="Predictor output|Predictor must"):
        run_image_path_inference(
            _runtime(MALFORMED[case]),  # type: ignore[arg-type]
            paths,
            existing,
            mode=mode,  # type: ignore[arg-type]
            tile_overlap=4,
        )
    assert existing.read_bytes() == b"previous result"
    assert [p.name for p in out_dir.iterdir()] == ["prediction.png"]


@pytest.mark.parametrize(
    ("given", "message"),
    [({"AF": "af"}, r"missing=\['LF'\]"), ({"AF": "af", "LF": "lf", "TH": "th"}, "extra=")],
)
def test_input_names_must_match_the_contract(
    tmp_path: Path, given: dict[str, str], message: str
) -> None:
    paths = {name: tmp_path / f"{stem}.png" for name, stem in given.items()}
    for path in paths.values():
        _random_image(path, (16, 16))
    with pytest.raises(ValueError, match=message):
        run_image_path_inference(_runtime(InMemoryPredictor()), paths, tmp_path / "out.png")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_names": ()}, "non-empty"),
        ({"input_names": ("AF", "AF")}, "unique"),
        ({"input_names": ("AF", " ")}, "non-empty"),
        ({"input_names": ["AF"]}, "tuple"),
        ({"output_names": ()}, "output_names must be a non-empty"),
        ({"output_names": ("HE", "HE")}, "output_names must be unique"),
        ({"output_names": ["HE"]}, "output_names must be a non-empty tuple"),
        ({"image_size": (0, 16)}, "positive"),
        ({"image_size": (16,)}, "positive"),
        ({"image_size": (16.0, 16)}, "positive"),
        ({"output_semantics": "multi_output"}, "Unsupported output_semantics"),
        ({"output_semantics": "scalar"}, "Unsupported output_semantics"),
        ({"value_range": (0, 1)}, "Unsupported value_range"),
    ],
)
def test_invalid_contracts_are_rejected(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "input_names": ("AF",),
        "output_names": ("HE",),
        "image_size": (16, 16),
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        PredictionContract(**values)  # type: ignore[arg-type]


def test_missing_output_path_without_runtime_default_fails_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = _pair(tmp_path / "in", (16, 16))
    runtime = _runtime(lambda inputs: pytest.fail("predictor must not run"))
    with pytest.raises(ValueError, match="explicit output path"):
        run_image_path_inference(runtime, paths)
    with pytest.raises(ValueError, match="explicit output path"):
        run_image_path_inference(runtime, {"AF": tmp_path / "in/AF", "LF": tmp_path / "in/LF"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in"]


def test_output_may_not_replace_an_input(tmp_path: Path) -> None:
    paths = _pair(tmp_path, (16, 16))
    with pytest.raises(ValueError, match="overwrite input LF"):
        run_image_path_inference(_runtime(InMemoryPredictor()), paths, paths["LF"])


def test_directory_outputs_that_collide_are_rejected_before_prediction(tmp_path: Path) -> None:
    for modality in ("AF", "LF"):
        _random_image(tmp_path / modality / "a.png", (16, 16))
        _random_image(tmp_path / modality / "a.tif", (16, 16))
    runtime = _runtime(lambda inputs: pytest.fail("predictor must not run"))
    with pytest.raises(ValueError, match="would both be written to"):
        run_image_path_inference(
            runtime,
            {"AF": tmp_path / "AF", "LF": tmp_path / "LF"},
            tmp_path / "out",
            output_format="png",
        )


def test_same_basename_in_different_subdirectories_does_not_collide(tmp_path: Path) -> None:
    for modality in ("AF", "LF"):
        _random_image(tmp_path / modality / "x" / "a.png", (16, 16))
        _random_image(tmp_path / modality / "y" / "a.png", (16, 16))
    result = run_image_path_inference(
        _runtime(InMemoryPredictor()),
        {"AF": tmp_path / "AF", "LF": tmp_path / "LF"},
        tmp_path / "out",
        recursive=True,
    )
    assert isinstance(result, DirectoryInferenceResult)
    assert {r.output_paths["HE"] for r in result.results} == {
        tmp_path / "out/x/HE/a_generated.png",
        tmp_path / "out/y/HE/a_generated.png",
    }


def test_directory_extension_mismatch_names_the_modality(tmp_path: Path) -> None:
    _random_image(tmp_path / "AF" / "a.png", (16, 16))
    _random_image(tmp_path / "LF" / "a.tif", (16, 16))
    with pytest.raises(ValueError, match=r"Input modality LF relative paths differ"):
        run_image_path_inference(
            lambda: pytest.fail("runtime must not load"),
            {"AF": tmp_path / "AF", "LF": tmp_path / "LF"},
        )


def test_directory_output_path_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    for modality in ("AF", "LF"):
        _random_image(tmp_path / modality / "a.png", (16, 16))
    existing = tmp_path / "outputs"
    existing.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        run_image_path_inference(
            _runtime(InMemoryPredictor()), {"AF": tmp_path / "AF", "LF": tmp_path / "LF"}, existing
        )


def test_predictor_identity_is_reported_and_outputs_are_named_by_output(tmp_path: Path) -> None:
    for modality in ("AF", "LF"):
        _random_image(tmp_path / modality / "a.png", (16, 16))
    runtime = InferenceRuntime(
        InMemoryPredictor(),
        PredictionContract(("AF", "LF"), ("HE",), (16, 16)),
        CPU,
        predictor_identity="my-model",
    )
    result = run_image_path_inference(
        runtime, {"AF": tmp_path / "AF", "LF": tmp_path / "LF"}, tmp_path / "out"
    )
    assert isinstance(result, DirectoryInferenceResult)
    assert result.predictor_identity == "my-model"
    assert result.results[0].output_paths == {"HE": tmp_path / "out" / "HE" / "a_generated.png"}


def _two_outputs() -> InferenceRuntime:
    return _runtime(InMemoryPredictor(("PAS", "HE")), outputs=("PAS", "HE"))


def test_several_outputs_never_share_one_output_file(tmp_path: Path) -> None:
    paths = _pair(tmp_path / "in", (16, 16))
    runtime = _runtime(lambda inputs: pytest.fail("predictor must not run"), outputs=("PAS", "HE"))

    with pytest.raises(ValueError, match="cannot hold the 2 outputs"):
        run_image_path_inference(runtime, paths, tmp_path / "out" / "result.png")
    existing = tmp_path / "existing"
    existing.write_text("a file")
    with pytest.raises(ValueError, match="cannot hold the 2 outputs"):
        run_image_path_inference(runtime, paths, existing)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("size", "mode"), [((16, 16), "resize"), ((37, 21), "tile")])
def test_several_outputs_are_published_into_an_output_directory(
    tmp_path: Path, size: tuple[int, int], mode: str
) -> None:
    paths = _pair(tmp_path / "in", size, name="sample.png")
    predictor = InMemoryPredictor(("PAS", "HE"))

    result = run_image_path_inference(
        _runtime(predictor, outputs=("PAS", "HE")),
        paths,
        tmp_path / "out",
        mode=mode,  # type: ignore[arg-type]
        tile_overlap=4,
    )

    assert isinstance(result, SingleInferenceResult)
    assert result.output_paths == {
        "PAS": tmp_path / "out" / "PAS" / "sample_generated.png",
        "HE": tmp_path / "out" / "HE" / "sample_generated.png",
    }
    source = _read(paths["LF"]).astype(int)
    he = _read(result.output_paths["HE"]).astype(int)
    pas = _read(result.output_paths["PAS"]).astype(int)
    assert he.shape == pas.shape == source.shape  # every output on the input grid
    assert np.abs(he - source).max() <= 1
    assert np.abs(pas - (255 - source)).max() <= 1
    if mode == "tile":
        # One traversal: one predictor call per tile, not one per output.
        assert len(predictor.calls) == 6


def test_directory_inference_publishes_every_output_per_input(tmp_path: Path) -> None:
    for modality in ("AF", "LF"):
        _random_image(tmp_path / modality / "a.png", (16, 16))
        _random_image(tmp_path / modality / "b.png", (16, 16), seed=1)

    result = run_image_path_inference(
        _two_outputs(), {"AF": tmp_path / "AF", "LF": tmp_path / "LF"}, tmp_path / "out"
    )

    assert isinstance(result, DirectoryInferenceResult)
    assert sorted(
        path.relative_to(tmp_path / "out")
        for single in result.results
        for path in single.output_paths.values()
    ) == [
        Path("HE/a_generated.png"),
        Path("HE/b_generated.png"),
        Path("PAS/a_generated.png"),
        Path("PAS/b_generated.png"),
    ]


def test_reordered_output_mapping_is_rejected(tmp_path: Path) -> None:
    paths = _pair(tmp_path, (16, 16))
    runtime = _runtime(InMemoryPredictor(("HE", "PAS")), outputs=("PAS", "HE"))

    with pytest.raises(ValueError, match="must be exactly"):
        run_image_path_inference(runtime, paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("output_name", [".", "..", "../HE", "HE/PAS", "HE\\PAS", "1HE", "H&E", ""])
def test_standalone_contract_rejects_unsafe_output_names_before_prediction(
    tmp_path: Path, output_name: str
) -> None:
    paths = _pair(tmp_path / "in", (16, 16))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="output_name"):
        PredictionContract(("AF", "LF"), (output_name,), (16, 16))
    with pytest.raises(ValueError, match="output_name"):
        run_image_path_inference(
            lambda: _runtime(
                lambda inputs: pytest.fail("predictor must not run"), outputs=(output_name,)
            ),
            paths,
            out,
        )
    assert not out.exists()


@pytest.mark.parametrize("output_name", ["HE", "PAS", "H_E", "H-E", "HE2"])
def test_standalone_contract_accepts_safe_output_names(tmp_path: Path, output_name: str) -> None:
    paths = _pair(tmp_path / "in", (16, 16), name="s.png")

    def predictor(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {output_name: inputs["LF"]}

    result = run_image_path_inference(
        _runtime(predictor, outputs=(output_name,)), paths, tmp_path / "out.png"
    )

    assert isinstance(result, SingleInferenceResult)
    assert result.output_paths == {output_name: tmp_path / "out.png"}


def test_tiled_publication_matches_prechange_png_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _pair(tmp_path / "in", (37, 21), name="sample.png")
    originals = {name: path.read_bytes() for name, path in paths.items()}
    names = ("PAS", "HE")
    model = InMemoryPredictor(names)
    coordinates = [(x, y) for y in (0, 5) for x in (0, 12, 21)]
    sums = {name: torch.zeros(3, 21, 37) for name in names}
    weights = torch.zeros(1, 21, 37)
    closed: list[Path] = []
    original_close = PillowRegionImageReader.close

    def close(reader: PillowRegionImageReader) -> None:
        closed.append(reader.path)
        original_close(reader)

    def predictor(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = model(inputs)
        x, y = coordinates[len(model.calls) - 1]
        for name, value in outputs.items():
            sums[name][:, y : y + 16, x : x + 16] += denormalize_model_output(value)[0]
        weights[:, y : y + 16, x : x + 16] += 1
        return outputs

    monkeypatch.setattr(PillowRegionImageReader, "close", close)
    result = run_image_path_inference(
        _runtime(predictor, outputs=names), paths, tmp_path / "out", mode="tile", tile_overlap=4
    )
    expected_paths = {name: tmp_path / "out" / name / "sample_generated.png" for name in names}
    assert isinstance(result, SingleInferenceResult)
    assert result == SingleInferenceResult(paths, expected_paths, (16, 16), "tile", "cpu")
    assert tuple(result.output_paths) == names
    assert closed == list(paths.values())
    assert model.calls == [("AF", "LF")] * len(coordinates)
    source = _read(paths["LF"])
    for name, path in expected_paths.items():
        reference = tmp_path / "reference" / f"{name}.png"
        save_rgb((sums[name] / weights.clamp_min(1.0)).clamp(0, 1), reference)
        assert path.read_bytes() == reference.read_bytes()
        with Image.open(path) as image:
            assert image.size == (37, 21) and image.mode == "RGB"
            assert image.info == {}
        np.testing.assert_array_equal(_read(path), source if name == "HE" else 255 - source)
        assert list(path.parent.iterdir()) == [path]
    assert {name: path.read_bytes() for name, path in paths.items()} == originals


@pytest.mark.parametrize("failure", ["predictor", "writer", "verification"])
def test_tiled_failure_closes_readers_and_preserves_per_file_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    paths = _pair(tmp_path / "in", (37, 21), name="sample.png")
    destinations = {
        name: tmp_path / "out" / name / "sample_generated.png" for name in ("PAS", "HE")
    }
    for path in destinations.values():
        path.parent.mkdir(parents=True)
        path.write_bytes(b"previous result")
    closed: list[Path] = []
    original_close = PillowRegionImageReader.close
    original_save = output_writer.save_image
    original_verify = PngImagePlugin.PngImageFile.verify
    model = InMemoryPredictor(("PAS", "HE"))

    def close(reader: PillowRegionImageReader) -> None:
        closed.append(reader.path)
        original_close(reader)

    def predictor(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if failure == "predictor" and model.calls:
            raise RuntimeError("predictor failure")
        return model(inputs)

    def save(output: torch.Tensor, partial: Path) -> None:
        assert closed == list(paths.values())
        original_save(output, partial)
        if failure == "writer" and partial.parent == destinations["HE"].parent:
            raise OSError("writer failure")

    def verify(image: PngImagePlugin.PngImageFile) -> None:
        original_verify(image)
        assert isinstance(image.filename, str)
        if failure == "verification" and Path(image.filename).parent == destinations["HE"].parent:
            raise OSError("verification failure")

    monkeypatch.setattr(PillowRegionImageReader, "close", close)
    monkeypatch.setattr(output_writer, "save_image", save)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "verify", verify)
    with pytest.raises((RuntimeError, OSError), match=f"{failure} failure"):
        run_image_path_inference(
            _runtime(predictor, outputs=("PAS", "HE")),
            paths,
            tmp_path / "out",
            mode="tile",
            tile_overlap=4,
        )
    assert closed == list(paths.values())
    assert destinations["HE"].read_bytes() == b"previous result"
    if failure == "predictor":
        assert destinations["PAS"].read_bytes() == b"previous result"
    else:
        np.testing.assert_array_equal(_read(destinations["PAS"]), 255 - _read(paths["LF"]))
    for path in destinations.values():
        assert list(path.parent.iterdir()) == [path]
