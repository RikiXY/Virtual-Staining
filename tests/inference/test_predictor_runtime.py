"""Direct inference with a caller-constructed predictor: no checkpoint, config or run."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from virtual_staining.inference.single import (
    DirectoryInferenceResult,
    InferenceRuntime,
    PredictionContract,
    SingleInferenceResult,
    run_image_path_inference,
)

CPU = torch.device("cpu")


class InMemoryPredictor(torch.nn.Module):
    """Returns the LF input unchanged and checks both modalities share coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.calls: list[tuple[str, ...]] = []

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        self.calls.append(tuple(inputs))
        assert torch.equal(inputs["AF"], inputs["LF"]), "modalities read different regions"
        return inputs["LF"] * self.scale


def _random_image(path: Path, size: tuple[int, int], seed: int = 0) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def _runtime(
    predictor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    names: tuple[str, ...] = ("AF", "LF"),
    size: tuple[int, int] = (16, 16),
) -> InferenceRuntime:
    return InferenceRuntime(predictor, PredictionContract(names, size), CPU)


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
    assert sorted(r.output_path.relative_to(directory.output_dir) for r in directory.results) == [
        Path("a_target_generated.png"),
        Path("nested/b_target_generated.png"),
    ]
    assert set(predictor.calls) == {("AF", "LF")}
    assert np.abs(_read(tiled.output_path).astype(int) - _read(large["LF"])).max() <= 1
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
    return lambda inputs: transform(inputs["LF"])


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
    "multi_output": _shift(lambda x: (x, x)),
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
        ({"image_size": (0, 16)}, "positive"),
        ({"image_size": (16,)}, "positive"),
        ({"image_size": (16.0, 16)}, "positive"),
        ({"output_semantics": "multi_output"}, "Unsupported output_semantics"),
        ({"output_semantics": "scalar"}, "Unsupported output_semantics"),
        ({"value_range": (0, 1)}, "Unsupported value_range"),
    ],
)
def test_invalid_contracts_are_rejected(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"input_names": ("AF",), "image_size": (16, 16), **kwargs}
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
    assert {r.output_path for r in result.results} == {
        tmp_path / "out/x/a_target_generated.png",
        tmp_path / "out/y/a_target_generated.png",
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


def test_explicit_artifact_direction_names_outputs(tmp_path: Path) -> None:
    for modality in ("AF", "LF"):
        _random_image(tmp_path / modality / "a.png", (16, 16))
    runtime = InferenceRuntime(
        InMemoryPredictor(),
        PredictionContract(("AF", "LF"), (16, 16), artifact_direction="A_to_B"),
        CPU,
        predictor_identity="my-model",
    )
    result = run_image_path_inference(
        runtime, {"AF": tmp_path / "AF", "LF": tmp_path / "LF"}, tmp_path / "out"
    )
    assert isinstance(result, DirectoryInferenceResult)
    assert result.predictor_identity == "my-model"
    assert result.artifact_direction == "A_to_B"
    assert result.results[0].output_path.name == "a_A_to_B_generated.png"
