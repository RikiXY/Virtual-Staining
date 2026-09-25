from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.config_helpers import cyclegan_config_data, write_config_data
from tests.conftest import ManifestDataset
from tests.image_helpers import write_rgb_image
from virtual_staining.applications import infer as infer_app
from virtual_staining.applications.infer import infer
from virtual_staining.applications.infer_images import infer_images
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.inference.outputs import generated_path_for_record
from virtual_staining.inference.runner import (
    inference_direction,
    inference_input_names,
    load_inference_generator,
    predict_batch,
)
from virtual_staining.inference.single import DirectoryInferenceResult
from virtual_staining.methods.cyclegan import CycleGANInferenceAdapter, CycleGANMethod
from virtual_staining.training.checkpoints import MethodCheckpointManager
from virtual_staining.utils.artifacts import generated_filename

_CPU = torch.device("cpu")


def _config_path(tmp_path: Path, direction: str | None, dataset_root: Path | None = None) -> Path:
    data = cyclegan_config_data(tmp_path)
    if dataset_root is not None:
        data["dataset_root"] = str(dataset_root)
    if direction is not None:
        data["inference"]["direction"] = direction
    return write_config_data(tmp_path / f"run_{direction}.yaml", data)


def _trained(tmp_path: Path, dataset_root: Path | None = None) -> CycleGANMethod:
    config = RunConfig.from_yaml(_config_path(tmp_path, None, dataset_root))
    torch.manual_seed(0)
    method = CycleGANMethod(config, _CPU, seed=7)
    batch = {"domain_a": torch.rand(2, 3, 32, 32) * 2 - 1, "domain_b": torch.rand(2, 3, 32, 32)}
    method.step(batch, epoch=0, global_step=0)
    layout = RunLayout.from_project(config.project)
    ensure_run_directories(layout)
    MethodCheckpointManager(
        method, layout.checkpoints_dir, image_size=config.project.image_size, device=_CPU
    ).save(0)
    return method


def test_direction_resolution_and_required_inputs(tmp_path: Path) -> None:
    default = RunConfig.from_yaml(_config_path(tmp_path, None))
    reverse = RunConfig.from_yaml(_config_path(tmp_path, "B_to_A"))

    assert inference_direction(default) == "A_to_B"
    assert inference_input_names(default) == ("label_free",)
    assert inference_direction(reverse) == "B_to_A"
    assert inference_input_names(reverse) == ("stained",)


@pytest.mark.parametrize(
    ("direction", "input_name", "source"),
    [("A_to_B", "label_free", "G_A_to_B"), ("B_to_A", "stained", "G_B_to_A")],
)
def test_same_checkpoint_loads_either_direction(
    tmp_path: Path, direction: str, input_name: str, source: str
) -> None:
    method = _trained(tmp_path)
    config = RunConfig.from_yaml(_config_path(tmp_path, direction))

    generator, path = load_inference_generator(config, RunLayout.from_project(config.project), _CPU)

    assert path.name == "ep000.pth"
    assert isinstance(generator, CycleGANInferenceAdapter)
    assert generator.input_names == (input_name,)
    assert not generator.training
    expected = method._models()[source]
    expected.eval()
    for key, value in expected.state_dict().items():
        assert torch.equal(value, generator.generator.state_dict()[key])
    image = torch.rand(1, 3, 32, 32) * 2 - 1
    prediction = predict_batch(generator, {input_name: image}, _CPU)
    assert prediction.shape == (1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(prediction, (expected(image) * 0.5 + 0.5).clamp(0, 1))
    with pytest.raises(ValueError, match="Expected inputs"):
        generator({"wrong": image})


def test_direction_aware_names_do_not_collide_and_pix2pix_is_unchanged() -> None:
    assert generated_filename("s1", ".TIF") == "s1_target_generated.tif"
    assert generated_filename("s1", ".tif", "A_to_B") == "s1_A_to_B_generated.tif"
    assert generated_filename("s1", ".tif", "B_to_A") == "s1_B_to_A_generated.tif"


def test_infer_images_both_directions_share_output_root_recursively(tmp_path: Path) -> None:
    _trained(tmp_path)
    inputs = tmp_path / "inputs"
    write_rgb_image(inputs / "slide1" / "p0.png", size=(32, 32))
    write_rgb_image(inputs / "slide2" / "nested" / "p1.png", size=(32, 32))
    output = tmp_path / "generated"

    forward = infer_images(
        _config_path(tmp_path, "A_to_B"),
        (f"label_free={inputs}",),
        output,
        recursive=True,
    )
    backward = infer_images(
        _config_path(tmp_path, "B_to_A"),
        (str(inputs),),
        output,
        recursive=True,
    )

    assert isinstance(forward, DirectoryInferenceResult)
    assert isinstance(backward, DirectoryInferenceResult)
    assert set(forward.input_dirs) == {"label_free"}
    assert set(backward.input_dirs) == {"stained"}
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*.png")) == [
        "slide1/p0_A_to_B_generated.png",
        "slide1/p0_B_to_A_generated.png",
        "slide2/nested/p1_A_to_B_generated.png",
        "slide2/nested/p1_B_to_A_generated.png",
    ]
    with pytest.raises(ValueError, match="Unknown input modality: label_free"):
        infer_images(_config_path(tmp_path, "B_to_A"), (f"label_free={inputs}",), output)


def test_manifest_inference_reads_direction_specific_domain(
    tmp_path: Path, manifest_dataset: ManifestDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        infer_app, "load_manifest_or_raise", lambda _project: manifest_dataset.manifest
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    _trained(runs, dataset_root=manifest_dataset.root)
    record = manifest_dataset.test_records[0]
    output_dir = RunLayout.from_project(
        RunConfig.from_yaml(_config_path(runs, None, manifest_dataset.root)).project
    ).output_test_dir
    results: dict[str, Any] = {}

    for direction in ("A_to_B", "B_to_A"):
        config_path = _config_path(runs, direction, manifest_dataset.root)
        results[direction] = infer(RunConfig.from_yaml(config_path), config_path)

    for direction in ("A_to_B", "B_to_A"):
        expected = generated_path_for_record(record, output_dir, direction)
        assert results[direction].generated_paths == [expected]
        assert expected.is_file()
