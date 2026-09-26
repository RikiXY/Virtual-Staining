# pyright: reportArgumentType=false
# Plain lists of batches stand in for DataLoaders.
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.config_helpers import cyclegan_config_data, write_config_data
from virtual_staining.config.losses import parse_loss_config
from virtual_staining.config.method import MethodConfig
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.config.training import TrainingConfig
from virtual_staining.methods.cyclegan import CycleGANMethod
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.models.io_contract import denormalize_model_output
from virtual_staining.training import preview as preview_module
from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder
from virtual_staining.training.preview import ValidationPreview, ValidationPreviewWriter

_CPU = torch.device("cpu")
_ADVERSARIAL = {
    "generator": [{"name": "adversarial_bce", "weight": 1.0}, {"name": "l1", "weight": 100.0}],
    "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
}


def _pix2pix(tmp_path: Path, losses: dict[str, Any] = _ADVERSARIAL) -> Pix2PixMethod:
    project = ProjectConfig(
        dataset_root=tmp_path / "dataset",
        results_path=tmp_path / "results",
        run_name="run",
        image_size=(32, 32),
    )
    training = TrainingConfig(
        batch_size=2,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=0,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=1,
        losses=parse_loss_config(losses),
    )
    config = RunConfig(
        project=project,
        method=MethodConfig(),
        model=ModelConfig.from_mapping(
            {
                "inputs": ["LF", "AF"],
                "target": "stained",
                "generator": {"base_channels": 4},
                "discriminator": {"ndf": 4},
            }
        ),
        training=training,
        inference=None,
        preprocessing=None,
        evaluation=None,
    )
    torch.manual_seed(0)
    return Pix2PixMethod(config, _CPU)


def _pix2pix_batch(seed: int, size: int = 2) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "inputs": {
            "LF": torch.rand(size, 3, 32, 32, generator=generator) * 2 - 1,
            "AF": torch.rand(size, 3, 32, 32, generator=generator) * 2 - 1,
        },
        "target": torch.rand(size, 3, 32, 32, generator=generator) * 2 - 1,
        "masks": {},
    }


def _cyclegan(tmp_path: Path) -> CycleGANMethod:
    config = RunConfig.from_yaml(
        write_config_data(tmp_path / "run.yaml", cyclegan_config_data(tmp_path))
    )
    torch.manual_seed(0)
    return CycleGANMethod(config, _CPU, seed=7)


def _cyclegan_batch(seed: int, size: int = 2) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "domain_a": torch.rand(size, 3, 32, 32, generator=generator) * 2 - 1,
        "domain_b": torch.rand(size, 3, 32, 32, generator=generator) * 2 - 1,
    }


def _models(method: Pix2PixMethod | CycleGANMethod) -> list[torch.nn.Module]:
    if isinstance(method, Pix2PixMethod):
        return [method.generator, method.discriminator]
    return list(method._models().values())


_CASES = [
    pytest.param(_pix2pix, _pix2pix_batch, id="pix2pix"),
    pytest.param(_cyclegan, _cyclegan_batch, id="cyclegan"),
]


class _RecordingSink:
    def __init__(self, fail: bool = False) -> None:
        self.previews: list[ValidationPreview] = []
        self.fail = fail

    def wants(self, epoch: int, batch_index: int) -> bool:
        return batch_index < 5

    def write(self, preview: ValidationPreview) -> None:
        if self.fail:
            raise OSError("disk full")
        self.previews.append(preview)


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, Path, dict[str, Any]]]:
    calls: list[tuple[torch.Tensor, Path, dict[str, Any]]] = []
    monkeypatch.setattr(
        preview_module,
        "save_image",
        lambda tensor, path, **kwargs: calls.append((tensor.clone(), Path(path), kwargs)),
    )
    return calls


@pytest.mark.parametrize(("build", "batch"), _CASES)
def test_disabled_previews_do_no_io_and_match_enabled_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build: Any, batch: Any
) -> None:
    method = build(tmp_path)
    loader = [batch(seed) for seed in range(7)]
    preview_dir = tmp_path / "val"
    enabled = method.validate(loader, epoch=3, preview_sink=ValidationPreviewWriter(preview_dir))
    written = sorted(path.name for path in preview_dir.iterdir())

    def no_io(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preview I/O with previews disabled")

    monkeypatch.setattr(preview_module, "save_image", no_io)
    monkeypatch.setattr(Path, "mkdir", no_io)
    disabled = method.validate(loader, epoch=3)

    assert disabled == enabled
    stems = [f"epoch3_batch{index}" for index in range(5)]
    if isinstance(method, Pix2PixMethod):
        expected = [
            f"{stem}_{role}.tif" for stem in stems for role in ("input", "output", "target")
        ]
    else:
        expected = [f"{stem}_preview.tif" for stem in stems]
    assert written == sorted(expected)


def test_pix2pix_preview_payload_and_saved_files(tmp_path: Path, saved: list[Any]) -> None:
    method = _pix2pix(tmp_path)
    loader = [_pix2pix_batch(1)]
    sink = _RecordingSink()
    method.validate(loader, epoch=0, preview_sink=sink)
    (payload,) = sink.previews

    assert (payload.epoch, payload.batch_index) == (0, 0)
    assert tuple(payload.images) == ("input", "output", "target")
    assert all(t.grad_fn is None and not t.requires_grad for t in payload.images.values())
    # Payload references the loader tensors in place: nothing was copied or moved.
    assert payload.images["input"].data_ptr() == loader[0]["inputs"]["LF"].data_ptr()
    assert payload.images["target"].data_ptr() == loader[0]["target"].data_ptr()

    method.validate(loader, epoch=0, preview_sink=ValidationPreviewWriter(tmp_path / "val"))
    with torch.no_grad():
        expected_output = method.generator.eval()(loader[0]["inputs"])
    assert [(path.name, kwargs) for _, path, kwargs in saved] == [
        ("epoch0_batch0_input.tif", {}),
        ("epoch0_batch0_output.tif", {}),
        ("epoch0_batch0_target.tif", {}),
    ]
    assert torch.equal(saved[0][0], denormalize_model_output(loader[0]["inputs"]["LF"][0]))
    assert torch.equal(saved[1][0], denormalize_model_output(expected_output[0]))
    assert torch.equal(saved[2][0], denormalize_model_output(loader[0]["target"][0]))


def test_cyclegan_preview_payload_and_grid(tmp_path: Path, saved: list[Any]) -> None:
    method = _cyclegan(tmp_path)
    loader = [_cyclegan_batch(1)]
    sink = _RecordingSink()
    metrics = method.validate(loader, epoch=4, preview_sink=sink)
    (payload,) = sink.previews

    assert metrics.image == {}
    assert tuple(payload.images) == ("real_A", "fake_B", "real_B", "fake_A")
    assert all(t.grad_fn is None and not t.requires_grad for t in payload.images.values())
    assert payload.images["real_A"].data_ptr() == loader[0]["domain_a"].data_ptr()

    method.validate(loader, epoch=4, preview_sink=ValidationPreviewWriter(tmp_path / "val"))
    with torch.no_grad():
        method.G_A_to_B.eval()
        method.G_B_to_A.eval()
        fake_b = method.G_A_to_B(loader[0]["domain_a"])
        fake_a = method.G_B_to_A(loader[0]["domain_b"])
    ((grid, path, kwargs),) = saved
    assert (path.name, kwargs) == ("epoch4_batch0_preview.tif", {"nrow": 4})
    expected = torch.stack(
        [loader[0]["domain_a"][0], fake_b[0], loader[0]["domain_b"][0], fake_a[0]]
    )
    assert torch.equal(grid, denormalize_model_output(expected))


@pytest.mark.parametrize(("build", "batch"), _CASES)
@pytest.mark.parametrize("fail", [False, True])
def test_validation_restores_mixed_model_modes(
    tmp_path: Path, build: Any, batch: Any, fail: bool
) -> None:
    method = build(tmp_path)
    models = _models(method)
    for index, model in enumerate(models):
        model.train(index % 2 == 0)
    before = [model.training for model in models]

    if fail:
        with pytest.raises(OSError, match="disk full"):
            method.validate([batch(1)], epoch=0, preview_sink=_RecordingSink(fail=True))
    else:
        method.validate([batch(1)], epoch=0, preview_sink=_RecordingSink())

    assert [model.training for model in models] == before
    assert all(
        not submodule.training
        for model, was_training in zip(models, before, strict=True)
        if not was_training
        for submodule in model.modules()
    )


def test_pix2pix_skips_discriminator_without_adversarial_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _pix2pix(tmp_path, {"generator": [{"name": "l1", "weight": 1.0}]})

    def forbidden(*_args: object) -> None:
        raise AssertionError("discriminator must not run")

    monkeypatch.setattr(method.discriminator, "forward", forbidden)
    metrics = method.validate([_pix2pix_batch(1)], epoch=0, preview_sink=_RecordingSink())

    assert metrics.losses["loss_D"] == 0.0


@pytest.mark.parametrize(("build", "batch"), _CASES)
def test_short_empty_and_uneven_loaders_keep_step_averaging(
    tmp_path: Path, build: Any, batch: Any
) -> None:
    method = build(tmp_path)
    sink = _RecordingSink()

    empty = method.validate([], epoch=0, preview_sink=sink)
    assert sink.previews == []
    empty_loss = 0.0 if isinstance(method, Pix2PixMethod) else math.nan
    assert repr(empty) == repr(method.validate([], epoch=0))
    assert empty.losses["loss_G"] == pytest.approx(empty_loss, nan_ok=True)

    full, partial = batch(1, size=2), batch(2, size=1)
    combined = method.validate([full, partial], epoch=0, preview_sink=sink)
    assert [preview.batch_index for preview in sink.previews] == [0, 1]
    per_batch = [method.validate([b], epoch=0).losses["loss_G"] for b in (full, partial)]
    assert combined.losses["loss_G"] == pytest.approx(sum(per_batch) / 2)
    assert combined == method.validate([full, partial], epoch=0)


def test_writer_times_preview_io_only_when_writing(tmp_path: Path, saved: list[Any]) -> None:
    method = _pix2pix(tmp_path)
    loader = [_pix2pix_batch(seed) for seed in range(6)]
    recorder = TrainingBenchmarkRecorder(_CPU, warmup_batches=0)
    phases_during_save: list[int] = []
    writer = ValidationPreviewWriter(tmp_path / "val", benchmark_recorder=recorder)
    original_write = writer.write

    def write(preview: ValidationPreview) -> None:
        phases_during_save.append(len(recorder._phases))
        original_write(preview)

    writer.write = write  # type: ignore[method-assign]
    recorder.start_run()
    method.validate(loader, epoch=0)
    assert recorder._phases == []
    method.validate(loader, epoch=0, preview_sink=writer)
    recorder.finish_run()

    assert phases_during_save == [0, 1, 2, 3, 4]
    assert len(saved) == 15
    assert recorder.report()["summary"]["phases"]["preview_io"]["count"] == 5
