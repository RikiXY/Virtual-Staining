from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

from virtual_staining.training.config import InferenceConfig, TrainingConfig


def _make_namespace(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        dataset_root="data/root",
        results_path="results",
        run_name="my_run",
        image_size=[256, 256],
        batch_size=8,
        epochs=50,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        l1_weight=25.0,
        seed=42,
        num_workers=4,
        validate_rate=10,
        checkpoint_rate=10,
        log_rate=15,
        resume=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_from_args_basic() -> None:
    config = TrainingConfig.from_args(_make_namespace())
    assert config.dataset_root == Path("data/root")
    assert config.run_name == "my_run"
    assert config.image_size == (256, 256)
    assert config.epochs == 50
    assert config.seed == 42
    assert config.resume is None


def test_run_root_derived() -> None:
    config = TrainingConfig.from_args(_make_namespace(results_path="results", run_name="exp_01"))
    assert config.run_root == Path("results") / "exp_01"


def test_from_args_with_resume() -> None:
    config = TrainingConfig.from_args(_make_namespace(resume="checkpoints/ep049.pth"))
    assert config.resume == "checkpoints/ep049.pth"


def test_frozen() -> None:
    config = TrainingConfig.from_args(_make_namespace())
    with pytest.raises((AttributeError, TypeError)):
        config.epochs = 999  # type: ignore[misc]


def test_from_yaml(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        image_size: [128, 128]
        batch_size: 4
        epochs: 20
        lr_g: 0.0002
        lr_d: 0.0002
        beta1: 0.5
        beta2: 0.999
        l1_weight: 25.0
        seed: 7
        num_workers: 2
        validate_rate: 5
        checkpoint_rate: 5
        log_rate: 10
    """)
    yaml_file = tmp_path / "train.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = TrainingConfig.from_yaml(yaml_file)
    assert config.run_name == "yaml_run"
    assert config.image_size == (128, 128)
    assert config.epochs == 20
    assert config.seed == 7
    assert config.batch_size == 4
    assert config.resume is None
    assert config.run_root == Path("/results") / "yaml_run"


def test_from_args_partial_namespace() -> None:
    """from_args() falls back to dataclass defaults when optional fields are absent (SUPPRESS)."""
    args = argparse.Namespace(dataset_root="/data", run_name="test_run", epochs=10)
    config = TrainingConfig.from_args(args)
    assert config.results_path == Path("local_workspace/results")
    assert config.image_size == (256, 256)
    assert config.batch_size == 8
    assert config.lr_g == pytest.approx(2e-4)
    assert config.l1_weight == pytest.approx(25.0)
    assert config.seed is None
    assert config.resume is None
    assert config.log_rate == 15


def test_to_yaml_round_trip(tmp_path: Path) -> None:
    config = TrainingConfig.from_args(_make_namespace(seed=42, epochs=20))
    yaml_file = tmp_path / "config.yaml"
    config.to_yaml(yaml_file)
    assert yaml_file.exists()
    loaded = TrainingConfig.from_yaml(yaml_file)
    assert loaded == config


def test_from_yaml_defaults_for_optional_fields(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: minimal_run
        epochs: 10
    """)
    yaml_file = tmp_path / "minimal.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = TrainingConfig.from_yaml(yaml_file)
    assert config.batch_size == 8
    assert config.lr_g == pytest.approx(2e-4)
    assert config.l1_weight == pytest.approx(25.0)
    assert config.log_rate == 15
    assert config.resume is None


def test_training_from_run_yaml_section(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
          batch_size: 2
          l1_weight: 50
          train_dir: /custom/train
          val_dir: /custom/val
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = TrainingConfig.from_yaml(yaml_file)
    assert config.dataset_root == Path("/data")
    assert config.run_root == Path("/results") / "section_run"
    assert config.image_size == (512, 512)
    assert config.epochs == 30
    assert config.batch_size == 2
    assert config.l1_weight == pytest.approx(50)
    assert config.dataset_train_dir == Path("/custom/train")
    assert config.dataset_val_dir == Path("/custom/val")


def test_inference_from_run_yaml_section(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint: checkpoints/ep030.pth
          test_dir: /custom/test
          output_dir: /results/section_run/custom_output
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = InferenceConfig.from_yaml(yaml_file)
    assert config.test_dir == Path("/custom/test")
    assert config.output_test_dir == Path("/results") / "section_run" / "custom_output"
    assert config.checkpoint == Path("/results/section_run/checkpoints/ep030.pth")
    assert config.image_size == (512, 512)


def test_inference_latest_checkpoint_policy(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "section_run"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "ep009.pth").write_bytes(b"")
    (checkpoint_dir / "ep019.pth").write_bytes(b"")

    yaml_content = textwrap.dedent(f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint_policy: latest
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = InferenceConfig.from_yaml(yaml_file)
    assert config.checkpoint == checkpoint_dir / "ep019.pth"
    assert config.output_test_dir == run_root / "output_test"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"run_name": ""}, "run_name"),
        ({"image_size": [0, 256]}, "image_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"epochs": 0}, "epochs"),
        ({"lr_g": 0.0}, "lr_g"),
        ({"lr_d": -0.1}, "lr_d"),
        ({"beta1": -0.1}, "beta1"),
        ({"beta2": 1.0}, "beta2"),
        ({"l1_weight": -1.0}, "l1_weight"),
        ({"num_workers": -1}, "num_workers"),
        ({"validate_rate": 0}, "validate_rate"),
        ({"checkpoint_rate": 0}, "checkpoint_rate"),
        ({"log_rate": 0}, "log_rate"),
    ],
)
def test_training_from_args_invalid_values_raise_value_error(
    overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        TrainingConfig.from_args(_make_namespace(**overrides))


def test_non_square_image_size_training_from_args() -> None:
    """image_size=[320, 256] must parse as (width=320, height=256) in TrainingConfig."""
    config = TrainingConfig.from_args(_make_namespace(image_size=[320, 256]))
    assert config.image_size == (320, 256)


def test_non_square_image_size_training_from_yaml(tmp_path: Path) -> None:
    """image_size: [320, 256] in YAML must parse as (width=320, height=256)."""
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: ns_run
        image_size: [320, 256]
        epochs: 5
    """)
    yaml_file = tmp_path / "ns_train.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    config = TrainingConfig.from_yaml(yaml_file)
    assert config.image_size == (320, 256)


def test_non_square_image_size_inference_from_yaml(tmp_path: Path) -> None:
    """image_size: [320, 256] must be preserved through InferenceConfig parsing."""
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: ns_run
        image_size: [320, 256]
        inference:
          checkpoint: checkpoints/ep010.pth
    """)
    yaml_file = tmp_path / "ns_infer.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    config = InferenceConfig.from_yaml(yaml_file)
    assert config.image_size == (320, 256)


def test_training_from_yaml_invalid_image_size_raises_value_error(
    tmp_path: Path,
) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: bad_training
        image_size: [256, -1]
        epochs: 10
    """)
    yaml_file = tmp_path / "bad_training.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="image_size"):
        TrainingConfig.from_yaml(yaml_file)


@pytest.mark.parametrize(
    ("inference_yaml", "field"),
    [
        ("checkpoint: checkpoints/ep010.pth\nrun_name: ''", "run_name"),
        ("checkpoint: checkpoints/ep010.pth\nimage_size: [0, 256]", "image_size"),
        ("checkpoint: ''", "checkpoint"),
        ("checkpoint: '   '", "checkpoint"),
    ],
)
def test_inference_from_yaml_invalid_values_raise_value_error(
    tmp_path: Path, inference_yaml: str, field: str
) -> None:
    yaml_content = (
        "dataset_root: /data\n"
        "results_path: /results\n"
        "run_name: inference_run\n"
        "image_size: [256, 256]\n"
        "inference:\n"
        f"{textwrap.indent(inference_yaml, '  ')}\n"
    )
    yaml_file = tmp_path / "bad_inference.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        InferenceConfig.from_yaml(yaml_file)
