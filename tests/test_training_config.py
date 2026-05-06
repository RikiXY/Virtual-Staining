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
    config = TrainingConfig.from_args(
        _make_namespace(results_path="results", run_name="exp_01")
    )
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
        training:
          image_size: [512, 512]
          epochs: 30
          batch_size: 2
          l1_weight: 50
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


def test_inference_from_run_yaml_section(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        training:
          image_size: [512, 512]
          epochs: 30
        inference:
          checkpoint: /results/section_run/checkpoints/ep030.pth
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = InferenceConfig.from_yaml(yaml_file)
    assert config.test_dir == Path("/data") / "dataset_test"
    assert config.output_test_dir == Path("/results") / "section_run" / "output_test"
    assert config.checkpoint == Path("/results/section_run/checkpoints/ep030.pth")
    assert config.image_size == (512, 512)
