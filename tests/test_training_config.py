from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

from virtual_staining.training.config import TrainingConfig


def _make_namespace(**overrides):
    defaults = dict(
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


def test_from_args_basic():
    config = TrainingConfig.from_args(_make_namespace())
    assert config.dataset_root == Path("data/root")
    assert config.run_name == "my_run"
    assert config.image_size == (256, 256)
    assert config.epochs == 50
    assert config.seed == 42
    assert config.resume is None


def test_run_root_derived():
    config = TrainingConfig.from_args(_make_namespace(results_path="results", run_name="exp_01"))
    assert config.run_root == Path("results") / "exp_01"


def test_from_args_with_resume():
    config = TrainingConfig.from_args(_make_namespace(resume="checkpoints/ep049.pth"))
    assert config.resume == "checkpoints/ep049.pth"


def test_frozen():
    config = TrainingConfig.from_args(_make_namespace())
    with pytest.raises((AttributeError, TypeError)):
        config.epochs = 999  # type: ignore[misc]


def test_from_yaml(tmp_path):
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
    yaml_file.write_text(yaml_content)

    config = TrainingConfig.from_yaml(yaml_file)
    assert config.run_name == "yaml_run"
    assert config.image_size == (128, 128)
    assert config.epochs == 20
    assert config.seed == 7
    assert config.batch_size == 4
    assert config.resume is None
    assert config.run_root == Path("/results") / "yaml_run"


def test_from_yaml_defaults_for_optional_fields(tmp_path):
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: minimal_run
        epochs: 10
    """)
    yaml_file = tmp_path / "minimal.yaml"
    yaml_file.write_text(yaml_content)

    config = TrainingConfig.from_yaml(yaml_file)
    assert config.batch_size == 8
    assert config.lr_g == pytest.approx(2e-4)
    assert config.l1_weight == pytest.approx(25.0)
    assert config.log_rate == 15
    assert config.resume is None
