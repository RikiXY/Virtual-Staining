from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from virtual_staining.config import RunConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.training.config import TrainingConfig


def _make_project(**overrides: object) -> ProjectConfig:
    defaults: dict[str, object] = {
        "dataset_root": Path("data/root"),
        "results_path": Path("results"),
        "run_name": "my_run",
        "image_size": (256, 256),
    }
    defaults.update(overrides)
    return ProjectConfig(
        dataset_root=defaults["dataset_root"],  # type: ignore[arg-type]
        results_path=defaults["results_path"],  # type: ignore[arg-type]
        run_name=defaults["run_name"],  # type: ignore[arg-type]
        image_size=defaults["image_size"],  # type: ignore[arg-type]
    )


def _make_training_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "batch_size": 8,
        "epochs": 50,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "l1_weight": 25.0,
        "seed": 42,
        "num_workers": 4,
        "validate_rate": 10,
        "checkpoint_rate": 10,
        "log_rate": 15,
        "resume": None,
        "train_dir": None,
        "val_dir": None,
    }
    defaults.update(overrides)
    return TrainingConfig(
        batch_size=defaults["batch_size"],  # type: ignore[arg-type]
        epochs=defaults["epochs"],  # type: ignore[arg-type]
        lr_g=defaults["lr_g"],  # type: ignore[arg-type]
        lr_d=defaults["lr_d"],  # type: ignore[arg-type]
        beta1=defaults["beta1"],  # type: ignore[arg-type]
        beta2=defaults["beta2"],  # type: ignore[arg-type]
        l1_weight=defaults["l1_weight"],  # type: ignore[arg-type]
        seed=defaults["seed"],  # type: ignore[arg-type]
        num_workers=defaults["num_workers"],  # type: ignore[arg-type]
        validate_rate=defaults["validate_rate"],  # type: ignore[arg-type]
        checkpoint_rate=defaults["checkpoint_rate"],  # type: ignore[arg-type]
        log_rate=defaults["log_rate"],  # type: ignore[arg-type]
        resume=defaults["resume"],  # type: ignore[arg-type]
        train_dir=defaults["train_dir"],  # type: ignore[arg-type]
        val_dir=defaults["val_dir"],  # type: ignore[arg-type]
    )


def test_frozen() -> None:
    config = _make_training_config()
    with pytest.raises((AttributeError, TypeError)):
        config.epochs = 999  # type: ignore[misc]


def test_run_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        image_size: [128, 128]
        epochs: 20
        batch_size: 4
        seed: 7
    """)
    yaml_file = tmp_path / "train.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.dataset_root == Path("/data")
    assert run_config.project.run_root == Path("/results") / "yaml_run"
    assert run_config.project.image_size == (128, 128)
    assert run_config.training is not None
    assert run_config.training.epochs == 20
    assert run_config.training.batch_size == 4
    assert run_config.training.seed == 7


def test_training_from_run_yaml_defaults(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: minimal_run
        epochs: 10
    """)
    yaml_file = tmp_path / "minimal.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.training is not None
    config = run_config.training
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

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.dataset_root == Path("/data")
    assert run_config.project.run_root == Path("/results") / "section_run"
    assert run_config.project.image_size == (512, 512)
    assert run_config.training is not None
    assert run_config.training.epochs == 30
    assert run_config.training.batch_size == 2
    assert run_config.training.l1_weight == pytest.approx(50)
    assert run_config.training.train_dir == Path("/custom/train")
    assert run_config.training.val_dir == Path("/custom/val")


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

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.inference is not None
    config = run_config.inference
    assert config.test_dir == Path("/custom/test")
    assert config.output_dir == Path("/results/section_run/custom_output")
    assert config.checkpoint_path == Path("checkpoints/ep030.pth")


def test_inference_latest_checkpoint_policy(tmp_path: Path) -> None:
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

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.inference is not None
    config = run_config.inference
    assert config.checkpoint_policy == "latest"
    assert config.checkpoint_path is None
    assert config.output_dir is None


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
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
def test_training_invalid_values_raise_value_error(
    overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _make_training_config(**overrides).validate()


def test_project_config_rejects_empty_run_name(tmp_path: Path) -> None:
    yaml_content = "dataset_root: /data\nresults_path: /results\nrun_name: ''\nepochs: 10\n"
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="run_name"):
        RunConfig.from_yaml(yaml_file)


def test_project_config_rejects_invalid_image_size(tmp_path: Path) -> None:
    yaml_content = (
        "dataset_root: /data\nresults_path: /results\nrun_name: run\n"
        "image_size: [0, 256]\nepochs: 10\n"
    )
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="image_size"):
        RunConfig.from_yaml(yaml_file)


def test_non_square_image_size_training_from_yaml(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: ns_run
        image_size: [320, 256]
        epochs: 5
    """)
    yaml_file = tmp_path / "ns_train.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.image_size == (320, 256)


def test_non_square_image_size_inference_from_yaml(tmp_path: Path) -> None:
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
    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.image_size == (320, 256)


def test_training_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        training:
          epochs: 10
          bathc_size: 4
    """)
    yaml_file = tmp_path / "typo.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="bathc_size"):
        RunConfig.from_yaml(yaml_file)


def test_training_from_yaml_unknown_top_level_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        typo_field: oops
        training:
          epochs: 10
    """)
    yaml_file = tmp_path / "typo_top.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="typo_field"):
        RunConfig.from_yaml(yaml_file)


def test_training_from_yaml_unknown_flat_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        epochs: 10
        bathc_size: 4
    """)
    yaml_file = tmp_path / "typo_flat.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="bathc_size"):
        RunConfig.from_yaml(yaml_file)


def test_inference_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        inference:
          checkpoint: checkpoints/ep010.pth
          checkpont_policy: latest
    """)
    yaml_file = tmp_path / "typo_inf.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="checkpont_policy"):
        RunConfig.from_yaml(yaml_file)


def test_training_from_yaml_invalid_image_size_raises_value_error(tmp_path: Path) -> None:
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
        RunConfig.from_yaml(yaml_file)


@pytest.mark.parametrize(
    ("inference_yaml", "field"),
    [
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
        RunConfig.from_yaml(yaml_file)
