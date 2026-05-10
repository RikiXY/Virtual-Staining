from __future__ import annotations

import textwrap
from pathlib import Path
from typing import cast

import pytest

from virtual_staining.config import RunConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.training.config import InferenceConfig, TrainingConfig


def _make_project(**overrides: object) -> ProjectConfig:
    dataset_root = Path("data/root")
    results_path = Path("results")
    run_name = "my_run"
    image_size = (256, 256)

    if "dataset_root" in overrides:
        dataset_root = cast(Path, overrides["dataset_root"])
    if "results_path" in overrides:
        results_path = cast(Path, overrides["results_path"])
    if "run_name" in overrides:
        run_name = cast(str, overrides["run_name"])
    if "image_size" in overrides:
        image_size = cast(tuple[int, int], overrides["image_size"])

    return ProjectConfig(
        dataset_root=dataset_root,
        results_path=results_path,
        run_name=run_name,
        image_size=image_size,
    )


def _make_training_config(**overrides: object) -> TrainingConfig:
    batch_size = 8
    epochs = 50
    lr_g = 2e-4
    lr_d = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    l1_weight = 25.0
    seed: int | None = 42
    num_workers = 4
    validate_rate = 10
    checkpoint_rate = 10
    log_rate = 15
    resume: str | None = None
    train_dir: Path | None = None
    val_dir: Path | None = None
    project: ProjectConfig | None = _make_project()

    if "batch_size" in overrides:
        batch_size = cast(int, overrides["batch_size"])
    if "epochs" in overrides:
        epochs = cast(int, overrides["epochs"])
    if "lr_g" in overrides:
        lr_g = cast(float, overrides["lr_g"])
    if "lr_d" in overrides:
        lr_d = cast(float, overrides["lr_d"])
    if "beta1" in overrides:
        beta1 = cast(float, overrides["beta1"])
    if "beta2" in overrides:
        beta2 = cast(float, overrides["beta2"])
    if "l1_weight" in overrides:
        l1_weight = cast(float, overrides["l1_weight"])
    if "seed" in overrides:
        seed = cast(int | None, overrides["seed"])
    if "num_workers" in overrides:
        num_workers = cast(int, overrides["num_workers"])
    if "validate_rate" in overrides:
        validate_rate = cast(int, overrides["validate_rate"])
    if "checkpoint_rate" in overrides:
        checkpoint_rate = cast(int, overrides["checkpoint_rate"])
    if "log_rate" in overrides:
        log_rate = cast(int, overrides["log_rate"])
    if "resume" in overrides:
        resume = cast(str | None, overrides["resume"])
    if "train_dir" in overrides:
        train_dir = cast(Path | None, overrides["train_dir"])
    if "val_dir" in overrides:
        val_dir = cast(Path | None, overrides["val_dir"])
    if "project" in overrides:
        project = cast(ProjectConfig | None, overrides["project"])

    return TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        lr_g=lr_g,
        lr_d=lr_d,
        beta1=beta1,
        beta2=beta2,
        l1_weight=l1_weight,
        seed=seed,
        num_workers=num_workers,
        validate_rate=validate_rate,
        checkpoint_rate=checkpoint_rate,
        log_rate=log_rate,
        resume=resume,
        train_dir=train_dir,
        val_dir=val_dir,
        project=project,
    )


def test_training_config_uses_project_fields() -> None:
    config = _make_training_config()
    assert config.dataset_root == Path("data/root")
    assert config.run_name == "my_run"
    assert config.image_size == (256, 256)
    assert config.run_root == Path("results") / "my_run"
    assert config.epochs == 50
    assert config.resume is None


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


def test_training_from_yaml_shim(tmp_path: Path) -> None:
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
    assert config.project is not None
    assert config.project.image_size == (256, 256)


def test_training_to_yaml_round_trip(tmp_path: Path) -> None:
    config = _make_training_config(seed=42, epochs=20)
    yaml_file = tmp_path / "config.yaml"
    config.to_yaml(yaml_file)
    loaded = TrainingConfig.from_yaml(yaml_file)
    assert loaded == config


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
    assert run_config.training.dataset_train_dir == Path("/custom/train")
    assert run_config.training.dataset_val_dir == Path("/custom/val")


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
    assert config.output_test_dir == Path("/results/section_run/custom_output")
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
        ({"project": _make_project(run_name="")}, "run_name"),
        ({"project": _make_project(image_size=(0, 256))}, "image_size"),
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
    config = TrainingConfig.from_yaml(yaml_file)
    assert config.image_size == (320, 256)


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
    config = InferenceConfig.from_yaml(yaml_file)
    assert config.image_size == (320, 256)


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
        TrainingConfig.from_yaml(yaml_file)


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
        TrainingConfig.from_yaml(yaml_file)


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
        TrainingConfig.from_yaml(yaml_file)


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
        InferenceConfig.from_yaml(yaml_file)


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
