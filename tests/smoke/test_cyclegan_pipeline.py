from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.applications.train import train
from virtual_staining.config.run import RunConfig


def _write_domain(root: Path, offset: int) -> None:
    for split in ("train", "val", "test"):
        directory = root / split
        directory.mkdir(parents=True)
        values = np.arange(64 * 64 * 3, dtype=np.uint16).reshape(64, 64, 3)
        image = ((values + offset) % 255).astype(np.uint8)
        Image.fromarray(image).save(directory / f"{split}.png")


def test_cyclegan_application_smoke(tmp_path: Path) -> None:
    _write_domain(tmp_path / "dataset" / "domain_a", 0)
    _write_domain(tmp_path / "dataset" / "domain_b", 31)
    config_data = {
        "dataset_root": str(tmp_path / "dataset"),
        "results_path": str(tmp_path / "results"),
        "run_name": "cyclegan_smoke",
        "image_size": [64, 64],
        "method": {"name": "cyclegan"},
        "data": {
            "pairing": "unpaired",
            "domains": {"label_free": "domain_a", "stained": "domain_b"},
        },
        "model": {
            "inputs": ["label_free"],
            "target": "stained",
            "generator": {
                "architecture": "resnet",
                "base_channels": 8,
                "blocks": 1,
                "norm": "instance",
            },
            "discriminator": {"ndf": 8, "norm": "instance"},
        },
        "training": {
            "batch_size": 1,
            "epochs": 1,
            "num_workers": 0,
            "validate_rate": 1,
            "checkpoint_rate": 1,
            "log_rate": 1,
            "losses": {
                "generator": [
                    {"name": "adversarial_lsgan", "weight": 1.0},
                    {"name": "cycle_l1", "weight": 10.0},
                    {"name": "identity_l1", "weight": 0.0, "enabled": False},
                ],
                "discriminator": [{"name": "adversarial_lsgan", "weight": 1.0}],
            },
        },
        "inference": {"checkpoint_policy": "latest", "direction": "A_to_B"},
        "evaluation": {"protocol": "unpaired", "save_graphs": True},
    }
    config_path = tmp_path / "cyclegan.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = RunConfig.from_yaml(config_path)

    training_result = train(config, config_path)
    inference_result = infer(config, config_path)
    evaluate(config, config_path)

    assert training_result.best_checkpoint_path is not None
    assert inference_result.num_samples == 1
    metadata_path = (
        tmp_path / "results" / "cyclegan_smoke" / "evaluation" / "unpaired_evaluation.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["paired_metrics_available"] is False
    assert not (metadata_path.parent / "per_image_metrics.csv").exists()
    assert (metadata_path.parent / "unpaired_image_statistics.csv").exists()
    assert (metadata_path.parent / "unpaired_feature_distributions.png").exists()
    assert (metadata_path.parent / "unpaired_feature_wasserstein.png").exists()
