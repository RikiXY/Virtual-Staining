"""Explicit composition of existing stage owners, plus independent stage examples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.applications.train import train
from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.data.alignment import AlignmentResult
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.evaluation.evaluator import EvaluationSample, evaluate_pair, evaluate_samples
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.runner import load_inference_generator
from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    run_image_path_inference,
)
from virtual_staining.metrics import resolve_metrics
from virtual_staining.training.trainer import Trainer

from .definitions import definitions
from .registration import BACKEND

CPU = torch.device("cpu")
METRICS = [{"name": "scaled_max_error", "options": {"scale": 2}}, {"name": "mae"}]


def mapping(root, **options):
    return {
        "dataset_root": str(root / "dataset"),
        "results_path": str(root / "results"),
        "run_name": "composed",
        "image_size": [8, 8],
        "method": {"name": "external_reconstruction", "options": options},
        "model": {"inputs": ["LF", "AF"], "outputs": ["HE"]},
        "data": {"pairing": "paired", "group_validation": "unavailable"},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "seed": 17,
            "num_workers": 0,
            "validate_rate": 1,
            "checkpoint_rate": 1,
        },
        "inference": {"checkpoint_policy": "latest"},
        "evaluation": {"metrics": METRICS, "bootstrap_iterations": 0},
    }


def pixels(x, y):
    return np.stack((20 + 2 * x + y, 40 + x, 60 + y), axis=-1).astype(np.uint8)


def prepare_example(root):
    """No inventory or tracked run: three explicit sets, four patches each."""
    sources = root / "sources"
    sources.mkdir(parents=True)
    sets = []
    for index in range(3):
        assets = {}
        for name in ("LF", "AF", "HE"):
            path = sources / f"S{index}-{name}.png"
            y, x = np.indices((16, 16) if name == "LF" else (20, 20))
            Image.fromarray(pixels(x, y) + index * 40).save(path)
            path.chmod(0o444)
            assets[name] = SlideAsset(name, path, already_aligned=name == "LF")
        sets.append(SlideSet(f"S{index}", (assets["LF"], assets["AF"]), (assets["HE"],), "LF"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.iterdir()}
    dataset = root / "dataset"
    config = PreprocessingConfig.from_mapping(
        {
            "inputs": {
                "inventory": "absent.csv",
                "modalities": ["LF", "AF"],
                "target_modalities": ["HE"],
                "reference": "LF",
            },
            "patching": {"patch_size": [8, 8], "grid_movement": [8, 8], "margin": 0},
            "masks": {"generation": "never"},
            "filtering": {"foreground": {"enabled": False}},
            "split": {"unit": "set", "train": 0.34, "val": 0.33, "test": 0.33},
            "io": {"tiled": False, "backend": "pillow"},
        },
        dataset_root=dataset,
        default_image_size=(8, 8),
    )
    result = DatasetBuilder(config, tuple(sets), registration_backend=BACKEND).run_all()
    assert sorted(p.name for p in root.iterdir()) == ["dataset", "sources"]
    assert not (dataset / "absent.csv").exists()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.iterdir()}
    # DatasetBuilder validates before publication; inspect its documented CSV/JSON
    # artifacts here. The train application validates them again on consumption.
    metadata = json.loads((dataset / "manifests/manifest_metadata.json").read_text())
    assert metadata["input_modalities"] == ["LF", "AF"]
    assert metadata["target_modalities"] == ["HE"]
    manifest = dataset / "manifests/manifest.csv"
    with manifest.open() as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 12 and {row["split"] for row in records} == {"train", "val", "test"}
    assert result.train_count == result.val_count == result.test_count == 4
    for record in records:
        px, py = int(record["x"]), int(record["y"])
        y, x = np.mgrid[py : py + 8, px : px + 8]
        for name in ("LF", "AF", "HE"):
            role = "target" if name == "HE" else "input"
            absolute = (dataset / record[f"{role}__{name}"]).resolve()
            assert absolute.is_relative_to(dataset.resolve())
            expected = pixels(x, y) if name == "LF" else pixels(x + 2, y + 1)
            expected = expected + int(record["set_id"][1:]) * 40
            with Image.open(absolute) as image:
                np.testing.assert_array_equal(np.asarray(image), expected)
    with (dataset / "manifests/slide_sets.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for row in rows:
        for name in ("LF", "AF", "HE"):
            aligned = AlignmentResult.from_dict(json.loads(row[f"{name}__alignment_metadata"]))
            assert aligned.candidate.reference.name == "LF"
            assert aligned.candidate.moving.name == name
            if name != "LF":
                assert aligned.attempt.runtime.backend == "external_translation"
                assert aligned.attempt.runtime.backend_version == "1"
                assert aligned.qc is None
                np.testing.assert_array_equal(
                    aligned.candidate.map_points(np.array([[2.0, 1.0]])), [[0.0, 0.0]]
                )
    return manifest


def caller_loader():
    """Caller-owned tensors, independent of manifests and prepared files."""
    source = torch.linspace(-1, 1, 3 * 8 * 8).reshape(3, 8, 8)
    return DataLoader(
        [
            {"inputs": {"LF": source, "AF": -source}, "targets": {"HE": source / 2}}
            for _ in range(4)
        ],
        batch_size=2,
    )


def standalone_train(root, architecture="tiny_conv"):
    config = RunConfig.from_mapping(mapping(root, architecture=architecture), definitions())
    runtime = config.method.definition.build_training_runtime(config, CPU, seed=17)
    layout = RunLayout(root / "standalone_run")
    loader = caller_loader()
    trainer = Trainer(config.training, layout, runtime, loader, loader, CPU)
    result = trainer.train(seed=17)
    assert result.best_checkpoint_path.is_file()
    assert not layout.metadata_dir.exists() and not layout.config_dir.exists()
    assert not config.project.dataset_root.exists()
    return config, layout, runtime


def predictor_only(root):
    root.mkdir(parents=True)
    # Build directly from consumer components; no RunConfig or checkpoint is involved.
    from .method import TinyNetwork

    predictor = TinyNetwork(("LF", "AF"), "HE", 4, False).to(CPU).eval()
    paths = {}
    for name in ("LF", "AF"):
        paths[name] = root / f"{name}.png"
        Image.new("RGB", (8, 8), (20, 40, 60)).save(paths[name])
    runtime = InferenceRuntime(predictor, PredictionContract(("LF", "AF"), ("HE",), (8, 8)), CPU)
    result = run_image_path_inference(runtime, paths, root / "prediction.png")
    assert result.output_paths == {"HE": root / "prediction.png"}
    assert result.checkpoint_path is None
    with Image.open(result.output_paths["HE"]) as image:
        assert image.size == (8, 8) and image.mode == "RGB"
    return paths


def explicit_evaluation(root):
    root.mkdir(parents=True)
    target, generated = root / "target.png", root / "generated.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(target)
    Image.new("RGB", (8, 8), (12, 20, 30)).save(generated)
    metrics = resolve_metrics(METRICS, definitions().metrics)
    pair, shape = evaluate_pair(target, generated, metrics=metrics)
    assert shape == (8, 8, 3)
    assert np.isclose(pair["scaled_max_error"].value, 4 / 255)
    samples = [
        EvaluationSample("ok", "HE", "S1", target, generated),
        EvaluationSample("missing", "HE", "S1", target, root / "missing.png"),
    ]
    result = evaluate_samples(
        samples, root / "report", metrics=metrics, input_failures="permissive"
    )
    assert (result.num_evaluated, result.num_excluded) == (1, 1)
    assert result.coverage_rows[1]["reason"] == "missing_generated"
    for path in (result.metrics_csv, result.summary_csv, result.coverage_csv, result.result_json):
        assert path.is_file()
    return result


def composed(root):
    manifest = prepare_example(root)
    supplied = definitions()
    raw = mapping(root)
    path = root / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = RunConfig.from_yaml(path, supplied)
    assert RunConfig.from_mapping(config.to_dict(), supplied).to_dict() == config.to_dict()
    trained = train(config, path)
    assert trained.best_checkpoint_path.is_file()
    layout = RunLayout.from_project(config.project)
    model, checkpoint = load_inference_generator(config, layout, CPU)
    with torch.no_grad():
        assert tuple(model(next(iter(caller_loader()))["inputs"])) == ("HE",)
    inferred = infer(config, path)
    assert inferred.num_samples == 4 and all(p.is_file() for p in inferred.generated_paths)
    evaluate(config, path)
    report = layout.evaluation_dir / "evaluation_result.json"
    assert report.is_file()
    with layout.per_image_metrics.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and all(float(row["scaled_max_error"]) >= 0 for row in rows)
    return {
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "inference": [str(p) for p in inferred.generated_paths],
        "evaluation": str(report),
        "custom_metric": [row["scaled_max_error"] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="New output directory (must not exist)")
    args = parser.parse_args()
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    standalone_train(root / "training")
    predictor_only(root / "predictor")
    explicit_evaluation(root / "explicit_evaluation")
    evidence = composed(root / "composed")
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
