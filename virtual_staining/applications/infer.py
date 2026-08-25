from __future__ import annotations

import logging
from pathlib import Path

from torchvision.utils import save_image

from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import load_manifest_or_raise
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.inference.runner import (
    InferenceResult,
    build_inference_transform,
    load_inference_generator,
    predict_batch,
    resolve_inference_device,
)
from virtual_staining.utils.artifacts import generated_filename

logger = logging.getLogger(__name__)


def infer(config: RunConfig, config_path: Path) -> InferenceResult:
    """Load a checkpoint, run the generator on the test split, and write outputs."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    with ExperimentSession.open(config=config, config_path=config_path, stage="infer") as session:
        device = resolve_inference_device()
        logger.info("Inference device: %s", device)
        generator, checkpoint_path = load_inference_generator(config, session.paths, device)
        transform = build_inference_transform(config.project.image_size)
        output_dir = config.inference.output_dir or session.paths.output_test_dir

        manifest = load_manifest_or_raise(config.project)
        if not set(config.model.inputs).issubset(manifest.metadata.input_modalities):
            raise ValueError("model.inputs must be a subset of manifest input modalities")
        if config.model.target != manifest.metadata.target_modality:
            raise ValueError("model.target must equal manifest target modality")
        manifest.validate(check_files_exist=True, require_splits={"test"})
        test_manifest = manifest.filter_split("test")
        dataset = PairedManifestDataset(
            test_manifest, input_names=config.model.inputs, transform=transform
        )
        logger.info("Loaded manifest: %s test samples", len(dataset))
        session.result(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            test_sample_count=len(test_manifest.records),
            device=str(device),
            inferred_count=0,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        result = InferenceResult(output_dir=output_dir)
        if len(dataset) == 0:
            logger.warning(
                "No test pairs found in manifest: %s",
                DatasetLayout.from_project(config.project).manifest_path,
            )
            return result

        for idx in range(len(dataset)):
            sample = dataset[idx]
            inputs = {name: tensor.unsqueeze(0) for name, tensor in sample["inputs"].items()}
            record = test_manifest.records[idx]
            output = predict_batch(generator, inputs, device)[0]
            out_path = output_dir / generated_filename(record.sample_id, record.target_path.suffix)
            save_image(output, out_path)
            result.generated_paths.append(out_path)
            result.num_samples += 1
            session.result(inferred_count=result.num_samples)
    logger.info("Inference complete: %s samples -> %s", result.num_samples, output_dir)
    return result
