from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from virtual_staining.checkpoint_contract import (
    CHECKPOINT_FORMAT_VERSION,
    ValidatedCheckpoint,
    read_checkpoint,
    validate_checkpoint,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.inference.runner import resolve_inference_device
from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    predict_single_patch,
    validate_patch_image,
)
from virtual_staining.methods.builtin import config_from_pix2pix_checkpoint


def _read_ui_checkpoint(path: Path) -> tuple[RunConfig, ValidatedCheckpoint]:
    """Resolve built-in Pix2Pix metadata through the same definitions as the CLI."""
    payload = read_checkpoint(path)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported format version; expected {CHECKPOINT_FORMAT_VERSION}.")
    try:
        config = config_from_pix2pix_checkpoint(payload)
        checkpoint = validate_checkpoint(
            payload, config.method.definition.checkpoint_identity(config), path
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed checkpoint reconstruction metadata.") from exc
    if len(config.model.inputs) != 1:
        raise ValueError("The single-patch UI requires exactly one input modality.")
    if len(config.model.outputs) != 1:
        raise ValueError("The single-patch UI requires exactly one output modality.")
    return config, checkpoint


logger = logging.getLogger(__name__)


class UIInferenceError(RuntimeError):
    """An expected inference workflow failure safe to present to a UI user."""


@dataclass(frozen=True)
class ModelDescriptor:
    """UI-facing description of a validated, single-patch compatible checkpoint."""

    identifier: str
    checkpoint_filename: str
    input_domains: tuple[str, ...]
    target_domain: str
    architecture_id: str
    model_class: str
    checkpoint_schema_version: int
    image_size: tuple[int, int]
    channels_per_input: int

    @property
    def transformation(self) -> str:
        source = " + ".join(self.input_domains) if self.input_domains else "Input"
        return f"{source} → {self.target_domain}"

    @property
    def display_name(self) -> str:
        return f"{self.transformation} · {self.checkpoint_filename}"


@dataclass(frozen=True)
class CatalogIssue:
    checkpoint: str
    reason: str


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[ModelDescriptor, ...]
    issues: tuple[CatalogIssue, ...]
    checkpoint_directory_exists: bool


@dataclass(frozen=True)
class ResultProvenance:
    """Reproducibility context known during one UI inference operation."""

    transformation: str
    model_identifier: str
    checkpoint_filename: str
    checkpoint_schema_version: int
    source_domains: tuple[str, ...]
    target_domain: str
    architecture_id: str
    model_class: str
    required_image_size: tuple[int, int]
    required_input_channels: int
    input_image_size: tuple[int, int]
    input_image_mode: str
    runtime_device: str
    source_filename: str
    generated_filename: str

    def to_dict(self) -> dict[str, Any]:
        """Return portable JSON data without machine-specific checkpoint paths."""
        return {
            "schema_version": 1,
            "transformation": self.transformation,
            "model": {
                "identifier": self.model_identifier,
                "checkpoint_filename": self.checkpoint_filename,
                "checkpoint_schema_version": self.checkpoint_schema_version,
                "source_domains": list(self.source_domains),
                "target_domain": self.target_domain,
                "architecture_id": self.architecture_id,
                "model_class": self.model_class,
                "required_image_size": list(self.required_image_size),
                "required_input_channels": self.required_input_channels,
            },
            "input": {
                "filename": self.source_filename,
                "image_size": list(self.input_image_size),
                "image_mode": self.input_image_mode,
            },
            "output": {"filename": self.generated_filename},
            "runtime": {"device": self.runtime_device},
        }


@dataclass(frozen=True)
class UIInferenceResult:
    source_image: Image.Image
    generated_image: Image.Image
    provenance: ResultProvenance


@dataclass(frozen=True)
class SavedInferenceResult:
    image_path: Path
    sidecar_path: Path
    provenance: ResultProvenance


class UIInferenceService:
    """Stable adapter from UI concepts to the current checkpoint inference core.

    Model reconstruction is intentionally hidden here so presentation code does
    not depend on today's generator class or package layout.
    """

    def __init__(
        self,
        checkpoint_directory: Path,
        default_output_directory: Path,
        *,
        working_directory: Path | None = None,
    ) -> None:
        self.checkpoint_directory = Path(checkpoint_directory).expanduser()
        self.default_output_directory = Path(default_output_directory).expanduser()
        self.working_directory = working_directory or Path.cwd()
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._checkpoint_paths: dict[str, Path] = {}

    def discover_models(self) -> ModelCatalog:
        """Discover current-format checkpoints and report invalid files non-fatally."""
        root = self._resolve_from_working_directory(self.checkpoint_directory)
        self._descriptors.clear()
        self._checkpoint_paths.clear()
        if not root.is_dir():
            return ModelCatalog((), (), False)

        issues: list[CatalogIssue] = []
        for checkpoint_path in sorted(root.rglob("*.pth")):
            identifier = checkpoint_path.relative_to(root).as_posix()
            try:
                config, _ = _read_ui_checkpoint(checkpoint_path)
                descriptor = ModelDescriptor(
                    identifier=identifier,
                    checkpoint_filename=checkpoint_path.name,
                    input_domains=config.model.inputs,
                    target_domain=config.model.outputs[0],
                    architecture_id=config.method.options.generator.name,
                    model_class="ConcatUNetGenerator",
                    checkpoint_schema_version=CHECKPOINT_FORMAT_VERSION,
                    image_size=config.project.image_size,
                    channels_per_input=3,
                )
            except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible UI checkpoint %s: %s", checkpoint_path, exc)
                issues.append(CatalogIssue(identifier, _checkpoint_issue_message(exc)))
                continue
            self._descriptors[identifier] = descriptor
            self._checkpoint_paths[identifier] = checkpoint_path

        return ModelCatalog(
            tuple(sorted(self._descriptors.values(), key=lambda item: item.display_name.lower())),
            tuple(issues),
            True,
        )

    def validate_input(self, model_identifier: str, image: Image.Image) -> ModelDescriptor:
        """Validate an uploaded patch against the selected descriptor."""
        descriptor = self._get_descriptor(model_identifier)
        try:
            validate_patch_image(
                image,
                descriptor.image_size,
                descriptor.channels_per_input,
            )
        except ValueError as exc:
            raise UIInferenceError(str(exc)) from exc
        return descriptor

    def run_inference(
        self,
        model_identifier: str,
        image: Image.Image,
        source_filename: str,
    ) -> UIInferenceResult:
        """Load the selected checkpoint and run the shared strict patch inference path."""
        descriptor = self.validate_input(model_identifier, image)
        checkpoint_path = self._checkpoint_paths[model_identifier]
        try:
            config, checkpoint = _read_ui_checkpoint(checkpoint_path)
            if (config.model.inputs, config.model.outputs, config.project.image_size) != (
                descriptor.input_domains,
                (descriptor.target_domain,),
                descriptor.image_size,
            ):
                raise ValueError("Checkpoint changed; refresh the model catalog.")
            device = resolve_inference_device()
            runtime = InferenceRuntime(
                predictor=config.method.definition.build_inference_model(
                    config,
                    checkpoint,
                    direction=None,
                    device=device,
                ),
                contract=PredictionContract(
                    config.model.inputs, config.model.outputs, config.project.image_size
                ),
                device=device,
                checkpoint_path=checkpoint_path,
            )
            generated = predict_single_patch(runtime, image)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("UI inference failed for checkpoint %s", checkpoint_path)
            raise UIInferenceError(
                "Inference could not be completed with the selected model. "
                "The checkpoint may be unreadable or incompatible."
            ) from exc

        safe_source = _safe_source_filename(source_filename)
        generated_filename = f"{Path(safe_source).stem or 'image'}_generated.png"
        provenance = ResultProvenance(
            transformation=descriptor.transformation,
            model_identifier=descriptor.identifier,
            checkpoint_filename=descriptor.checkpoint_filename,
            checkpoint_schema_version=descriptor.checkpoint_schema_version,
            source_domains=descriptor.input_domains,
            target_domain=descriptor.target_domain,
            architecture_id=descriptor.architecture_id,
            model_class=descriptor.model_class,
            required_image_size=descriptor.image_size,
            required_input_channels=descriptor.channels_per_input,
            input_image_size=image.size,
            input_image_mode=image.mode,
            runtime_device=str(runtime.device),
            source_filename=safe_source,
            generated_filename=generated_filename,
        )
        return UIInferenceResult(image.copy(), generated, provenance)

    def save_result(
        self,
        result: UIInferenceResult,
        output_directory: str | Path | None = None,
    ) -> SavedInferenceResult:
        """Save a generated PNG and portable JSON provenance without overwriting files."""
        raw_directory = (
            self.default_output_directory if output_directory is None else Path(output_directory)
        )
        if not str(raw_directory).strip():
            raise UIInferenceError("Output folder must not be empty.")
        directory = self._resolve_from_working_directory(raw_directory.expanduser())

        try:
            if directory.exists() and not directory.is_dir():
                raise NotADirectoryError(f"Output folder is not a directory: {raw_directory}")
            directory.mkdir(parents=True, exist_ok=True)
            image_path, sidecar_path = _available_output_paths(
                directory, result.provenance.source_filename
            )
            saved_provenance = replace(
                result.provenance,
                generated_filename=image_path.name,
            )
            sidecar_json = json.dumps(saved_provenance.to_dict(), indent=2) + "\n"
            image_file = image_path.open("xb")
            sidecar_created = False
            try:
                with image_file, sidecar_path.open("x", encoding="utf-8") as sidecar:
                    sidecar_created = True
                    result.generated_image.save(image_file, format="PNG")
                    sidecar.write(sidecar_json)
            except Exception:
                image_path.unlink(missing_ok=True)
                if sidecar_created:
                    sidecar_path.unlink(missing_ok=True)
                raise
        except (OSError, ValueError) as exc:
            logger.exception("Could not save UI inference result in %s", directory)
            raise UIInferenceError(
                "The result could not be saved. Check that the output folder is writable."
            ) from exc
        return SavedInferenceResult(image_path, sidecar_path, saved_provenance)

    def _get_descriptor(self, identifier: str) -> ModelDescriptor:
        try:
            return self._descriptors[identifier]
        except KeyError as exc:
            raise UIInferenceError(
                "The selected model is no longer available. Refresh the model catalog."
            ) from exc

    def _resolve_from_working_directory(self, path: Path) -> Path:
        return path if path.is_absolute() else self.working_directory / path


def _safe_source_filename(filename: str) -> str:
    safe = Path(filename.replace("\\", "/")).name
    return safe if safe not in {"", ".", ".."} else "image.png"


def _available_output_paths(directory: Path, source_filename: str) -> tuple[Path, Path]:
    stem = Path(_safe_source_filename(source_filename)).stem or "image"
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        image_path = directory / f"{stem}_generated{suffix}.png"
        sidecar_path = image_path.with_suffix(".json")
        if not image_path.exists() and not sidecar_path.exists():
            return image_path, sidecar_path
        index += 1


def _checkpoint_issue_message(exc: Exception) -> str:
    message = str(exc)
    if "format version" in message:
        return "Unsupported checkpoint schema; this UI requires the current v4 format."
    if isinstance(exc, (EOFError, pickle.UnpicklingError)):
        return "Unreadable checkpoint file."
    return message or "Checkpoint is not compatible with single-patch inference."
