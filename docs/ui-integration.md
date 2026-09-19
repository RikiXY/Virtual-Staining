# UI branch integration notes

The `feat/ui` branch was inspected read-only and was not merged or rebased. Its presentation
code primarily depends on two application adapters:

- `virtual_staining.applications.api.ApplicationService` for run discovery, metric summaries,
  representative samples, comparisons, and panels.
- `virtual_staining.applications.ui_inference.UIInferenceService` for checkpoint discovery,
  validation, inference, and result provenance.

Most experiment/evaluation APIs remain compatible for paired Pix2Pix runs. When rebasing the UI:

1. Extend `ModelDescriptor` and `ResultProvenance` with neutral `method`, `pairing`, and
   `supported_directions` fields. Do not infer capability from a concrete generator class.
2. Update checkpoint discovery to accept method checkpoint v4 as well as legacy Pix2Pix v3.
   Read component and direction metadata from `checkpoint["method"]`; retain the v3 adapter.
3. Add direction selection for CycleGAN and derive the required upload domain from the selected
   direction. The shared `inference_input_names` and `load_inference_generator` functions already
   resolve this behavior for CLI/application inference.
4. Treat an unpaired evaluation report as a valid qualitative result with unavailable paired
   metrics. `ApplicationService` currently assumes `summary.csv` and `per_image_metrics.csv`
   imply a completed evaluation; it should also recognize `unpaired_evaluation.json` and avoid
   metric rankings/representative-by-metric controls for that run.
5. Replace UI test fixtures that directly construct `ConcatUNetGenerator` and
   `PatchGANDiscriminator` checkpoints with method-neutral v4 fixtures, while retaining one v3
   fixture to verify backward-compatible discovery.

The UI should continue importing application adapters rather than `methods` or concrete models.
