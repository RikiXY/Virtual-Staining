# Model Card: Pix2Pix Virtual Staining

## Model Summary

| Property | Value |
|---|---|
| Task | Image-to-image translation for virtual staining |
| Architecture | Pix2Pix-style conditional GAN with U-Net generator and PatchGAN discriminator |
| Input | Label-free microscopy patch (RGB, configurable size; default `256x256`) |
| Output | Virtually stained microscopy patch (RGB, same size as input) |
| Framework | PyTorch |
| Language | Python 3.11+ |

## Intended Use

This model generates virtually stained histology-like image patches from paired
label-free microscopy inputs.

Intended uses:

- Research on virtual staining workflows.
- Benchmarking paired image-to-image translation models.
- Educational, reproducibility, and proof-of-concept demonstrations.

## Not Intended For

- Clinical diagnosis or treatment decisions.
- Replacement of pathologist-reviewed stained slides.
- Unsupervised or unvalidated medical use.
- Deployment in regulated clinical settings without appropriate validation and approval.

## Architecture

The repository implements a Pix2Pix-style conditional GAN.

**Generator**

- U-Net generator implemented in PyTorch.
- Default encoder/decoder width starts at 64 channels and increases by depth.
- Downsampling uses max pooling followed by double-convolution blocks.
- Upsampling uses transposed convolutions by default (`bilinear: false` in the example config).
- Skip connections join encoder activations to matching decoder stages.
- Output activation is `tanh`, producing values in `[-1, 1]`.

**Discriminator**

- PatchGAN discriminator operating on the concatenated input and target/generated image pair.
- Default input channel count is 6 (`3 + 3` for RGB source and RGB target/generated).
- Uses a final patchwise logit map rather than a single image-level prediction.
- Keeps the standard PatchGAN receptive field of approximately `70x70`.
- Uses raw logits by default (`use_sigmoid: false`).

**Training Loss**

- Adversarial term: `BCEWithLogitsLoss`.
- Reconstruction term: `L1Loss`.
- Combined generator objective: adversarial loss plus weighted L1 loss.
- Default L1 weight in the example training config: `25.0`.

## Training Data

The model is trained on paired label-free / stained microscopy images after
preprocessing and patch extraction.

- Patches are extracted from aligned full-size source/target image pairs.
- Patch size is configurable; the standard example configuration uses `256x256`.
- Default data split is patch-level train/validation/test.
- Quality filters remove patches using foreground ratio, white ratio, and largest
  white component ratio thresholds.

## Evaluation Metrics

The evaluation pipeline computes the following metrics on the test split:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index |
| PCC (gray) | Pearson Correlation Coefficient on grayscale images |
| PCC (RGB) | Mean Pearson Correlation Coefficient across RGB channels |

This repository does not publish fixed benchmark values in the codebase. Metric
values should be taken from the run-specific evaluation outputs generated for a
particular dataset and experiment.

## Limitations

- **Patch-level split**: the default split draws train, validation, and test
  patches from the same slide. Reported test metrics therefore reflect same-slide
  internal validation, not independent slide-level or patient-level generalization.
- **Registration sensitivity**: supervision quality depends on alignment between
  label-free and stained images. Registration errors directly degrade training quality.
- **Dataset specificity**: performance depends on the tissue type, staining process,
  acquisition setup, and preprocessing assumptions represented in the paired data.
- **Patch-based scope**: the model operates on isolated patches, so whole-slide use
  may introduce seams or context-related artifacts at tile boundaries.
- **No uncertainty estimate**: outputs are deterministic image predictions and do
  not provide calibrated confidence or uncertainty measures.

## Failure Modes

- Hallucinated stain-like texture in background or low-information regions.
- Poor color fidelity when evaluation data differ from the training distribution.
- Artifacts near tissue boundaries or in patches with limited foreground context.
- Degraded output quality when paired training images are imperfectly aligned.
- Overly optimistic conclusions if patch-level metrics are interpreted as evidence
  of independent clinical or cross-site generalization.

## Ethical Considerations

This model produces synthetic stained images that may look visually plausible, but
they are generated outputs rather than ground-truth stained slides.

**It must not be used for clinical interpretation.** Misuse could contribute to
incorrect diagnostic conclusions, overconfidence in synthetic imagery, or unsafe
deployment claims. Users are responsible for ensuring that any research, demo, or
deployment context is appropriately validated and compliant with local requirements.
