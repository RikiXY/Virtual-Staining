# Model Card: Pix2Pix Virtual Staining

## Model Summary

This card describes the Pix2Pix reference implementation and its intended use and
limitations. It does not describe a released trained-model artifact, a particular
checkpoint, a frozen publication study, or measured performance. Datasets, weights,
configurations, and measured results belong to each concrete experiment.

| Property | Value |
|---|---|
| Task | Image-to-image translation for virtual staining |
| Architecture | Pix2Pix-style conditional GAN with U-Net generator and PatchGAN discriminator |
| Input | N ordered named RGB patches (`model.inputs`, e.g. `[AF, LF]`; configurable size, default `256x256`) |
| Output | M ordered named virtually stained RGB patches (`model.outputs`, e.g. `[HE]` or `[HE, PAS]`), same grid as the inputs |
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

This card covers the Pix2Pix reference method, a Pix2Pix-style conditional GAN. The
repository's second built-in method, CycleGAN, is not described by this card.

**Generator**

- U-Net generator implemented in PyTorch over the channel-concatenated named inputs
  (`3*N` channels) with `3*M` output channels split back into the named outputs; one
  output is the `M=1` case of the same model.
- Default encoder/decoder width starts at 64 channels and increases by depth.
- Downsampling uses max pooling followed by double-convolution blocks.
- Upsampling uses transposed convolutions by default (`bilinear: false` in the example config).
- Skip connections join encoder activations to matching decoder stages.
- Output activation is `tanh`, producing values in `[-1, 1]`.

**Discriminator**

- One joint conditional PatchGAN discriminator scoring all named inputs together with
  all real or all generated outputs.
- Its input channel count is `3*N + 3*M` (6 for one input and one output).
- Uses a final patchwise logit map rather than a single image-level prediction.
- Keeps the standard PatchGAN receptive field of approximately `70x70`.
- Uses raw logits by default (`use_sigmoid: false`).

**Training Loss**

- Adversarial term: `BCEWithLogitsLoss`.
- Reconstruction term: `L1Loss`.
- Combined generator objective: one joint adversarial term plus, per reconstruction term,
  the weighted arithmetic mean of that term over the outputs (one output keeps its
  exact scale). This training mean is not an evaluation score.
- Default L1 weight in the example training config: `25.0`.
- Several outputs are an engineering capability; nothing in this repository shows that
  predicting several stains jointly helps any of them.

## Training Data

Training uses experiment-specific paired label-free / stained microscopy images
after preprocessing and patch extraction.

- Patches are extracted from source slide sets onto one reference grid for every
  named input and target.
- Patch size is configurable; the standard example configuration uses `256x256`.
- Default preparation split is patch-level train/validation/test; `set`, `specimen`,
  and `patient` grouped splits are also supported when the corresponding real
  metadata are supplied.
- Quality filters remove patches using foreground ratio, white ratio, and largest
  white component ratio thresholds.

## Evaluation Metrics

Built-in paired evaluation metrics available to request against corresponding
test references include:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index |
| PCC (gray) | Pearson Correlation Coefficient on grayscale images |
| PCC (RGB) | Mean Pearson Correlation Coefficient across RGB channels |

Only requested metrics are evaluated, and results remain separate for each named
output. See [the run-output contract](docs/run_format.md) for validity, coverage,
aggregation, and persisted result semantics.

This card reports no fixed benchmark values. Metric
values should be taken from the run-specific evaluation outputs generated for a
particular dataset and experiment.

## Limitations

- **Split independence**: default patch-level splitting can place patches from the
  same specimen or patient in different partitions and does not establish
  unseen-specimen or unseen-patient performance. Grouped splits exist, but their
  labels must be real rather than inferred; a set is not automatically a patient
  or specimen. Actual independence depends on the experiment's split choice and
  available metadata; unknown metadata must narrow the claim.
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
