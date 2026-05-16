# Dataset Card: Virtual Staining Paired Patch Dataset

## Dataset Summary

| Property | Value |
|---|---|
| Task | Virtual histological staining / paired image-to-image translation |
| Modalities | Label-free microscopy input + stained histology target |
| Format | Paired RGB image patches written from user-supplied source images |
| Patch size | Configurable; default `256x256` pixels |
| Splits | Train / Val / Test (`patch` level) |
| Index file | `manifests/manifest.csv` |

## Source Data

This repository does not ship raw microscopy data. The dataset is built from a
user-provided paired image set placed under `dataset_root`.

- The input modality is expected to be a label-free source image.
- The target modality is expected to be the corresponding stained image.
- The default filenames in the example configuration are `label_free.tif` and
  `stained.tif`.
- The preprocessing code accepts `.tif`, `.tiff`, and `.png` inputs.

Dataset creators and users are responsible for ensuring that their source data
are legally shareable and appropriately governed for their institution and use case.

## Acquisition Process

The intended workflow assumes that the source and target images depict the same
tissue section under two imaging conditions: a label-free source modality and a
stained target modality.

The repository does not enforce any acquisition hardware, stain protocol, or
institution-specific procedure. Instead, it assumes the user provides one paired
source image and one paired target image, then performs computational alignment
of the target image to the source reference frame.

## Preprocessing Pipeline

`vs-prepare` performs the following steps:

1. **Input validation and loading**: reads the paired source and target images
   from `dataset_root` and validates the configured filenames and sizes.
2. **Tissue mask computation**: computes foreground masks for both images using
   thresholding, connected-component analysis, and repeated grid-based mask
   passes across multiple scales.
3. **Affine registration**: aligns the target image to the source image using
   mask-constrained SIFT feature matching and `cv2.estimateAffinePartial2D`.
   The repository writes `alignment_metadata.json` and warps target patches on
   demand during extraction.
4. **Patch extraction**: crops by the configured `margin`, extracts patches at
   `image_size`, and steps the extraction grid using `grid_movement`.
5. **Quality filtering**: rejects patch pairs when foreground coverage is too
   low, white/background coverage is too high, or the largest white connected
   component exceeds the configured threshold.
6. **Split assignment**: randomly assigns accepted patches to `train`, `val`,
   and `test` using the configured ratios and random seed.
7. **Manifest writing**: writes accepted-patch records to `manifests/manifest.csv`
   and discarded-patch records to `manifests/discarded_manifest.csv`, with
   per-patch filter diagnostics in `discarded_patches/discarded_log.csv`.

## Splits

The dataset builder writes accepted patches into:

- `splits/train/`
- `splits/val/`
- `splits/test/`

`manifests/manifest.csv` is the canonical index of accepted patches and their
split assignments. Downstream training, inference, and evaluation stages rely on
this manifest rather than discovering files ad hoc.

## Known Leakage Risk

The default split is **patch-level**, not slide-level or patient-level.

Patches extracted from the **same full-size image pair / slide** can be assigned
to different splits. As a result, `splits/test/` is suitable for same-slide
internal validation, but it is **not** a fully independent estimate of
generalization to unseen slides, patients, institutions, or acquisition settings.

Reported metrics from this default split may therefore overestimate real-world
generalization. The current pipeline does **not** implement slide-level,
patient-level, or spatial-block split strategies.

## Metrics

The repository evaluates generated images on the test split using:

- MAE
- MSE
- RMSE
- PSNR
- SSIM
- PCC (grayscale)
- PCC per RGB channel and RGB mean

See [`docs/run_format.md`](/home/andrea/projects/Virtual-Staining/docs/run_format.md)
for the exact evaluation output columns.

## Known Biases

- **Tissue-type bias**: performance will reflect the tissue types present in the
  user-supplied paired images and may not transfer to unseen tissues.
- **Protocol bias**: target appearance depends on the staining protocol,
  scanner, microscope, illumination, and acquisition settings used to produce
  the target images.
- **Registration bias**: misalignment between source and target images can
  corrupt supervision even when the pipeline completes successfully.
- **Patch-selection bias**: filtering removes patches with high background or
  insufficient foreground, which can skew the retained dataset toward clearer
  tissue regions.
- **Boundary/context bias**: patch-based training reduces global tissue context
  and may underrepresent whole-slide structure.

## Privacy and Licensing

- **Raw images**: not included in this repository.
- **Code license**: the preprocessing pipeline is released under the MIT License.
  See [LICENSE](/home/andrea/projects/Virtual-Staining/LICENSE).
- **Privacy**: if source images originate from patient tissue, users are
  responsible for de-identification, ethics review, consent handling, and any
  required IRB or institutional approval.
- **Redistribution**: this repository does not grant rights to redistribute any
  user-supplied microscopy data. Dataset creators are responsible for their own
  licensing and sharing constraints.
