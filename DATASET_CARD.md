# Dataset Card: Virtual Staining Paired Patch Dataset

## Dataset Summary

This card describes the paired dataset/preparation format and its scientific
limitations, not a frozen publication cohort or a particular biological dataset.
The repository establishes no patient population, tissue distribution, independent
patient generalization, or clinical validity.

| Property | Value |
|---|---|
| Task | Virtual histological staining / paired image-to-image translation |
| Modalities | Named label-free inputs + one or more named stained histology targets |
| Format | Aligned RGB image patches (every input and target of a sample on one grid) written from user-supplied source images |
| Patch size | Configurable; default `256x256` pixels |
| Splits | Train / Val / Test; `patch` (default), `set`, `specimen`, or `patient` |
| Index file | `manifests/manifest.csv` (schema `4.0`) |

## Source Data

This repository does not ship raw microscopy data. The dataset is built from a
user-provided paired image set placed under `dataset_root`.

- The input modality is expected to be a label-free source image.
- Every target modality (`preprocessing.inputs.target_modalities`, e.g. `[HE, PAS]`) is
  expected to be a corresponding stained image of the same set; all configured targets
  are required for every set.
- Assets are listed in the wide `inputs/slide_sets.csv` inventory
  (`input__<name>_path`, `target__<name>_path`, ...); see
  [dataset format](docs/dataset_format.md).
- The preprocessing code accepts `.tif`, `.tiff`, and `.png` inputs.

Dataset creators and users are responsible for ensuring that their source data
are legally shareable and appropriately governed for their institution and use case.

## Acquisition Process

Paired supervision assumes that the named inputs and targets depict corresponding
tissue. Dataset creators must establish whether acquisition supports that assumption.

The repository does not enforce any acquisition hardware, stain protocol, or
institution-specific procedure. Instead, it assumes the user provides the named input
images and every named target image of a set, then performs computational alignment of
each non-reference image to the reference input's frame. Filenames, matching
dimensions, identity alignment, successful SIFT or an injected registration backend,
transform fit, and foreground masks do not prove biological correspondence.

Registration execution success, transform geometry, independent QC, and biological
qualification are separate. Missing QC is not acceptance, and preparation foreground
masks are not independent registration-QC evidence.

## Preprocessing Pipeline

Preparation covers:

1. **Source validation**: validates the inventory and required named source assets.
2. **Reference-frame alignment**: maps every moving input and target directly to
   one explicit reference frame using shared registration geometry. Without an
   alternate backend, preparation retains built-in identity/SIFT behavior.
   Programmatic callers can supply another registration implementation through the
   [public preparation boundary](docs/library_api.md#injecting-registration-into-preparation).
3. **Bounded patch extraction**: extracts all named images on the reference grid
   using the configured patch size, margin, and grid step. Transformed reads are
   bounded; no full aligned whole-slide image is materialized.
4. **Foreground/white-area filtering**: rejects samples when configured foreground coverage is too
   low, white/background coverage is too high, or the largest white connected
   component exceeds the configured threshold.
5. **Split assignment**: assigns accepted samples to `train`, `val`, and `test`
   using the selected split unit, ratios and seed, or supplied grouped assignments.
6. **Manifest publication**: writes accepted-patch records to `manifests/manifest.csv`
   and discarded-patch records to `manifests/discarded_manifest.csv`.
   Discarded patch images are saved only when `save_discarded_patches` is
   enabled in preprocessing config.
7. **Preparation provenance**: records configuration, source identity, split
   assignments, and per-asset registration results. Persisted details belong to the
   [dataset format](docs/dataset_format.md).

## Splits

The dataset builder writes accepted patches into:

- `splits/train/`
- `splits/val/`
- `splits/test/`

`manifests/manifest.csv` is the canonical index of accepted patches and their
split assignments. Downstream training, inference, and evaluation stages rely on
this manifest rather than discovering files ad hoc.

Supported split units are `patch` (default), `set`, `specimen`, and `patient`.
`set` keeps all patches of a supplied set together; `specimen` and `patient`
require real `specimen_id` and `patient_id` metadata, respectively. A set is not
automatically a patient, specimen, or independent biological unit. Group labels
must reflect the actual dataset; software support does not validate their truth.
Missing or unknown metadata must narrow the generalization claim, not be guessed.
`slide` is not a supported split-unit value; spatial-block splitting is not implemented.

## Known Leakage Risk

With the default **patch-level** split, patches from the same source image, set,
specimen, or patient can enter different partitions. Such test metrics do not
establish unseen-patient or unseen-specimen performance and may overestimate
generalization. Grouped splitting is available as described above; actual
independence depends on the chosen experiment and the available metadata. Neither
split choice alone establishes transfer to new institutions or acquisition settings.

## Metrics

Built-in paired evaluation metrics available to request include:

- MAE
- MSE
- RMSE
- PSNR
- SSIM
- PCC (grayscale)
- PCC per RGB channel and RGB mean

Only requested metrics are evaluated, against appropriate corresponding references.
Results remain separate for each named output. See
[`docs/run_format.md`](docs/run_format.md) for validity, coverage, aggregation, and
persisted result semantics.

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
  See [LICENSE](LICENSE).
- **Privacy**: if source images originate from patient tissue, users are
  responsible for de-identification, ethics review, consent handling, and any
  required IRB or institutional approval.
- **Data and weights rights**: the software license grants no rights to
  user-provided microscopy data, trained weights, or third-party data. Dataset
  creators are responsible for their own licensing and sharing constraints.
