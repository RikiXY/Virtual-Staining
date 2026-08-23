# Dataset Format

## Pair inventory

New datasets provide `inputs/pairs.csv`. Required columns are `pair_id`,
`source_path`, and `target_path`; optional columns are `already_aligned`, mask
paths, patient/specimen IDs, and source/target slide IDs. Paths are relative to
`dataset_root`. Pair IDs must match `[A-Za-z0-9][A-Za-z0-9._-]*`.

```csv
pair_id,source_path,target_path,already_aligned,patient_id,specimen_id
P001,raw/source/S001.svs,raw/target/T001.svs,false,PT001,SP001
P002,raw/source/S002.svs,raw/target/T002.svs,true,PT002,SP002
```

The pair inventory is required; single-pair datasets use the same format with
one row.

## Prepared layout

```text
dataset_root/
├── inputs/pairs.csv
├── splits/{train,val,test}/<pair_id>/
├── manifests/
│   ├── manifest.csv
│   ├── discarded_manifest.csv
│   └── pairs.csv
├── metadata/
│   ├── pairs/<pair_id>.json
│   ├── split_assignment.csv
│   ├── excluded_pairs.csv
│   ├── dataset_build.json
│   ├── dataset_fingerprint.json
│   └── input_hashes.json
├── discarded_patches/<pair_id>/
└── resolved_masks/<pair_id>/
```

Dataset-global output directories are initialized once. Pair processing never
deletes another pair's files.

## Patch manifest v2

`manifests/manifest.csv` is the authoritative training, inference, and
evaluation contract. Its exact columns are:

```text
sample_id,pair_id,split,input_path,target_path,foreground_mask_path,
input_modality,target_modality,x,y,width,height
```

- `sample_id` is globally unique and opaque, currently
  `<pair_id>__x<8 digits>_y<8 digits>`.
- `x` and `y` are level-0 coordinates in the original source image, including
  any configured margin offset.
- `foreground_mask_path` is authoritative and may be blank. Version 2 never
  guesses mask filenames.
- Output ordering is `pair_id`, `y`, then `x`.

The loader accepts this exact v2 column set only. Older manifests must be
rebuilt with `vs prepare`.

`manifests/pairs.csv` is the normalized cohort contract. It contains one row
per input pair, its assigned split, biological IDs, mask inputs, processing
status, alignment method, and pair metadata path.

## Leakage-safe splitting

Inventory configurations explicitly choose `patch`, `pair`, `specimen`, or
`patient`. Pair/specimen/patient groups are assigned before patch extraction by
a stable hash of the seed and group ID. Every nonzero split receives at least
one independent group or preparation fails. `metadata/split_assignment.csv`
freezes the result; an optional assignment file must exactly match the cohort.

Patch mode supports leakage ablations. Training pools and shuffles all training
patches each epoch; validation and test remain unshuffled.

## Masks and alignment

Mask layout (`none`, `shared`, `separate`, or `auto`) and generation (`never`,
`if_missing`, or `always`) are independent policies. Partial or mixed supplied
layouts are rejected. Shared masks require declared identity alignment and
compatible geometry. Maskless operation requires foreground filtering to be
disabled and alignment to resolve to identity.

Every processed pair has either an identity or `affine_sift` alignment result.
Registration failure aborts or creates an explicit excluded-pair record,
according to `alignment.on_failure`; it never silently falls back to identity.
Target patches and target masks continue to be warped on demand.

## WSI readers and provenance

`io.backend` accepts `auto`, `pillow`, or `openslide`. OpenSlide is optional;
install the `wsi` extra plus the native OpenSlide library. Reader metadata
records pyramid levels, downsampling, MPP when available, and vendor.

The dataset fingerprint includes canonical preprocessing policy, normalized
pair inventory, pair/group metadata, and every source, target, and supplied
mask hash. Inventory row order does not affect semantic identity. Normal builds
may reuse cached content hashes when path, size, and nanosecond mtime match;
`inputs.hash_verification: always` rereads every input for publication builds.

## Evaluation

Evaluation preserves patch-level outputs and writes pair summaries plus
specimen/patient summaries when those IDs are complete. Metrics
are averaged within each independent unit before aggregate statistics and
deterministic percentile bootstrap confidence intervals are computed.
