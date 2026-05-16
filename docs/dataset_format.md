# Dataset Format

## Directory Layout

`vs-prepare` writes its outputs into the dataset root folder specified by
`dataset_root` in the run YAML.

```
local_workspace/datasets/<name>/
├── raw/                        # (reserved for future use)
├── processed/                  # intermediate aligned images
├── splits/
│   ├── train/                  # accepted train-split patch pairs
│   ├── val/                    # accepted validation-split patch pairs
│   └── test/                   # accepted test-split patch pairs
├── discarded_patches/
│   ├── source/                 # optional discarded source patch images
│   ├── target/                 # optional discarded target patch images
│   └── discarded_log.csv       # per-patch filter diagnostics
├── manifests/
│   ├── manifest.csv            # all accepted patches with split assignment
│   └── discarded_manifest.csv  # discarded patches (split = "discarded")
├── config/
│   ├── input.yaml              # exact copy of the user-supplied run YAML
│   └── resolved.yaml           # fully expanded effective run config
└── metadata/
    ├── config_hash.txt         # sha256 of config/resolved.yaml
    ├── dataset_build.json      # build statistics and provenance
    ├── dataset_fingerprint.json # cache identity for prepare reuse decisions
    └── environment.json        # runtime environment snapshot
```

## Manifest as Pipeline Contract

`manifests/manifest.csv` is the authoritative record of which accepted patches
exist and which dataset split they belong to. All downstream pipeline stages
depend on it:

- **Training** (`vs-train`) loads the manifest, validates it, filters to the
  `train` and `val` splits, and fails loudly if the manifest is missing.
- **Inference** (`vs-infer`) loads the manifest, validates it, filters to the
  `test` split, and fails loudly if the manifest is missing.
- **Evaluation** (`vs-evaluate`) loads the manifest, validates it, iterates the
  `test` split records, and pairs each target image with the generated image
  expected for that manifest record.

All pipeline stages therefore require `manifests/manifest.csv` to exist.

To use a non-default manifest (for example a curated subset or a
cross-validation fold), set `manifest_path:` in the run YAML. If omitted, the
default path is `<dataset_root>/manifests/manifest.csv`. See the commented
example in [`config/runs/example.yaml`](../config/runs/example.yaml).

## manifest.csv Columns

All accepted patches (train, val, test) are indexed in `manifests/manifest.csv`.
Paths are relative to the dataset root.

Accepted source and target files are stored side by side under the split
directory named by the manifest `split` value. Example manifest row:

```csv
00512_09216,train,splits/train/00512_09216_source.tif,splits/train/00512_09216_target.tif,label_free,stained,512,9216,256,256
```

Split assignment is deterministic for a fixed `seed` and `sample_id`: each
accepted patch is assigned by a stable hash and written directly to its final
`splits/<split>/` directory during preparation. Configured split ratios are
therefore approximate, not exact-count guarantees.

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | Unique identifier for the patch (e.g. `00512_09216`) |
| `split` | string | Dataset split: `train`, `val`, or `test` |
| `input_path` | path | Relative path to the input (source) patch image |
| `target_path` | path | Relative path to the target patch image |
| `input_modality` | string | Imaging modality of the input (e.g. `label_free`) |
| `target_modality` | string | Imaging modality of the target (e.g. `stained`) |
| `x` | int | Left edge of the patch in the full-size source image (pixels) |
| `y` | int | Top edge of the patch in the full-size source image (pixels) |
| `width` | int | Patch width in pixels |
| `height` | int | Patch height in pixels |

### Sample Identity Contract

`sample_id` encodes the top-left coordinate of a patch within the source image:
`f"{x:05}_{y:05}"` (zero-padded 5-digit integers, for example `00512_09216`
for a patch starting at column 512, row 9216).

**Uniqueness scope**: `sample_id` is unique within one `vs-prepare` run over a
single source/target image pair.

**Not globally unique**: if `vs-prepare` is run separately on two different
slides and both runs contain the same patch coordinates, the resulting manifests
will contain colliding `sample_id` values. Such manifests cannot be concatenated
safely under the current contract.

**Downstream invariants that depend on uniqueness**:

- `DatasetManifest.validate()` rejects duplicate `sample_id` values.
- Generated image filenames use `{sample_id}_target_generated.<ext>`.
- Evaluation skipped-sample logs use `sample_id` as the row key.
- Reproducibility and manifest-hash workflows include `sample_id` values.

**Future multi-slide support**: would require a different identity scheme such
as `{slide_id}_{x:05}_{y:05}` to avoid collisions across slides. Any change to
the `sample_id` format would also require a manifest schema version bump and
updates to downstream filename and evaluation contracts.

Until multi-slide support is added, every manifest is expected to correspond to
exactly one source/target image pair.

## discarded_manifest.csv

Patches that failed quality filtering are recorded in
`manifests/discarded_manifest.csv` with `split = "discarded"`.
The same columns as `manifest.csv` apply.

Detailed per-patch filter diagnostics (foreground ratio, white ratio, component
ratio, and failure reasons) are written to `discarded_patches/discarded_log.csv`.
This CSV is the canonical audit trail for discarded patches and is always
written. Discarded patch image files are written only when
`preprocessing.save_discarded_patches: true`; otherwise the `source/` and
`target/` subdirectories under `discarded_patches/` may be absent.

## Tissue Mask Strategies

`preprocessing.mask_strategy` controls how tissue foreground masks are
generated. The default is `connected_components`, which preserves the historical
multi-scale bright-component heuristic. The alternative `hsv` strategy marks
pixels as tissue when they are sufficiently saturated or darker than bright
background, then applies morphology cleanup to remove small speckles and fill
small holes.

Use `preprocessing.source_mask_strategy` and
`preprocessing.target_mask_strategy` to override the global strategy for one
modality. Valid values are `connected_components` and `hsv`. The HSV strategy is
faster and useful for stained or visibly colored tissue, but it can be less
appropriate for very pale or low-saturation label-free inputs.

## Low-Resolution Mask Filtering

When `preprocessing.mask_scale < 1.0`, masks are computed on downsampled image
copies. By default, those masks are resized back to full image resolution before
patch filtering, preserving the historical pixel-space behavior.

Set `preprocessing.lowres_mask_filtering: true` to keep those masks in their
downsampled mask space during patch admission. Source patch windows are mapped
from full-resolution image coordinates into mask coordinates, and target mask
patches are warped from mask space on demand. This reduces long-lived mask
memory, but foreground ratios are approximate near mask boundaries because they
are measured on fewer mask pixels. The saved patch images, manifest coordinates,
split paths, and discarded log schema are unchanged.

## Region-Based Image I/O

Set `preprocessing.tiled_io: true` to use the region-reader preparation path.
This path opens source and target images through the shared image-reader
abstraction, reads low-resolution previews for mask generation and global
alignment, and reads only the source and target regions required for each patch
during streaming. It supports the local image formats handled by Pillow,
including PNG, JPEG, TIFF, and BMP files.

For whole-slide-scale inputs, pair `tiled_io: true` with `mask_scale < 1.0`;
the preview scale is controlled by `mask_scale`. In tiled mode, foreground
filtering uses mask-space patch windows even when `lowres_mask_filtering` is not
set explicitly. The affine model is still global and rigid/affine; non-rigid
whole-slide registration and cloud/object-store streaming are not implemented.

## dataset_build.json Fields

`metadata/dataset_build.json` records build statistics and is written once
`vs-prepare` completes successfully.

| Field | Type | Description |
|---|---|---|
| `dataset_name` | string | Name of the dataset root directory |
| `status` | string | Always `"completed"` on success |
| `started_at` | ISO 8601 | UTC timestamp when the build started |
| `completed_at` | ISO 8601 | UTC timestamp when the build finished |
| `num_patches_total` | int | Total patches extracted before filtering |
| `num_patches_valid` | int | Patches that passed all quality filters |
| `num_patches_discarded` | int | Patches removed by quality filters |
| `num_train` | int | Patches assigned to the train split |
| `num_val` | int | Patches assigned to the val split |
| `num_test` | int | Patches assigned to the test split |
| `seed` | int | Random seed used for the train/val/test split |

## dataset_fingerprint.json Fields

`metadata/dataset_fingerprint.json` records the cache identity of the prepared
dataset. It is written after a successful `vs-prepare` run and is the intended
input for later reuse-or-rebuild decisions.

`vs-prepare` may skip rebuilding and reuse the existing dataset only when this
fingerprint matches the current preprocessing request and the required prepared
outputs still exist. Matching fingerprint metadata alone is not sufficient.

The fingerprint is derived from:

- the resolved `preprocessing` section only
- the configured `dataset_root`
- source image provenance
- target image provenance

Source and target provenance include absolute path, file size, `mtime_ns`, and
SHA-256 content hash so reuse decisions fail closed if image contents change.

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Fingerprint metadata schema version |
| `fingerprint` | string | Canonical SHA-256 dataset fingerprint |
| `prepared_at` | ISO 8601 | UTC timestamp when the fingerprint was recorded |
| `dataset_root` | path | Absolute path of the dataset root this fingerprint applies to |
| `preprocessing` | object | Canonical resolved preprocessing payload only |
| `preprocessing_hash` | string | SHA-256 of the canonical preprocessing payload |
| `source` | object | Source image provenance record |
| `target` | object | Target image provenance record |

Each provenance record contains:

| Field | Type | Description |
|---|---|---|
| `path` | path | Absolute filesystem path of the input image |
| `size` | int | File size in bytes |
| `mtime_ns` | int | File modification time in nanoseconds |
| `sha256` | string | SHA-256 of the file contents |

## Config Snapshots

`config/input.yaml` preserves the exact YAML passed to `vs-prepare`.

`config/resolved.yaml` contains the fully expanded effective run configuration
for the prepare stage, including the resolved preprocessing section that drove
dataset creation.

`metadata/config_hash.txt` stores the SHA-256 hash of `config/resolved.yaml`,
and `metadata/environment.json` records the runtime environment captured at
prepare start.
