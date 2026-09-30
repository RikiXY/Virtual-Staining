# Dataset format

## Wide slide-set inventory

`inputs/slide_sets.csv` is a wide inventory, symmetric in named inputs and named
targets. Paths are relative to `dataset_root`. Every input and target name is a machine
identifier matching `[A-Za-z][A-Za-z0-9_-]*` (`HE` for H&E); names are never sanitized.
Input and target name sets are disjoint; the reference names an input, and mask
mappings name their corresponding input or target.

Each input `<name>` has the columns `input__<name>_path`, `input__<name>_aligned`,
`input__<name>_mask`, `input__<name>_slide_id`, and each target `<name>` the columns
`target__<name>_path`, `target__<name>_aligned`, `target__<name>_mask`,
`target__<name>_slide_id`; the `_path` and `_aligned` columns are required, the others
optional. `set_id` is required; `patient_id` and `specimen_id` are optional. For inputs
`LF,AF` and targets `HE,PAS`:

```text
set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,target__HE_path,target__HE_aligned,target__PAS_path,target__PAS_aligned
S001,raw/lf/S001.svs,true,raw/af/S001.svs,false,raw/he/S001.svs,false,raw/pas/S001.svs,
```

Every configured input and target is required for every set; the reference input must
be marked aligned. Headers must be unique and use only the fields above. Set IDs must
be unique and match `[A-Za-z0-9][A-Za-z0-9._-]*`. Paths must be relative,
non-traversing, and remain inside `dataset_root` after symlink resolution. Input and
target images within a set must be distinct physical files, including through symlinks
and hard links.

## Authoring the inventory

`vs inventory preview|write` builds the raw inventory from explicit asset mappings;
it does not prepare a dataset. Unpaired `data.domains` collections do not use this
inventory. The [Python API](library_api.md#authoring-the-slide-set-inventory) exposes
the same authoring operation.

```bash
vs inventory preview --dataset-root DATASET \
  --input LF=raw/LF --input 'AF=raw/AF/**/*.svs' \
  --target HE=raw/HE --target PAS=raw/PAS --reference LF \
  [--input-mask AF=masks/AF] [--target-mask HE=masks/HE] [--target-mask PAS=masks/PAS] \
  [--key relative-path|relative-stem] [--metadata meta.csv]
vs inventory write ... [--output inputs/slide_sets.csv]
```

**Mappings.** Every input (ordered, one per `--input NAME=SPEC`), every target (ordered,
one per `--target NAME=SPEC`), and each optional mask is named explicitly; nothing is
inferred from folder names and no other directory is scanned. A spec is relative to
`dataset_root` and is either:

- a *directory*: every regular file below it, recursively; or
- a *glob* (`*`, `?`, `[...]`, `**` for any number of directories): its *anchor* is the
  longest leading path without glob characters (`raw/AF` for `raw/AF/**/*.svs`).

Keys are paths relative to the directory or glob anchor (`raw/LF/case1/S001.svs` under
`raw/LF` is `case1/S001.svs`), listed in sorted POSIX order. Specs must be relative and
non-traversing, with no symlinks along the path or below the anchor. Matches must be regular files, and every required mapping must be
non-empty. No image is opened.

**Key rule.** `relative-path` (default) matches the full relative path including the
extension: `case1/S001.svs` matches only `case1/S001.svs`. `relative-stem` removes only
the final extension, so `case1/S001.svs` and `case1/S001.tif` both become `case1/S001`
(but `S001.ome.tiff` becomes `S001.ome`). A key forms a set only when every input and
every target have exactly one file with that key. A key missing from any required
mapping, two files with the same key in one mapping (e.g. `S001.svs` and `S001.tif`
under `relative-stem`), or two mappings resolving to the same file is an error. All
discrepancies are listed together; incomplete keys are never silently dropped, and files
are never paired by position, even when every mapping holds the same number of files.

**Set IDs.** By default `set_id` is the key's file name without its final extension
(`case1/S001.svs` → `S001`). It is never sanitized: it must already match
`[A-Za-z0-9][A-Za-z0-9._-]*`, and all set IDs must be unique, so
`patient1/S001.tif` and `patient2/S001.tif` match as distinct sets but collide on
`S001` until the metadata CSV supplies explicit `set_id`s.

**Metadata CSV** (optional, relative to `dataset_root`). Joined on a required `key`
column holding keys in the selected key rule's form. Other allowed columns are only
existing inventory fields: `set_id`, `patient_id`, `specimen_id`,
`input__<name>_aligned`, `input__<name>_slide_id`, `target__<name>_aligned`,
`target__<name>_slide_id` (target-specific, e.g. `target__PAS_slide_id`). Keys must be unique
and refer to discovered asset keys; modality fields must name requested modalities. Alignment values are `true`, `false`, or blank; masks come
from the mask mappings, not metadata columns. Patient, specimen and slide IDs are
never invented.

**Alignment.** The reference input is written `true` unless metadata says otherwise, and
metadata declaring it `false` is an error; `true` means identity to the declared reference
coordinate system only, not correspondence certification. Every other input and every
target keep an explicitly supplied `true`/`false`, otherwise the field stays blank
(unknown). Alignment is never inferred from names, keys, directories, or dimensions.

**Masks.** `--input-mask NAME=SPEC` and `--target-mask NAME=SPEC` use the same key rule
and fill `input__<name>_mask` / `target__<name>_mask`; a set without a mask leaves it
blank.
Duplicate mask keys and mask keys matching no set are errors. Masks are never generated.

**Rendering and publication.** Columns are `set_id`, each input's then each target's path
and alignment in request order, then only the optional columns that carry a value, in
the order per-input mask and slide ID, per-target mask and slide ID, `patient_id`,
`specimen_id`. Rows are sorted by `set_id`; paths are `dataset_root`-relative POSIX paths.
The same request yields the same bytes. `preview` writes nothing. `write` requires a
valid preview with unchanged source membership and metadata, and the published CSV
must resolve to the previewed sets. The default output is `inputs/slide_sets.csv`;
`--output` must stay inside `dataset_root` without traversing symlinks (relative paths
resolve from that root). Existing destinations are never replaced; only the output's
parent directory may be created.

Name matching establishes no biological independence, patient or specimen identity,
spatial correspondence, or registration validity beyond metadata you supply.

## Prepared layout

```text
dataset_root/
├── config/{input.yaml,resolved.yaml}
├── inputs/slide_sets.csv
├── splits/{train,val,test}/<set_id>/
├── manifests/
│   ├── manifest.csv
│   ├── discarded_manifest.csv
│   ├── manifest_metadata.json
│   └── slide_sets.csv
├── metadata/
│   ├── config_hash.txt
│   ├── environment.json
│   ├── split_assignment.csv
│   ├── excluded_sets.csv
│   ├── dataset_build.json
│   ├── dataset_fingerprint.json
│   └── consumed_data/prepare/{snapshot.json,rows.csv}
└── discarded_patches/<set_id>/

```

`metadata/dataset_build.json` is the successful dataset provenance record.
`metadata/dataset_fingerprint.json` stores the semantic preprocessing,
canonical inventory, source-file hashes, and a `sha256:` fingerprint. It is
preparation lineage: it answers what dataset this configuration and these sources
build, and it decides whether an existing complete dataset can be reused.

`metadata/consumed_data/prepare/` is the consumed-data snapshot of the raw assets the
last preparation attempt selected: every input slide, every target slide, and supplied mask
with its modality and set/specimen/patient IDs (format in
[Run Output Format](run_format.md#consumed-data-snapshots)). It is written before any
reuse decision or build. Under `data.hash_policy: content`, reuse requires re-verified
source digests; the fingerprint records the snapshot as `source_snapshot_id`. Splits are assigned during preparation, so this snapshot makes
no cross-split claim. Experiment runs record the manifest hash and fingerprint as
`sources` of their own stage snapshots; they do not write generic run, event, or stage
metadata into the dataset directory.

Every non-reference input and every target is aligned directly to the reference frame.
No full aligned whole-slide image is created. Patches share reference-grid positions;
each accepted sample contains all named images and, when requested, every target's
foreground mask with verified dimensions. Failed samples leave no partial files.
Rejected positions appear in `discarded_manifest.csv` with split `discarded`; an
incomplete rebuild does not leave a consumable manifest or successful build record.

Alignment matrices map moving full-resolution `(x, y)` to reference full-resolution
`(x, y)`. Array shapes are `(height, width)` and output sizes `(width, height)`.
Preview estimation accounts for each image's actual per-axis scale, including resize
rounding; masks use nearest-neighbor resampling. Mask IoU measures estimation-space
overlap and is diagnostic, with no
rejection threshold. Serialized keypoint fields `n_keypoints_src` and
`n_keypoints_tgt` mean reference and moving keypoints respectively.

## Patch manifest v4

`manifests/manifest.csv` is the only prepared-record contract, and schema `4.0` is the
only accepted version: earlier manifests are rejected, never converted. Its columns are
ordered by manifest metadata:

```text
sample_id,set_id,split,input__LF,input__AF,target__HE,target__PAS,foreground_mask__HE,foreground_mask__PAS,x,y,width,height
```

`sample_id` is `<set_id>__x<8 digits>_y<8 digits>`. Every record holds one relative path
per named input and per named target (all required; there are no sparse targets) and a
`foreground_mask__<target>` column for every target, which may be blank. Patch files are
`<sample_id>__input__<name>`, `<sample_id>__target__<name>` and
`<sample_id>__foreground_mask__<name>` with the source extension. `x`, `y` are the
upper-left patch origin and `width`, `height` the extent, in reference level-0 pixels
with integer pixel centers; they are never model-resized tensor coordinates.
`manifest_metadata.json` holds the canonical identity plus the builder-owned producer
fields `created_at`, `record_count`, and `splits`:

```json
{
  "schema_version": "4.0",
  "input_modalities": ["LF", "AF"],
  "target_modalities": ["HE", "PAS"],
  "reference_modality": "LF",
  "coordinate_space": "reference_level0_pixels",
  "pixel_center": "integer"
}
```

The CSV must use the exact metadata-defined header order with one cell per column;
metadata permits only the canonical fields above and the typed producer fields
(`created_at`: string, `record_count`: integer, `splits`: mapping). Named modalities
must be unique safe identifiers. Retained sample IDs must be unique across splits
(`train`, `val`, `test`); a `discarded` row may repeat a retained sample ID. Input and
target paths must not be reused, and all referenced paths must be relative,
non-traversing, and contained in the dataset root after symlink resolution when files
are checked. Coordinates are non-negative integers and extents positive integers;
referenced files must exist when file validation is requested.

Runtime tensor shapes and model-order selection are documented in
[Named runtime samples](library_api.md#named-runtime-samples).
