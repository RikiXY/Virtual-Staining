# Dataset format

## Wide slide-set inventory

`inputs/slide_sets.csv` is a wide inventory, symmetric in named inputs and named
targets. Paths are relative to `dataset_root`. Every input and target name is a machine
identifier matching `[A-Za-z][A-Za-z0-9_-]*` (`HE` for H&E); names are never sanitized.

Each input `<name>` has the columns `input__<name>_path`, `input__<name>_aligned`,
`input__<name>_mask`, `input__<name>_slide_id`, and each target `<name>` the columns
`target__<name>_path`, `target__<name>_aligned`, `target__<name>_mask`,
`target__<name>_slide_id`; the `_path` and `_aligned` columns are required, the others
optional. `set_id`, `patient_id`, and `specimen_id` complete the row. For inputs `LF,AF`
and targets `HE,PAS`:

```text
set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,target__HE_path,target__HE_aligned,target__PAS_path,target__PAS_aligned
S001,raw/lf/S001.svs,true,raw/af/S001.svs,false,raw/he/S001.svs,false,raw/pas/S001.svs,
```

One target is the one-item case of the same columns. Every configured target is required
for every set. The configured reference input must be marked aligned. Missing,
duplicate, unknown, or superseded columns (the singular `target_path`, `target_aligned`,
`target_mask`, `target_slide_id`) are rejected, as are unsafe or symlink-escaping paths,
malformed set IDs, and any two assets of one set that are the same physical file
(input/target or target/target reuse, including through symlinks and hard links).

`DatasetLayout` in `virtual_staining.data.layout` is the single owner of these
dataset paths. `ProjectConfig` supplies YAML values only; it does not construct
persistent dataset paths.

## Authoring the inventory

`vs inventory preview|write` (library: `applications.inventory_authoring`) builds this
same wide CSV from explicit asset mappings. It is a raw-input authoring aid only: it
never writes a prepared manifest, manifest metadata, split assignment, patches,
fingerprints, or consumed-data snapshots, and CycleGAN `data.domains` collections are
not paired inventory and never need it.

```bash
vs inventory preview --dataset-root DATASET \
  --input LF=raw/LF --input 'AF=raw/AF/**/*.svs' \
  --target HE=raw/HE --target PAS=raw/PAS --reference LF \
  [--input-mask AF=masks/AF] [--target-mask HE=masks/HE] [--target-mask PAS=masks/PAS] \
  [--key relative-path|relative-stem] [--metadata meta.csv]
vs inventory write ... [--output inputs/slide_sets.csv]
```

The library request has the same shape:

```python
InventoryRequest(
    dataset_root=root,
    inputs=(("LF", "raw/LF"), ("AF", "raw/AF")),
    targets=(("HE", "raw/HE"), ("PAS", "raw/PAS")),
    reference="LF",
    target_masks=(("HE", "masks/HE"), ("PAS", "masks/PAS")),
)
```

**Mappings.** Every input (ordered, one per `--input NAME=SPEC`), every target (ordered,
one per `--target NAME=SPEC`), and each optional mask is named explicitly; nothing is
inferred from folder names and no other directory is scanned. A spec is relative to
`dataset_root` and is either:

- a *directory*: every regular file below it, recursively; or
- a *glob* (`*`, `?`, `[...]`, `**` for any number of directories): its *anchor* is the
  longest leading path without glob characters (`raw/AF` for `raw/AF/**/*.svs`).

Keys are paths relative to the directory or glob anchor (`raw/LF/case1/S001.svs` under
`raw/LF` is `case1/S001.svs`), listed in sorted POSIX order. Absolute specs, `..`,
symlinked files, symlinked directories (below the anchor or on the way to it), glob
matches that are directories, non-regular files, and required mappings that match no
file are errors. No image is opened.

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
`target__<name>_slide_id` (target-specific, e.g. `target__PAS_slide_id`). Duplicate
keys, keys matching no discovered asset key, unknown
columns, mask columns, modalities not in the request, and alignment values other than
`true`/`false`/blank are errors. Patient, specimen and slide IDs are never invented.

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
`specimen_id`. Rows are sorted by `set_id` and all
paths are `dataset_root`-relative POSIX paths, so the same request yields the same
bytes. `preview` writes nothing. `write` refuses an invalid preview, reruns discovery
and requires the same membership and rows, writes a sibling temporary file, loads it
with the canonical slide-set loader, and only publishes it (by hard link) when it
resolves to the previewed sets. The default output is `inputs/slide_sets.csv`; another
`--output` must stay inside `dataset_root` (relative paths are relative to it). An
existing destination is never replaced and there is no overwrite option; only the output's
parent directory may be created.

Name matching establishes no biological independence, patient or specimen identity,
spatial correspondence, or registration validity beyond metadata you supply.

## Prepared layout

```text
dataset_root/
├── inputs/slide_sets.csv
├── splits/{train,val,test}/<set_id>/
├── manifests/
│   ├── manifest.csv
│   ├── discarded_manifest.csv
│   ├── manifest_metadata.json
│   └── slide_sets.csv
├── metadata/
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
build, and it decides whether an existing complete dataset can be reused. These
dataset-owned artifacts are built by `data/provenance.py`.

`metadata/consumed_data/prepare/` is the consumed-data snapshot of the raw assets the
last preparation attempt selected: every input slide, every target slide, and supplied mask
with its modality and set/specimen/patient IDs (format in
[Run Output Format](run_format.md#consumed-data-snapshots)). It is written before any
reuse decision or build. Under `data.hash_policy: content` its verified digests feed
the fingerprint in place of the size/mtime hash cache, so reuse is claimed only after
the selected sources were re-verified; the fingerprint records the snapshot as
`source_snapshot_id`. Splits are assigned during preparation, so this snapshot makes
no cross-split claim. Experiment runs record the manifest hash and fingerprint as
`sources` of their own stage snapshots; they do not write generic run, event, or stage
metadata into the dataset directory.

Every non-reference input and every target is aligned directly to the reference
coordinate frame with the existing alignment code. No full aligned whole-slide image is
created. All inputs and targets are extracted at the same reference-grid position; a
sample is committed to a split only after every one of its images (and, with
`masks.save_patch_masks`, every target's foreground mask) has been written and its
dimensions verified; a sample that fails leaves none of its files behind. Rejected positions are recorded in `discarded_manifest.csv` with
split `discarded`. A rebuild first withdraws `dataset_build.json` and the manifests, so a
preparation that fails part way never looks consumable.

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

The loader enforces the exact header order and rejects duplicate headers, rows with
missing or extra cells, malformed metadata types and unknown metadata fields, unsafe
identifiers, missing or extra named images, absolute or traversing paths, paths whose
symlink resolution leaves the dataset root (when files are checked), duplicate sample
IDs, cross-split sample conflicts, duplicate input paths, duplicate target paths,
input/target path reuse, invalid coordinates or extents, and missing files when
requested.

## Named runtime samples

`PairedManifestDataset` returns the selected names in configured order:

```python
{
    "inputs": {"LF": lf_tensor, "AF": af_tensor},
    "targets": {"PAS": pas_tensor, "HE": he_tensor},
    "masks": {"foreground_mask": {"PAS": pas_mask, "HE": he_mask}},
}
```

`masks` is `{}` unless a configured loss needs foreground masks; then every selected
target carries its own `1HW` mask and a mask of one target is never used for another.
Collation gives `NCHW` images and `N1HW` masks under the same names. Model configuration
selects `model.inputs` from the manifest input modalities and `model.outputs` from its
target modalities, each in any order; model order is authoritative.
