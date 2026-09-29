# Dataset format

## Wide slide-set inventory

`inputs/slide_sets.csv` is a wide inventory. Paths are relative to `dataset_root`.
Configured input names must match `[A-Za-z][A-Za-z0-9_-]*`.

For modalities `LF,AF`, required columns are:

```text
set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,target_path,target_aligned
S001,raw/lf/S001.svs,true,raw/af/S001.svs,false,raw/he/S001.svs,false
```

Optional columns are `input__<modality>_mask`, `input__<modality>_slide_id`,
`target_mask`, `target_slide_id`, `patient_id`, and `specimen_id`.
The configured reference input must be marked aligned. Missing or legacy columns
are rejected.

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
  --target-modality HE --target raw/HE --reference LF \
  [--input-mask AF=masks/AF] [--target-mask masks/HE] \
  [--key relative-path|relative-stem] [--metadata meta.csv]
vs inventory write ... [--output inputs/slide_sets.csv]
```

**Mappings.** Every input (ordered, one per `--input NAME=SPEC`), the target, and each
optional mask is named explicitly; nothing is inferred from folder names and no other
directory is scanned. A spec is relative to `dataset_root` and is either:

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
the target have exactly one file with that key. A key missing from any required mapping,
two files with the same key in one mapping (e.g. `S001.svs` and `S001.tif` under
`relative-stem`), or a target that is the same file as an input is an error. All
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
`input__<modality>_aligned`, `input__<modality>_slide_id`, `target_aligned`,
`target_slide_id`. Duplicate keys, keys matching no discovered asset key, unknown
columns, mask columns, modalities not in the request, and alignment values other than
`true`/`false`/blank are errors. Patient, specimen and slide IDs are never invented.

**Alignment.** The reference input is written `true` unless metadata says otherwise, and
metadata declaring it `false` is an error; `true` means identity to the declared reference
coordinate system only, not correspondence certification. Every other input and the
target keep an explicitly supplied `true`/`false`, otherwise the field stays blank
(unknown). Alignment is never inferred from names, keys, directories, or dimensions.

**Masks.** `--input-mask NAME=SPEC` and `--target-mask SPEC` use the same key rule and
fill `input__<modality>_mask` / `target_mask`; a set without a mask leaves it blank.
Duplicate mask keys and mask keys matching no set are errors. Masks are never generated.

**Rendering and publication.** Columns are `set_id`, each input's path and alignment in
request order, `target_path`, `target_aligned`, then only the optional columns that
carry a value, in the order per-input mask and slide ID, `target_mask`,
`target_slide_id`, `patient_id`, `specimen_id`. Rows are sorted by `set_id` and all
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
last preparation attempt selected: every input slide, target slide, and supplied mask
with its modality and set/specimen/patient IDs (format in
[Run Output Format](run_format.md#consumed-data-snapshots)). It is written before any
reuse decision or build. Under `data.hash_policy: content` its verified digests feed
the fingerprint in place of the size/mtime hash cache, so reuse is claimed only after
the selected sources were re-verified; the fingerprint records the snapshot as
`source_snapshot_id`. Splits are assigned during preparation, so this snapshot makes
no cross-split claim. Experiment runs record the manifest hash and fingerprint as
`sources` of their own stage snapshots; they do not write generic run, event, or stage
metadata into the dataset directory.

Every non-reference input and the target is aligned directly to the reference
coordinate frame. No full aligned whole-slide image is created.

## Patch manifest v3

`manifests/manifest.csv` is the only prepared-record contract. Its columns are
ordered by manifest metadata:

```text
sample_id,set_id,split,input__LF,input__AF,target_path,foreground_mask_path,x,y,width,height
```

`sample_id` is `<set_id>__x<8 digits>_y<8 digits>`. Each record contains one
relative path per named input, one target path, and optionally the target
foreground mask. `manifest_metadata.json` contains exactly the v3 identity:

```json
{
  "schema_version": "3.0",
  "input_modalities": ["LF", "AF"],
  "reference_modality": "LF",
  "target_modality": "H&E"
}
```

The loader rejects v1/v2 columns, absolute or traversing paths, missing names,
input/target collisions, duplicate paths, and missing files when requested.

## Named runtime samples

Datasets return:

```python
{"inputs": {"LF": lf_tensor, "AF": af_tensor}, "target": target_tensor, "masks": {}}
```

Model configuration selects an ordered subset with `model.inputs` and a target
with `model.target`.
