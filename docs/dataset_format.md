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
