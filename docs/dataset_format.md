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
│   └── dataset_fingerprint.json
└── discarded_patches/<set_id>/

```

`metadata/dataset_build.json` is the successful dataset provenance record.
`metadata/dataset_fingerprint.json` stores the semantic preprocessing,
canonical inventory, source-file hashes, and a `sha256:` fingerprint. These
dataset-owned artifacts are built by `data/provenance.py`. Experiment runs read
this fingerprint and the manifest hash as lineage; they do not write generic
run, event, or stage metadata into the dataset directory.

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
