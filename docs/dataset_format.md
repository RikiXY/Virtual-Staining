# Dataset Format

## Directory Layout

`vs-prepare` writes its outputs into the dataset root folder specified by
`dataset_root` in the run YAML.

```
local_workspace/datasets/<name>/
├── raw/                        # (reserved for future use)
├── processed/                  # intermediate aligned images
├── splits/                     # (reserved for future use)
├── dataset_train/              # train-split patches (input + target pairs)
├── dataset_val/                # validation-split patches
├── dataset_test/               # test-split patches
├── discarded_patches/
│   ├── source/                 # source patches that failed quality filters
│   ├── target/                 # corresponding target patches
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
    └── environment.json        # runtime environment snapshot
```

## manifest.csv Columns

All accepted patches (train, val, test) are indexed in `manifests/manifest.csv`.
Paths are relative to the dataset root.

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

## discarded_manifest.csv

Patches that failed quality filtering are recorded in
`manifests/discarded_manifest.csv` with `split = "discarded"`.
The same columns as `manifest.csv` apply.

Detailed per-patch filter diagnostics (foreground ratio, white ratio, component
ratio, and failure reasons) are written to `discarded_patches/discarded_log.csv`.

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

## Config Snapshots

`config/input.yaml` preserves the exact YAML passed to `vs-prepare`.

`config/resolved.yaml` contains the fully expanded effective run configuration
for the prepare stage, including the resolved preprocessing section that drove
dataset creation.

`metadata/config_hash.txt` stores the SHA-256 hash of `config/resolved.yaml`,
and `metadata/environment.json` records the runtime environment captured at
prepare start.
