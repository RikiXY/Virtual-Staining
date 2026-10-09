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
it does not prepare a dataset. Paired mode is the default and authors wide
`slide_sets.csv`; explicit `--pairing unpaired` authors long-form raw `paths.csv`.
Experiment-time `data.domains` collections remain separate. The
[Python API](library_api.md#authoring-the-slide-set-inventory) exposes the same
authoring operation.

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
Programmatic registration injection adds a `registration` record containing the caller's
`identifier`, `version`, frozen JSON `options`, and `qc_disposition`; all participate in
the fingerprint. Without injection this record is absent. Callable identities and per-asset
execution results are not fingerprint inputs. See the
[preparation injection API](library_api.md#injecting-registration-into-preparation).
Supplied registration evidence adds a sorted `registration_evidence` list to that same
fingerprint. Each entry records `set_id`, `modality`, owning `asset` geometry, `kind`,
`grid`, optional `source`, and `values_sha256`: a digest of boolean values packed in
C order with little-endian bit order. Arrays are not expanded into JSON. Absent evidence
adds no record; changing content, geometry, binding or source changes preparation identity.
These input identities are separate from per-asset registration/QC execution results.

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

Alignment metadata uses the versioned result format below. Preparation's existing
alignment flags declare coordinates only; its foreground masks are not tissue-support
or observation-validity evidence. Candidate estimation does not certify correspondence.
Each `<modality>__alignment_metadata` cell in `metadata/slide_sets.csv` retains the complete
result below, including injected results and QC outcomes when an explicit disposition
continues or excludes a set. A skipped backend failure retains its typed attempt/failure;
an unattempted asset has an empty cell. QC absence is serialized as null, never acceptance.

### Persisted alignment geometry and results

Transforms retain `virtual_staining.alignment/1` (`kind: transform`). Results emitted
by `AlignmentResult.metadata` and their attempt records use
`virtual_staining.alignment.result/2`, with `kind: result` and `kind: attempt`
respectively. Their `from_dict()` methods accept only these canonical representations;
results using the superseded `virtual_staining.alignment/1` schema are rejected.
Superseded 2×3 geometry and unknown fields/families are also rejected.

A transform contains `direction: moving_level0_to_reference_level0`, a finite float64
homogeneous 3×3 `matrix`, `family` (`identity`, `similarity`, or `affine`), and explicit
`moving`/`reference` frames with `name`, `(height, width)` `shape`, and nullable `(x, y)`
`mpp`. Integer `(x, y)` denotes a pixel centre, x right and y down. The pixel-cell
extent is `[-0.5, width-0.5) × [-0.5, height-0.5)`; index boxes are half-open.
`moving_grid` and `reference_grid` are both null or contain estimation-grid `shape`
and `grid_to_level0` matrices, including crop, resize and centre offsets. Matrix
coefficients retain double precision through JSON round trips. No physical scale is
inferred from dimensions; reflection and deformation are unsupported.

A result contains `backend_status`, `method`, nullable `candidate`, backend
`diagnostics`, nullable `qc`, a required `attempt`, a nullable success/provenance
`reason`, and the `request` declarations. The request
keeps relationship, transform permissions, existing alignment, purpose, diagnostic
region and correspondence evidence separate. QC records `status` (`accepted`,
`rejected`, `insufficient_evidence`), nullable metrics, missing evidence and reasons.
Backend failure has no candidate or QC decision and requires a typed
`attempt.failure`: validated `category`, optional stable `subcode`, and a nonempty
human-readable `message`. Success has no failure. QC rejection remains independent
of backend outcome. A `same_coordinate_frame` declaration requires equal native
shape and compatible known per-axis MPP, including when loading a supplied result.

`RegistrationAttempt` records stage, outcome, optional run/case/attempt IDs, UTC Unix
start/end seconds, monotonic duration seconds, and `RegistrationRuntime` evidence.
Runtime fields cover backend/version, model/checkpoint identity and hash, supplied
input metadata/fingerprint references, requested/resolved moving/reference grids,
support mode, device, precision, determinism, seed, scientific parameter hash, and
optional resource measurements (CPU/GPU/temp-disk peak bytes and reader count).
Resolution reuses the explicit grid-to-level-0 maps rather than inferring physical
spacing. An optional attempt QC snapshot must agree with result QC when supplied.
Fallback decisions and next-attempt IDs record caller decisions without executing
retries. Diagnostic artifacts are references only: at most 16, each at most 2048
characters. Unavailable runtime evidence is null; reader handles and image/evidence
arrays are never serialized.

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

## Independent unpaired preparation

`vs prepare` also accepts exactly two ordered, independent raw-image domains through
`data.pairing: unpaired`. The [minimal prepare-only YAML](../config/runs/minimal_unpaired_prepare.yaml)
uses `preprocessing.inputs.inventory: inputs/paths.csv` and
`preprocessing.inputs.domains: [LF, HE]`. There is no reference, target, alignment,
model, method, results directory, run name, or training direction. The domain order
is preserved as inventory/output metadata; it does not assign model inputs/outputs.
`data.domains` retains its experiment-time collection meaning.

Hand-author one long-form CSV under `dataset_root`. Only `domain,path` are required:

```csv
domain,path,set_id,specimen_id,patient_id
LF,raw/LF/image_a.tif,set_a,specimen_a,patient_1
HE,raw/HE/image_b.png,set_b,specimen_b,patient_2
HE,raw/HE/image_c.jpg,set_c,specimen_c,patient_3
```

The IDs above are illustrative placeholders, not claims about real images. Patient,
specimen and set IDs must be supplied by the dataset owner when their split or group
policy requires them. Rows have no positional correspondence. Different counts,
dimensions, resolutions, tissue support and retained-patch counts are expected.
The runnable [software inventory](../examples/unpaired/paths.csv) intentionally supplies
no biological IDs and uses the explicit patch-split engineering exception.

### Authoring independent raw domains

```bash
vs inventory preview --pairing unpaired --dataset-root DATASET \
  --domain LF=raw/LF --domain 'HE=raw/HE/**/*.tif' --metadata metadata.csv
vs inventory write --pairing unpaired --dataset-root DATASET \
  --domain LF=raw/LF --domain 'HE=raw/HE/**/*.tif' --metadata metadata.csv \
  --output inputs/paths.csv
```

Exactly two distinct domain names are required, in the requested order. Each mapping
selects one recursive directory or glob using the paired command's traversal conventions.
Selections must stay within `dataset_root`; traversal, symlinked sources or directories,
nonregular files, missing/empty selections and unsupported matches fail. Overlapping
selections and hard-link aliases fail even within one domain. Distinct files with the
same basename are allowed. Use a glob to exclude unrelated files from a source directory.
No names, positions, dimensions or counts are matched between domains.
`--input`, `--target`, `--reference`, mask-mapping flags and `--key` are paired-only.

`--metadata` is optional. Its CSV joins by the **exact discovered root-relative path**:

```csv
path,set_id,specimen_id,patient_id
raw/LF/image_a.png,S01,SP01,P01
raw/HE/sample_1.tif,S02,SP02,P02
```

Supply real dataset-owner identities in place of these illustrative values. Partial
metadata is allowed; absent IDs remain unknown, never inferred from paths. Duplicate
keys, undiscovered paths, malformed/unknown columns, invalid IDs and contradictory or
incomplete parent identities fail. Optional `mask_path` uses the canonical root-relative
locator and file validation; authoring rejects symlinked masks. Mask dimensions and
pixels are checked during preparation. Missing patient IDs do not invalidate a raw
inventory, but do not satisfy a patient split.

Output columns are `domain,path`, then only optional columns carrying a value in the
order `set_id,specimen_id,patient_id,mask_path`. Rows follow domain request order, then
sorted exact root-relative POSIX paths. The canonical `load_unpaired_inventory()` reader
validates both the in-memory preview CSV and staged CSV, including metadata readback.
The result is a normal editable raw inventory with no authoring cache or sidecar dependency.

Preview lists domain specifications, counts, membership, issues and missing-ID limitations;
it creates no files or directories and performs no image decoding or content hashing.
Ordinary formats are recognized by extension; other formats require OpenSlide format
detection. This does not establish image integrity, biological identity, independence
or correspondence. Write repeats discovery and metadata reading, compares membership
and source/mask device, inode, size and modification/change timestamps, then publishes
without replacing an existing destination, including a competing writer's file.
Only owned temporary files are cleaned after failure; an output parent directory may
remain. These checks do not freeze sources against subsequent changes or verify pixels.
The default output is `inputs/paths.csv`; output containment rules match paired mode.

To prepare, set `dataset_root` in the existing
[minimal unpaired prepare config](../config/runs/minimal_unpaired_prepare.yaml) to
`DATASET`, retain `preprocessing.inputs.inventory: inputs/paths.csv` and the ordered
`preprocessing.inputs.domains: [LF, HE]`, and supply explicit patient IDs for its patient
split. Then run:

```bash
vs config check --config config/runs/minimal_unpaired_prepare.yaml --stages prepare
vs config check --config config/runs/minimal_unpaired_prepare.yaml --stages prepare --assets
vs prepare --config config/runs/minimal_unpaired_prepare.yaml
```

Raw `preprocessing.inputs.domains` names the two inventory domains without model roles.
Experiment-time `data.domains` maps those names to prepared split directories or globs;
use the collections reported by preparation. Authoring assigns no training direction.

### Raw inventory validation

Headers must be unique; unknown columns, short/long rows, surrounding whitespace,
empty or unknown domains, duplicate paths/physical files (including aliases and hard
links), missing files and unsupported image formats fail. The ordinary image extensions
are PNG, JPEG, BMP and TIFF; other extensions require OpenSlide recognition. Domain names match
`[A-Za-z][A-Za-z0-9_-]*` exactly; optional IDs match
`[A-Za-z0-9][A-Za-z0-9._-]*`. Paths are normalized, relative to `dataset_root`, and must
stay inside it after symlink resolution. The inventory itself must also be contained
there. An internal symlink is allowed only when it does not reuse another selected
physical file. Repeated set/specimen IDs must declare consistent parent IDs, including
consistent missingness. No identifier is derived from a filename.

The only additional optional asset column is `mask_path`, a root-relative image-local
binary mask with the same native dimensions as its source. Masks use known values
0/255; unknown/nonbinary values fail. Image decoding is deferred to preparation:
config-only commands read no assets, and `--assets` checks inventory membership,
groups and split feasibility without decoding or content hashing.

```bash
vs config check --config unpaired_prepare.yaml --stages prepare
vs config check --config unpaired_prepare.yaml --stages prepare --assets
vs config resolve --config unpaired_prepare.yaml --stages prepare
vs prepare --config unpaired_prepare.yaml
```

Conditional preparation options reuse the existing catalogue:

| Option | Independent-image meaning |
| --- | --- |
| `inputs.domains` | Exactly two unique names; replaces `modalities`, `reference`, `target_modalities`. |
| `patching` | Existing size, movement and margin in each source's native pixels. Outputs are RGB PNG patches; no shared origins or geometry. |
| `masks.generation` | `never` uses supplied masks or remains maskless; maskless processing requires `foreground.enabled: false`. `if_missing` generates only absent masks; `always` regenerates them. |
| `masks.strategy`, `scale` | Existing connected-component/HSV algorithms. At scale 1, generate masks on native patch windows to keep tiled reads bounded. An explicitly smaller scale generates an image-local overview mask; patch pixels remain native resolution. |
| `masks.lowres_filtering` | Evaluate foreground fraction on the overview-mask window before nearest-neighbor expansion when enabled. |
| `masks.save_resolved_masks`, `save_patch_masks` | Save resolved overviews (or resolved native windows at scale 1) and accepted patch masks outside image collections. Maskless processing writes no invented masks. |
| `filtering.foreground.policy` | Defaults to `all`: the single image's foreground. Explicit `reference`, `target`, `intersection`, and `union` policies are rejected, even when filtering is disabled. |
| White/background filters | Apply the existing thresholds separately to each image patch. No cross-domain rejection. |
| `alignment` | Unsupported for unpaired images, including programmatic registration injection. |
| `io` | Existing backend and tiled selection. Tiled native extraction requests only patch windows; no implicit downsampling, rescaling or reorientation. Explicit memory budgets reject excessive estimated working arrays. Pillow may decode a whole ordinary image internally; use native OpenSlide-compatible tiled TIFFs for bounded WSI access. |
| `split` | Assign all supplied groups across both domains before decoding. Requested IDs are mandatory. Shared explicit set/specimen/patient identities connect the chosen split units; contradictory frozen assignments fail. |
| `split.assignment_file` | Existing exact `group_id,unit,split` CSV; must cover all actual selected-unit IDs. Unsupported for patch splitting. |
| `data.group_validation` | Existing group checks. Patch splitting requires `unavailable`; shared groups and the limited independence claim are recorded. It is never described as patient-independent. |
| `inputs.hash_verification` | Accepted existing option; unpaired reuse always verifies source content, including under the weaker `data.hash_policy: membership` snapshot policy. |

Group fractions allocate independent connected groups, not equal patch counts.
Every nonzero split must retain at least one image from each domain or preparation
fails; it does not move groups or invent patches to fill a missing collection. A split
with fraction zero remains empty and must not be selected by a consumer requiring
nonempty collections. Duplicate content across disjoint splits fails, including
supplied raw masks; patch splitting does not exempt content leakage.

### Grouped preparation template for your data

The [two-column example](../examples/unpaired/prepare.yaml) remains valid without
biological metadata. The richer [CSV template](../examples/unpaired/paths_grouped.csv)
and [prepare-only YAML](../examples/unpaired/prepare_grouped.yaml) illustrate nested
patient/specimen/set identities. **They are templates for your data:** the image paths
and `EXAMPLE_` identifiers are placeholders, not bundled images or biological ground
truth. Replace them with actual paths and recorded identifiers; do not assign these
example identities to real images.

The CSV illustrates six patients, twelve specimens and twenty-four sets, with unequal
LF/HE image counts. Its first rows show multiple images in one set, multiple sets in
one specimen, different specimens from one patient, and shared groups across domains:

```csv
domain,path,set_id,specimen_id,patient_id
LF,raw/LF/image_000.png,EXAMPLE_S001_A1,EXAMPLE_SP001_A,EXAMPLE_P001
LF,raw/LF/image_001.png,EXAMPLE_S001_A1,EXAMPLE_SP001_A,EXAMPLE_P001
LF,raw/LF/image_002.png,EXAMPLE_S001_A2,EXAMPLE_SP001_A,EXAMPLE_P001
HE,raw/HE/image_003.tif,EXAMPLE_S001_A2,EXAMPLE_SP001_A,EXAMPLE_P001
HE,raw/HE/image_004.tif,EXAMPLE_S001_B1,EXAMPLE_SP001_B,EXAMPLE_P001
LF,raw/LF/image_005.png,EXAMPLE_S001_B2,EXAMPLE_SP001_B,EXAMPLE_P001
```

`patient_id` identifies the declared patient, `specimen_id` a specimen from that
patient, and `set_id` a collection within that specimen. Multiple images can share
these IDs, including across domains. Shared IDs express group membership; they do
not establish image pairing, alignment or anatomical correspondence.

`split.unit` selects the IDs whose assignments are recorded. `data.group_validation`
selects the required independence evidence. Edit these fields in the same YAML:

| `preprocessing.split.unit` | `data.group_validation` | Behavior |
| --- | --- | --- |
| `patient` | `patient` | Keep all images of each supplied patient together. |
| `specimen` | `patient` | Assign specimens, keeping specimens sharing a patient together. |
| `set` | `patient` | Assign sets, keeping shared specimens and patients together transitively. |
| `patch` | `unavailable` | Explicit engineering exception; source images and patients can span splits. |

Identifiers are optional **CSV columns**, but conditionally mandatory values:
patient/specimen/set splitting requires that ID on every row; explicit group
validation also requires its selected ID. Specimen splitting without patient IDs
may use specimen validation, but cannot establish patient independence. Patient
splitting can work without specimen IDs. Missing values are never inferred from
filenames. Reusing a child ID with conflicting or inconsistently missing parents
fails. Known higher-level IDs connect finer split units even under weaker validation;
`unavailable` does not waive leakage checks for grouped splits. Patch splitting
records its limitation in preparation and consumed-data snapshots and is unsuitable
for independent biological performance claims.

Copy the CSV to `<dataset_root>/inputs/paths.csv`, replace its rows with your data,
and edit `dataset_root`, patch geometry and fractions in the YAML. The example values
are illustrative, not study recommendations. Config-only checking requires no images;
after supplying the inventory and images, preflight and prepare from the repository root:

```bash
vs config check --config examples/unpaired/prepare_grouped.yaml --stages prepare
vs config check --config examples/unpaired/prepare_grouped.yaml --stages prepare --assets
vs prepare --config examples/unpaired/prepare_grouped.yaml
```

For frozen grouped partitions, set `preprocessing.split.assignment_file` to a
dataset-relative CSV with exactly `group_id,unit,split`. Include **every actual ID
of the selected unit**, assigning all connected IDs to the same split. Seed changes
do not override it. Missing/unexpected/duplicate groups, malformed rows, wrong units,
invalid split names and shared identities crossing splits fail before publication.

Inspect each logged build root: `metadata/split_assignment.csv` records assignments,
`metadata/groups.csv` records every accepted patch's supplied identities, and
`metadata/images.json` joins patches to raw sources and coordinates.
`metadata/prepared_data/` records output membership and validation limitations. Use
`splits/{train,val,test}/{LF,HE}/` with the consumer configuration below.
The regression tests load this template with temporary synthetic images and verify
all four split units against published patches and metadata, including the unpaired
resolver and CycleGAN data adapter. This verifies software behavior, not real-data
biological independence.

### Unpaired output and consumption

The command logs a deterministic build root:
`<dataset_root>/prepared_unpaired/<fingerprint>/`. It contains:

```text
config/{input.yaml,resolved.yaml}
splits/{train,val,test}/{LF,HE}/<source_digest>__x00000000_y00000000.png
metadata/groups.csv
metadata/split_assignment.csv
metadata/images.json
metadata/dataset_fingerprint.json
metadata/dataset_build.json
metadata/config_hash.txt
metadata/environment.json
metadata/consumed_data/prepare/{snapshot.json,rows.csv}
metadata/prepared_data/{snapshot.json,rows.csv}
masks/                         # only if requested
discarded_patches/              # only if requested
```

`source_digest` hashes the explicit domain and root-relative source path; it is a file
identity, not a biological ID. `images.json` associates accepted/excluded origins with
actual raw source paths, domains, supplied group metadata, geometry and rejection
reasons. The canonical group sidecar columns are exactly
`path,domain,split,set_id,specimen_id,patient_id`. No paired manifest is created.

Use the logged build root as the later experiment's `dataset_root` and select collections
explicitly (the model and direction remain the experiment owner's choices):

```yaml
dataset_root: local_workspace/datasets/unpaired_example/prepared_unpaired/<fingerprint>
data:
  pairing: unpaired
  domains:
    LF: 'splits/{split}/LF/*.png'
    HE: 'splits/{split}/HE/*.png'
  group_metadata: metadata/groups.csv
  hash_policy: content
  group_validation: patient
```

These are directly consumed by `resolve_domain_collections()` and
`UnpairedImageDataset`. For a prepared patch split, use `group_validation: unavailable`;
the training adapter recognizes the verified preparation sidecar/assignment evidence
and records the engineering exception. Preparation snapshots remain dataset-local;
training writes its own consumed-data evidence later. No train/infer/evaluate run
metadata is created by preparation.

Publication builds in an owned temporary sibling directory, verifies source stability,
accepted output membership and output hashes, then publishes the complete directory using an exclusive native rename on Linux/macOS
(no overwrite, even for an empty destination). Filesystems lacking this operation fail
explicitly.
The fingerprint covers resolved configuration/stage identity, selected inventory bytes,
explicit source/group metadata, verified source content/stat information and frozen
assignments. Changed input/configuration creates a different build root; older builds
and paired outputs are retained. Reuse requires the matching fingerprint and every
recorded artifact with its original hash, without extra files or symlinks. An incomplete
or edited destination causes an explicit collision error and is preserved. Failed
attempts leave `failure-*.json` evidence beside builds and remove only their owned
temporary directory. Users manage retention of older builds; preparation never deletes
raw images, caller files or earlier builds.
