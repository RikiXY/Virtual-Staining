# Reproducibility

Each pipeline stage records the supplied configuration and its resolved,
canonical form:

```text
input YAML
    -> RunConfig.from_yaml()
    -> typed domain configs
    -> RunConfig.to_dict()
    -> YAML with sorted keys
    -> SHA-256
```

The input snapshot is an exact copy of the YAML passed to the command. The
resolved snapshot contains parsed values and defaults from every configured
domain. Loading that resolved YAML produces the same `RunConfig`:

```python
RunConfig.from_yaml(resolved_path) == config
```

Sorted YAML keys make the resolved file and its `sha256:<hex>` hash stable for
equivalent effective configurations, regardless of input key order. The hash
identifies the resolved configuration bytes; it does not include source data or
the software environment.

Training writes `config/resolved.yaml` and `metadata/config_hash.txt` under the
run directory. Inference and evaluation use stage-specific resolved YAML and
hash files so they do not overwrite training provenance. Preparation writes the
same snapshot set under the dataset root.

Configuration hashes are only one part of the provenance record. Dataset
manifests and source files have separate SHA-256 values, while environment JSON
records the Python, dependency, platform, and accelerator context needed to
interpret or reproduce a run.

See [Run Output Format](run_format.md) and [Dataset Format](dataset_format.md)
for artifact locations and schemas.
