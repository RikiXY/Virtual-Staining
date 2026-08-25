# Reproducibility

Each pipeline stage records the supplied configuration and its resolved,
canonical form:

```text
input YAML
    -> RunConfig.from_yaml()
    -> config/ typed section dataclasses and strict parsers
    -> RunConfig.to_dict()
    -> YAML with sorted keys
    -> SHA-256

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

Each run stage writes `config/<stage>/input.yaml`,
`config/<stage>/resolved.yaml`, and `metadata/environments/<stage>.json`.
The resolved YAML hash is stored in the stage record and event rather than in
a standalone hash file. Preparation keeps its dataset-local
`config/input.yaml`, `config/resolved.yaml`, `metadata/config_hash.txt`, and
`metadata/environment.json` snapshots; dataset fingerprint construction and
source-file hashing belong to `data/provenance.py`, while run snapshot writers
belong to `experiment/snapshots.py`.
`DatasetLayout` owns dataset-local provenance paths and `RunLayout` owns
run-local stage paths. The shared `RuntimeInfo` collector supplies the same
runtime facts to environment snapshots and status diagnostics.

See [Run Output Format](run_format.md) and [Dataset Format](dataset_format.md)
for artifact locations and schemas.
