# Marimo

This directory is the user-facing home for optional Marimo documentation and
apps. Marimo is installed only through the `analysis` dependency group; it is
not part of the core training, inference, or evaluation runtime.

```bash
uv sync --frozen --group analysis
```

Open the editor:

```bash
make marimo
```

After a Marimo notebook exists, pass its path through `MARIMO_APP`:

```bash
make marimo MARIMO_APP=docs/marimo/evaluation_explorer.py
make marimo-run MARIMO_APP=docs/marimo/evaluation_explorer.py
```

Export a static HTML snapshot:

```bash
make marimo-export MARIMO_APP=docs/marimo/evaluation_explorer.py
```

The default export path is `docs/marimo/exports/<app>.html`. Override it with
`MARIMO_EXPORT_PATH=...` when needed, or pass extra export flags through
`MARIMO_EXPORT_ARGS=...`. The Make target defaults to `-f` so rerunning an
export refreshes the existing snapshot.

Static exports are snapshots, not pipeline artifacts. They must not contain
private absolute paths, private images, patient identifiers, or local run names.
When no public fixture data exists, export the app in its documented empty state.

The planned Marimo files are:

| File | Purpose |
|---|---|
| `alignment.py` | Alignment and preparation mechanics |
| `dataset_preparation.py` | Manifest-first dataset preparation walkthrough |
| `model_reference.py` | Model, dataset interface, and loss reference |
| `training_workflow.py` | Train, infer, evaluate workflow documentation |
| `evaluation_explorer.py` | Artifact-driven evaluation explorer |

Do not add placeholder Marimo files. Add each `.py` file only when its owning
implementation task builds a useful surface.
