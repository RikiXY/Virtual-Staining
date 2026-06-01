# Marimo

This directory is the user-facing home for optional Marimo documentation and
apps. Marimo is installed only through the `analysis` dependency group; it is
not part of the core training, inference, or evaluation runtime.
Plotly is also installed through this group and is used only inside Marimo apps,
for example the interactive histograms in `research/alignment.py`.

```bash
uv sync --frozen --group analysis
```

Open the editor:

```bash
make marimo
```

After a Marimo notebook exists, pass its path through `MARIMO_APP`:

```bash
make marimo MARIMO_APP=docs/marimo/research/alignment.py
make marimo-run MARIMO_APP=docs/marimo/inspection/alignment_explorer.py
```

Export a static HTML snapshot:

```bash
make marimo-export MARIMO_APP=docs/marimo/research/alignment.py
```

The default export path is `docs/marimo/exports/<app>.html`; for maintained
exports, mirror the source folder under `docs/marimo/exports/research/` or
`docs/marimo/exports/inspection/`. Override the path with
`MARIMO_EXPORT_PATH=...` when needed, or pass extra export flags through
`MARIMO_EXPORT_ARGS=...`. The Make target defaults to `-f` so rerunning an export
refreshes the existing snapshot.

Static exports are snapshots, not pipeline artifacts. They must not contain
private absolute paths, private images, patient identifiers, or local run names.
When no public fixture data exists, export the app in its documented empty state.

The planned Marimo files are:

| File | Purpose |
|---|---|
| `research/alignment.py` | Alignment research/example walkthrough using committed demo images |
| `inspection/alignment_explorer.py` | Read-only prepared alignment artifact inspector |
| `research/dataset_preparation.py` | Manifest-first dataset preparation walkthrough |
| `inspection/dataset_explorer.py` | Read-only prepared dataset inspector |
| `research/model_reference.py` | Model, dataset interface, and loss reference |
| `research/training_workflow.py` | Train, infer, evaluate workflow documentation |
| `inspection/evaluation_explorer.py` | Artifact-driven evaluation explorer |

Do not add placeholder Marimo files. Add each `.py` file only when its owning
implementation task builds a useful surface.
