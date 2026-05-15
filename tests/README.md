# Test Suite Structure

Tests are grouped by the subsystem they exercise:

- `cli/`: command-line contracts, Makefile integration, and run queues.
- `config/`: YAML parsing, validation, and resolved configuration contracts.
- `data/`: manifests, preprocessing, datasets, and dataset build outputs.
- `evaluation/`: metrics, summaries, plotting, ranking, and comparisons.
- `experiment/`: metadata, environment snapshots, and reproducibility artifacts.
- `inference/`: inference outputs and inference/evaluation compatibility.
- `models/`: model configuration and factory behavior.
- `smoke/`: end-to-end pipeline smoke tests.
- `training/`: checkpoints, runners, trainer behavior, and training result contracts.
- `utils/`: shared utility modules.

Shared test helpers stay at the `tests/` root:

- `conftest.py` contains pytest fixtures.
- `config_helpers.py` writes small YAML configs used across CLI and application tests.
- `image_helpers.py` creates small RGB images and source/target image pairs.
- `manifest_helpers.py` builds synthetic manifests for data and pipeline tests.

Prefer a local helper inside a test module when only that module uses it. Move a helper to the
root only when at least two domains need the same setup.
