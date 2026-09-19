# Test Suite Structure

Tests follow the production owner whose public behavior they protect.

- applications/: stage orchestration and user-visible application behavior.
- cli/: command-line parsing, rendering, and queue dispatch contracts.
- config/: configuration parsing, validation, and serialization only.
- data/: manifests, preprocessing, splitting, patching, filtering, slide-set processing, and dataset building.
- evaluation/: metrics, summaries, plots, ranking, selection, and comparison behavior.
- experiment/: run metadata, environment snapshots, and reproducibility artifacts.
- inference/: inference dispatch, outputs, tiling, and WSI runtime behavior.
- models/: model configuration and architecture contracts.
- smoke/: end-to-end workflow coverage; keep this distinct from unit tests.
- training/: augmentation, losses, training steps, checkpoints, history, and trainer behavior.
- utils/: reusable utility contracts.
- architecture/: a small set of intentional dependency boundaries, not an exhaustive mirror of the current import graph.

Shared helpers at the test root construct canonical current data/configuration shapes. They must not silently translate legacy fields into current fields. Compatibility tests that need legacy input must spell that input out locally.

Prefer tests through public contracts. Direct private-function tests are reserved for narrow pure parsers, mathematically sensitive algorithms, or characterization boundaries where the public surface would hide the behavior being protected. Monkeypatch private workers only when necessary to isolate expensive or external work; do not preserve a private name merely because a test imports it.

tests/data/test_alignment_legacy.py is temporary characterization coverage for the existing SIFT-based registration implementation. Keep it stable enough to detect regressions, but do not expand or polish that implementation before the replacement registration system lands.

Test files should map cleanly to current ownership. Mixed application/configuration or builder/processor tests should be split when the distinction improves navigation; there is no target test-count reduction.

The existing slow marker policy remains the default. Add markers only when a concrete runtime need appears.

Shared helpers:

- conftest.py: pytest fixtures.
- config_helpers.py: canonical YAML/run configuration setup.
- image_helpers.py: small synthetic image creation.
- manifest_helpers.py: synthetic manifest construction.
