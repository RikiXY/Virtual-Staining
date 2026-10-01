# Reproducibility

Tracked stages preserve the supplied YAML and its canonical resolved form. The input
snapshot is an exact copy; the resolved snapshot contains parsed values and defaults.
Loading the resolved YAML with the same definitions produces the same `RunConfig`.

Sorted YAML keys make the resolved file and its `sha256:<hex>` hash stable for equivalent
effective configurations, regardless of input key order. This hash identifies config
bytes only, excluding source data and the software environment. Artifact locations and
stage bindings are documented in [Run Output Format](run_format.md#file-descriptions).

Data identity is separate: [consumed-data snapshots](run_format.md#consumed-data-snapshots)
describe the files selected by each tracked stage, while the
[dataset fingerprint](dataset_format.md#prepared-layout) describes preparation lineage.
Neither file identity nor matching configuration proves biological independence or
scientific validity.

Environment snapshots record package versions, including OpenSlide Python, pyvips,
PyTorch, NumPy, OpenCV, and Albumentations, plus optional CUDA/GPU facts. Version recording
is best-effort provenance; `vs status` checks whether required packages and native WSI
libraries are actually usable.
