# Translation methods

Virtual Staining supports two deliberately small baselines behind one method boundary:

- `pix2pix` performs supervised paired translation from one or more named inputs to one target.
- `cyclegan` performs unpaired image-to-image translation between exactly two RGB domains. It
  uses two ResNet generators, two PatchGAN discriminators, LSGAN objectives, cycle-consistency
  L1, and optional identity L1.

The method owns how models, losses, optimizers, batches, and checkpoint state interact. The
shared trainer owns epoch lifecycle, progress, history, checkpoint selection, provenance, and
reporting. This is the extension boundary; it is not an arbitrary N-to-M graph API.

## Data semantics

Pix2Pix remains backward compatible: omitting `method` and `data` selects `pix2pix` and
`paired`. Its manifest rows identify registered source/target patches.

CycleGAN requires explicit independent domain roots:

```yaml
method: {name: cyclegan}
data:
  pairing: unpaired
  domains:
    label_free: label_free
    stained: stained
model:
  inputs: [label_free]
  target: stained
```

Relative domain roots are resolved under `dataset_root`. Each root must contain `train`, `val`,
and `test` directories. The domains may contain different numbers of images and filenames do
not need to match. `vs prepare` remains the registered paired preprocessing path; it reports an
explicit error for unpaired configuration instead of applying registration to unrelated images.
An existing mixed split directory can be reused without copying files by using a `{split}` glob,
for example `splits/{split}/*_source.tif` and `splits/{split}/*_target.tif`.
The first CycleGAN baseline requires `training.augmentation.enabled: false`; paired geometric
augmentation is not silently reused for independent domains.

See `config/runs/cyclegan.example.yaml` for a complete CycleGAN baseline.

## Inference and evaluation

Set `inference.direction` to `A_to_B` or `B_to_A`. The ordinary `vs infer` and
`vs infer-images` commands reconstruct the correct generator from the checkpoint. Tiled image
inference remains available because the directional generator implements the existing predictor
contract.

Evaluation protocol is independent from training pairing:

```yaml
evaluation:
  protocol: paired  # auto, paired, or unpaired
  save_graphs: true
```

Use `paired` when the held-out manifest contains aligned references, including for CycleGAN
trained unpaired. It reuses MAE, RMSE, PSNR, SSIM, PCC, summaries, and the existing plots. Use
`unpaired` for independent test pools. It writes `unpaired_image_statistics.csv`,
`unpaired_evaluation.json`, and optional RGB/luminance distribution plots; these are preliminary
dataset-level diagnostics and do not measure sample fidelity or biological correctness. `auto`
uses the training pairing for backward compatibility. Validation grids remain qualitative and do
not imply physical recovery or invertibility.

For an unpaired target pool outside `data.domains`, provide either a domain directory containing
`test/` or a `{split}` glob:

```yaml
evaluation:
  protocol: unpaired
  real_target: "external_target/{split}/*.tif"
  save_graphs: true
```

## Custom components

A custom PyTorch component is imported directly:

```yaml
model:
  inputs: [label_free]
  target: stained
  generator:
    architecture: custom
    class_path: my_package.models:MyGenerator
    params: {features: 32}
```

The class must subclass `torch.nn.Module`; constructor keyword arguments come from `params`.
For Pix2Pix, the generator accepts the named tensor mapping used by `ConcatUNetGenerator`. For
CycleGAN, it accepts and returns an RGB NCHW tensor. A custom discriminator must also subclass
`nn.Module`; its calling convention must match the chosen method.

A custom method uses the same direct import mechanism:

```yaml
method:
  name: custom
  class_path: my_package.methods:MyMethod
  params: {temperature: 0.5}
```

Its constructor receives `config`, `device`, and `params`. The runtime exposes `name`, `pairing`,
`loss_names`, and `optimizers`, plus `train_mode`, `step`, `validate`, `component_metadata`,
`state_dict`, `load_state_dict`, and `load_legacy_v3`. Invalid runtimes fail during method
resolution with the missing contract members listed. `optimizers` contains exactly two groups,
matching the shared generator/discriminator scheduler and progress fields. There is intentionally no discovery,
registration decorator, or plugin manager.

## Checkpoints

New training runs use method checkpoint v4. It stores the method and pairing, component types
and custom class paths, all model/optimizer/scaler/scheduler states, image and normalization
contracts, and the full resolved configuration. Existing v3 Pix2Pix checkpoints remain loadable
for inference and Pix2Pix resume. A v3 checkpoint is never reinterpreted as CycleGAN.
