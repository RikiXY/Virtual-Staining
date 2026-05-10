import argparse
from pathlib import Path
from typing import cast

import torch
from torch.amp import autocast
from torchvision import transforms
from torchvision.utils import save_image

from virtual_staining.common.dimensions import to_torchvision_hw
from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import PairedHistologyDataset
from virtual_staining.models.config import ModelConfig
from virtual_staining.models.factory import build_generator
from virtual_staining.training.checkpoints import _check_generator_arch
from virtual_staining.training.config import InferenceConfig
from virtual_staining.training.runner import run_training


# --------------------- Working paths ---------------------
# This function builds all the main project paths starting from a single
# root. It avoids scattering hardcoded strings throughout the rest of
# the file.
def build_workspace_paths(run_root: str | Path) -> dict:
    root = Path(run_root)

    return {
        "run_root": root,
        "logs_dir": root / "logs",
        "checkpoints_dir": root / "checkpoints",
        "output_val_dir": root / "output_val",
        "output_test_dir": root / "output_test",
        "output_train_dir": root / "output_train",
    }


# --------------------- Command-line arguments ---------------------
# We separate parser and logic.
# The script supports two distinct modes: training and testing.
def build_parser():
    parser = argparse.ArgumentParser(
        prog="python src/pix2pix.py",
        description="Train or test the Pix2Pix model on a paired histology dataset.",
        epilog=(
            "Examples:\n"
            "  python src/pix2pix.py train --config config/runs/example.yaml\n"
            "\n"
            "  python src/pix2pix.py test --config config/runs/example.yaml\n"
            "\n"
            "Use 'python src/pix2pix.py <command> --help' "
            "to see the options for a specific command."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the Pix2Pix model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="config/runs/example.yaml",
        help="path to the run config YAML (default: config/runs/example.yaml)",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Run inference on the test set",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    test_parser.add_argument(
        "--config",
        type=str,
        default="config/runs/example.yaml",
        help="path to the run config YAML (default: config/runs/example.yaml)",
    )

    return parser


def is_amp_enabled(device):
    # Mixed precision with `autocast` and `GradScaler` is mainly useful
    # on CUDA GPUs. On CPU we leave everything disabled to avoid unnecessary
    # complexity and keep behaviour as straightforward as possible.
    return isinstance(device, torch.device) and device.type == "cuda"


# --------------------- Main ---------------------
def main(argv=None) -> None:
    """Backward-compatible CLI entry point for src/pix2pix.py."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "train":
        config_path = Path(args.config).resolve()
        config = RunConfig.from_yaml(config_path)
        run_training(config, config_path)
        return

    if args.mode == "test":
        config = InferenceConfig.from_yaml(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Starting test for run: {config.run_root}")
        print(f"Using checkpoint: {config.checkpoint}")
        test_inference(
            checkpoint_path=config.checkpoint,
            test_folder=str(config.test_dir),
            output_folder=str(config.output_test_dir),
            image_size=config.image_size,
            device=device,
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


def test_inference(checkpoint_path, test_folder, output_folder, image_size=(256, 256), device=None):
    """
    Runs inference on the test set and saves the generated images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig()
    G = build_generator(model_config.generator).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_image_size = checkpoint.get("image_size")
    if checkpoint_image_size is not None:
        checkpoint_image_size = tuple(checkpoint_image_size)
        requested_image_size = tuple(image_size)

        if checkpoint_image_size != requested_image_size:
            raise ValueError(
                "Image size mismatch between checkpoint and inference. "
                f"Checkpoint was trained with image_size={checkpoint_image_size}, "
                f"but inference is using image_size={requested_image_size}. "
                "Set image_size in the run config or use a matching "
                "checkpoint."
            )

    checkpoint_arch = checkpoint.get("architecture")
    if checkpoint_arch is None:
        raise ValueError(
            "Checkpoint has no architecture metadata. "
            "Only checkpoints saved with the current version are supported."
        )
    _check_generator_arch(checkpoint_arch, G)

    G.load_state_dict(checkpoint["generator_state_dict"])
    G.eval()

    # Preprocessing must stay consistent with training,
    # otherwise the model would receive inputs with a different distribution.
    transform = transforms.Compose(
        [
            transforms.Resize(to_torchvision_hw(image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    dataset = PairedHistologyDataset(test_folder, transform=transform)

    if not dataset.pairs:
        print(f"No test source files found in: {test_folder}")
        return

    amp_enabled = is_amp_enabled(device)

    with torch.no_grad():
        for idx in range(len(dataset)):
            source_tensor, _ = dataset[idx]
            source_tensor = cast(torch.Tensor, source_tensor)
            source_path = dataset.pairs[idx][0]
            prefix = source_path.stem[: -len("_source")]
            out_filename = f"{prefix}_target_generated{source_path.suffix.lower()}"

            img_tensor = source_tensor.unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=amp_enabled):
                fake_target = G(img_tensor)

            # Bring the output back to the range suitable for saving.
            fake_target = (fake_target * 0.5) + 0.5
            fake_target = fake_target.clamp(0, 1)

            out_path = Path(output_folder) / out_filename
            save_image(fake_target, out_path)

    print(f"Test completed. Images saved in {output_folder}")


if __name__ == "__main__":
    main()
