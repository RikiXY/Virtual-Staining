import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from virtual_staining.data.dataset import PairedHistologyDataset
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.training.config import TrainingConfig
from virtual_staining.training.trainer import Trainer


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
        "  python src/pix2pix.py train "
        "--dataset-root local_workspace/datasets/inverted_256 "
        "--run-name inv_P-256_L1-25 "
        "--epochs 100 "
        "--image-size 256 256\n"
        "\n"
        "  python src/pix2pix.py test "
        "--dataset-root local_workspace/datasets/inverted_256 "
        "--run-path local_workspace/results/inv_P-256_L1-25 "
        "--checkpoint local_workspace/results/inv_P-256_L1-25/checkpoints/ep099.pth\n"
        "\n"
        "Use 'python src/pix2pix.py <command> --help' "
        "to see the options for a specific command."
    ),
    formatter_class=argparse.RawTextHelpFormatter
)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the Pix2Pix model",
        formatter_class=argparse.RawTextHelpFormatter
    )
    train_parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root containing dataset_train/ and dataset_val/"
    )
    train_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the output run directory to create"
    )
    train_parser.add_argument(
        "--results-path",
        type=str,
        default="local_workspace/results",
        help="Base directory where the new run folder will be created (default: local_workspace/results)"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, a random seed is generated."
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the DataLoader (default: 8)"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of DataLoader workers (default: min(4, cpu_count))"
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
        help=(
            "Resize images before training as HEIGHT WIDTH "
            "(default: 256 256). Use 512 512 for 512x512 patch experiments."
        )
    )
    train_parser.add_argument(
        "--log-rate",
        type=int,
        default=15,
        help="Log every N batches (default: 15)"
    )
    train_parser.add_argument(
        "--checkpoint-rate",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (default: 10)"
    )
    train_parser.add_argument(
        "--validate-rate",
        type=int,
        default=10,
        help="Run validation every N epochs (default: 10)"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from"
    )
    train_parser.add_argument(
        "--l1-weight",
        type=float,
        default=25.0,
        help="Weight of the L1 reconstruction loss (default: 25.0)"
    )
    train_parser.add_argument(
        "--lr-g",
        type=float,
        default=2e-4,
        help="Learning rate for the generator (default: 2e-4)"
    )
    train_parser.add_argument(
        "--lr-d",
        type=float,
        default=2e-4,
        help="Learning rate for the discriminator (default: 2e-4)"
    )
    train_parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5)"
    )
    train_parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999)"
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Run inference on the test set",
        formatter_class=argparse.RawTextHelpFormatter
    )
    test_parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root containing dataset_test/"
    )
    test_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to use for inference"
    )
    test_parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="Path to an existing training run containing checkpoints/ and output folders"
    )
    test_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
        help=(
            "Resize images before inference as HEIGHT WIDTH "
            "(default: 256 256). Must match the image size used during training."
        )
    )

    return parser


def is_amp_enabled(device):
    # Mixed precision with `autocast` and `GradScaler` is mainly useful
    # on CUDA GPUs. On CPU we leave everything disabled to avoid unnecessary
    # complexity and keep behaviour as straightforward as possible.
    return isinstance(device, torch.device) and device.type == "cuda"


# --------------------- Determinism ---------------------
def set_seed(seed):
    # Fixing the seed greatly reduces variability and helps when comparing experiments.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- Main ---------------------
def main(config: TrainingConfig) -> None:
    seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_dir = Path(config.dataset_root) / "dataset_train"
    val_dir = Path(config.dataset_root) / "dataset_val"

    train_loader = DataLoader(
        PairedHistologyDataset(train_dir, transform=transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        PairedHistologyDataset(val_dir, transform=transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    ).train(seed=seed)


def test_inference(
    checkpoint_path,
    test_folder,
    output_folder,
    image_size=(256, 256),
    device=None
):
    """
    Runs inference on the test set and saves the generated images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    G = UNetGenerator().to(device)
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
                "Pass the correct --image-size or use a matching checkpoint."
            )
    G.load_state_dict(checkpoint["generator_state_dict"])
    G.eval()

    # Preprocessing must stay consistent with training,
    # otherwise the model would receive inputs with a different distribution.
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = PairedHistologyDataset(test_folder, transform=transform)

    if not dataset.pairs:
        print(f"No test source files found in: {test_folder}")
        return

    amp_enabled = is_amp_enabled(device)

    with torch.no_grad():
        for idx, (source_tensor, _) in enumerate(dataset):
            source_path = dataset.pairs[idx][0]
            prefix = source_path.stem[:-len("_source")]
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
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        main(TrainingConfig.from_args(args))

    elif args.mode == "test":
        test_dir = Path(args.dataset_root) / "dataset_test"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        run_root = Path(args.run_path)
        paths = build_workspace_paths(run_root)

        print(f"Starting test for run: {run_root}")
        print(f"Using checkpoint: {args.checkpoint}")
        test_inference(
            checkpoint_path=args.checkpoint,
            test_folder=str(test_dir),
            output_folder=str(paths["output_test_dir"]),
            image_size=tuple(args.image_size),
            device=device
        )
