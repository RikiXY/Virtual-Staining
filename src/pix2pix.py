import argparse
from pathlib import Path

import torch

from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.inference.runner import run_inference
from virtual_staining.models.config import ModelConfig
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
    Backward-compatible shim. Delegates to virtual_staining.inference.runner.
    """
    del device

    output_path = Path(output_folder)
    project = ProjectConfig(
        dataset_root=Path(test_folder).parent,
        results_path=output_path.parent,
        run_name=output_path.name,
        image_size=tuple(image_size),
    )
    inference = InferenceConfig(
        checkpoint_path=Path(checkpoint_path),
        test_dir=Path(test_folder),
        output_dir=output_path,
        project=project,
    )
    config = RunConfig(
        project=project,
        model=ModelConfig(),
        training=None,
        inference=inference,
        preprocessing=None,
        evaluation=None,
    )
    run_inference(config, Path(checkpoint_path))
    print(f"Test completed. Images saved in {output_folder}")


if __name__ == "__main__":
    main()
