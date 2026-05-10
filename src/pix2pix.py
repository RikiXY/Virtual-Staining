import argparse
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.inference.runner import run_inference
from virtual_staining.training.runner import run_training


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


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "train":
        config_path = Path(args.config).resolve()
        config = RunConfig.from_yaml(config_path)
        run_training(config, config_path)
        return

    if args.mode == "test":
        config_path = Path(args.config).resolve()
        config = RunConfig.from_yaml(config_path)
        run_inference(config, config_path)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
