"""
Integrated preprocessing pipeline for paired histopathology samples.

This script loads two paired full-size images (a source image and a target
image), computes tissue masks, aligns the target image to the source reference,
extracts paired patches, and creates the `dataset_train`, `dataset_val`, and
`dataset_test` splits.
"""

import argparse
from dataclasses import replace
from pathlib import Path

from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.results import DatasetBuildResult


def main(config: PreprocessingConfig) -> DatasetBuildResult:
    return DatasetBuilder(config).run_all()


def _apply_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> PreprocessingConfig:
    """Apply any CLI-specified fields on top of a YAML-loaded config."""
    kw: dict = {}
    if hasattr(args, "path"):
        kw["dataset_root"] = Path(args.path)
    if hasattr(args, "source_name"):
        kw["source_name"] = args.source_name
    if hasattr(args, "target_name"):
        kw["target_name"] = args.target_name
    if hasattr(args, "image_size"):
        kw["image_size"] = tuple(args.image_size)
    if hasattr(args, "grid_movement"):
        kw["grid_movement"] = tuple(args.grid_movement)
    if hasattr(args, "margin"):
        kw["margin"] = args.margin
    if hasattr(args, "seed"):
        kw["seed"] = args.seed
    if hasattr(args, "save_masks"):
        kw["save_masks"] = args.save_masks
    if hasattr(args, "min_foreground_ratio"):
        kw["min_foreground_ratio"] = args.min_foreground_ratio
    if hasattr(args, "max_white_ratio"):
        kw["max_white_ratio"] = args.max_white_ratio
    if hasattr(args, "white_threshold"):
        kw["white_threshold"] = args.white_threshold
    if hasattr(args, "max_largest_white_component_ratio"):
        kw["max_largest_white_component_ratio"] = args.max_largest_white_component_ratio
    return replace(config, **kw) if kw else config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage=(
            "python src/prepare_dataset.py [--config CONFIG]\n"
            "       [--path PATH --source-name SOURCE_NAME --target-name TARGET_NAME]\n"
            "       [--seed SEED] [--save-masks] [--image-size WIDTH HEIGHT]\n"
            "       [--grid-movement STEP_X STEP_Y] [--margin MARGIN]\n"
            "       [--min-foreground-ratio F] [--max-white-ratio F]\n"
            "       [--white-threshold N] [--max-largest-white-component-ratio F]"
        ),
        description="Prepares masks, aligns images, extracts subimages, and splits the dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to a preprocessing config YAML (CLI flags override matching fields)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=argparse.SUPPRESS,
        help="path to the folder containing the source and target images"
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=argparse.SUPPRESS,
        help="Source image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=argparse.SUPPRESS,
        help="Target image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=argparse.SUPPRESS,
        help="if set, also saves the subimage masks"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=argparse.SUPPRESS,
        help="Patch size used for extraction (default: 256 256)"
    )
    parser.add_argument(
        "--grid-movement",
        type=int,
        nargs=2,
        metavar=("STEP_X", "STEP_Y"),
        default=argparse.SUPPRESS,
        help="Grid step used for patch extraction (default: 256 256)"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=argparse.SUPPRESS,
        help="Margin cropped from each border before patch extraction (default: 200)"
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum foreground tissue ratio for a patch to be kept (default: 0.25)"
    )
    parser.add_argument(
        "--max-white-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum near-white pixel ratio for a patch to be kept (default: 0.7)"
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=argparse.SUPPRESS,
        help="Grayscale intensity threshold for classifying a pixel as near-white (default: 250)"
    )
    parser.add_argument(
        "--max-largest-white-component-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum ratio of the largest white connected component for a patch to be kept (default: 0.20)"
    )
    args = parser.parse_args()

    if args.config:
        config = PreprocessingConfig.from_yaml(args.config)
        config = _apply_overrides(config, args)
    else:
        missing = [flag for flag, attr in [
            ("--path", "path"),
            ("--source-name", "source_name"),
            ("--target-name", "target_name"),
        ] if not hasattr(args, attr)]
        if missing:
            parser.error(
                "the following arguments are required when --config is not given: "
                + ", ".join(missing)
            )
        config = PreprocessingConfig.from_args(args)

    main(config)
