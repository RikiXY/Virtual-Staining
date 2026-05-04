"""
Integrated preprocessing pipeline for paired histopathology samples.

This script loads two paired full-size images (a source image and a target
image), computes tissue masks, aligns the target image to the source reference,
extracts paired patches, and creates the `dataset_train`, `dataset_val`, and
`dataset_test` splits.
"""

import argparse
from pathlib import Path

from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.utils.cli import apply_namespace_overrides


_PREPROCESSING_OVERRIDES = {
    "path": ("dataset_root", Path),
    "source_name": "source_name",
    "target_name": "target_name",
    "image_size": ("image_size", tuple),
    "grid_movement": ("grid_movement", tuple),
    "margin": "margin",
    "seed": "seed",
    "save_masks": "save_masks",
    "min_foreground_ratio": "min_foreground_ratio",
    "max_white_ratio": "max_white_ratio",
    "white_threshold": "white_threshold",
    "max_largest_white_component_ratio": "max_largest_white_component_ratio",
}


def main(config: PreprocessingConfig) -> DatasetBuildResult:
    return DatasetBuilder(config).run_all()


def _apply_overrides(
    config: PreprocessingConfig, args: argparse.Namespace
) -> PreprocessingConfig:
    """Apply any CLI-specified fields on top of a YAML-loaded config."""
    return apply_namespace_overrides(config, args, _PREPROCESSING_OVERRIDES)


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
        default="config/preprocessing.yaml",
        help=(
            "path to a preprocessing config YAML "
            "(default: config/preprocessing.yaml; CLI flags override fields)"
        ),
    )
    parser.add_argument(
        "--path",
        type=str,
        default=argparse.SUPPRESS,
        help="path to the folder containing the source and target images",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=argparse.SUPPRESS,
        help="Source image filename with extension (.tif, .tiff, .png)",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=argparse.SUPPRESS,
        help="Target image filename with extension (.tif, .tiff, .png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=argparse.SUPPRESS,
        help="if set, also saves the subimage masks",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=argparse.SUPPRESS,
        help="Patch size used for extraction (default: 256 256)",
    )
    parser.add_argument(
        "--grid-movement",
        type=int,
        nargs=2,
        metavar=("STEP_X", "STEP_Y"),
        default=argparse.SUPPRESS,
        help="Grid step used for patch extraction (default: 256 256)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=argparse.SUPPRESS,
        help="Margin cropped from each border before patch extraction (default: 200)",
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum foreground tissue ratio for a patch to be kept (default: 0.25)",
    )
    parser.add_argument(
        "--max-white-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum near-white pixel ratio for a patch to be kept (default: 0.7)",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=argparse.SUPPRESS,
        help="Grayscale intensity threshold for classifying a pixel as near-white (default: 250)",
    )
    parser.add_argument(
        "--max-largest-white-component-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum ratio of the largest white connected component for a patch to be kept (default: 0.20)",
    )
    args = parser.parse_args()

    config = PreprocessingConfig.from_yaml(args.config)
    config = _apply_overrides(config, args)

    main(config)
