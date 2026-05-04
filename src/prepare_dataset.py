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


def main(config: PreprocessingConfig) -> DatasetBuildResult:
    return DatasetBuilder(config).run_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage=(
            "python src/prepare_dataset.py --path PATH\n"
            "       --source-name SOURCE_NAME --target-name TARGET_NAME\n"
            "       [--seed SEED] [--save-masks] [--image-size WIDTH HEIGHT]\n"
            "       [--grid-movement STEP_X STEP_Y] [--margin MARGIN]\n"
            "       [--min-foreground-ratio F] [--max-white-ratio F]\n"
            "       [--white-threshold N] [--max-largest-white-component-ratio F]"
        ),
        description="Prepares masks, aligns images, extracts subimages, and splits the dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to the folder containing <source_name>.tif and <target_name>.tif"
    )
    parser.add_argument(
        "--source-name",
        type=str,
        required=True,
        help="Source image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--target-name",
        type=str,
        required=True,
        help="Target image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="if set, also saves the subimage masks"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(256, 256),
        help="Patch size used for extraction (default: 256 256)"
    )
    parser.add_argument(
        "--grid-movement",
        type=int,
        nargs=2,
        metavar=("STEP_X", "STEP_Y"),
        default=(256, 256),
        help="Grid step used for patch extraction (default: 256 256)"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=200,
        help="Margin cropped from each border before patch extraction (default: 200)"
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.25,
        help="Minimum foreground tissue ratio for a patch to be kept (default: 0.25)"
    )
    parser.add_argument(
        "--max-white-ratio",
        type=float,
        default=0.7,
        help="Maximum near-white pixel ratio for a patch to be kept (default: 0.7)"
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Grayscale intensity threshold for classifying a pixel as near-white (default: 250)"
    )
    parser.add_argument(
        "--max-largest-white-component-ratio",
        type=float,
        default=0.20,
        help="Maximum ratio of the largest white connected component for a patch to be kept (default: 0.20)"
    )
    args = parser.parse_args()
    main(PreprocessingConfig.from_args(args))
