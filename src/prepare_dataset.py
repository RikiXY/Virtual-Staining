"""
Integrated preprocessing pipeline for paired histopathology samples.

This script loads two paired full-size images (a source image and a target
image), computes tissue masks, aligns the target image to the source reference,
extracts paired patches, and creates the `dataset_train`, `dataset_val`, and
`dataset_test` splits.
"""

import argparse

from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.results import DatasetBuildResult


def main(config: PreprocessingConfig) -> DatasetBuildResult:
    return DatasetBuilder(config).run_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python src/prepare_dataset.py --config CONFIG",
        description="Prepares masks, aligns images, extracts subimages, and splits the dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/runs/example.yaml",
        help="path to the run config YAML (default: config/runs/example.yaml)",
    )
    args = parser.parse_args()

    config = PreprocessingConfig.from_yaml(args.config)

    main(config)
