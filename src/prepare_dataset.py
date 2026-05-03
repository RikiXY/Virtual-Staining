"""
Integrated preprocessing pipeline for paired histopathology samples.

This script loads two paired full-size images (a source image and a target
image), computes tissue masks, aligns the target image to the source reference,
extracts paired patches, and creates the `dataset_train`, `dataset_val`, and
`dataset_test` splits.
"""

import argparse
import csv
import random
from pathlib import Path

import cv2

from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    calculate_mask_with_multiple_parameters,
    align_from_scaled,
    divide_image_with_grid,
    divide_image_with_positions,
    is_valid_patch_pair,
    split_items,
    ensure_clean_directory,
    validate_image_filename,
)
from virtual_staining.data.results import DatasetBuildResult


def main(config: PreprocessingConfig) -> DatasetBuildResult:
    seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)
    print(f"Seed set to {seed}")

    print(f"Loading images from {config.dataset_root}")
    if not config.dataset_root.exists():
        raise FileNotFoundError(f"The path {config.dataset_root} does not exist.")
    source_file = validate_image_filename(config.source_name, "Source")
    target_file = validate_image_filename(config.target_name, "Target")

    source_stem = source_file.stem
    target_stem = target_file.stem
    source_suffix = source_file.suffix.lower()
    target_suffix = target_file.suffix.lower()

    source_image = cv2.imread(config.dataset_root / source_file.name)
    target_image = cv2.imread(config.dataset_root / target_file.name)
    if source_image is None or target_image is None:
        raise FileNotFoundError(
            f"Missing paired files. Expected '{config.source_name}' and "
            f"'{config.target_name}' inside: {config.dataset_root}"
        )
    print(f"Images loaded. Sizes: source={source_image.shape}, target={target_image.shape}")

    print("Calculating masks. This will take some time...")
    source_mask = calculate_mask_with_multiple_parameters(source_image, MASK_PARAMETER_GRID)
    target_mask = calculate_mask_with_multiple_parameters(target_image, MASK_PARAMETER_GRID)
    print("Masks calculated")
    if config.save_masks:
        cv2.imwrite(config.dataset_root / f"mask_{source_stem}{source_suffix}", source_mask)
        cv2.imwrite(config.dataset_root / f"mask_{target_stem}{target_suffix}", target_mask)
    print("Masks saved")

    print("Aligning images. This will also take some time...")
    aligned_target, aligned_target_mask, warp_matrix = align_from_scaled(
        source_image,
        target_image,
        mask1=source_mask,
        mask2=target_mask,
        scale=0.5,
    )
    print("Images aligned")
    cv2.imwrite(config.dataset_root / f"aligned_{target_stem}{target_suffix}", aligned_target)
    cv2.imwrite(
        config.dataset_root / f"aligned_mask_{target_stem}{target_suffix}",
        aligned_target_mask,
    )
    print("Aligned images saved")

    print("Extracting sub-images")
    m = config.margin
    source_images, source_masks, positions = divide_image_with_grid(
        source_image[m:-m, m:-m],
        config.image_size,
        config.grid_movement,
        source_mask[m:-m, m:-m],
    )
    target_images = divide_image_with_positions(
        aligned_target[m:-m, m:-m],
        config.image_size,
        positions,
    )
    target_masks = divide_image_with_positions(
        aligned_target_mask[m:-m, m:-m],
        config.image_size,
        positions,
    )
    print(f"Total pairs extracted: {len(source_images)}")

    named_source_images = []
    named_target_images = []
    discarded_source_images = []
    discarded_target_images = []
    discarded_log_rows = []

    for (x, y), source_img, source_patch_mask, target_img, target_patch_mask in zip(
        positions,
        source_images,
        source_masks,
        target_images,
        target_masks,
    ):
        patch_source_name = f"{x:05}_{y:05}_source{source_suffix}"
        patch_target_name = f"{x:05}_{y:05}_target{target_suffix}"

        is_valid, debug_info = is_valid_patch_pair(
            source_img=source_img,
            target_img=target_img,
            source_mask=source_patch_mask,
            target_mask=target_patch_mask,
            min_foreground_ratio=config.min_foreground_ratio,
            max_white_ratio=config.max_white_ratio,
            white_threshold=config.white_threshold,
            max_largest_white_component_ratio=config.max_largest_white_component_ratio,
        )

        if is_valid:
            named_source_images.append((source_img, patch_source_name))
            named_target_images.append((target_img, patch_target_name))
        else:
            discarded_source_images.append((source_img, patch_source_name))
            discarded_target_images.append((target_img, patch_target_name))

            discarded_log_rows.append(
                {
                    "sample_id": f"{x:05}_{y:05}",
                    "source_name": patch_source_name,
                    "target_name": patch_target_name,
                    "source_foreground_ratio": debug_info["source_foreground_ratio"],
                    "target_foreground_ratio": debug_info["target_foreground_ratio"],
                    "source_white_ratio": debug_info["source_white_ratio"],
                    "target_white_ratio": debug_info["target_white_ratio"],
                    "source_largest_white_component_ratio": debug_info["source_largest_white_component_ratio"],
                    "target_largest_white_component_ratio": debug_info["target_largest_white_component_ratio"],
                    "reasons": ";".join(debug_info["reasons"]),
                }
            )

    print("Pairs renamed")

    discarded_root = config.dataset_root / "discarded_patches"
    discarded_source_dir = discarded_root / "source"
    discarded_target_dir = discarded_root / "target"

    ensure_clean_directory(config.dataset_root / "dataset_train")
    ensure_clean_directory(config.dataset_root / "dataset_val")
    ensure_clean_directory(config.dataset_root / "dataset_test")
    ensure_clean_directory(discarded_source_dir)
    ensure_clean_directory(discarded_target_dir)

    for source_img, patch_source_name in discarded_source_images:
        cv2.imwrite(discarded_source_dir / patch_source_name, source_img)
    for target_img, patch_target_name in discarded_target_images:
        cv2.imwrite(discarded_target_dir / patch_target_name, target_img)

    discarded_log_path = discarded_root / "discarded_log.csv"
    with open(discarded_log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source_name",
                "target_name",
                "source_foreground_ratio",
                "target_foreground_ratio",
                "source_white_ratio",
                "target_white_ratio",
                "reasons",
                "source_largest_white_component_ratio",
                "target_largest_white_component_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(discarded_log_rows)

    print("Dividing dataset into training, validation, and test")
    images = list(zip(named_source_images, named_target_images))
    split = split_items(images, [config.train_ratio, config.val_ratio, config.test_ratio])
    print(
        f"Number of pairs for training: {len(split[0])}, "
        f"validation: {len(split[1])}, test: {len(split[2])}"
    )

    for i, subset in enumerate(split):
        subset_name = ["dataset_train", "dataset_val", "dataset_test"][i]
        subset_dir = config.dataset_root / subset_name
        for source_img, target_img in subset:
            cv2.imwrite(subset_dir / source_img[1], source_img[0])
            cv2.imwrite(subset_dir / target_img[1], target_img[0])

    print("Dataset saved")
    return DatasetBuildResult(
        train_count=len(split[0]),
        val_count=len(split[1]),
        test_count=len(split[2]),
        skipped_count=len(discarded_source_images),
        output_root=config.dataset_root,
    )


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
