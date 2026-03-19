"""
Create a simple side-by-side comparison image:
[input] [output] [target]

Example:
    python tools/make_comparison_panel.py input.tif output.tif target.tif result.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side comparison image from three input files."
    )
    parser.add_argument("input_image", type=Path, help="Path to the input image.")
    parser.add_argument("output_image", type=Path, help="Path to the output image.")
    parser.add_argument("target_image", type=Path, help="Path to the target image.")
    parser.add_argument("save_path", type=Path, help="Path where the final image will be saved.")
    return parser.parse_args()


def open_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise SystemExit(f"Error: file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Error: not a file: {path}")

    with Image.open(path) as img:
        return img.convert("RGB")


def main() -> None:
    args = parse_args()

    input_img = open_rgb(args.input_image)
    output_img = open_rgb(args.output_image)
    target_img = open_rgb(args.target_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    images = [input_img, output_img, target_img]
    titles = ["INPUT", "OUTPUT", "TARGET"]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison image to: {args.save_path}")


if __name__ == "__main__":
    main()