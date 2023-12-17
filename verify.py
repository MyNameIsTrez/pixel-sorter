import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _get_colors_and_counts(filepath):
    img = Image.open(filepath).convert("RGBA")

    arr = np.array(img)

    # Set the RGB values of pixels with an alpha of 0 all to 0
    arr[arr[:, :, 3] == 0] = 0

    # Arrange all pixels into a tall column of 4 RGBA values and find unique rows (colors)
    colors, counts = np.unique(arr.reshape(-1, 4), axis=0, return_counts=1)

    return colors, counts


def verify(input_image_path, output_image_path):
    print("Verifying...")

    input_colors, input_counts = _get_colors_and_counts(input_image_path)
    output_colors, output_counts = _get_colors_and_counts(output_image_path)

    colors_equal = np.array_equal(input_colors, output_colors)
    counts_equal = np.array_equal(input_counts, output_counts)

    assert colors_equal, "❌ The set of colors of the input and output aren't identical!"

    assert counts_equal, "❌ The color counts of the input and output aren't identical!"

    print("🎉 The input and output have identical color counts!")


def add_parser_arguments(parser):
    parser.add_argument(
        "input_image_path",
        type=Path,
        help="Path to the input image",
    )
    parser.add_argument(
        "output_image_path",
        type=Path,
        help="Path to the output image",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    verify(args.input_image_path, args.output_image_path)


if __name__ == "__main__":
    main()
