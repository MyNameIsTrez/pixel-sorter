import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color


def verify(input_lab_npy_path, output_rgb_image_path):
    print("Loading input LAB image...")
    pixels = np.load(input_lab_npy_path)

    print("Running lab2rgb()...")
    pixels[:, :, :3] = color.lab2rgb(pixels[:, :, :3])

    pixels[:, :, :3] *= 255

    pixels = np.round(pixels).astype(np.uint8)

    Image.fromarray(pixels).save(output_rgb_image_path)

    print("Done!")


def add_parser_arguments(parser):
    parser.add_argument(
        "input_lab_npy_path",
        type=Path,
        help="Path to the LAB input npy",
    )
    parser.add_argument(
        "output_rgb_image_path",
        type=Path,
        help="Path to the RGB output image",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    verify(args.input_lab_npy_path, args.output_rgb_image_path)


if __name__ == "__main__":
    main()
