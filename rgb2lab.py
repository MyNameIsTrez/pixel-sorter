import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color


def verify(input_rgb_image_path, output_lab_npy_path):
    print("Loading input RGB image...")
    input_img = Image.open(input_rgb_image_path).convert("RGBA")
    pixels = np.array(input_img, dtype=np.float32)

    pixels[:, :, :3] /= 255

    print("Running rgb2lab()...")
    pixels[:, :, :3] = color.rgb2lab(pixels[:, :, :3])

    np.save(output_lab_npy_path, pixels)

    print("Done!")


def add_parser_arguments(parser):
    parser.add_argument(
        "input_rgb_image_path",
        type=Path,
        help="Path to the RGB input image",
    )
    parser.add_argument(
        "output_lab_npy_path",
        type=Path,
        help="Path to the LAB output npy",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    verify(args.input_rgb_image_path, args.output_lab_npy_path)


if __name__ == "__main__":
    main()
