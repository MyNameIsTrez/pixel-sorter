import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color


def verify(input_rgb_image_path, output_lab_npy_path):
    # "The L* values range from 0 to 100; the a* and b* values range from -128 to 127."
    # https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.lab2rgb
    signed_to_unsigned = 128

    # Unsigned integer precision loss compensation multiplier
    # 82 is the lowest integer value that works
    precision_compensation = 82

    print("Loading input RGB image")
    input_img = Image.open(input_rgb_image_path).convert("RGBA")
    pixels = np.array(input_img, dtype=np.float32)

    print("/= 255")
    # I do [:, :, :3] everywhere so only RGB is affected, and not a potential A
    pixels[:, :, :3] /= 255

    print("Running rgb2lab()")
    pixels[:, :, :3] = color.rgb2lab(pixels[:, :, :3])

    print("+= signed_to_unsigned")
    pixels[:, :, :3] += signed_to_unsigned

    print("*= precision_compensation")
    pixels[:, :, :3] *= precision_compensation

    print("Rounding to uint16")
    pixels = np.round(pixels).astype(np.uint16)

    print("Setting LAB values to 0 that have an alpha of 0")
    pixels[pixels[:, :, 3] == 0] = 0

    print("Saving output LAB image")
    np.save(output_lab_npy_path, pixels)

    print("Done! 🎉")


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
