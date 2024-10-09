import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color


def verify(input_lab_npy_path, output_rgb_image_path):
    # "The L* values range from 0 to 100; the a* and b* values range from -128 to 127."
    # https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.lab2rgb
    signed_to_unsigned = 128

    # Unsigned integer precision loss compensation multiplier
    # 82 is the lowest integer value that works
    precision_compensation = 82

    print("Loading input LAB image")
    pixels = np.load(input_lab_npy_path)

    print(f"Setting LAB values to a temporary value that have an alpha of 0")
    t = signed_to_unsigned * precision_compensation
    pixels[pixels[:, :, 3] == 0] = [t, t, t, 0]

    print("pixels.astype(np.float32)")
    pixels = pixels.astype(np.float32)

    print("/= precision_compensation")
    pixels[:, :, :3] /= precision_compensation

    print("-= signed_to_unsigned")
    pixels[:, :, :3] -= signed_to_unsigned

    print("Running lab2rgb()")
    pixels[:, :, :3] = color.lab2rgb(pixels[:, :, :3])

    print("*= 255")
    pixels[:, :, :3] *= 255

    print("Rounding to uint8")
    pixels = np.round(pixels).astype(np.uint8)

    print("Saving output RGB image")
    Image.fromarray(pixels).save(output_rgb_image_path)

    print("Done! 🎉")


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
