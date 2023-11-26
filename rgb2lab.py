import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color


def verify(input_rgb_image_path, output_lab_image_path):
    print("Loading input RGB image...")
    input_img = Image.open(input_rgb_image_path).convert("RGBA")
    pixels = np.array(input_img, dtype=np.float32)

    pixels[:, :, :3] /= 255

    print("Running rgb2lab()...")
    pixels[:, :, :3] = color.rgb2lab(pixels[:, :, :3])

    pixels[:, :, 0] *= 2.55
    pixels[:, :, 1] += 128
    pixels[:, :, 2] += 128

    # Assert that all values lie between 0 and 255
    # pixels[0, 0] = [256, 2, 3, 255]
    assert np.all((pixels >= 0) & (pixels <= 255))

    pixels = np.round(pixels).astype(np.uint8)

    Image.fromarray(pixels).save(output_lab_image_path)

    print("Done!")


def add_parser_arguments(parser):
    parser.add_argument(
        "input_rgb_image_path",
        type=Path,
        help="The path to the RGB input image",
    )
    parser.add_argument(
        "output_lab_image_path",
        type=Path,
        help="The path to the LAB output image",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    verify(args.input_rgb_image_path, args.output_lab_image_path)


if __name__ == "__main__":
    main()
