import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def shuffle(input_image_path, output_image_path):
    input_img = Image.open(input_image_path).convert("RGBA")
    input_arr = np.array(input_img)
    original_input_arr = input_arr.copy()

    # Filter away pixels with an alpha of 0
    input_arr = input_arr[input_arr[:, :, 3] != 0]

    np.random.shuffle(input_arr)

    output_arr = np.zeros((input_img.height, input_img.width, 4), dtype=np.uint8)

    offset = 0
    pixel_index = 0
    width = original_input_arr.shape[1]
    for row in original_input_arr:
        for pixel in row:
            if pixel[3] != 0:
                index = offset
                x = index % width
                y = int(index / width)
                pixel = input_arr[pixel_index]
                pixel_index += 1
                output_arr[y, x] = pixel
            offset += 1

    output_img = Image.fromarray(output_arr)
    output_img.save(output_image_path)


def add_parser_arguments(parser):
    parser.add_argument(
        "input_image_path",
        type=Path,
        help="The path to the input image",
    )
    parser.add_argument(
        "output_image_path",
        type=Path,
        help="The path to the output image",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    shuffle(args.input_image_path, args.output_image_path)


if __name__ == "__main__":
    main()
