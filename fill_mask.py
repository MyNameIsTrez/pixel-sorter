import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def sanity_check_inputs(input_arr, input_mask_arr):
    input_arr_pixel_count = input_arr.shape[0]
    # Filter away black pixels
    input_mask_arr_pixel_count = len(input_mask_arr[input_mask_arr[:, :, 0] != 0])

    assert (
        (input_mask_arr == (0, 0, 0, 0)) | (input_mask_arr == (255, 255, 255, 255))
    ).all(), "❌ The input mask has a pixel that isn't black nor white!"

    assert (
        input_arr_pixel_count == input_mask_arr_pixel_count
    ), "❌ The color counts of the input image and input mask image aren't identical!"


def fill_mask(input_image_path, input_mask_image_path, output_filled_mask_image_path):
    input_img = Image.open(input_image_path).convert("RGBA")
    input_arr = np.array(input_img)
    # Filter away pixels with an alpha of 0
    input_arr = input_arr[input_arr[:, :, 3] != 0]

    input_mask_img = Image.open(input_mask_image_path).convert("RGBA")
    input_mask_arr = np.array(input_mask_img)

    sanity_check_inputs(input_arr, input_mask_arr)

    output_arr = np.zeros(
        (input_mask_img.width, input_mask_img.height, 4), dtype=np.uint8
    )

    offset = 0
    pixel_index = 0
    width = input_mask_arr.shape[1]
    for row in input_mask_arr:
        for pixel in row:
            if pixel[0] == 255:
                index = offset
                x = index % width
                y = int(index / width)
                pixel = input_arr[pixel_index]
                pixel_index += 1
                output_arr[y, x] = pixel
            offset += 1

    output_img = Image.fromarray(output_arr)
    output_img.save(output_filled_mask_image_path)


def add_parser_arguments(parser):
    parser.add_argument(
        "input_image_path",
        type=Path,
        help="Path to the input image",
    )
    parser.add_argument(
        "input_mask_image_path",
        type=Path,
        help="Path to the input mask image",
    )
    parser.add_argument(
        "output_filled_mask_image_path",
        type=Path,
        help="Path to the output filled mask image",
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    fill_mask(
        args.input_image_path,
        args.input_mask_image_path,
        args.output_filled_mask_image_path,
    )


if __name__ == "__main__":
    main()
