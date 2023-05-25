import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def shuffle(input_image_path, output_image_path):
    input = Image.open(input_image_path).convert("RGB")

    arr = np.reshape(input.getdata(), (input.width * input.height, 3))

    np.random.shuffle(arr)

    arr = np.reshape(arr, (input.height, input.width, 3))

    output = Image.fromarray(arr.astype("uint8"))

    output.save(output_image_path)


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
