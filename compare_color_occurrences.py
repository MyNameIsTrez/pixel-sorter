import numpy as np
from PIL import Image

filename = "elephant.png"
# filename = "small.png"
# filename = "tiny.png"


def get_colors_and_counts(filepath):
    img = Image.open(filepath).convert("RGB")

    arr = np.array(img)

    # Arrange all pixels into a tall column of 3 RGB values and find unique rows (colors)
    colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=1)
    return colors, counts


def main():
    input_colors, input_counts = get_colors_and_counts(f"input/{filename}")
    output_colors, output_counts = get_colors_and_counts(f"output/{filename}")

    colors_equal = np.array_equal(input_colors, output_colors)
    input_counts_equal = np.array_equal(input_counts, output_counts)
    print(colors_equal)
    print(input_counts_equal)


if __name__ == "__main__":
    main()
