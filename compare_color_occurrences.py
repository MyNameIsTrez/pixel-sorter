import numpy as np
from PIL import Image

# filename = "all_colors.png"
filename = "elephant.png"
# filename = "grid.png"
# filename = "palette.png"
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
    counts_equal = np.array_equal(input_counts, output_counts)

    assert (
        colors_equal
    ), "❌ The set of colors in the output isn't identical to the set of colors in the input!"

    assert (
        counts_equal
    ), "❌ The counts of colors in the output isn't identical to the counts of colors in the input!"

    print("🎉 Images have identical color occurrences!")


if __name__ == "__main__":
    main()
