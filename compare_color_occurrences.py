import numpy as np
from PIL import Image

# filename = "all_colors.png"
filename = "big_palette.png"
# filename = "elephant.png"
# filename = "grid.png"
# filename = "palette.png"
# filename = "small.png"
# filename = "tiny.png"


def get_color_counts(filepath):
    img = Image.open(filepath).convert("RGB")

    arr = np.array(img)

    # Arrange all pixels into a tall column of 3 RGB values and find unique rows (colors)
    _, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=1)
    return counts


def main():
    input_counts = get_color_counts(f"input/{filename}")
    output_counts = get_color_counts(f"output/{filename}")

    counts_equal = np.array_equal(input_counts, output_counts)

    assert (
        counts_equal
    ), "❌ The color counts of the input and the output weren't identical!"

    print("🎉 Images have identical color occurrences!")


if __name__ == "__main__":
    main()
