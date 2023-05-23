import math

from skimage import color


def main():
    min_l = math.inf
    min_a = math.inf
    min_b = math.inf

    max_l = -math.inf
    max_a = -math.inf
    max_b = -math.inf

    for r in range(256):
        print(f"\rCalculating LAB min, max and range. Progress: {r}/255")

        for g in range(256):
            for b in range(256):
                rgb = (r / 255, g / 255, b / 255)
                lab = color.rgb2lab(rgb)

                min_l = min(min_l, lab[0])
                min_a = min(min_a, lab[1])
                min_b = min(min_b, lab[2])

                max_l = max(max_l, lab[0])
                max_a = max(max_a, lab[1])
                max_b = max(max_b, lab[2])

    print("\n")

    range_l = max_l - min_l
    range_a = max_a - min_a
    range_b = max_b - min_b

    print("real ranges:")
    print(f"range_l: {range_l}, range_a: {range_a}, range_b: {range_b}")


if __name__ == "__main__":
    main()
