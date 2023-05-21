import math

from PIL import Image


def main():
    pixel_count = 256**3
    side_length = math.isqrt(pixel_count)

    img = Image.new("RGB", (side_length, side_length), "black")
    pixels = img.load()

    for b in range(256):
        for g in range(256):
            for r in range(256):
                col = r + ((b % 16) * 256)
                row = g + ((b // 16) * 256)

                pixels[col, row] = (r, g, b)

    img.save("input/all_colors.png")


if __name__ == "__main__":
    main()
