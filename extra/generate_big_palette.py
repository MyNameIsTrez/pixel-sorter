import math
import numpy as np
from PIL import Image


def main(desired_width, desired_height):
    img = Image.open("input/palette.png")
    img_w, img_h = img.size

    duplicates_width = math.ceil(desired_width / 16)
    duplicates_height = math.ceil(desired_height / 16)

    background = Image.new("RGB", (img_w * duplicates_width, img_h * duplicates_height))

    for duplicate_height in range(duplicates_height):
        for duplicate_width in range(duplicates_width):
            offset = (16 * duplicate_width, 16 * duplicate_height)
            background.paste(img, offset)

    background = background.crop((0, 0, desired_width, desired_height))

    background_px = np.reshape(
        background.getdata(), (background.width * background.height, 3)
    )

    # np.random.shuffle(background_px)

    background_px = np.reshape(background_px, (background.height, background.width, 3))

    res = Image.fromarray(background_px.astype("uint8"))

    res.save("big_palette.png")


if __name__ == "__main__":
    main(1080, 2340)
