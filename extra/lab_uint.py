# Verifies all_colors.png unsigned integer translation loses no information

import numpy as np
from PIL import Image
from skimage import color


def main():
    # "The L* values range from 0 to 100; the a* and b* values range from -128 to 127."
    # https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.lab2rgb
    signed_to_unsigned = 128

    # Unsigned integer precision loss compensation multiplier
    # 82 is the lowest integer value that works
    precision_compensation = 82

    print("Loading input RGB image")
    input_img = Image.open("input/all_colors.png").convert("RGBA")
    pixels = np.array(input_img, dtype=np.float32)

    print("/= 255")
    # I do [:, :, :3] everywhere so only RGB is affected, and not a potential A
    pixels[:, :, :3] /= 255

    print("Running rgb2lab()")
    pixels[:, :, :3] = color.rgb2lab(pixels[:, :, :3])

    print("+= signed_to_unsigned")
    pixels[:, :, :3] += signed_to_unsigned

    print("*= precision_compensation")
    pixels[:, :, :3] *= precision_compensation

    print("Rounding to uint32")
    pixels = np.round(pixels).astype(np.uint16)

    print("pixels.astype(np.float32)")
    pixels = pixels.astype(np.float32)

    print("/= precision_compensation")
    pixels[:, :, :3] /= precision_compensation

    print("-= signed_to_unsigned")
    pixels[:, :, :3] -= signed_to_unsigned

    print("pixels.copy()")
    new_pixels = pixels.copy()

    print("Running lab2rgb()")
    new_pixels[:, :, :3] = color.lab2rgb(pixels[:, :, :3])

    print("*= 255")
    new_pixels[:, :, :3] *= 255

    print("Rounding to uint8")
    new_pixels = np.round(new_pixels).astype(np.uint8)

    assert (
        np.array(input_img, dtype=np.uint8) == new_pixels
    ).all(), "❌ There was loss of information during the translation!"

    print("🎉 There was no loss of information during the translation!")


if __name__ == "__main__":
    main()
