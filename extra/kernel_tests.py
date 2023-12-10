import numpy as np
from scipy import ndimage, signal

pixels = np.array(
    [
        [[1.0, 2.0], [1.0, 2.0]],
        [[1.0, 2.0], [1.0, 2.0]],
    ]
)

print("pixels:")
print(pixels)

kernel = np.array(
    [
        [[0.5, 0.5], [1.0, 1.0], [0.5, 0.5]],
        [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        [[0.5, 0.5], [1.0, 1.0], [0.5, 0.5]],
    ]
)

# kernel = np.array(
#     [
#         [[0.5], [1.0], [0.5]],
#         [[1.0], [0.0], [1.0]],
#         [[0.5], [1.0], [0.5]],
#     ]
# )

print("kernel:")
print(kernel)

expected = np.array(
    [
        [[2.5, 5.0], [2.5, 5.0]],
        [[2.5, 5.0], [2.5, 5.0]],
    ]
)

print("expected:")
print(expected)

print("actual:")

# These results are wrong
# print(ndimage.correlate(pixels, kernel, mode="constant"))
# print(ndimage.convolve(pixels, kernel, mode="constant"))

# These results are correct
# print(ndimage.correlate(pixels, kernel[:, :, :1], mode="constant"))
# print(ndimage.convolve(pixels, kernel[:, :, :1], mode="constant"))
print(signal.convolve(pixels, kernel[:, :, :1], mode="same"))
# fftconvolve reportedly can be faster on extremely huge arrays, but can also cause out-of-memory:
# https://stackoverflow.com/a/15217324/13279557
# print(signal.fftconvolve(pixels, kernel[:, :, :1], mode="same"))

# pixels isn't modified by the signal.convolve()
print("pixels:")
print(pixels)
