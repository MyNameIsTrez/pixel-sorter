# Source:
# https://github.com/inducer/pyopencl/blob/main/examples/median-filter.py

import os
from pathlib import Path

import numpy as np
import pyopencl as cl
from imageio import imread, imsave

# Read in image
img = imread("input/median_filter_input.jpg").astype(np.float32)
print(img.shape)

img = np.mean(img, axis=2)
print(img.shape)

os.environ["PYOPENCL_CTX"] = "0"
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

code = Path("median_filter.cl").read_text()

# Kernel function instantiation
prg = cl.Program(ctx, code).build()
# Allocate memory for variables on the device
img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
width_g = cl.Buffer(
    ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1])
)
height_g = cl.Buffer(
    ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0])
)
# Call Kernel. Automatically takes care of block/grid distribution
prg.medianFilter(queue, img.shape, None, img_g, result_g, width_g, height_g)
result = np.empty_like(img)
cl.enqueue_copy(queue, result, result_g)

# Show the blurred image
imsave("output/median_filter_output.jpg", result, mode="RGB")
