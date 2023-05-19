# Source:
# https://gist.github.com/pointhi/801a3c19498728801e909b623c1e15a1

# This script is a simple test to do image operations on pyopencl in combination with PIL
#
# based on the code of: https://gist.github.com/likr/3735779

import os
from pathlib import Path

import numpy
import pyopencl as cl
from PIL import Image

filename = "elephant.png"
# filename = "blue.png"

# initialize OpenCL
os.environ["PYOPENCL_CTX"] = "0"
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# load and build OpenCL function
prg = cl.Program(ctx, Path("checkerboard.cl").read_text()).build()

# load and convert source image
src_img = Image.open(f"input/{filename}").convert(
    "RGBA"
)  # This example code only works with RGBA images
src = numpy.array(src_img)

# get size of source image (note height is stored at index 0)
h = src.shape[0]
w = src.shape[1]

# build a 2D OpenCL Image from the numpy array
src_buf = cl.image_from_array(ctx, src, 4)

# build destination OpenCL Image
fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

# execute OpenCL function
prg.convert(queue, (w, h), None, src_buf, dest_buf)

# copy result back to host
dest = numpy.empty_like(src)
cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))

# convert image and save it
dest_img = Image.fromarray(dest)
dest_img.save(f"output/{filename}", "PNG")
