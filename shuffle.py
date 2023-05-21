import os
from pathlib import Path

import numpy
import pyopencl as cl
from PIL import Image

# filename = "all_colors.png"
# filename = "elephant.png"
filename = "small.png"
# filename = "tiny.png"


def main():
    # Initialize OpenCL
    os.environ["PYOPENCL_CTX"] = "0"
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Load and build OpenCL function
    prg = cl.Program(ctx, Path("shuffle.cl").read_text()).build()

    # Load and convert source image
    # This example code only works with RGBA images
    src_img = Image.open(f"input/{filename}").convert("RGBA")
    src = numpy.array(src_img)

    # Get size of source image (note height is stored at index 0)
    h = src.shape[0]
    w = src.shape[1]
    # print(f"width: {w}, height: {h}")

    # Build a 2D OpenCL Image from the numpy array
    src_buf = cl.image_from_array(ctx, src, 4)

    # Build destination OpenCL Image
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))
    cl.enqueue_copy(queue, dest_buf, src, origin=(0, 0), region=(w, h))

    # Execute OpenCL function
    assert w % 2 == 0, "This program currently doesn't support images with an odd width"
    prg.shuffle_(queue, (int(w / 2) * h, 1), None, src_buf, dest_buf)

    # Copy result back to host
    dest = numpy.empty_like(src)
    cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))

    # Convert image and save it
    dest_img = Image.fromarray(dest)
    dest_img.save(f"output/{filename}")


if __name__ == "__main__":
    main()
