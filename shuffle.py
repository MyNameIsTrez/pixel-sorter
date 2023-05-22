import os
import time
from pathlib import Path

import numpy as np
import pyopencl as cl
from PIL import Image

# filename = "all_colors.png"
# filename = "elephant.png"
# filename = "grid.png"
filename = "palette.png"
# filename = "small.png"
# filename = "tiny.png"


def main():
    start_time = time.time()

    # Initialize OpenCL
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # TODO: Try to find useful optimization flags
    # Load and build OpenCL function
    prg = cl.Program(ctx, Path("shuffle.cl").read_text()).build()

    # Load and convert source image
    # This example code only works with RGBA images
    src_img = Image.open(f"input/{filename}").convert("RGBA")
    src = np.array(src_img)

    # Get size of source image (note height is stored at index 0)
    h = src.shape[0]
    w = src.shape[1]
    # print(f"width: {w}, height: {h}")

    # Build a 2D OpenCL Image from the numpy array
    src_buf = cl.image_from_array(ctx, src, 4)

    # Build destination OpenCL Image
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

    # TODO: Try to make this WRITE_ONLY again for optimization purposes?
    dest_buf = cl.Image(ctx, cl.mem_flags.READ_WRITE, fmt, shape=(w, h))

    cl.enqueue_copy(queue, dest_buf, src, origin=(0, 0), region=(w, h))

    assert w % 2 == 0, "This program doesn't support images with an odd width"
    thread_dimensions = (int(w / 2) * h, 1)

    rand1 = np.uint32(42424242)
    rand2 = np.uint32(69696969)

    # Execute OpenCL function
    prg.shuffle_(queue, thread_dimensions, None, src_buf, dest_buf, rand1, rand2)

    # Copy result back to host
    dest = np.empty_like(src)
    cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))

    # Convert image and save it
    dest_img = Image.fromarray(dest)
    dest_img.save(f"output/{filename}")

    print(f"Program took {time.time() - start_time:.0f} seconds")


if __name__ == "__main__":
    main()
