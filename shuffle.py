import os
import time
from pathlib import Path

import numpy as np
import pyopencl as cl
from PIL import Image

# TODO: Let CLI ask for input and output filepaths instead
# filename = "all_colors.png"
filename = "big_palette.png"
# filename = "elephant.png"
# filename = "grid.png"
# filename = "palette.png"
# filename = "small.png"
# filename = "tiny.png"

# TODO: REMOVE THESE FROM HERE
ITERATIONS_IN_KERNEL_PER_CALL = 1

SECONDS_BETWEEN_STATUS_UPDATES = 10


def print_status(python_iteration, start_time):
    print(
        f"Iteration {(python_iteration + 1) * ITERATIONS_IN_KERNEL_PER_CALL:.0f} ({python_iteration + 1:.0f} * {ITERATIONS_IN_KERNEL_PER_CALL:.0f}) at {time.time() - start_time:.1f} seconds"
    )


def save_result(src, queue, dest_buf, w, h):
    # Copy result back to host
    dest = np.empty_like(src)
    cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))

    # Convert image and save it
    dest_img = Image.fromarray(dest)
    dest_img.save(f"output/{filename}")


def main():
    start_time = time.time()

    # Initialize OpenCL
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    opencl_code = Path("shuffle.cl").read_text()

    # You can set ITERATIONS_IN_KERNEL_PER_CALL higher than 1 in some cases
    # which can speed up the program by 3 times,
    # but make sure to run compare_color_occurrences.py
    # if you do set it higher, since it can mess some images up!
    defines = {
        "ITERATIONS_IN_KERNEL_PER_CALL": "1",
        "KERNEL_RADIUS": "10",
        "MODE": "LCG",
    }

    defines_str = "\n".join(
        (
            f"#define {define_name} {define_value}"
            for define_name, define_value in defines.items()
        )
    )

    opencl_code = defines_str + "\n\n" + opencl_code

    # TODO: Try to find useful optimization flags
    # Load and build OpenCL function
    prg = cl.Program(ctx, opencl_code).build(
        options="-DMAKE_VSCODE_HIGHLIGHTER_HAPPY=1"
    )

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

    # TODO: Having global_size be super large may be bad?
    # TODO: Should local_size be bigger than 1?
    # thread_dimensions = (16, 1)
    thread_dimensions = (int(w / 2) * h, 1)
    # print(int(w / 2) * h)
    # thread_dimensions = (4, 1)

    rand1 = np.uint32(42424242)
    rand2 = np.uint32(69696969)

    python_iteration = 0

    last_printed_time = 0

    opencl_shuffle = prg.shuffle_

    # TODO: Fix wrong elephant color count with ITERATIONS_IN_KERNEL_PER_CALL 2
    # opencl_shuffle(queue, thread_dimensions, None, src_buf, dest_buf, rand1, rand2)
    # save_result(src, queue, dest_buf, w, h)
    # print_status(python_iteration, start_time)

    try:
        while True:
            python_iteration += 1

            if time.time() > last_printed_time + SECONDS_BETWEEN_STATUS_UPDATES:
                save_result(src, queue, dest_buf, w, h)

                print_status(python_iteration, start_time)

                last_printed_time = time.time()

            # TODO: Add wraparound code
            rand1 = np.uint32(rand1 + 1)

            # The .wait() is crucial!
            # The reason being that the OpenCL kernel call is async,
            # so without it you end up being unable to use Ctrl+C
            # to stop the program!
            opencl_shuffle(
                queue, thread_dimensions, None, src_buf, dest_buf, rand1, rand2
            ).wait()
    except KeyboardInterrupt:
        save_result(src, queue, dest_buf, w, h)

        # print(f"Program took {time.time() - start_time:.1f} seconds")
        print_status(python_iteration, start_time)


if __name__ == "__main__":
    main()
