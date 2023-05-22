import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyopencl as cl
from PIL import Image


def print_status(
    saved_results, python_iteration, iterations_in_kernel_per_call, start_time
):
    print(
        f"Frame {saved_results}, iteration {(python_iteration + 1) * iterations_in_kernel_per_call:.0f} ({python_iteration + 1:.0f} * {iterations_in_kernel_per_call:.0f}) at {time.time() - start_time:.1f} seconds"
    )


def save_result(
    src,
    queue,
    dest_buf,
    w,
    h,
    output_image_path,
    no_overwriting_output,
    saved_results,
    saved_image_leading_zero_count,
):
    # Copy result back to host
    dest = np.empty_like(src)
    cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))

    # Convert the array to an image
    dest_img = Image.fromarray(dest)

    saved_results += 1

    # Save the image with/without overwriting the old image
    if no_overwriting_output:
        dest_img.save(
            f"{output_image_path.with_suffix('')}_{saved_results:0{saved_image_leading_zero_count}d}{output_image_path.suffix}"
        )
    else:
        dest_img.save(output_image_path)

    return saved_results


def get_opencl_code(iterations_in_kernel_per_call, kernel_radius, shuffle_mode):
    opencl_code = Path("sort.cl").read_text()

    defines = {
        "ITERATIONS_IN_KERNEL_PER_CALL": iterations_in_kernel_per_call,
        "KERNEL_RADIUS": kernel_radius,
        "SHUFFLE_MODE": shuffle_mode,
    }

    defines_str = "\n".join(
        (
            f"#define {define_name} {define_value}"
            for define_name, define_value in defines.items()
        )
    )

    return defines_str + "\n\n" + opencl_code


def add_parser_arguments(parser):
    parser.add_argument(
        "input_image_path",
        type=Path,
        help="The path to an input image to sort",
    )
    parser.add_argument(
        "output_image_path",
        type=Path,
        help="The path where to save the output image to",
    )
    parser.add_argument(
        "-it",
        "--iterations-in-kernel-per-call",
        type=int,
        default=1,
        help="Setting this higher than 1 can massively speed up the program, but it also messes up on some input images! If you do set it higher, verify the output image with compare_color_occurrences.py",
    )
    parser.add_argument(
        "-s",
        "--seconds-between-saves",
        type=int,
        default=1,
        help="How often the current output image gets saved",
    )
    parser.add_argument(
        "-k",
        "--kernel-radius",
        type=int,
        default=10,
        help="The radius of neighbors that get compared against the current pixel's color",
    )
    parser.add_argument(
        "-m",
        "--shuffle-mode",
        type=str,
        default="LCG",
        help="The shuffle mode: LCG is faster, while PHILOX is higher quality",
    )
    parser.add_argument(
        "-n",
        "--no-overwriting-output",
        action="store_true",
        help="Save all output images, instead of the default behavior of overwriting",
    )
    parser.add_argument(
        "-z",
        "--saved-image-leading-zero-count",
        type=int,
        default=0,
        help="The number of leading zeros on saved images; this has no effect if the -n switch isn't passed!",
    )


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    if args.shuffle_mode == "PHILOX":
        raise Exception("PHILOX creates shuffle collisions, so meanwhile just use LCG")

    # Initialize OpenCL
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # TODO: Try to find useful optimization flags
    # Load and build OpenCL function
    prg = cl.Program(
        ctx,
        get_opencl_code(
            args.iterations_in_kernel_per_call, args.kernel_radius, args.shuffle_mode
        ),
    ).build(options="-DMAKE_VSCODE_HIGHLIGHTER_HAPPY=1")

    # Load and convert source image
    # This example code only works with RGBA images
    src_img = Image.open(args.input_image_path).convert("RGBA")
    src = np.array(src_img)

    # Get size of source image (note height is stored at index 0)
    h = src.shape[0]
    w = src.shape[1]
    # print(f"width: {w}, height: {h}")

    # Build a 2D OpenCL Image from the numpy array
    src_buf = cl.image_from_array(ctx, src, 4)
    # src_buf = cl.Image(ctx, cl.mem_flags.READ_WRITE, fmt, shape=(w, h))
    # cl.enqueue_copy(queue, src_buf, src, origin=(0, 0), region=(w, h))

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

    global_local_work_sizes = None

    rand1 = np.uint32(42424242)
    rand2 = np.uint32(69696969)

    python_iteration = 0

    saved_results = 0

    last_printed_time = 0

    opencl_sort = prg.sort

    # TODO: Fix wrong elephant color count with ITERATIONS_IN_KERNEL_PER_CALL 2
    # opencl_sort(queue, thread_dimensions, None, src_buf, dest_buf, rand1, rand2)
    # save_result()
    # print_status()

    try:
        while True:
            python_iteration += 1

            if time.time() > last_printed_time + args.seconds_between_saves:
                saved_results = save_result(
                    src,
                    queue,
                    dest_buf,
                    w,
                    h,
                    args.output_image_path,
                    args.no_overwriting_output,
                    saved_results,
                    args.saved_image_leading_zero_count,
                )

                print_status(
                    saved_results,
                    python_iteration,
                    args.iterations_in_kernel_per_call,
                    start_time,
                )

                last_printed_time = time.time()

            # Numpy handles unsigned wraparound for us
            rand1 = np.uint32(rand1 + 1)

            # TODO: Why does removing the wait() suddenly fix tiny.png wrong count issues??

            # The .wait() is crucial!
            # The reason being that the OpenCL kernel call is async,
            # so without it you end up being unable to use Ctrl+C
            # to stop the program!
            opencl_sort(
                queue,
                thread_dimensions,
                global_local_work_sizes,
                src_buf,
                dest_buf,
                rand1,
                rand2,
            ).wait()

            # if python_iteration > 1510:
            # if python_iteration > 5000:
            #     return

    except KeyboardInterrupt:
        saved_results = save_result(
            src,
            queue,
            dest_buf,
            w,
            h,
            args.output_image_path,
            args.no_overwriting_output,
            saved_results,
            args.saved_image_leading_zero_count,
        )

        print_status(
            saved_results,
            python_iteration,
            args.iterations_in_kernel_per_call,
            start_time,
        )


if __name__ == "__main__":
    main()
