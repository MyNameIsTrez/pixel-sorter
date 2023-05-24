import argparse
import os
import time
from pathlib import Path

import humanize
import numpy as np
import pyopencl as cl
from PIL import Image
from scipy import ndimage
from skimage import color

import count_colors


def unpack_rgb_from_pixels(pixels):
    lab = pixels[:, :, :3]

    rgb = color.lab2rgb(lab)

    rgb *= 255

    pixels[:, :, :3] = rgb

    return pixels


def pack_lab_into_pixels(pixels):
    rgb = pixels[:, :, :3]

    rgb /= 255

    lab = color.rgb2lab(rgb)

    pixels[:, :, :3] = lab

    return pixels


def print_status(
    saved_results,
    python_iteration,
    iterations_in_kernel_per_call,
    start_time,
    thread_count,
):
    iteration = (python_iteration + 1) * iterations_in_kernel_per_call

    print(
        f"Frame {saved_results}, {time.time() - start_time:.1f} seconds, iteration {iteration:.0f} ({python_iteration + 1:.0f} * {iterations_in_kernel_per_call:.0f}), {humanize.intword(iteration * thread_count)} attempted swaps"
    )


def save_result(
    pixels,
    queue,
    pixels_buf,
    width,
    height,
    output_image_path,
    no_overwriting_output,
    saved_results,
    saved_image_leading_zero_count,
    color_comparison,
):
    # Copy result back to host
    saved = np.empty_like(pixels)
    cl.enqueue_copy(queue, saved, pixels_buf, origin=(0, 0), region=(width, height))

    if color_comparison == "LAB":
        saved = unpack_rgb_from_pixels(saved)

    saved = np.round(saved).astype(np.uint8)

    # Convert the array to an image
    saved_img = Image.fromarray(saved)

    saved_results += 1

    # Save the image with/without overwriting the old image
    if no_overwriting_output:
        saved_img.save(
            f"{output_image_path.with_suffix('')}_{saved_results:0{saved_image_leading_zero_count}d}{output_image_path.suffix}"
        )
    else:
        saved_img.save(output_image_path)

    return saved_results


def initialize_neighbor_totals_buf(
    queue, neighbor_totals_buf, pixels, width, height, kernel_radius
):
    kernel_diameter = kernel_radius * 2 + 1

    kernel = np.ones((kernel_diameter, kernel_diameter, 4))

    # mode=constant: The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter (which is 0 by default).
    # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html#scipy.ndimage.convolve
    neighbor_totals = ndimage.convolve(pixels, kernel, mode="constant")

    cl.enqueue_copy(
        queue,
        neighbor_totals_buf,
        neighbor_totals,
        origin=(0, 0),
        region=(width, height),
    )


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
        help="Setting this higher than 1 can give a massive speedup, but the end of the program may tell you it messed up the output image!",
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
        help="The radius of neighbors that get compared against the current pixel's color; a higher radius means more blur",
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
        help="Save all output images, instead of the default behavior of overwriting; this turns off count_colors() being ran at the end",
    )
    parser.add_argument(
        "-z",
        "--saved-image-leading-zero-count",
        type=int,
        default=0,
        help="The number of leading zeros on saved images; this has no effect if the -n switch isn't passed!",
    )
    parser.add_argument(
        "-c",
        "--color-comparison",
        type=str,
        default="LAB",
        help="The color space in which pixels are compared: LAB mimics how the human eye percieves color, while RGB is easier to implement",
    )


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    if args.shuffle_mode == "PHILOX":
        raise Exception(
            "PHILOX is broken, creating shuffle collisions, so use LCG for now"
        )

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
    pixels_img = Image.open(args.input_image_path).convert("RGBA")
    pixels = np.array(pixels_img, dtype=np.float32)

    # Get size of source image
    height = pixels.shape[0]
    width = pixels.shape[1]
    # print(f"width: {width}, height: {height}")

    # TODO: Remove?
    if args.color_comparison == "LAB":
        pixels = pack_lab_into_pixels(pixels)

    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    pixels_buf = cl.Image(ctx, cl.mem_flags.READ_WRITE, fmt, shape=(width, height))
    cl.enqueue_copy(queue, pixels_buf, pixels, origin=(0, 0), region=(width, height))

    # TODO: The fmt channel_type FLOAT might be lossy with large enough kernels!
    # Make sure to try a huge kernel on big_palette.png, and let count_colors.py do its thing!
    neighbor_totals_buf = cl.Image(
        ctx, cl.mem_flags.READ_WRITE, fmt, shape=(width, height)
    )
    initialize_neighbor_totals_buf(
        queue, neighbor_totals_buf, pixels, width, height, args.kernel_radius
    )

    assert width % 2 == 0, "This program doesn't support images with an odd width"

    thread_count = int(width / 2) * height
    thread_dimensions = (thread_count, 1)

    # TODO: What does setting global_local_work_sizes to None do?
    # TODO: Do I want to customize it?
    global_local_work_sizes = None

    rand1 = np.uint32(42424242)
    rand2 = np.uint32(69696969)

    python_iteration = 0

    saved_results = 0

    last_printed_time = time.time()

    opencl_sort = prg.sort

    # TODO: Fix wrong elephant color count with ITERATIONS_IN_KERNEL_PER_CALL 2
    # opencl_sort()
    # save_result()
    # print_status()

    try:
        while True:
            python_iteration += 1

            if time.time() > last_printed_time + args.seconds_between_saves:
                saved_results = save_result(
                    pixels,
                    queue,
                    pixels_buf,
                    width,
                    height,
                    args.output_image_path,
                    args.no_overwriting_output,
                    saved_results,
                    args.saved_image_leading_zero_count,
                    args.color_comparison,
                )

                print_status(
                    saved_results,
                    python_iteration,
                    args.iterations_in_kernel_per_call,
                    start_time,
                    thread_count,
                )

                last_printed_time = time.time()

            # Numpy handles unsigned wraparound for us
            rand1 = np.uint32(rand1 + 1)

            # TODO: Why does removing the wait() suddenly fix tiny.png wrong count issues??

            # The .wait() at the end of this line is crucial!
            # The reason being that the OpenCL kernel call is async,
            # so without it you end up being unable to use Ctrl+C
            # to stop the program!
            opencl_sort(
                queue,
                thread_dimensions,
                global_local_work_sizes,
                pixels_buf,
                neighbor_totals_buf,
                rand1,
                rand2,
            ).wait()

    except KeyboardInterrupt:
        saved_results = save_result(
            pixels,
            queue,
            pixels_buf,
            width,
            height,
            args.output_image_path,
            args.no_overwriting_output,
            saved_results,
            args.saved_image_leading_zero_count,
            args.color_comparison,
        )

        print_status(
            saved_results,
            python_iteration,
            args.iterations_in_kernel_per_call,
            start_time,
            thread_count,
        )

        if not args.no_overwriting_output:
            count_colors.count_colors(args.input_image_path, args.output_image_path)


if __name__ == "__main__":
    main()
