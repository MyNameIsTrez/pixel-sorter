import argparse
import math
import os
import time
from pathlib import Path

import humanize
import numpy as np
import pyopencl as cl
from PIL import Image
from scipy import signal
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
    prev_python_iteration,
    python_iteration,
    iterations_in_kernel_per_call,
    start_time,
    thread_count,
):
    prev_iteration = (prev_python_iteration + 1) * iterations_in_kernel_per_call
    iteration = (python_iteration + 1) * iterations_in_kernel_per_call

    prev_attempted_swaps = prev_iteration * thread_count
    attempted_swaps = iteration * thread_count

    attempted_swaps_difference = attempted_swaps - prev_attempted_swaps

    print(
        f"Frame {saved_results}"
        f", {humanize.precisedelta(time.time() - start_time)}"
        f", iteration {iteration:.0f}"
        f" ({python_iteration + 1:.0f} * {iterations_in_kernel_per_call:.0f})"
        f", {humanize.intword(attempted_swaps, '%.3f')} attempted swaps"
        f" ({humanize.intword(attempted_swaps_difference, '%+.3f')})"
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
    cl.enqueue_copy(
        queue, saved, pixels_buf, origin=(0, 0), region=(width, height)
    ).wait()

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
    queue, neighbor_totals_buf, pixels, width, height, kernel
):
    print("Running convolve(pixels, kernel)...")

    # [:, :, :3] means only grabbing the R out of RGBA
    # Play around with kernel_tests.py to see how convolve() works.
    # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
    neighbor_totals = signal.convolve(pixels, kernel[:, :, :3], mode="same")

    # In practice seeding neighbor_totals with random values works fine as well,
    # since sort.cl will overwrite the initial values quickly anyways
    # neighbor_totals = np.random.rand(width, height, 4)

    print("Copying neighbor_totals to neighbor_totals_buf...")
    cl.enqueue_copy(
        queue,
        neighbor_totals_buf,
        neighbor_totals,
        origin=(0, 0),
        region=(width, height),
    ).wait()


def get_kernel(kernel_radius):
    kernel_diameter = kernel_radius * 2 + 1

    kernel = np.zeros((kernel_diameter, kernel_diameter, 4), dtype=np.float32)

    kernel_radius_squared = kernel_radius**2

    # Create kernel
    for dy in range(-kernel_radius, kernel_radius + 1):
        for dx in range(-kernel_radius, kernel_radius + 1):
            distance_squared = dx * dx + dy * dy
            if distance_squared > kernel_radius_squared:
                continue

            x = kernel_radius + dx
            y = kernel_radius + dy

            # TODO: Not sure if it should be x, y instead?
            kernel[y, x] = 1 / (distance_squared + 1)

    # print(kernel)

    return kernel


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
        default=10,
        help="How often the current output image gets saved",
    )
    parser.add_argument(
        "-k",
        "--kernel-radius",
        type=int,
        default=math.inf,
        help="The radius of neighbors that get compared against the current pixel's color; a higher radius means more blur, but is quadratically slower",
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
        default=4,
        help="The number of leading zeros on saved images; this has no effect if the -n switch isn't passed!",
    )
    parser.add_argument(
        "-c",
        "--color-comparison",
        type=str,
        default="LAB",
        help="The color space in which pixels are compared: LAB mimics how the human eye percieves color, while RGB is easier to implement",
    )
    parser.add_argument(
        "-w",
        "--workgroup-size",
        type=int,
        default=8,
        help="The workgroup size; the actually used workgroup size can be lower, and will be printed",
    )


def main():
    start_time = time.time()

    print("Setting up argument parser...")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_parser_arguments(parser)
    args = parser.parse_args()

    if args.shuffle_mode == "PHILOX":
        raise Exception(
            "PHILOX is broken, creating shuffle collisions, so use LCG for now"
        )

    print("Initializing OpenCL...")
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    print("Loading input image...")
    # This example code only works with RGBA images
    pixels_img = Image.open(args.input_image_path).convert("RGBA")
    pixels = np.array(pixels_img, dtype=np.float32)

    # Get size of source image
    height = pixels.shape[0]
    width = pixels.shape[1]

    assert width % 2 == 0, "This program doesn't support images with an odd width"

    kernel_radius = args.kernel_radius
    max_kernel_radius = max(width, height) - 1
    kernel_radius = min(kernel_radius, max_kernel_radius)
    print(f"Using kernel radius {kernel_radius}")

    print("Building sort.cl...")
    # TODO: Try to find useful optimization flags
    prg = cl.Program(
        ctx,
        get_opencl_code(
            args.iterations_in_kernel_per_call, kernel_radius, args.shuffle_mode
        ),
    ).build(options="-DMAKE_VSCODE_HIGHLIGHTER_HAPPY=1")

    # How many work-items to have (one for every pair of pixels)
    pair_count = int(width / 2)
    thread_count = pair_count * height
    global_size = (thread_count, 1)

    # Work groups have to be able to exactly consume all work-items, with no leftovers
    workgroup_size = args.workgroup_size

    while thread_count % workgroup_size != 0:
        workgroup_size -= 1

    # Don't allow the workgroup_size to be odd
    # if workgroup_size % 2 == 1:
    #     workgroup_size -= 1

    print(f"Using workgroup-size {workgroup_size}")

    print("Packing LAB colors into input image pixels...")
    if args.color_comparison == "LAB":
        pixels = pack_lab_into_pixels(pixels)

    rgba_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    print("Creating pixels_buf for sort.cl...")
    pixels_buf = cl.Image(
        ctx, cl.mem_flags.READ_WRITE, rgba_format, shape=(width, height)
    )
    cl.enqueue_copy(
        queue, pixels_buf, pixels, origin=(0, 0), region=(width, height)
    ).wait()

    print("Creating image kernel...")
    kernel = get_kernel(kernel_radius)
    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]

    print("Creating kernel_buf for sort.cl...")
    # TODO: Try using kernel_format:
    # kernel_format = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT)
    kernel_buf = cl.Image(
        ctx, cl.mem_flags.READ_WRITE, rgba_format, shape=(kernel_width, kernel_height)
    )
    cl.enqueue_copy(
        queue,
        kernel_buf,
        kernel,
        origin=(0, 0),
        region=(kernel_width, kernel_height),
    ).wait()

    # TODO: The rgba_format channel_type FLOAT might be lossy with large enough kernels!
    # Make sure to try a huge kernel on big_palette.png, and let count_colors.py do its thing!
    neighbor_totals_buf = cl.Image(
        ctx, cl.mem_flags.READ_WRITE, rgba_format, shape=(width, height)
    )
    initialize_neighbor_totals_buf(
        queue, neighbor_totals_buf, pixels, width, height, kernel
    )

    print("Creating updated_buf for sort.cl...")
    updated_buf = cl.Image(
        ctx, cl.mem_flags.READ_WRITE, rgba_format, shape=(width, height)
    )

    # How many work-items to put in a work-group, i.e. how to partition work-items.
    # Items in a work-group can work together, e.g. they can share fast local memory.
    # Because the global size is partitioned with the local size into groups,
    # both must have the same dimension, e.g. g.s=(1,10) and l.s=(1,2) gives 5 groups.
    # If you don't care about work-groups, just put None.
    # Source is Harry's comment below this answer: https://stackoverflow.com/a/50373589/13279557
    #
    # Setting this to None to go would mean an implementation-defined workgroups size would be used,
    # which crashes your GPU after a few minutes when input/all_colors_shuffled.png is the input
    # when a huge kernel_size is used (30 on my GPU).
    # Source: https://stackoverflow.com/a/25443544/13279557
    local_size = (workgroup_size, 1)

    rand1 = np.uint32(42424242)
    rand2 = np.uint32(69696969)

    python_iteration = 0
    prev_python_iteration = 0

    saved_results = 0

    last_printed_time = time.time()

    opencl_sort = prg.sort

    # TODO: Fix wrong elephant color count with ITERATIONS_IN_KERNEL_PER_CALL 2
    # opencl_sort()
    # save_result()
    # print_status()

    print("Running sort.cl...")
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
                    prev_python_iteration,
                    python_iteration,
                    args.iterations_in_kernel_per_call,
                    start_time,
                    thread_count,
                )

                last_printed_time = time.time()
                prev_python_iteration = python_iteration

            # Numpy handles unsigned wraparound for us
            rand1 = np.uint32(rand1 + 1)

            # TODO: Why does removing the wait() suddenly fix tiny.png wrong count issues??

            # The .wait() at the end of this line is crucial!
            # The reason being that the OpenCL kernel call is async,
            # so without it you end up being unable to use Ctrl+C
            # to stop the program!
            # Here's the documentation of all the arguments:
            # https://documen.tician.de/pyopencl/runtime_program.html#pyopencl.Kernel.__call__
            opencl_sort(
                queue,
                global_size,
                local_size,
                pixels_buf,
                neighbor_totals_buf,
                updated_buf,
                kernel_buf,
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
            prev_python_iteration,
            python_iteration,
            args.iterations_in_kernel_per_call,
            start_time,
            thread_count,
        )

        if not args.no_overwriting_output:
            count_colors.count_colors(args.input_image_path, args.output_image_path)


if __name__ == "__main__":
    main()
