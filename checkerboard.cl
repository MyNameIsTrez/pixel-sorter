__kernel void convert(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	// CLK_NORMALIZED_COORDS_FALSE means the x and y coordinates won't be normalized to between 0 and 1
	// CLK_ADDRESS_CLAMP_TO_EDGE means the x and y coordinates are clamped to be within the image's size
	// CLK_FILTER_NEAREST means not having any pixel neighbor interpolation occur
	// Sources:
	// https://man.opencl.org/sampler_t.html
	// https://registry.khronos.org/OpenCL/specs/opencl-1.1.pdf
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	const int x = get_global_id(0);
	const int y = get_global_id(1);

    const int2 pos = (int2)(x, y);
    const uint4 pix = read_imageui(src, sampler, pos);

    // A simple test operation: delete pixels to form a checkerboard pattern
    if ((x + ((y + 1) % 2)) % 2 == 0) {
        pix.x = 0;
        pix.y = 0;
        pix.z = 0;
    }

    write_imageui(dest, pos, pix);
}
