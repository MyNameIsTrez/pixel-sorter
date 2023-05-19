__kernel void grayscale(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	// CLK_NORMALIZED_COORDS_FALSE means the x and y coordinates won't be normalized to between 0 and 1
	// CLK_ADDRESS_CLAMP_TO_EDGE means the x and y coordinates are clamped to be within the image's size
	// CLK_FILTER_NEAREST means not having any pixel neighbor interpolation occur
	// Sources:
	// https://man.opencl.org/sampler_t.html
	// https://registry.khronos.org/OpenCL/specs/opencl-1.1.pdf
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

    int2 pos = (int2)(x, y);
    uint4 pix = read_imageui(src, sampler, pos);

	// Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
	// Source: https://en.wikipedia.org/wiki/Grayscale
	int Y = 0.2126 * pix.x + 0.7152 * pix.y + 0.0722 * pix.z;
	pix.x = Y;
	pix.y = Y;
	pix.z = Y;

    write_imageui(dest, pos, pix);
}
