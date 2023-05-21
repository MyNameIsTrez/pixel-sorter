#define uint64_t ulong

uint64_t round_up_power_2(
	uint64_t a
) {
	if(a & (a - 1))
	{
		uint64_t i;
		for(i = 0; a > 1; i++)
		{
			a >>= 1ull;
		}

		return 1ull << (i + 1ull);
	}

	return a;
}

uint64_t lcg(
	uint64_t capacity,
	uint64_t val
) {
	uint64_t modulus = round_up_power_2(capacity);

	// TODO: Ask authors what I should do in place of random_function() here:
	uint64_t multiplier_rand = 42424242;

	// Must be odd so it is coprime to modulus
	uint64_t multiplier = (multiplier_rand * 2 + 1) % modulus;

	// TODO: Ask authors what I should do in place of random_function() here:
	uint64_t addition_rand = 69696969;

	uint64_t addition = addition_rand % modulus;

	// Modulus must be power of two
	// assert((modulus & (modulus - 1)) == 0);
	// TODO: Replace with proper assert() somehow
	if (!((modulus & (modulus - 1)) == 0)) {
		printf("Assertion failure: Modulus wasn't power of two!\n");
	}

	return ((val * multiplier) + addition) & (modulus - 1);
}

void swap(
	read_only image2d_t src,
    write_only image2d_t dest,
	int width,
	int shuffled_i1,
	int shuffled_i2
) {
	// TODO: Move code to new get_pos()

	int x1 = shuffled_i1 % width;
	int y1 = (int)(shuffled_i1 / width);
    int2 pos1 = (int2)(x1, y1);

	int x2 = shuffled_i2 % width;
	int y2 = (int)(shuffled_i2 / width);
    int2 pos2 = (int2)(x2, y2);

	// printf("shuffled_i1: %d, x1: %d, y1: %d\n", shuffled_i1, x1, y1);
	// printf("shuffled_i2: %d, x2: %d, y2: %d\n", shuffled_i2, x2, y2);

	// CLK_NORMALIZED_COORDS_FALSE means the x and y coordinates won't be normalized to between 0 and 1
	// CLK_ADDRESS_CLAMP_TO_EDGE means the x and y coordinates are clamped to be within the image's size
	// CLK_FILTER_NEAREST means not having any pixel neighbor interpolation occur
	// Sources:
	// https://man.opencl.org/sampler_t.html
	// https://registry.khronos.org/OpenCL/specs/opencl-1.1.pdf
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    uint4 pix1 = read_imageui(src, sampler, pos1);

    uint4 pix2 = read_imageui(src, sampler, pos2);

	// printf(
	// 	"pix1: [%d,%d,%d,%d], pix2: [%d,%d,%d,%d]\n",
	// 	pix1.x, pix1.y, pix1.z, pix1.w,
	// 	pix2.x, pix2.y, pix2.z, pix2.w
	// );

    write_imageui(dest, pos1, pix2);

    write_imageui(dest, pos2, pix1);
}

int get_shuffled_index(
	int i,
	int num_pixels
) {
	// assert(i < num_pixels);
	// TODO: Replace with proper assert() somehow
	if (!(i < num_pixels)) {
		printf("Assertion failure: i < num_pixels was false!\n");
	}
	int shuffled = i;

	// This loop is guaranteed to terminate if i < num_pixels
	do {
		// shuffled = philox( shuffled );

		shuffled = lcg(num_pixels, shuffled);

		// printf("shuffled: %d\n", shuffled);
	} while (shuffled >= num_pixels);

	// printf("shuffled: %d\n", shuffled);

	return shuffled;
}

kernel void grayscale(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	// TODO: Test if using get_image_dim() instead of these two calls is faster
	int width = get_image_width(src);
	int height = get_image_height(src);

	int pixel_count = width * height;

	// lcg(pixel_count, 0);
// 	printf("width: %d, height: %d, pixel_count: %d, lcg(0): %d\n", width, height, pixel_count, lcg(pixel_count, 0));
// }

	int gid = get_global_id(0);
	int i1 = gid * 2;
	int i2 = i1 + 1;

	// int x = get_global_id(0);
	// int y = get_global_id(1);
	// int i = y * width + x;

	// printf("x: %d, y: %d", x, y);

	// printf("i1: %d, i2: %d, r.v[0]: %d\n", i1, i2, r.v[0]);

	// uint R = r.v[0] & 255;
	// uint G = r.v[0] & 255;
	// uint B = r.v[0] & 255;
	// uint v = lcg(pixel_count, i) & 255;
	// uint R = v;
	// uint G = v;
	// uint B = v;
	// uint A = 255;
	// uint4 pix = (uint4)(R, G, B, A);

	// int shuffled_i1 = get_shuffled_index(i1, pixel_count);
	// int shuffled_i2 = get_shuffled_index(i2, pixel_count);
	int shuffled_i1 = i1;
	int shuffled_i2 = i2;
	swap(src, dest, width, shuffled_i1, shuffled_i2);
}
