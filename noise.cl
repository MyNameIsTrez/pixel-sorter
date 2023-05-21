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

int2 get_pos(
	int shuffled_i,
	int width
) {
	int x = shuffled_i % width;
	int y = (int)(shuffled_i / width);
	return (int2)(x, y);
}

void swap(
	read_only image2d_t src,
	write_only image2d_t dest,
	int width,
	int shuffled_i1,
	int shuffled_i2
) {
	int2 pos1 = get_pos(shuffled_i1, width);
	int2 pos2 = get_pos(shuffled_i2, width);

	// CLK_NORMALIZED_COORDS_FALSE means the x and y coordinates won't be normalized to between 0 and 1
	// CLK_ADDRESS_CLAMP_TO_EDGE means the x and y coordinates are clamped to be within the image's size
	// CLK_FILTER_NEAREST means not having any pixel neighbor interpolation occur
	// Sources:
	// https://man.opencl.org/sampler_t.html
	// https://registry.khronos.org/OpenCL/specs/opencl-1.1.pdf
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	uint4 pix1 = read_imageui(src, sampler, pos1);
	uint4 pix2 = read_imageui(src, sampler, pos2);

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
		shuffled = lcg(num_pixels, shuffled);
	} while (shuffled >= num_pixels);

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

	int gid = get_global_id(0);
	int i1 = gid * 2;
	int i2 = i1 + 1;

	int shuffled_i1 = get_shuffled_index(i1, pixel_count);
	int shuffled_i2 = get_shuffled_index(i2, pixel_count);
	swap(src, dest, width, shuffled_i1, shuffled_i2);
}
