#define uint32_t uint
#define uint64_t ulong

#define NUM_ROUNDS 24

#define KERNEL_SIDE_LENGTH 2

// uint64_t round_up_to_power_of_2(
// 	uint64_t a
// ) {
// 	if(a & (a - 1))
// 	{
// 		uint64_t i;
// 		for(i = 0; a > 1; i++)
// 		{
// 			a >>= 1ull;
// 		}

// 		return 1ull << (i + 1ull);
// 	}

// 	return a;
// }

// uint64_t lcg(
// 	uint64_t capacity,
// 	uint64_t val
// ) {
// 	uint64_t modulus = round_up_to_power_of_2(capacity);

// 	// TODO: Ask authors what I should do in place of random_function() here:
// 	uint64_t multiplier_rand = 42424242;

// 	// Must be odd so it is coprime to modulus
// 	uint64_t multiplier = (multiplier_rand * 2 + 1) % modulus;

// 	// TODO: Ask authors what I should do in place of random_function() here:
// 	uint64_t addition_rand = 69696969;

// 	uint64_t addition = addition_rand % modulus;

// 	// Modulus must be power of two
// 	// assert((modulus & (modulus - 1)) == 0);
// 	// TODO: Replace with proper assert() somehow
// 	if (!((modulus & (modulus - 1)) == 0)) {
// 		printf("Assertion failure: Modulus wasn't power of two!\n");
// 	}

// 	// printf("modulus: %d, multiplier: %d, addition: %d, returned: %d\n", modulus, multiplier, addition, ((val * multiplier) + addition) & (modulus - 1));

// 	return ((val * multiplier) + addition) & (modulus - 1);
// }

uint32_t mulhilo(
	uint64_t a,
	uint32_t b,
	uint32_t *hip
) {
    uint64_t product = a * convert_ulong(b);
    *hip = product >> 32;
    return convert_uint(product);
}

uint64_t get_cipher_bits(uint64_t capacity)
{
	if(capacity == 0)
		return 0;

	uint64_t i = 0;
	capacity--;
	while(capacity != 0)
	{
		i++;
		capacity >>= 1;
	}

	return max(i, convert_ulong(4));
}

uint64_t philox(
	uint64_t capacity,
	uint64_t val
) {
	uint64_t M0 = 0xD2B74407B1CE6E93;
    uint32_t key[NUM_ROUNDS];

	uint64_t total_bits = get_cipher_bits(capacity);

	// Half bits rounded down
	uint64_t left_side_bits = total_bits / 2;
	uint64_t left_side_mask = (1ull << left_side_bits) - 1;

	// Half the bits rounded up
	uint64_t right_side_bits = total_bits - left_side_bits;
	uint64_t right_side_mask = (1ull << right_side_bits) - 1;
	for(int i = 0; i < NUM_ROUNDS; i++)
	{
		// TODO: Ask authors what I should do in place of random_function() here:
		key[i] = 42424242;
	}

	uint32_t state[2] = {
		convert_uint(val >> right_side_bits),
		convert_uint(val & right_side_mask)
	};

	for(int i = 0; i < NUM_ROUNDS; i++)
	{
		uint32_t hi;
		uint32_t lo = mulhilo(M0, state[0], &hi);
		lo = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
		state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
		state[1] = lo & right_side_mask;
	}

	// Combine the left and right sides together to get result
	return convert_ulong(state[0]) << right_side_bits | convert_ulong(state[1]);
}

uint4 get_pixel(
	read_only image2d_t src,
	int2 pos
) {
	// CLK_NORMALIZED_COORDS_FALSE means the x and y coordinates won't be normalized to between 0 and 1
	// CLK_ADDRESS_CLAMP_TO_EDGE means the x and y coordinates are clamped to be within the image's size
	// CLK_FILTER_NEAREST means not having any pixel neighbor interpolation occur
	// Sources:
	// https://man.opencl.org/sampler_t.html
	// https://registry.khronos.org/OpenCL/specs/opencl-1.1.pdf
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	return read_imageui(src, sampler, pos);
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
	int2 pos1,
	int2 pos2
) {
	uint4 pixel1 = get_pixel(src, pos1);
	uint4 pixel2 = get_pixel(src, pos2);

	write_imageui(dest, pos1, pixel2);
	write_imageui(dest, pos2, pixel1);
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
		// shuffled = lcg(num_pixels, shuffled);
		shuffled = philox(num_pixels, shuffled);
	} while (shuffled >= num_pixels);

	return shuffled;
}

int get_squared_color_difference(
	read_only image2d_t src,
	uint4 pixel,
	uint4 neighbor_pixel
) {
	int r_diff = pixel.x - neighbor_pixel.x;
	int g_diff = pixel.y - neighbor_pixel.y;
	int b_diff = pixel.z - neighbor_pixel.z;

	return (
		r_diff * r_diff +
		g_diff * g_diff +
		b_diff * b_diff
	);
}

int get_score(
	read_only image2d_t src,
	int width,
	int height,
	int2 center,
	uint4 pixel
) {
	int score = 0;

	for (int dy = -KERNEL_SIDE_LENGTH; dy <= KERNEL_SIDE_LENGTH; dy++) {
		for (int dx = -KERNEL_SIDE_LENGTH; dx <= KERNEL_SIDE_LENGTH; dx++) {
			int2 neighbor = (int2){center.x + dx, center.y + dy};

			if (neighbor.x < 0 || neighbor.y >= width
			 || neighbor.y < 0 || neighbor.y >= height)
				continue;

			uint4 neighbor_pixel = get_pixel(src, neighbor);

			score += get_squared_color_difference(src, pixel, neighbor_pixel);
		}
	}

	return score;
}

bool should_swap(
	read_only image2d_t src,
	int width,
	int height,
	int2 pos1,
	int2 pos2
) {
	uint4 pixel1 = get_pixel(src, pos1);
	uint4 pixel2 = get_pixel(src, pos2);

	int i1_score_difference = -get_score(src, width, height, pos1, pixel1) + get_score(src, width, height, pos1, pixel2);
	int i2_score_difference = -get_score(src, width, height, pos2, pixel2) + get_score(src, width, height, pos2, pixel1);

	return i1_score_difference + i2_score_difference < 0;
}

kernel void shuffle_(
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

	int2 pos1 = get_pos(shuffled_i1, width);
	int2 pos2 = get_pos(shuffled_i2, width);

	if (should_swap(src, width, height, pos1, pos2)) {
		swap(src, dest, pos1, pos2);
	}
}
