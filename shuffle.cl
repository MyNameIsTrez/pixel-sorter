#define ITERATION_COUNT 1e4
#define KERNEL_RADIUS 1
#define MODE PHILOX

#define NUM_PHILOX_ROUNDS 24

enum MODE {
	LCG,
	PHILOX,
};

typedef uint u32;
typedef ulong u64;

u64 round_up_to_power_of_2(
	u64 a
) {
	if(a & (a - 1))
	{
		u64 i;
		for(i = 0; a > 1; i++)
		{
			a >>= 1ull;
		}

		return 1ull << (i + 1ull);
	}

	return a;
}

u64 lcg(
	u64 capacity,
	u64 val,
	u32 multiplier_rand,
	u32 addition_rand
) {
	u64 modulus = round_up_to_power_of_2(capacity);

	// Must be odd so it is coprime to modulus
	u64 multiplier = (multiplier_rand * 2 + 1) % modulus;

	u64 addition = addition_rand % modulus;

	// Modulus must be power of two
	// assert((modulus & (modulus - 1)) == 0);
	// TODO: Replace with proper assert() somehow
	if (!((modulus & (modulus - 1)) == 0)) {
		printf("Assertion failure: Modulus wasn't power of two!\n");
	}

	// printf("val: %d, modulus: %d, multiplier_rand: %d, multiplier: %d, addition: %d, returned: %d\n", val, modulus, multiplier_rand, multiplier, addition, ((val * multiplier) + addition) & (modulus - 1));

	return ((val * multiplier) + addition) & (modulus - 1);
}

// Source: https://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
u32 rand(
	uint2 *state
) {
    enum { A=4294883355U };
    u32 x=(*state).x, c=(*state).y;  // Unpack the state
    u32 res=x^c;                     // Calculate the result
    u32 hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);               // Pack the state back up
    return res;                       // Return the next result
}

u32 mulhilo(
	u64 a,
	u32 b,
	u32 *hip
) {
    u64 product = a * convert_ulong(b);
    *hip = product >> 32;
    return convert_uint(product);
}

u64 get_cipher_bits(u64 capacity)
{
	if(capacity == 0)
		return 0;

	u64 i = 0;
	capacity--;
	while(capacity != 0)
	{
		i++;
		capacity >>= 1;
	}

	return max(i, convert_ulong(4));
}

u64 philox(
	u64 capacity,
	u64 val,
	uint2 *rand_state
) {
	u64 M0 = 0xD2B74407B1CE6E93;
    u32 key[NUM_PHILOX_ROUNDS];

	u64 total_bits = get_cipher_bits(capacity);

	// Half bits rounded down
	u64 left_side_bits = total_bits / 2;
	u64 left_side_mask = (1ull << left_side_bits) - 1;

	// Half the bits rounded up
	u64 right_side_bits = total_bits - left_side_bits;
	u64 right_side_mask = (1ull << right_side_bits) - 1;

	for(int i = 0; i < NUM_PHILOX_ROUNDS; i++)
	{
		key[i] = rand(rand_state);
	}

	u32 state[2] = {
		convert_uint(val >> right_side_bits),
		convert_uint(val & right_side_mask)
	};

	for(int i = 0; i < NUM_PHILOX_ROUNDS; i++)
	{
		u32 hi;
		u32 lo = mulhilo(M0, state[0], &hi);
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

void set_pixel(
	write_only image2d_t dest,
	int2 pos,
	uint4 pixel
) {
	write_imageui(dest, pos, pixel);
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

	set_pixel(dest, pos1, pixel2);
	set_pixel(dest, pos2, pixel1);
}

int get_shuffled_index(
	int i,
	int num_pixels,
	u32 rand1,
	u32 rand2,
	uint2 *rand_state
) {
	// assert(i < num_pixels);
	// TODO: Replace with proper assert() somehow
	if (!(i < num_pixels)) {
		printf("Assertion failure: i < num_pixels was false!\n");
	}
	int shuffled = i;

	// This loop is guaranteed to terminate if i < num_pixels
	do {
		if (MODE == LCG) {
			shuffled = lcg(num_pixels, shuffled, rand1, rand2);
		} else {
			shuffled = philox(num_pixels, shuffled, rand_state);
		}
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
	uint4 pixel,
	int gid
) {
	int score = 0;

	int dy_min = -min(center.y, KERNEL_RADIUS);
	int dy_max = min(height - 1 - center.y, KERNEL_RADIUS);

	int dx_min = -min(center.x, KERNEL_RADIUS);
	int dx_max = min(width - 1 - center.x, KERNEL_RADIUS);

	for (int dy = dy_min; dy <= dy_max; dy++) {
		for (int dx = dx_min; dx <= dx_max; dx++) {

			int2 neighbor = (int2){center.x + dx, center.y + dy};

			if ((dx == 0 && dy == 0)
			|| neighbor.x < 0 || neighbor.x >= width
			|| neighbor.y < 0 || neighbor.y >= height) {
				continue;
			}

			// printf("center: {%d,%d}, neighbor: {%d,%d}, dims: {%d,%d}\n", center.x, center.y, neighbor.x, neighbor.y, width, height);

			uint4 neighbor_pixel = get_pixel(src, neighbor);

			score += get_squared_color_difference(src, pixel, neighbor_pixel);

			// TODO: Not sure whether squared_color_difference is a good idea?
			// The advantage of it is that it fixes the issues on palette.png
			// with a KERNEL_RADIUS of 15 where none of the pixels get moved.
			// I think it can work if it's tuned a bit more to be less aggressive?

			// int squared_color_difference = get_squared_color_difference(src, pixel, neighbor_pixel);

			// int distance_squared = dx * dx + dy * dy;

			// score += squared_color_difference / distance_squared;
			// printf("squared_color_difference: %d, distance_squared: %d, squared_color_difference / distance_squared: %d, score: %d\n", squared_color_difference, distance_squared, squared_color_difference / distance_squared, score);

			// if (gid == 1 && center.x == 3 && center.y == 1 && pixel.x == 150) {
			// if (gid == 1 && center.x == 1 && center.y == 0 && pixel.x == 150) {
			// 	printf("gid: %d, neighbor: {%d,%d}, score: %d, dims: {%d,%d}, pixel: {%d,%d,%d}, neighbor_pixel: {%d,%d,%d}\n", gid, neighbor.x, neighbor.y, score, width, height, pixel.x, pixel.y, pixel.z, neighbor_pixel.x, neighbor_pixel.y, neighbor_pixel.z);
			// }
		}
	}

	// if (gid == 1 && center.x == 3 && center.y == 1 && pixel.x == 150) {
	// 	printf("score: %d\n", score);
	// }

	return score;
}

bool should_swap(
	read_only image2d_t src,
	int width,
	int height,
	int2 pos1,
	int2 pos2,
	int gid
) {
	uint4 pixel1 = get_pixel(src, pos1);
	uint4 pixel2 = get_pixel(src, pos2);

	int i1_old_score = get_score(src, width, height, pos1, pixel1, gid);
	int i1_new_score = get_score(src, width, height, pos1, pixel2, gid);
	int i1_score_difference = -i1_old_score + i1_new_score;

	int i2_old_score = get_score(src, width, height, pos2, pixel2, gid);
	int i2_new_score = get_score(src, width, height, pos2, pixel1, gid);
	int i2_score_difference = -i2_old_score + i2_new_score;

	int score_difference = i1_score_difference + i2_score_difference;

	// if (gid == 1) {
	// if (pos1.x == 1 && pos1.y == 0 && pos2.x == 3 && pos2.y == 1) {
		// printf("{0, 0}: {%d,%d,%d}", get_pixel(src, (int2)(0, 0)).x, get_pixel(src, (int2)(0, 0)).y, get_pixel(src, (int2)(0, 0)).z);
		// printf("{1, 0}: {%d,%d,%d}", get_pixel(src, (int2)(1, 0)).x, get_pixel(src, (int2)(1, 0)).y, get_pixel(src, (int2)(1, 0)).z);
		// printf("{2, 0}: {%d,%d,%d}", get_pixel(src, (int2)(2, 0)).x, get_pixel(src, (int2)(2, 0)).y, get_pixel(src, (int2)(2, 0)).z);
		// printf("{0, 1}: {%d,%d,%d}", get_pixel(src, (int2)(0, 1)).x, get_pixel(src, (int2)(0, 1)).y, get_pixel(src, (int2)(0, 1)).z);
		// printf("{1, 1}: {%d,%d,%d}", get_pixel(src, (int2)(1, 1)).x, get_pixel(src, (int2)(1, 1)).y, get_pixel(src, (int2)(1, 1)).z);
		// printf("{2, 1}: {%d,%d,%d}", get_pixel(src, (int2)(2, 1)).x, get_pixel(src, (int2)(2, 1)).y, get_pixel(src, (int2)(2, 1)).z);

		// printf("Y: gid %d, swap pos1 {%d,%d} with pos2 {%d,%d}, score difference: %d from i1 {%d,%d,%d}, i2 {%d,%d,%d}\n", gid, pos1.x, pos1.y, pos2.x, pos2.y, score_difference, i1_old_score, i1_new_score, i1_score_difference, i2_old_score, i2_new_score, i2_score_difference);
	// }

	return score_difference < 0;
}

kernel void shuffle_(
	read_only image2d_t src,
	write_only image2d_t dest,
	u32 rand1,
	u32 rand2
) {
	// TODO: Move as much as possible out of this loop!

	// TODO: Test if using get_image_dim() instead of these two calls is faster
	int width = get_image_width(src);
	int height = get_image_height(src);

	int pixel_count = width * height;

	int gid = get_global_id(0);
	int i1 = gid * 2;
	int i2 = i1 + 1;

	// TODO: Maybe base this off of the gid?
	uint2 rand_state = (uint2)(rand1, rand2);

	// double taken = 0;
	// double not_taken = 0;

	for (int iteration = 0; iteration < ITERATION_COUNT; iteration++) {
		// TODO: Is this defined to wrap around in OpenCL?
		rand1++;

		int shuffled_i1 = get_shuffled_index(i1, pixel_count, rand1, rand2, &rand_state);
		int shuffled_i2 = get_shuffled_index(i2, pixel_count, rand1, rand2, &rand_state);

		int2 pos1 = get_pos(shuffled_i1, width);
		int2 pos2 = get_pos(shuffled_i2, width);

		// printf("i1: %d, i2: %d, shuffled_i1: %d, shuffled_i2: %d, pos1: {%d,%d}, pos2: {%d,%d}", i1, i2, shuffled_i1, shuffled_i2, pos1.x, pos1.y, pos2.x, pos2.y);

		// TODO: Stop unnecessarily passing gid to a bunch of functions!
		if (should_swap(src, width, height, pos1, pos2, gid)) {
			swap(src, dest, pos1, pos2);

			// Copy the dst buffer to the src buffer
			set_pixel(src, pos1, get_pixel(dest, pos1));
			set_pixel(src, pos2, get_pixel(dest, pos2));

			// taken++;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// not_taken++;
	}

	// printf("Percentage of swaps taken: %f%\n", taken / ((not_taken == 0) ? 1 : not_taken) * 100);
}
