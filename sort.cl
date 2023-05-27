// These #defines in here are deliberately never reached
// They are solely here to make my VS Code highlighter
// not whine about the *real* defines not being in this file
#ifndef MAKE_VSCODE_HIGHLIGHTER_HAPPY
#define ITERATIONS_IN_KERNEL_PER_CALL 0
#define KERNEL_RADIUS 0
#define SHUFFLE_MODE 0
#endif

#define KERNEL_RADIUS_SQUARED (KERNEL_RADIUS * KERNEL_RADIUS)
#define NUM_PHILOX_ROUNDS 24

enum SHUFFLE_MODES {
	LCG,
	PHILOX,
};

typedef uint u32;
typedef ulong u64;

void set_pixel(
	write_only image2d_t pixels,
	int2 pos,
	float4 pixel
) {
	write_imagef(pixels, pos, pixel);
}

void mark_neighbors_as_updated(
	read_only image2d_t updated,
	int width,
	int height,
	int2 center
) {
	int dy_min = -min(center.y, KERNEL_RADIUS);
	int dy_max = min(height - 1 - center.y, KERNEL_RADIUS);

	int dx_min = -min(center.x, KERNEL_RADIUS);
	int dx_max = min(width - 1 - center.x, KERNEL_RADIUS);

	for (int dy = dy_min; dy <= dy_max; dy++) {
		for (int dx = dx_min; dx <= dx_max; dx++) {

			int2 neighbor = (int2){center.x + dx, center.y + dy};

			if (neighbor.x < 0 || neighbor.x >= width
			|| neighbor.y < 0 || neighbor.y >= height) {
				continue;
			}

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > KERNEL_RADIUS_SQUARED) {
				continue;
			}

			set_pixel(updated, neighbor, 1);
		}
	}
}

float get_squared_color_difference(
	read_only image2d_t pixels,
	float4 pixel,
	float4 neighbor_pixel
) {
	float r_diff = pixel.x - neighbor_pixel.x;
	float g_diff = pixel.y - neighbor_pixel.y;
	float b_diff = pixel.z - neighbor_pixel.z;

	return (
		r_diff * r_diff +
		g_diff * g_diff +
		b_diff * b_diff
	);
}

float4 get_pixel(
	read_only image2d_t pixels,
	int2 pos
) {
	// CLK_NORMALIZED_COORDS_FALSE means the x and y coordinates won't be normalized to between 0 and 1
	// CLK_ADDRESS_CLAMP_TO_EDGE means the x and y coordinates are clamped to be within the image's size
	// CLK_FILTER_NEAREST means not having any pixel neighbor interpolation occur
	// Sources:
	// https://man.opencl.org/sampler_t.html
	// https://registry.khronos.org/OpenCL/specs/opencl-1.1.pdf
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	return read_imagef(pixels, sampler, pos);
}

// TODO: Figure out a way to make this one work
// void update_neighbor_total(
// 	write_only image2d_t neighbor_totals,
// 	int2 pos,
// 	float4 old_pixel,
// 	float4 new_pixel
// ) {
// 	float4 old_neighbor_total = get_pixel(neighbor_totals, pos);
// 	float4 new_neighbor_total = old_neighbor_total - old_pixel + new_pixel;

// 	set_pixel(neighbor_totals, pos, new_neighbor_total);
// }

void update_neighbor_total(
	read_only image2d_t pixels,
	write_only image2d_t neighbor_totals,
	int width,
	int height,
	int2 center,
	int gid
) {
	float4 neighbor_total = 0;

	int dy_min = -min(center.y, KERNEL_RADIUS);
	int dy_max = min(height - 1 - center.y, KERNEL_RADIUS);

	int dx_min = -min(center.x, KERNEL_RADIUS);
	int dx_max = min(width - 1 - center.x, KERNEL_RADIUS);

	for (int dy = dy_min; dy <= dy_max; dy++) {
		for (int dx = dx_min; dx <= dx_max; dx++) {

			int2 neighbor = (int2){center.x + dx, center.y + dy};

			if (neighbor.x < 0 || neighbor.x >= width
			|| neighbor.y < 0 || neighbor.y >= height) {
				continue;
			}

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > KERNEL_RADIUS_SQUARED) {
				continue;
			}

			float4 neighbor_pixel = get_pixel(pixels, neighbor);

			neighbor_total += neighbor_pixel / (distance_squared + 1);

			// printf("center: {%d,%d}, neighbor: {%d,%d}, dims: {%d,%d}\n", center.x, center.y, neighbor.x, neighbor.y, width, height);

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

	set_pixel(neighbor_totals, center, neighbor_total);
}

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
    enum {
		A = 4294883355U
	};

	// Unpack the state
    u32 x=(*state).x;
	u32 c=(*state).y;

	// Calculate the result
    u32 res=x^c;

	// Step the RNG
	u32 hi=mul_hi(x,A);
    x=x*A+c;
    c=hi+(x<c);

	// Pack the state back up
    *state=(uint2)(x,c);

	// Return the next result
    return res;
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

// Source: https://github.com/djns99/CUDA-Shuffle/blob/master/include/shuffle/PhiloxShuffle.h
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

int2 get_pos(
	int shuffled_i,
	int width
) {
	int x = shuffled_i % width;
	int y = (int)(shuffled_i / width);
	return (int2)(x, y);
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
		if (SHUFFLE_MODE == LCG) {
			shuffled = lcg(num_pixels, shuffled, rand1, rand2);
		} else {
			shuffled = philox(num_pixels, shuffled, rand_state);
		}
	} while (shuffled >= num_pixels);

	return shuffled;
}

float4 get_averaged_score_pixel(
	read_only image2d_t pixels,
	read_only image2d_t neighbor_totals,
	read_only image2d_t kernel_,
	int width,
	int height,
	int2 center
) {
	float4 score = get_pixel(neighbor_totals, center);

	float kernel_area = 0;
	int2 kernel_center = (int2){KERNEL_RADIUS, KERNEL_RADIUS};

	int dy_min = -min(center.y, KERNEL_RADIUS);
	int dy_max = min(height - 1 - center.y, KERNEL_RADIUS);

	int dx_min = -min(center.x, KERNEL_RADIUS);
	int dx_max = min(width - 1 - center.x, KERNEL_RADIUS);

	// TODO: See if caching this in an image is faster
	for (int dy = dy_min; dy <= dy_max; dy++) {
		for (int dx = dx_min; dx <= dx_max; dx++) {

			int2 offset = (int2){dx, dy};

			int2 neighbor = center + offset;

			if (neighbor.x < 0 || neighbor.x >= width
			|| neighbor.y < 0 || neighbor.y >= height) {
				continue;
			}

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > KERNEL_RADIUS_SQUARED) {
				continue;
			}

			int2 kernel_pos = kernel_center + offset;

			float weight = get_pixel(kernel_, kernel_pos).x;

			kernel_area += weight;
		}
	}

	return score / kernel_area;
}

bool should_swap(
	read_only image2d_t pixels,
	read_only image2d_t neighbor_totals,
	read_only image2d_t kernel_,
	float4 pixel1,
	float4 pixel2,
	int width,
	int height,
	int2 pos1,
	int2 pos2,
	int gid
) {
	float4 i1_averaged = get_averaged_score_pixel(pixels, neighbor_totals, kernel_, width, height, pos1);
	float i1_old_score = get_squared_color_difference(pixels, pixel1, i1_averaged);
	float i1_new_score = get_squared_color_difference(pixels, pixel2, i1_averaged);
	float i1_score_difference = -i1_old_score + i1_new_score;

	float4 i2_averaged = get_averaged_score_pixel(pixels, neighbor_totals, kernel_, width, height, pos2);
	float i2_old_score = get_squared_color_difference(pixels, pixel2, i2_averaged);
	float i2_new_score = get_squared_color_difference(pixels, pixel1, i2_averaged);
	float i2_score_difference = -i2_old_score + i2_new_score;

	float score_difference = i1_score_difference + i2_score_difference;

	// if (gid == 1) {
	// if (pos1.x == 1 && pos1.y == 0 && pos2.x == 3 && pos2.y == 1) {
		// printf("{0, 0}: {%d,%d,%d}", get_pixel(pixels, (int2)(0, 0)).x, get_pixel(pixels, (int2)(0, 0)).y, get_pixel(pixels, (int2)(0, 0)).z);
		// printf("{1, 0}: {%d,%d,%d}", get_pixel(pixels, (int2)(1, 0)).x, get_pixel(pixels, (int2)(1, 0)).y, get_pixel(pixels, (int2)(1, 0)).z);
		// printf("{2, 0}: {%d,%d,%d}", get_pixel(pixels, (int2)(2, 0)).x, get_pixel(pixels, (int2)(2, 0)).y, get_pixel(pixels, (int2)(2, 0)).z);
		// printf("{0, 1}: {%d,%d,%d}", get_pixel(pixels, (int2)(0, 1)).x, get_pixel(pixels, (int2)(0, 1)).y, get_pixel(pixels, (int2)(0, 1)).z);
		// printf("{1, 1}: {%d,%d,%d}", get_pixel(pixels, (int2)(1, 1)).x, get_pixel(pixels, (int2)(1, 1)).y, get_pixel(pixels, (int2)(1, 1)).z);
		// printf("{2, 1}: {%d,%d,%d}", get_pixel(pixels, (int2)(2, 1)).x, get_pixel(pixels, (int2)(2, 1)).y, get_pixel(pixels, (int2)(2, 1)).z);

		// printf("Y: gid %d, swap pos1 {%d,%d} with pos2 {%d,%d}, score difference: %d from i1 {%d,%d,%d}, i2 {%d,%d,%d}\n", gid, pos1.x, pos1.y, pos2.x, pos2.y, score_difference, i1_old_score, i1_new_score, i1_score_difference, i2_old_score, i2_new_score, i2_score_difference);
	// }

	return score_difference < 0;
}

kernel void sort(
	read_only image2d_t pixels,
	read_only image2d_t neighbor_totals,
	read_only image2d_t updated,
	read_only image2d_t kernel_,
	u32 rand1,
	u32 rand2
) {
	// TODO: Move as much as possible out of this loop!

	// TODO: Test if using get_image_dim() instead of these two calls is faster
	int width = get_image_width(pixels);
	int height = get_image_height(pixels);

	int pixel_count = width * height;

	int gid = get_global_id(0);
	int i1 = gid * 2;
	int i2 = i1 + 1;

	// TODO: Maybe base this off of the gid?
	// TODO: Ask the author what I should do here instead, since this has wrong color counts
	uint2 rand_state = (uint2)(rand1, rand2);

	// double taken = 0;
	// double not_taken = 0;

	for (int iteration = 0; iteration < ITERATIONS_IN_KERNEL_PER_CALL; iteration++) {
		// TODO: Is this defined to wrap around in OpenCL?
		rand1++;

		int shuffled_i1 = get_shuffled_index(i1, pixel_count, rand1, rand2, &rand_state);
		int shuffled_i2 = get_shuffled_index(i2, pixel_count, rand1, rand2, &rand_state);

		int2 pos1 = get_pos(shuffled_i1, width);
		int2 pos2 = get_pos(shuffled_i2, width);

		set_pixel(updated, pos1, 0);
		set_pixel(updated, pos2, 0);

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		float4 pixel1 = get_pixel(pixels, pos1);
		float4 pixel2 = get_pixel(pixels, pos2);

		// printf("i1: %d, i2: %d, shuffled_i1: %d, shuffled_i2: %d, pos1: {%d,%d}, pos2: {%d,%d}", i1, i2, shuffled_i1, shuffled_i2, pos1.x, pos1.y, pos2.x, pos2.y);

		// TODO: Stop unnecessarily passing gid to a bunch of functions!
		bool swapping = should_swap(pixels, neighbor_totals, kernel_, pixel1, pixel2, width, height, pos1, pos2, gid);

		// if (swapping) {
		// 	taken++;
		// } else {
		// 	not_taken++;
		// }

		// TODO: Not sure which of these two flags I should use,
		// cause either seems to work.
		// TODO: Not sure if this barrier is still necessary
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (swapping) {
			set_pixel(pixels, pos1, pixel2);
			mark_neighbors_as_updated(updated, width, height, pos1);

			set_pixel(pixels, pos2, pixel1);
			mark_neighbors_as_updated(updated, width, height, pos2);
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (swapping || get_pixel(updated, pos1).x != 0) {
			update_neighbor_total(pixels, neighbor_totals, width, height, pos1, gid);
		}

		if (swapping || get_pixel(updated, pos2).x != 0) {
			update_neighbor_total(pixels, neighbor_totals, width, height, pos2, gid);
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	// printf("Percentage of swaps taken: %f%\n", taken / ((not_taken == 0) ? 1 : not_taken) * 100);
}
