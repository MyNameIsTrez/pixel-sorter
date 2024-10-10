// These #defines in here are deliberately never reached
// They are solely here to make my VS Code highlighter
// not whine about the *real* defines not being in this file
#ifndef MAKE_VSCODE_HIGHLIGHTER_HAPPY
#define WIDTH 0
#define HEIGHT 0
#define OPAQUE_PIXEL_COUNT 0
#define ITERATIONS_IN_KERNEL_PER_CALL 0
#define KERNEL_RADIUS 0
#endif

#define KERNEL_RADIUS_SQUARED (KERNEL_RADIUS * KERNEL_RADIUS)

typedef uint u32;
typedef ulong u64;

void set_pixel(
	read_write image2d_t pixels,
	int2 pos,
	float4 pixel
) {
	write_imagef(pixels, pos, pixel);
}

void mark_neighbors_as_updated(
	read_write image2d_t updated,
	int2 center
) {
	// TODO: By padding the input image, it should be possible to get rid of these bounds variables
	int dy_min = -min(center.y, KERNEL_RADIUS);
	int dy_max = min(HEIGHT - 1 - center.y, KERNEL_RADIUS);

	int dx_min = -min(center.x, KERNEL_RADIUS);
	int dx_max = min(WIDTH - 1 - center.x, KERNEL_RADIUS);

	for (int dy = dy_min; dy <= dy_max; dy++) {
		for (int dx = dx_min; dx <= dx_max; dx++) {

			int2 neighbor = (int2){center.x + dx, center.y + dy};

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > KERNEL_RADIUS_SQUARED) {
				continue;
			}

			set_pixel(updated, neighbor, 1);
		}
	}
}

float get_squared_color_difference(
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
	read_write image2d_t pixels,
	int2 pos
) {
	// Samplerless: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/imageSamplerlessReadFunctions.html
	return read_imagef(pixels, pos);
}

void update_neighbor_total(
	read_write image2d_t pixels,
	read_write image2d_t neighbor_totals,
	read_write image2d_t kernel_,
	int2 center
) {
	float4 neighbor_total = 0;
	int2 kernel_center = (int2){KERNEL_RADIUS, KERNEL_RADIUS};

	// TODO: By padding the input image, it should be possible to get rid of these bounds variables
	int dy_min = -min(center.y, KERNEL_RADIUS);
	int dy_max = min(HEIGHT - 1 - center.y, KERNEL_RADIUS);

	int dx_min = -min(center.x, KERNEL_RADIUS);
	int dx_max = min(WIDTH - 1 - center.x, KERNEL_RADIUS);

	for (int dy = dy_min; dy <= dy_max; dy++) {
		for (int dx = dx_min; dx <= dx_max; dx++) {
			int2 offset = (int2){dx, dy};

			int2 neighbor = center + offset;

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > KERNEL_RADIUS_SQUARED) {
				continue;
			}

			float4 neighbor_pixel = get_pixel(pixels, neighbor);

			int2 kernel_pos = kernel_center + offset;

			float weight = get_pixel(kernel_, kernel_pos).x;

			neighbor_total += neighbor_pixel * weight;
		}
	}

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

int2 get_pos(
	int shuffled_i
) {
	int x = shuffled_i % WIDTH;
	int y = (int)(shuffled_i / WIDTH);
	return (int2)(x, y);
}

int get_shuffled_index(
	int i,
	u32 rand1,
	u32 rand2
) {
	// assert(i < OPAQUE_PIXEL_COUNT);
	// TODO: Replace with proper assert() somehow
	if (!(i < OPAQUE_PIXEL_COUNT)) {
		printf("Assertion failure: i < OPAQUE_PIXEL_COUNT was false!\n");
	}

	int shuffled = i;

	// This loop is guaranteed to terminate if i < OPAQUE_PIXEL_COUNT
	do {
		shuffled = lcg(OPAQUE_PIXEL_COUNT, shuffled, rand1, rand2);
	} while (shuffled >= OPAQUE_PIXEL_COUNT);

	return shuffled;
}

bool should_swap(
	read_write image2d_t neighbor_totals,
	float4 pixel1,
	float4 pixel2,
	int2 pos1,
	int2 pos2
) {
	float4 i1_neighbor_total = get_pixel(neighbor_totals, pos1);
	float i1_old_score = get_squared_color_difference(pixel1, i1_neighbor_total);
	float i1_new_score = get_squared_color_difference(pixel2, i1_neighbor_total);
	float i1_score_difference = -i1_old_score + i1_new_score;

	float4 i2_neighbor_total = get_pixel(neighbor_totals, pos2);
	float i2_old_score = get_squared_color_difference(pixel2, i2_neighbor_total);
	float i2_new_score = get_squared_color_difference(pixel1, i2_neighbor_total);
	float i2_score_difference = -i2_old_score + i2_new_score;

	float score_difference = i1_score_difference + i2_score_difference;

	return score_difference < 0;
}

kernel void sort(
	read_write image2d_t pixels,
	read_write image2d_t neighbor_totals,
	read_write image2d_t updated,
	read_write image2d_t kernel_,
	global int normal_to_opaque_index_lut[OPAQUE_PIXEL_COUNT],
	u32 rand1,
	u32 rand2
) {
	int gid = get_global_id(0);
	int i1 = gid * 2;
	int i2 = i1 + 1;

	for (int iteration = 0; iteration < ITERATIONS_IN_KERNEL_PER_CALL; iteration++) {
		// TODO: Is this defined to wrap around in OpenCL?
		rand1++;

		int shuffled_i1 = get_shuffled_index(i1, rand1, rand2);
		int shuffled_i2 = get_shuffled_index(i2, rand1, rand2);

		// TODO: Remap shuffled_i1 so it takes images with empty alpha=0 spots into account
		shuffled_i1 = normal_to_opaque_index_lut[shuffled_i1];
		shuffled_i2 = normal_to_opaque_index_lut[shuffled_i2];

		int2 pos1 = get_pos(shuffled_i1);
		int2 pos2 = get_pos(shuffled_i2);

		set_pixel(updated, pos1, 0);
		set_pixel(updated, pos2, 0);

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		float4 pixel1 = get_pixel(pixels, pos1);
		float4 pixel2 = get_pixel(pixels, pos2);

		// printf("i1: %d, i2: %d, shuffled_i1: %d, shuffled_i2: %d, pos1: {%d,%d}, pos2: {%d,%d}", i1, i2, shuffled_i1, shuffled_i2, pos1.x, pos1.y, pos2.x, pos2.y);

		bool swapping = should_swap(neighbor_totals, pixel1, pixel2, pos1, pos2);

		// TODO: Not sure which of these two flags I should use,
		// cause either seems to work.
		// TODO: Not sure if this barrier is still necessary
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (swapping) {
			set_pixel(pixels, pos1, pixel2);
			mark_neighbors_as_updated(updated, pos1);

			set_pixel(pixels, pos2, pixel1);
			mark_neighbors_as_updated(updated, pos2);
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (swapping || get_pixel(updated, pos1).x != 0) {
			update_neighbor_total(pixels, neighbor_totals, kernel_, pos1);
		}

		if (swapping || get_pixel(updated, pos2).x != 0) {
			update_neighbor_total(pixels, neighbor_totals, kernel_, pos2);
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
