#include "cnpy.h"

#include <algorithm>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <signal.h>

struct xy
{
	int x;
	int y;

	xy operator+(const xy &other)
	{
		return {
			x + other.x,
			y + other.y
		};
	}
};

struct rgb
{
	float r;
	float g;
	float b;

	rgb operator*(float multiplier)
	{
		return {
			r * multiplier,
			g * multiplier,
			b * multiplier
		};
	}

	rgb operator+=(const rgb &other)
	{
		r += other.r;
		g += other.g;
		b += other.b;

		return *this;
	}
};

static volatile sig_atomic_t running = true;
static void sigint_handler_running(int signum)
{
	(void)signum;
	running = false;
}

#ifdef DEBUG
static void print_neighbor_totals_error(const std::vector<float> &neighbor_totals_error)
{
	for (size_t i = 0; i < neighbor_totals_error.size(); i++)
	{
		if (i > 0)
		{
			std::cout << ", ";
		}
		std::cout << "[" << i << "] error: " << neighbor_totals_error[i];
	}
	std::cout << std::endl;
}
#endif

static size_t get_index(xy pos, int width)
{
	return pos.x + pos.y * width;
}

static rgb get_pixel(const std::vector<float> &pixels, xy pos, int width)
{
	size_t i = get_index(pos, width) * 4;

	float r = pixels[i + 0];
	float g = pixels[i + 1];
	float b = pixels[i + 2];

	return {r, g, b};
}

static void set_pixel(std::vector<float> &pixels, xy pos, int width, rgb rgb)
{
	size_t i = get_index(pos, width) * 4;

	pixels[i + 0] = rgb.r;
	pixels[i + 1] = rgb.g;
	pixels[i + 2] = rgb.b;
}

void update_neighbor_total(const std::vector<float> &pixels, std::vector<float> &neighbor_totals, const std::vector<float> &kernel, xy center, int width, int height, int kernel_radius)
{
	rgb neighbor_total = {};
	xy kernel_center = {kernel_radius, kernel_radius};

	// TODO: By padding the input image, it should be possible to get rid of these bounds variables
	int dy_min = -std::min(center.y, kernel_radius);
	int dy_max = std::min(height - 1 - center.y, kernel_radius);

	int dx_min = -std::min(center.x, kernel_radius);
	int dx_max = std::min(width - 1 - center.x, kernel_radius);

	for (int dy = dy_min; dy <= dy_max; dy++)
	{
		for (int dx = dx_min; dx <= dx_max; dx++)
		{
			xy offset = {dx, dy};

			xy neighbor = center + offset;

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > kernel_radius * kernel_radius)
			{
				continue;
			}

			rgb neighbor_pixel = get_pixel(pixels, neighbor, width);

			xy kernel_pos = kernel_center + offset;

			float weight = kernel[get_index(kernel_pos, width)];

			neighbor_total += neighbor_pixel * weight;
		}
	}

	set_pixel(neighbor_totals, center, width, neighbor_total);

	// TODO: REMOVE
#ifdef DEBUG
	std::cout << "Set neighbor_totals (x=" << center.x << ",y=" << center.y << ") to (r=" << neighbor_total.r << ",g=" << neighbor_total.g << ",b=" << neighbor_total.b << ")" << std::endl;
#endif
}

static void mark_neighbors_as_updated(std::vector<bool> &updated, xy center, int width, int height, int kernel_radius)
{
	// TODO: By padding the input image, it should be possible to get rid of these bounds variables
	int dy_min = -std::min(center.y, kernel_radius);
	int dy_max = std::min(height - 1 - center.y, kernel_radius);

	int dx_min = -std::min(center.x, kernel_radius);
	int dx_max = std::min(width - 1 - center.x, kernel_radius);

	for (int dy = dy_min; dy <= dy_max; dy++)
	{
		for (int dx = dx_min; dx <= dx_max; dx++)
		{

			xy neighbor = {center.x + dx, center.y + dy};

            int distance_squared = dx * dx + dy * dy;
			if (distance_squared > kernel_radius * kernel_radius)
			{
				continue;
			}

			updated[get_index(neighbor, width)] = true;
		}
	}
}

static float get_squared_color_difference(rgb pixel, rgb neighbor_pixel)
{
	float r_diff = pixel.r - neighbor_pixel.r;
	float g_diff = pixel.g - neighbor_pixel.g;
	float b_diff = pixel.b - neighbor_pixel.b;

	return (
		r_diff * r_diff +
		g_diff * g_diff +
		b_diff * b_diff
	);
}

static bool should_swap(std::vector<float> &neighbor_totals, rgb pixel1, rgb pixel2, xy pos1, xy pos2, int width)
{
	// TODO:
	/*
	I think this is fundamentally flawed, since we're comparing a single RGB with a sum of RGB?
	The example that is fucking me up is tiny3.png its pos1 that is at x=1 in the first iteration here
	I think the issue is that the neighbor total includes the own, center pixel?

	The hypothetical scenario:
	Imagine a 3D RGB cube where i1_neighbor_total is very low, like (r=10,g=10,b=10)
	Say pixel1 was (r=3,g=3,b=3), then it shouldn't be swapped when pixel2 is say (r=11,g=11,b=11),
	but this current code would swap it, since the difference between 10 and 11 is smaller
	*/

	rgb i1_neighbor_total = get_pixel(neighbor_totals, pos1, width);
	float i1_old_score = get_squared_color_difference(pixel1, i1_neighbor_total);
	float i1_new_score = get_squared_color_difference(pixel2, i1_neighbor_total);
	float i1_score_difference = -i1_old_score + i1_new_score;

	rgb i2_neighbor_total = get_pixel(neighbor_totals, pos2, width);
	float i2_old_score = get_squared_color_difference(pixel2, i2_neighbor_total);
	float i2_new_score = get_squared_color_difference(pixel1, i2_neighbor_total);
	float i2_score_difference = -i2_old_score + i2_new_score;

	float score_difference = i1_score_difference + i2_score_difference;

	return score_difference < 0;
}

static xy get_pos(int i, int width)
{
	return {i % width, i / width};
}

static uint64_t round_up_to_power_of_2(uint64_t n)
{
	// If n isn't a power of 2 already
	if(n & (n - 1))
	{
		uint64_t i;

		// TODO: Can "1ull" be replaced with "1" everywhere here?

		// Count the number of times n can be right-shifted
		for(i = 0; n > 1; i++)
		{
			n >>= 1ull;
		}

		// Use that number of times to round it up to a power of 2
		return 1ull << (i + 1ull);
	}

	return n;
}

static uint64_t lcg(uint64_t capacity, uint64_t val, uint32_t multiplier_rand, uint32_t addition_rand)
{
	uint64_t modulus = round_up_to_power_of_2(capacity);

	// Must be odd so it is coprime to modulus
	uint64_t multiplier = (multiplier_rand * 2 + 1) % modulus;

	uint64_t addition = addition_rand % modulus;

	// Modulus must be power of two
	assert((modulus & (modulus - 1)) == 0);

	return ((val * multiplier) + addition) & (modulus - 1);
}

static int get_shuffled_index(int i, uint32_t rand1, uint32_t rand2, int opaque_pixel_count)
{
	assert(i < opaque_pixel_count);

	int shuffled = i;

	do
	{
		shuffled = lcg(opaque_pixel_count, shuffled, rand1, rand2);
	} while (shuffled >= opaque_pixel_count);

	return shuffled;
}

static std::vector<float> get_neighbor_totals(const std::vector<float> &pixels, const std::vector<float> &kernel, int width, int height, int kernel_radius)
{
	std::vector<float> neighbor_totals(pixels.size(), 0);

	int kernel_diameter = kernel_radius * 2 + 1;

	// Play around with extra/kernel_tests.cpp to see how this convolving works
	// For every pixel
	for (int py = 0; py < height; py++)
	{
		for (int px = 0; px < width; px++)
		{
			float pr = pixels[(px + py * width) * 4 + 0];
			float pg = pixels[(px + py * width) * 4 + 1];
			float pb = pixels[(px + py * width) * 4 + 2];

			// Apply the kernel
			for (int kdy = -kernel_radius; kdy < kernel_radius + 1; kdy++)
			{
				for (int kdx = -kernel_radius; kdx < kernel_radius + 1; kdx++)
				{
					int x = px + kdx;
					int y = py + kdy;
					if (x < 0 || y < 0 || x >= width || y >= height)
					{
						continue;
					}

					int kx = kernel_radius + kdx;
					int ky = kernel_radius + kdy;
					float k = kernel[kx + ky * kernel_diameter];

					neighbor_totals[(x + y * width) * 4 + 0] += pr * k;
					neighbor_totals[(x + y * width) * 4 + 1] += pg * k;
					neighbor_totals[(x + y * width) * 4 + 2] += pb * k;
				}
			}
		}
	}

	return neighbor_totals;
}

#ifdef DEBUG
static std::vector<float> get_neighbor_totals_error(const std::vector<float> &neighbor_totals, const std::vector<float> &pixels, const std::vector<float> &kernel, int width, int height, int kernel_radius)
{
	std::vector<float> actual_neighbor_totals = get_neighbor_totals(pixels, kernel, width, height, kernel_radius);
	// TODO: Turn into a one-liner with C++ magic
	std::vector<float> neighbor_totals_error;
	for (size_t i = 0; i < neighbor_totals.size(); i++)
	{
		neighbor_totals_error.push_back(neighbor_totals[i] - actual_neighbor_totals[i]);
	}
	return neighbor_totals_error;
}
#endif

static void sort(std::vector<float> &pixels, std::vector<float> &neighbor_totals, std::vector<bool> &updated, const std::vector<float> &kernel, const std::vector<size_t> &normal_to_opaque_index_lut, int width, int height, uint32_t rand1, uint32_t rand2, int pair_count, int kernel_radius, uint64_t &attempted_swaps)
{
	int opaque_pixel_count = pair_count * 2;

#ifdef DEBUG
	uint64_t swaps = 0;
	uint64_t updated_neighbors = 0;
#endif

	// TODO: Use a C++ parallel-loop here, to get it closer to sort.cl
	for (int i1 = 0; i1 < pair_count; i1 += 2)
	{
#ifdef DEBUG
		std::vector<float> neighbor_totals_error = get_neighbor_totals_error(neighbor_totals, pixels, kernel, width, height, kernel_radius);
		bool no_error = std::all_of(neighbor_totals_error.begin(), neighbor_totals_error.end(), [](float f) { return f == 0.0f; });
		if (!no_error)
		{
			std::cout << "There were " << swaps << " swaps and " << updated_neighbors << " updated neighbors in " << attempted_swaps << " attempted swaps" << std::endl;
			print_neighbor_totals_error(neighbor_totals_error);
			abort();
		}
#endif

		int i2 = i1 + 1;

		int shuffled_i1 = get_shuffled_index(i1, rand1, rand2, opaque_pixel_count);
		int shuffled_i2 = get_shuffled_index(i2, rand1, rand2, opaque_pixel_count);

		shuffled_i1 = normal_to_opaque_index_lut[shuffled_i1];
		shuffled_i2 = normal_to_opaque_index_lut[shuffled_i2];

		xy pos1 = get_pos(shuffled_i1, width);
		xy pos2 = get_pos(shuffled_i2, width);

		updated[get_index(pos1, width)] = false;
		updated[get_index(pos2, width)] = false;

		rgb pixel1 = get_pixel(pixels, pos1, width);
		rgb pixel2 = get_pixel(pixels, pos2, width);

		bool swapping = should_swap(neighbor_totals, pixel1, pixel2, pos1, pos2, width);

		if (swapping)
		{
			set_pixel(pixels, pos1, width, pixel2);
			mark_neighbors_as_updated(updated, pos1, width, height, kernel_radius);

			set_pixel(pixels, pos2, width, pixel1);
			mark_neighbors_as_updated(updated, pos2, width, height, kernel_radius);
#ifdef DEBUG
			swaps++;
#endif
		}

		if (swapping || updated[get_index(pos1, width)])
		{
			update_neighbor_total(pixels, neighbor_totals, kernel, pos1, width, height, kernel_radius);
#ifdef DEBUG
			updated_neighbors++;
#endif
		}

		if (swapping || updated[get_index(pos2, width)])
		{
			update_neighbor_total(pixels, neighbor_totals, kernel, pos2, width, height, kernel_radius);
#ifdef DEBUG
			updated_neighbors++;
#endif
		}

		attempted_swaps++;
	}
}

static std::string humanize_uint64(uint64_t n)
{
	std::ostringstream ss;
	ss << std::fixed << std::setprecision(1);

	if (n < 1'000)
	{
		ss << n;
		return ss.str();
	}
	else if (n < 1'000'000)
	{
		double d = n / 1'000.0;
		ss << d;
		return ss.str() + " thousand";
	}
	else if (n < 1'000'000'000)
	{
		double d = n / 1'000'000.0;
		ss << d;
		return ss.str() + " million";
	}

	double d = n / 1'000'000'000.0;
	ss << d;
	return ss.str() + " billion";
}

static void print_status(int saved_results, uint64_t prev_attempted_swaps, uint64_t attempted_swaps, const std::chrono::steady_clock::time_point &start_time)
{
	const auto now = std::chrono::steady_clock::now();
	const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

	std::cout
		<< "Frame " << saved_results
		<< ", " << seconds << " seconds"
		<< ", " << humanize_uint64(attempted_swaps) << " attempted swaps"
		<< " (+" << humanize_uint64(attempted_swaps - prev_attempted_swaps) << ")"
		<< std::endl;
}

static void save_result(const std::vector<float> &pixels, const std::vector<size_t> &shape, const std::filesystem::path &output_npy_path)
{
	cnpy::npy_save(output_npy_path, pixels.data(), shape, "w");
}

static std::filesystem::path get_output_npy_path(
		const std::filesystem::path &output_npy_path,
		bool no_overwriting_output,
		int saved_image_leading_zero_count,
		int saved_results)
{
    if (no_overwriting_output)
	{
		// Create the string "_0000", assuming saved_results is 0 and saved_image_leading_zero_count is 4
		std::ostringstream ss;
		ss << std::setw(saved_image_leading_zero_count) << std::setfill('0') << saved_results;
		std::string saved_results_str(ss.str());

		// Append "_0000" to the output filename's stem
		std::filesystem::path saved_filename(output_npy_path.stem().string() + "_" + saved_results_str + output_npy_path.extension().string());

		// Stitch the parent directory path back to the front
		return output_npy_path.parent_path() / saved_filename;
	}
    else
	{
        return output_npy_path;
	}
}

static std::vector<size_t> get_normal_to_opaque_index_lut(const std::vector<float> &pixels)
{
    std::vector<size_t> normal_to_opaque_index_lut;

    size_t offset = 0;
	for (size_t i = 3; i < pixels.size(); i += 4)
	{
		float alpha = pixels[i];

		// TODO: Check that this produces the same result as sort.py's version of this code
		if (alpha != 0)
		{
			normal_to_opaque_index_lut.push_back(offset);
		}

		offset += 1;
	}

    return normal_to_opaque_index_lut;
}

// TODO: Profile whether it's faster to just recreate the kernel on the fly,
// TODO: since we still need these same loops to loop over it anyways!
static std::vector<float> get_kernel(int kernel_radius)
{
    int kernel_diameter = kernel_radius * 2 + 1;

	std::vector<float> kernel(kernel_diameter * kernel_diameter, 0);

	for (int dy = -kernel_radius; dy < kernel_radius + 1; dy++)
	{
		for (int dx = -kernel_radius; dx < kernel_radius + 1; dx++)
		{
            int distance_squared = dx * dx + dy * dy;
            if (distance_squared > kernel_radius * kernel_radius)
			{
                continue;
			}

            int x = kernel_radius + dx;
            int y = kernel_radius + dy;

            kernel[x + y * kernel_diameter] = 1 / static_cast<float>(distance_squared + 1);
		}
	}

	return kernel;
}

static int get_pair_count(const std::vector<float> &pixels)
{
	int opaque_pixel_count = 0;
	for (size_t i = 3; i < pixels.size(); i += 4)
	{
		if (pixels[i] != 0)
		{
			opaque_pixel_count++;
		}
	}

    // TODO: Get rid of this limitation by introducing x and y start offsets,
    // and alternating them
	assert(opaque_pixel_count % 2 == 0 && "The program currently doesn't support images with an odd number of pixels");

	return opaque_pixel_count / 2;
}

class Args
{
public:
	Args(int argc, char *argv[])
		: seconds_between_saves(1)
		, kernel_radius(100)
		, no_overwriting_output(false)
		, saved_image_leading_zero_count(4)
		, input_npy_path()
		, output_npy_path()
	{
		int c;

		char *program_name = argv[0];

		while (1)
		{
			int option_index = 0;
			static option long_options[] = {
				{"help", no_argument, 0, 'h'},
				{"seconds-between-saves", required_argument, 0, 's'},
				{"kernel-radius", required_argument, 0, 'k'},
				{"no-overwriting-output", no_argument, 0, 'n'},
				{"saved-image-leading-zero-count", required_argument, 0, 'z'},
				{0, 0, 0, 0}};

			c = getopt_long(argc, argv, "hi:s:k:nz:w:", long_options, &option_index);

			// Detect the end of the options
			if (c == -1)
			{
				break;
			}

			switch (c)
			{
			case 'h':
				print_help(program_name);
				exit(EXIT_FAILURE);

			case 's':
				seconds_between_saves = std::stoi(optarg);
				std::cout << "Set seconds_between_saves to " << seconds_between_saves << std::endl;
				break;

			case 'k':
				kernel_radius = std::stoi(optarg);
				std::cout << "Set kernel_radius to " << kernel_radius << std::endl;
				break;

			case 'n':
				no_overwriting_output = true;
				std::cout << "Set no_overwriting_output to " << no_overwriting_output << std::endl;
				break;

			case 'z':
				saved_image_leading_zero_count = std::stoi(optarg);
				std::cout << "Set saved_image_leading_zero_count to " << saved_image_leading_zero_count << std::endl;
				break;

			case '?':
				// getopt_long() already printed an error message
				exit(EXIT_FAILURE);

			default:
				abort();
			}
		}

		// If not exactly two positional arguments were provided
		if (optind != argc - 2)
		{
			print_help(program_name);
			exit(EXIT_FAILURE);
		}

		input_npy_path = argv[argc - 2];
		output_npy_path = argv[argc - 1];
	}

	int seconds_between_saves;
	int kernel_radius;
	bool no_overwriting_output;
	int saved_image_leading_zero_count;
	std::filesystem::path input_npy_path;
	std::filesystem::path output_npy_path;

private:
	void print_help(char *program_name)
	{
		std::cerr
			<< "Usage: " << program_name << " input_npy_path output_npy_path [-h] [-s SECONDS_BETWEEN_SAVES] [-k KERNEL_RADIUS] [-n] [-z SAVED_IMAGE_LEADING_ZERO_COUNT]\n\n";

		std::cerr
			<< "positional arguments:\n"
			"  input_npy_path        Input npy file path, generated by for example rgb2lab.py\n"
			"  output_npy_path       Output npy file path, which can then be used by for example lab2rgb.py\n\n";

		std::cerr
			<< "options:\n"
			"  -h, --help            show this help message and exit\n"
			"  -s SECONDS_BETWEEN_SAVES, --seconds-between-saves SECONDS_BETWEEN_SAVES\n"
			"                        How often the current output image gets saved (default: 1)\n"
			"  -k KERNEL_RADIUS, --kernel-radius KERNEL_RADIUS\n"
			"                        The radius of neighbors that get compared against the current pixel's color; a higher radius means better sorting, but is quadratically slower (default: 100)\n"
			"  -n, --no-overwriting-output\n"
			"                        Save all output images, instead of the default behavior of overwriting (default: False)\n"
			"  -z SAVED_IMAGE_LEADING_ZERO_COUNT, --saved-image-leading-zero-count SAVED_IMAGE_LEADING_ZERO_COUNT\n"
			"                        The number of leading zeros on saved images; this has no effect if the -n switch isn't passed! (default: 4)\n";
	}
};

int main(int argc, char *argv[])
{
	const auto start_time = std::chrono::steady_clock::now();

	std::cout << "Started program" << std::endl;

	Args args(argc, argv);

	cnpy::NpyArray arr = cnpy::npy_load(args.input_npy_path);
	// This won't ever turn into a dangling pointer, since the vector stays a constant size
	std::vector<float> pixels = arr.as_vec<float>();

	int height = arr.shape[0];
	int width = arr.shape[1];

	int kernel_radius = args.kernel_radius;
	int max_kernel_radius = std::max(width, height) - 1;
	kernel_radius = std::min(kernel_radius, max_kernel_radius);
    std::cout << "Using kernel radius " << kernel_radius << std::endl;

	int pair_count = get_pair_count(pixels);
    std::cout << "pair_count is " << pair_count << std::endl;

	std::vector<float> kernel = get_kernel(kernel_radius);

	std::vector<float> neighbor_totals = get_neighbor_totals(pixels, kernel, width, height, kernel_radius);

	std::vector<bool> updated(width * height, 0);

    uint32_t rand1 = 42424242;
    uint32_t rand2 = 69696969;

	uint64_t attempted_swaps = 0;
	uint64_t prev_attempted_swaps = 0;

	int saved_results = 0;

	std::vector<size_t> normal_to_opaque_index_lut = get_normal_to_opaque_index_lut(pixels);

	const std::filesystem::path output_npy_path = get_output_npy_path(
		args.output_npy_path,
		args.no_overwriting_output,
		args.saved_image_leading_zero_count,
		saved_results
	);

	auto last_printed_time = std::chrono::steady_clock::now();

	assert(signal(SIGINT, sigint_handler_running) != SIG_ERR);
	while (running)
	{
		// TODO: Profile whether getting the time here *every single loop* isn't too slow
		const auto now = std::chrono::steady_clock::now();

		if (now > last_printed_time + std::chrono::seconds(args.seconds_between_saves))
		{
			const std::filesystem::path output_npy_path = get_output_npy_path(
				args.output_npy_path,
				args.no_overwriting_output,
				args.saved_image_leading_zero_count,
				saved_results
			);

			save_result(pixels, arr.shape, output_npy_path);
			saved_results += 1;

			print_status(saved_results, prev_attempted_swaps, attempted_swaps, start_time);

			last_printed_time = std::chrono::steady_clock::now();
			prev_attempted_swaps = attempted_swaps;
		}

		// Using unsigned wraparound
		rand1++;

		sort(pixels, neighbor_totals, updated, kernel, normal_to_opaque_index_lut, width, height, rand1, rand2, pair_count, kernel_radius, attempted_swaps);
	}

	save_result(pixels, arr.shape, output_npy_path);
	saved_results += 1;

	print_status(saved_results, prev_attempted_swaps, attempted_swaps, start_time);

	std::cout << "Gootbye" << std::endl;

	return EXIT_SUCCESS;
}
