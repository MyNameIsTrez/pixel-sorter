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
			y + other.y};
	}
};

struct lab
{
	uint16_t l;
	uint16_t a;
	uint16_t b;

	// TODO: This can't return lab anymore, since it still assumes it's a float3
	// lab operator*(float multiplier)
	// {
	// 	return {
	// 		l * multiplier,
	// 		a * multiplier,
	// 		b * multiplier};
	// }

	// TODO: Remove?
	// lab operator/(float divisor)
	// {
	// 	return {
	// 		static_cast<uint16_t>(l / divisor),
	// 		static_cast<uint16_t>(a / divisor),
	// 		static_cast<uint16_t>(b / divisor)};
	// }

	lab operator+=(const lab &other)
	{
		l += other.l;
		a += other.a;
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
static void print_neighbor_totals_error(const std::vector<double> &neighbor_totals_error)
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

static lab get_pixel(const std::vector<uint16_t> &pixels, xy pos, int width)
{
	size_t i = get_index(pos, width) * 4;

	uint16_t l = pixels[i + 0];
	uint16_t a = pixels[i + 1];
	uint16_t b = pixels[i + 2];

	return {l, a, b};
}

static void set_pixel(std::vector<uint16_t> &pixels, xy pos, int width, lab lab)
{
	size_t i = get_index(pos, width) * 4;

	pixels[i + 0] = lab.l;
	pixels[i + 1] = lab.a;
	pixels[i + 2] = lab.b;
}

static void update_neighbors(std::vector<uint64_t> &neighbor_totals, xy center, lab old_pixel, lab new_pixel, int width, int height, int kernel_radius)
{
	// TODO: By padding the input image it should be possible to get rid of these bounds variables
	int dy_min = -std::min(center.y, kernel_radius);
	int dy_max = std::min(height - 1 - center.y, kernel_radius);

	int dx_min = -std::min(center.x, kernel_radius);
	int dx_max = std::min(width - 1 - center.x, kernel_radius);

	for (int dy = dy_min; dy <= dy_max; dy++)
	{
		for (int dx = dx_min; dx <= dx_max; dx++)
		{
			int distance_squared = dx * dx + dy * dy;
			if (distance_squared > kernel_radius * kernel_radius)
			{
				continue;
			}

			xy neighbor = {center.x + dx, center.y + dy};

			assert(neighbor_totals[get_index(neighbor, width) * 4 + 0] >= old_pixel.l);
			assert(neighbor_totals[get_index(neighbor, width) * 4 + 1] >= old_pixel.a);
			assert(neighbor_totals[get_index(neighbor, width) * 4 + 2] >= old_pixel.b);

			// Replace an old pixel with a new pixel in neighbor_totals
			neighbor_totals[get_index(neighbor, width) * 4 + 0] += -old_pixel.l + new_pixel.l;
			neighbor_totals[get_index(neighbor, width) * 4 + 1] += -old_pixel.a + new_pixel.a;
			neighbor_totals[get_index(neighbor, width) * 4 + 2] += -old_pixel.b + new_pixel.b;
		}
	}
}

static double get_color_difference(lab pixel, lab neighbor_pixel)
{
	double l_diff = pixel.l - neighbor_pixel.l;
	double a_diff = pixel.a - neighbor_pixel.a;
	double b_diff = pixel.b - neighbor_pixel.b;

	return (
		l_diff * l_diff +
		a_diff * a_diff +
		b_diff * b_diff);
}

static lab get_neighbor_average(const std::vector<uint64_t> &neighbor_totals, const std::vector<float> &neighbor_counts, xy pos, int width)
{
	size_t i = get_index(pos, width) * 4;

	uint64_t l = neighbor_totals[i + 0];
	uint64_t a = neighbor_totals[i + 1];
	uint64_t b = neighbor_totals[i + 2];

	float c = neighbor_counts[get_index(pos, width)];

	// TODO: Profile whether it's worth it to cache this division result in a new vector
	return {
		static_cast<uint16_t>(l / c),
		static_cast<uint16_t>(a / c),
		static_cast<uint16_t>(b / c)};
}

static bool should_swap(const std::vector<uint64_t> &neighbor_totals, const std::vector<float> &neighbor_counts, lab pixel1, lab pixel2, xy pos1, xy pos2, int width)
{
	lab i1_neighbor_average = get_neighbor_average(neighbor_totals, neighbor_counts, pos1, width);
	double i1_old_score = get_color_difference(pixel1, i1_neighbor_average);
	double i1_new_score = get_color_difference(pixel2, i1_neighbor_average);
	double i1_score_difference = -i1_old_score + i1_new_score;

	lab i2_neighbor_average = get_neighbor_average(neighbor_totals, neighbor_counts, pos2, width);
	double i2_old_score = get_color_difference(pixel2, i2_neighbor_average);
	double i2_new_score = get_color_difference(pixel1, i2_neighbor_average);
	double i2_score_difference = -i2_old_score + i2_new_score;

	double score_difference = i1_score_difference + i2_score_difference;

	return score_difference < 0;
}

static xy get_pos(int i, int width)
{
	return {i % width, i / width};
}

static uint64_t round_up_to_power_of_2(uint64_t n)
{
	// If n isn't a power of 2 already
	if (n & (n - 1))
	{
		uint64_t i;

		// TODO: Can "1ull" be replaced with "1" everywhere here?

		// Count the number of times n can be right-shifted
		for (i = 0; n > 1; i++)
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

static std::vector<uint64_t> get_neighbor_totals(const std::vector<uint16_t> &pixels, int width, int height, int kernel_radius)
{
	std::vector<uint64_t> neighbor_totals(pixels.size(), 0);

	// For every pixel
	for (int py = 0; py < height; py++)
	{
		for (int px = 0; px < width; px++)
		{
			// TODO: By padding the input image it should be possible to get rid of these bounds variables
			int kdy_min = -std::min(py, kernel_radius);
			int kdy_max = std::min(height - 1 - py, kernel_radius);

			int kdx_min = -std::min(px, kernel_radius);
			int kdx_max = std::min(width - 1 - px, kernel_radius);

			// Apply the kernel
			for (int kdy = kdy_min; kdy <= kdy_max; kdy++)
			{
				for (int kdx = kdx_min; kdx <= kdx_max; kdx++)
				{
					int distance_squared = kdx * kdx + kdy * kdy;
					if (distance_squared > kernel_radius * kernel_radius)
					{
						continue;
					}

					int x = px + kdx;
					int y = py + kdy;

					uint16_t l = pixels[(x + y * width) * 4 + 0];
					uint16_t a = pixels[(x + y * width) * 4 + 1];
					uint16_t b = pixels[(x + y * width) * 4 + 2];

					neighbor_totals[(px + py * width) * 4 + 0] += l;
					neighbor_totals[(px + py * width) * 4 + 1] += a;
					neighbor_totals[(px + py * width) * 4 + 2] += b;
				}
			}
		}
	}

	return neighbor_totals;
}

#ifdef DEBUG
static std::vector<double> get_neighbor_totals_error(const std::vector<uint64_t> &neighbor_totals, const std::vector<uint16_t> &pixels, int width, int height, int kernel_radius)
{
	std::vector<uint64_t> actual_neighbor_totals = get_neighbor_totals(pixels, width, height, kernel_radius);
	// TODO: Turn into a one-liner with C++ magic
	std::vector<double> neighbor_totals_error;
	for (size_t i = 0; i < neighbor_totals.size(); i++)
	{
		neighbor_totals_error.push_back(neighbor_totals[i] - static_cast<double>(actual_neighbor_totals[i]));
	}
	return neighbor_totals_error;
}
#endif

static void sort(std::vector<uint16_t> &pixels, std::vector<uint64_t> &neighbor_totals, const std::vector<float> &neighbor_counts, const std::vector<size_t> &normal_to_opaque_index_lut, int width, int height, uint32_t rand1, uint32_t rand2, int pair_count, int kernel_radius, uint64_t &attempted_swaps)
{
	int opaque_pixel_count = pair_count * 2;

#ifdef DEBUG
	uint64_t swaps = 0;
#endif

	// TODO: Use a C++ parallel-loop here, to get it closer to sort.cl
	for (int i1 = 0; i1 < pair_count; i1 += 2)
	{
#ifdef DEBUG
		std::vector<double> neighbor_totals_error = get_neighbor_totals_error(neighbor_totals, pixels, width, height, kernel_radius);
		bool no_error = std::all_of(neighbor_totals_error.begin(), neighbor_totals_error.end(), [](double d)
									{ return d == 0.0f; });
		if (!no_error)
		{
			std::cout << "There were " << swaps << " swaps in " << attempted_swaps << " attempted swaps" << std::endl;
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

		lab pixel1 = get_pixel(pixels, pos1, width);
		lab pixel2 = get_pixel(pixels, pos2, width);

		if (should_swap(neighbor_totals, neighbor_counts, pixel1, pixel2, pos1, pos2, width))
		{
			set_pixel(pixels, pos1, width, pixel2);
			update_neighbors(neighbor_totals, pos1, pixel1, pixel2, width, height, kernel_radius);

			set_pixel(pixels, pos2, width, pixel1);
			update_neighbors(neighbor_totals, pos2, pixel2, pixel1, width, height, kernel_radius);

#ifdef DEBUG
			swaps++;
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

static void save_result(const std::vector<uint16_t> &pixels, const std::vector<size_t> &shape, const std::filesystem::path &output_npy_path)
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

static std::vector<size_t> get_normal_to_opaque_index_lut(const std::vector<uint16_t> &pixels)
{
	std::vector<size_t> normal_to_opaque_index_lut;

	size_t offset = 0;
	for (size_t i = 3; i < pixels.size(); i += 4)
	{
		uint16_t alpha = pixels[i];

		// TODO: Check that this produces the same result as sort.py's version of this code
		if (alpha != 0)
		{
			normal_to_opaque_index_lut.push_back(offset);
		}

		offset += 1;
	}

	return normal_to_opaque_index_lut;
}

// Returns a vector of floats instead of ints,
// since these values will be used for float division
static std::vector<float> get_neighbor_counts(int kernel_radius, size_t pixels_size, int width, int height)
{
	std::vector<float> neighbor_counts(pixels_size, 0);

	// For every pixel
	for (int py = 0; py < height; py++)
	{
		for (int px = 0; px < width; px++)
		{
			// TODO: By padding the input image it should be possible to get rid of these bounds variables
			int kdy_min = -std::min(py, kernel_radius);
			int kdy_max = std::min(height - 1 - py, kernel_radius);

			int kdx_min = -std::min(px, kernel_radius);
			int kdx_max = std::min(width - 1 - px, kernel_radius);

			// Apply the kernel
			for (int kdy = kdy_min; kdy <= kdy_max; kdy++)
			{
				for (int kdx = kdx_min; kdx <= kdx_max; kdx++)
				{
					int distance_squared = kdx * kdx + kdy * kdy;
					if (distance_squared > kernel_radius * kernel_radius)
					{
						continue;
					}

					int x = px + kdx;
					int y = py + kdy;

					// This doesn't work, since neighbor_totals stores the original uint16s added together,
					// so doesn't work with fractions of lab values. Otherwise we can't subtract and add
					// to it without losing more and more precision the longer the program runs!
					// neighbor_counts[x + y * width] += 1 / static_cast<float>(distance_squared + 1);

					neighbor_counts[x + y * width]++;
				}
			}
		}
	}

	return neighbor_counts;
}

static int get_pair_count(const std::vector<uint16_t> &pixels)
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
	if (opaque_pixel_count % 2 != 0)
	{
		std::cerr << "The program currently doesn't support images with an odd number of pixels" << std::endl;
		exit(EXIT_FAILURE);
	}

	return opaque_pixel_count / 2;
}

class Args
{
public:
	Args(int argc, char *argv[])
		: seconds_between_saves(1), kernel_radius(100), no_overwriting_output(false), saved_image_leading_zero_count(4), input_npy_path(), output_npy_path()
	{
		int c;

		char *program_name = argv[0];

		while (true)
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
	std::vector<uint16_t> pixels = arr.as_vec<uint16_t>();

	int height = arr.shape[0];
	int width = arr.shape[1];

	int kernel_radius = args.kernel_radius;
	int max_kernel_radius = std::max(width, height) - 1;
	kernel_radius = std::min(kernel_radius, max_kernel_radius);
	std::cout << "Using kernel radius " << kernel_radius << std::endl;

	int pair_count = get_pair_count(pixels);
	std::cout << "pair_count is " << pair_count << std::endl;

	std::vector<uint64_t> neighbor_totals = get_neighbor_totals(pixels, width, height, kernel_radius);

	const std::vector<float> neighbor_counts = get_neighbor_counts(kernel_radius, pixels.size(), width, height);

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
		saved_results);

	auto last_printed_time = std::chrono::steady_clock::now();

	if (signal(SIGINT, sigint_handler_running) == SIG_ERR)
	{
		abort();
	}

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
				saved_results);

			save_result(pixels, arr.shape, output_npy_path);
			saved_results += 1;

			print_status(saved_results, prev_attempted_swaps, attempted_swaps, start_time);

			last_printed_time = std::chrono::steady_clock::now();
			prev_attempted_swaps = attempted_swaps;
		}

		// Using unsigned wraparound
		rand1++;

		sort(pixels, neighbor_totals, neighbor_counts, normal_to_opaque_index_lut, width, height, rand1, rand2, pair_count, kernel_radius, attempted_swaps);
	}

	save_result(pixels, arr.shape, output_npy_path);
	saved_results += 1;

	print_status(saved_results, prev_attempted_swaps, attempted_swaps, start_time);

	std::cout << "Gootbye" << std::endl;

	return EXIT_SUCCESS;
}
