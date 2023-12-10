#include "cnpy.h"

#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <signal.h>

static volatile sig_atomic_t running = true;
static void sigint_handler_running(int signum)
{
	(void)signum;
	running = false;
}

uint64_t round_up_to_power_of_2(uint64_t n)
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

uint64_t lcg(uint64_t capacity, uint64_t val, uint32_t multiplier_rand, uint32_t addition_rand)
{
	uint64_t modulus = round_up_to_power_of_2(capacity);

	// Must be odd so it is coprime to modulus
	uint64_t multiplier = (multiplier_rand * 2 + 1) % modulus;

	uint64_t addition = addition_rand % modulus;

	// Modulus must be power of two
	assert((modulus & (modulus - 1)) == 0);

	return ((val * multiplier) + addition) & (modulus - 1);
}

int get_shuffled_index(int i, uint32_t rand1, uint32_t rand2, int opaque_pixel_count)
{
	assert(i < opaque_pixel_count);

	int shuffled = i;

	do
	{
		shuffled = lcg(opaque_pixel_count, shuffled, rand1, rand2);
	} while (shuffled >= opaque_pixel_count);

	return shuffled;
}

void sort(const std::vector<float> &pixels, const std::vector<float> &neighbor_totals, const std::vector<bool> &updated, const std::vector<float> &kernel, const std::vector<size_t> &normal_to_opaque_index_lut, uint32_t rand1, uint32_t rand2, int pair_count, uint64_t &attempted_swaps)
{
	// TODO: Remove these
	(void)pixels;
	(void)neighbor_totals;
	(void)updated;
	(void)kernel;
	(void)normal_to_opaque_index_lut;
	(void)rand1;
	(void)rand2;

	int opaque_pixel_count = pair_count * 2;

	// TODO: Use a C++ parallel-loop here, to get it closer to sort.cl
	for (int i1 = 0; i1 < pair_count; i1 += 2)
	{
		int i2 = i1 + 1;

		int shuffled_i1 = get_shuffled_index(i1, rand1, rand2, opaque_pixel_count);
		int shuffled_i2 = get_shuffled_index(i2, rand1, rand2, opaque_pixel_count);

		// TODO: Remove these
		(void)shuffled_i1;
		(void)shuffled_i2;

		attempted_swaps++;
	}
}

std::string humanize_uint64(uint64_t n)
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
	else
	{
		double d = n / 1'000'000'000.0;
		ss << d;
		return ss.str() + " billion";
	}
}

void print_status(int saved_results, uint64_t prev_attempted_swaps, uint64_t attempted_swaps, const std::chrono::steady_clock::time_point &start_time)
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

void save_result(const std::vector<float> &pixels, const std::vector<size_t> &shape, const std::filesystem::path &output_npy_path)
{
	cnpy::npy_save(output_npy_path, pixels.data(), shape, "w");
}

std::filesystem::path get_output_npy_path(
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

std::vector<size_t> get_normal_to_opaque_index_lut(const std::vector<float> &pixels)
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

std::vector<float> get_neighbor_totals(const std::vector<float> &pixels, const std::vector<float> &kernel, int width, int height, int kernel_radius)
{
	std::vector<float> neighbor_totals(pixels.size(), 0);

	int kernel_diameter = kernel_radius * 2 + 1;

	// Play around with extra/kernel_tests.cpp to see how this convolving works
	// For every pixel
	for (int py = 0; py < height; py++)
	{
		for (int px = 0; px < width; px++)
		{
			float pr = pixels.at((px + py * width) * 2);
			float pg = pixels.at((px + py * width) * 2 + 1);

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
					float k = kernel.at(kx + ky * kernel_diameter);

					neighbor_totals.at((x + y * width) * 2) += pr * k;
					neighbor_totals.at((x + y * width) * 2 + 1) += pg * k;
				}
			}
		}
	}

	return neighbor_totals;
}

// TODO: Profile whether it's faster to just recreate the kernel on the fly,
// TODO: since we still need these same loops to loop over it anyways!
std::vector<float> get_kernel(int kernel_radius)
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

            kernel.at(x + y * kernel_diameter) = 1 / static_cast<float>(distance_squared + 1);
		}
	}

	return kernel;
}

int get_pair_count(const std::vector<float> &pixels)
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

// c++ -Wall -Wextra -Werror -Wpedantic -Wfatal-errors -fsanitize=address,undefined -g -std=c++17 main.cpp cnpy.cpp -lz -o a.out && ./a.out "../input_npy/heart.npy" "../output_npy/heart.npy"
// ./a.out "../input_npy/heart.npy" "../output_npy/heart.npy"
int main(int argc, char *argv[])
{
	const auto start_time = std::chrono::steady_clock::now();

	std::cout << "Started program" << std::endl;

	Args args(argc, argv);

	cnpy::NpyArray arr = cnpy::npy_load(args.input_npy_path);
	// This won't ever turn into a dangling pointer, since the vector stays a constant size
	std::vector<float> pixels = arr.as_vec<float>();

	int width = arr.shape.at(0);
	int height = arr.shape.at(1);

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

		sort(pixels, neighbor_totals, updated, kernel, normal_to_opaque_index_lut, rand1, rand2, pair_count, attempted_swaps);
	}

	save_result(pixels, arr.shape, output_npy_path);
	saved_results += 1;

	print_status(saved_results, prev_attempted_swaps, attempted_swaps, start_time);

	std::cout << "Gootbye" << std::endl;

	return EXIT_SUCCESS;
}
