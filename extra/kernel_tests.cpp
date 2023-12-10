#include <iostream>
#include <vector>

// c++ -Wall -Wextra -Werror -Wpedantic -Wfatal-errors -fsanitize=address,undefined -g -std=c++17 kernel_tests.cpp -o a.out && ./a.out
int main(void)
{
	std::vector<float> pixels = {
		1.0, 2.0,  1.0, 2.0,
		1.0, 2.0,  1.0, 2.0,
	};
	std::cout << "pixels:" << std::endl;
	for (float f : pixels)
	{
		std::cout << f << " ";
	}
	std::cout << std::endl;

	std::vector<float> kernel = {
        0.5, 1.0, 0.5,
        1.0, 0.0, 1.0,
        0.5, 1.0, 0.5,
	};
	std::cout << "kernel:" << std::endl;
	for (float f : kernel)
	{
		std::cout << f << " ";
	}
	std::cout << std::endl;

	std::vector<float> expected = {
        2.5, 5.0,  2.5, 5.0,
        2.5, 5.0,  2.5, 5.0,
	};
	std::cout << "expected:" << std::endl;
	for (float f : expected)
	{
		std::cout << f << " ";
	}
	std::cout << std::endl;

	int width = 2;
	int height = 2;
	int kernel_radius = 1;
    int kernel_diameter = kernel_radius * 2 + 1;
	std::vector<float> actual(pixels.size(), 0);
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

					actual.at((x + y * width) * 2) += pr * k;
					actual.at((x + y * width) * 2 + 1) += pg * k;
				}
			}
		}
	}
	std::cout << "actual:" << std::endl;
	for (float f : actual)
	{
		std::cout << f << " ";
	}
	std::cout << std::endl;
}
