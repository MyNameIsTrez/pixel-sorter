#include "cnpy.h"

// C:/ghcup/ghc/8.10.7/mingw/bin/c++.exe -Wall -Wextra -Werror -Wpedantic -Wfatal-errors -std=c++11 main.cpp cnpy.cpp -lz -o a.out && ./a.out
int main()
{
	cnpy::NpyArray arr = cnpy::npy_load("../input_npy/heart.npy");
	float *pixels = arr.data<float>();

	// TODO: Port sort.py its algorithm here

	cnpy::npy_save("../output_npy/heart_0000.npy", pixels, arr.shape, "w");
}
