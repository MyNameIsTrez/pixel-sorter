// #include <include/Random123/philox.h>

#include "foo.h"

__kernel void grayscale(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	int x = get_global_id(0);
	int y = get_global_id(1);

    int2 pos = (int2)(x, y);

	int width = get_image_width(src);

	uint4 pix = (uint4)(

		XD, 255, 255, 255);
	// uint4 pix = (uint4)(lcg(y * width + x) / (float)0x7fffffff * 255, 255, 255, 255);
	// uint4 pix = (uint4)(lcg(y * width + x) & 255, 255, 255, 255);
	// printf("x: %d, y: %d, width: %d, result: %d\n", x, y, width, pix.x);

    write_imageui(dest, pos, pix);
}
