#include <include/Random123/philox.h>

__kernel void grayscale(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	// int gid = get_global_id(0);
	// int i1 = gid * 2;
	// int i2 = i1 + 1;
	int x = get_global_id(0);
	int y = get_global_id(1);

	// printf("x: %d, y: %d", x, y);

    int2 pos = (int2)(x, y);

	// int width = get_image_width(src);

	philox2x32_ctr_t c={{}};
	philox2x32_ukey_t uk={{}};

	// Seed
	uk.v[0] = 0;

	philox2x32_key_t k = philox2x32keyinit(uk);

    c.v[0] = x;
    c.v[1] = y;
    // c.v[0] = gid;
	philox2x32_ctr_t r = philox2x32(c, k);
	// printf("i1: %d, i2: %d, r.v[0]: %d\n", i1, i2, r.v[0]);

	uint R = r.v[0] & 255;
	uint G = r.v[0] & 255;
	uint B = r.v[0] & 255;
	uint A = 255;
	uint4 pix = (uint4)(R, G, B, A);

    write_imageui(dest, pos, pix);
}
