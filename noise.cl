#include <include/Random123/philox.h>

__kernel void grayscale(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	int x = get_global_id(0);
	int y = get_global_id(1);

    int2 pos = (int2)(x, y);

	int width = get_image_width(src);

	philox4x32_ctr_t c={{}};
	philox4x32_ukey_t uk={{}};

	// Seed
	uk.v[0] = 0;

	philox4x32_key_t k = philox4x32keyinit(uk);

    c.v[0] = x;
    c.v[1] = y;
	philox4x32_ctr_t r = philox4x32(c, k);

	uint R = r.v[0] & 255;
	uint G = r.v[1] & 255;
	uint B = r.v[2] & 255;
	uint A = 255;
	uint4 pix = (uint4)(R, G, B, A);

    write_imageui(dest, pos, pix);
}
