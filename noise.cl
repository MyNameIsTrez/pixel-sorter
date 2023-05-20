// //noise values in range if 0.0 to 1.0
// float noise3D(float x, float y, float z) {
//     float iptr = 0.0f;
//     return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &iptr);
// }

// void fillRandom(float seed, __global float* buffer, int length) {
//     int gi = get_global_id(0);
//     float fgi = float(gi)/length;
//     buffer[gi] = noise3D(fgi, 0.0f, seed);
// }

// Source:
// https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/random_r.c;hb=glibc-2.26#l362
uint lcg(uint seed)
{
	uint a = 1103515245;
	uint c = 12345;
	uint m = 0x7fffffff;
    return (a * seed + c) & m;
}

__kernel void grayscale(
    read_only image2d_t src,
    write_only image2d_t dest
) {
	int x = get_global_id(0);
	int y = get_global_id(1);

    int2 pos = (int2)(x, y);

	int width = get_image_width(src);

	// uint4 pix = (uint4)(

	// 	, 255, 255, 255);
	uint4 pix = (uint4)(lcg(x) / (float)0x7fffffff * 255, 255, 255, 255);
	// uint4 pix = (uint4)(lcg(y * width + x) & 255, 255, 255, 255);
	// printf("x: %d, y: %d, width: %d, result: %d\n", x, y, width, pix.x);

    write_imageui(dest, pos, pix);
}
