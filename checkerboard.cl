__kernel void convert(
    read_only image2d_t src,
    write_only image2d_t dest
) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix = read_imageui(src, sampler, pos);

    // A simple test operation: delete pixels to form a checkerboard pattern
    if((get_global_id(0)+((get_global_id(1)+1)%2)) % 2 == 0) {
        pix.x = 0;
        pix.y = 0;
        pix.z = 0;
    }

    write_imageui(dest, pos, pix);
}
