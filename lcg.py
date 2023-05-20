def noise_2d(x, y, z):
    fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr)

# def lcg(seed):
#     # a = 1140671485
#     # c = 128201163
#     # m = 2**24
#     a = 1103515245
#     c = 12345
#     m = 0x7FFFFFFF
#     # TODO: Use & instead of % ?
#     return (a * seed + c) % m

# for i in range(16):
#     print(lcg(i) & 255, lcg(i) / 0x7FFFFFFF, lcg(i))

# n = 0
# for i in range(16):
#     n = lcg(n)
#     print(n, n & 255, n / 0x7FFFFFFF)
