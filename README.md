# Pixel Sorter

Sorts the pixels of an image based on their color.

Before vs after:
<p>
	<img src="media/palette_input.png" alt="This input palette isn't very sorted by color.">
	<img src="media/palette_output.png" alt="This output palette is pretty much optimally sorted by color.">
</p>

Here's the `before` palette pasted 4 times and sorted into the shape of a heart:

https://github.com/MyNameIsTrez/pixel-sorter/assets/32989873/3a71be22-17b2-4e5f-8d89-183eb4c8907f

Here's the `before` palette pasted 64 times and sorted:

<img src="media/palette_output_large.png" alt="This large output palette is color sorted.">

It essentially blurs an image, while retaining all of the original pixels:

<p><img src="media/blurry_elephant.png" alt="Half is the input toy elephant and the other half is the blurry output toy elephant."></p>

## How it works

It repeatedly attempts to swap two random pixels, only doing the swap if that'd place them next to pixels with more similar colors.

At the heart of the program lies my PyOpenCL port of [CUDA-Shuffle](https://github.com/djns99/CUDA-Shuffle)'s `LCGBijectiveFunction` and `PhiloxBijectiveFunction` shuffling classes.

Pixels are first converted from RGB to the CIELAB color space, in order to make later pixel color comparisons more accurate to how the human eye works. See [this Wikipedia article](https://en.wikipedia.org/wiki/Color_difference) on color difference:

> Uniform color space: a color space in which equivalent numerical differences represent equivalent visual differences, regardless of location within the color space.

Here's the same heart video from before, but with color comparisons done in RGB. The ugly green splotches are most noticeable:

https://github.com/MyNameIsTrez/pixel-sorter/assets/32989873/e36952c7-fbaf-4745-ad10-cd145d844d64

## Usage

1. Clone this repository.
2. `cd` into it.
3. Install requirements with `pip install -r requirements.txt`
4. Run `python sort.py -h` to see how the program is used

## How to turn the output images into videos

### webm
`ffmpeg -framerate 1 -i output/elephant_%04d.png -crf 0 -s 1024x662 -sws_flags neighbor -r 30 output/output.webm`

### mp4 (bigger files but more widely supported)
`ffmpeg -framerate 1 -i output/elephant_%04d.png -crf 0 -s 1024x662 -sws_flags neighbor -c:v libx264 -pix_fmt yuv420p -r 30 output/output.mp4`

### gif

`ffmpeg -framerate 10.0 -i local/media/gifs/%1d.png -s 160x160 -sws_flags neighbor -r 30 output/gif.gif`
