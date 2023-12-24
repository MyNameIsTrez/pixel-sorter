# Pixel Sorter

Sorts the pixels of an image based on their color.

Before vs after:
<p>
	<img src="media/palette_input.png" alt="This input palette isn't very sorted by color.">
	<img src="media/palette_output.png" alt="This output palette is pretty much optimally sorted by color.">
</p>

Here's the `before` palette pasted 4 times into the shape of a heart, and then sorted:

https://github.com/MyNameIsTrez/pixel-sorter/assets/32989873/3a71be22-17b2-4e5f-8d89-183eb4c8907f

Here's the `before` palette pasted 64 times and sorted:

<img src="media/palette_output_large.png" alt="This large output palette is color sorted.">

It essentially blurs an image, while retaining all of the original pixels:

<p><img src="media/blurry_elephant.png" alt="Half is the input toy elephant and the other half is the blurry output toy elephant."></p>

## How it works

It repeatedly attempts to swap two random pixels, only doing the swap if that'd place them next to pixels with more similar colors.

At the heart of the program lies my PyOpenCL port of [CUDA-Shuffle](https://github.com/djns99/CUDA-Shuffle)'s `LCGBijectiveFunction` shuffling class.

Pixels are first converted from RGB to the CIELAB color space, in order to make later pixel color comparisons more accurate to how the human eye works. See [this Wikipedia article](https://en.wikipedia.org/wiki/Color_difference) on color difference:

> Uniform color space: a color space in which equivalent numerical differences represent equivalent visual differences, regardless of location within the color space.

Here's the same heart video from before, but with color comparisons done in RGB. The ugly green splotches everywhere are most noticeable:

https://github.com/MyNameIsTrez/pixel-sorter/assets/32989873/e36952c7-fbaf-4745-ad10-cd145d844d64

## Usage

You need to install OpenCL if you don't have it already, but this program should let you know if you don't have it installed yet, so follow these steps regardless:

1. Clone this repository.
2. `cd` into it.
3. Install requirements with `pip install -r requirements.txt`
4. Run `python sort.py -h` to see how the program is used.

Press Ctrl+C once to stop the program. Look at the output image while it's running to decide whether you are satisfied with how sorted the result is.

## Other included programs

If you open this repository in VS Code, you can launch and configure these programs using the `.vscode/launch.json` file.

### verify.py

Verifies that the color counts of the input and output image are identical.

If the color counts aren't identical and you started the program with VS Code's Python debugger, the VS Code `Run and Debug` view on the left allows you to inspect the colors and counts of the input and output image.

### fill_mask.py

Puts the opaque pixels of an input image into the white pixels of an input mask, and writes the result to an output image.

The heart shape was created using this program, from `input/heart.png` and the heart-shaped mask `masks/heart_1024.png`.

### shuffle.py

Shuffles the opaque pixels of an input image, and writes the result to an output image.

### rgb2lab.py and lab2rgb.py

You'd think doing `magick input/heart.png -colorspace Lab output/heart_magick.tif`, then `magick output/heart_magick.tif output/heart_magick.png`, and finally `python verify.py input/heart.png output/heart_magick.png` would work. It doesn't however, which is because the `.tif` file stores the Lab values as integers, rather than floats.

So `rgb2lab.py` converts an RGB image to Lab floating-point values and writes it to a numpy `.npy` binary file, and `lab2rgb.py` unpacks that back to an RGB image. The purpose of this binary file is that the C++ implementation `sort.cpp`, and any other implementation in any other language, doesn't need to go through the hassle of including and calling an LAB library.

Example usage:
1. `python rgb2lab.py input/heart.png output/heart_rgb2lab.npy`
2. `python lab2rgb.py output/heart_rgb2lab.npy output/heart_lab2rgb.png`
3. `python verify.py input/heart.png output/heart_lab2rgb.png`

## How to turn the output images into videos

### webm
`ffmpeg -framerate 1 -i output/elephant_%04d.png -crf 0 -s 1024x662 -sws_flags neighbor -r 30 output/output.webm`

### mp4

mp4 files are larger and of lower quality than webm files, but are more widely supported.

`ffmpeg -framerate 1 -i output/elephant_%04d.png -crf 0 -s 1024x662 -sws_flags neighbor -c:v libx264 -pix_fmt yuv420p -r 30 output/output.mp4`

### gif

`ffmpeg -framerate 10.0 -i local/media/gifs/%1d.png -s 160x160 -sws_flags neighbor -r 30 output/gif.gif`

## Profiling cpp/main.cpp

Clone [FlameGraph](https://github.com/brendangregg/FlameGraph), open a terminal in that directory, and run these commands:

```
sudo perf record -F 99 -p $(pgrep a.out) -g -- sleep 10 &&
sudo perf script > out.perf &&
./stackcollapse-perf.pl out.perf > out.folded &&
./flamegraph.pl out.folded > kernel.svg
```

Open `kernel.svg` in your browser and click around.

Generate an image with `python3 lab2rgb.py output_npy/big_palette.npy output/big_palette.png`

You can view the generated assembly with Compiler Exampler [here](https://godbolt.org/z/Tjbrqn6b6).

`perf top -a` is an excellent sampler for finding assembly instructions that take the biggest percentage of the program's time, and it even shows the original lines alongside the assembly!
