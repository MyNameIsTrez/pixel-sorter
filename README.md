# PyOpenCL Color Sorter

Sorts the pixels of an image based on their color.

In other words, it blurs an image while retaining all of the original pixels:

<p><img src="media/blurry_elephant.png" alt="Half is the input toy elephant and the other half is the blurry output toy elephant."></p>

It does this by repeatedly attempting to swap two random pixels, only doing so if that'd place them next to pixel neighbors with more similar colors.

At the heart of the program lies my port of [CUDA-Shuffle](https://github.com/djns99/CUDA-Shuffle)'s `PhiloxBijectiveFunction` shuffling class.

## Usage

1. Clone this repository.
2. `cd` into it.
3. Install requirements with `pip install -r requirements.txt`
4. Run it with `py shuffle.py`

The resulting `.png` is put in the `output/` directory.

Modify the `filename` variable at the top of `shuffle.py` in order to have it load a different `.png` from the `input/` directory.

Modify the `iteration_count` variable at the top of `shuffle.py` in order to have the program sort for longer.

Modify the `MODE` variable at the top of `shuffle.cl` to either `LCG` or `PHILOX`, to switch between shuffling algorithms.
