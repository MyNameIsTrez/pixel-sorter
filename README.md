# PyOpenCL Color Sorter

Sorts the pixels of an image based on their color.

It does this by repeatedly swapping two random pixels, but only if that'd place them next to more similar neighbor pixel colors.

## Usage

1. Clone this repository
2. `cd` into it
3. Install requirements with `pip install -r requirements.txt`
4. Run it with `py noise.py`

The result is put in the `output/` directory

Modify the `filename` variable at the top of `noise.py` in order to have it load a different `.png` from the `input/` directory.
