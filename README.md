# Pyramid Codec

**Created by Mohammadreza Abolhassani**

## Overview

Pyramid Codec is a Python application that compresses and reconstructs color images using a **Laplacian image pyramid**. Instead of storing every level of the image pyramid, the codec stores only the smallest image together with the Laplacian residual images required to reconstruct the original image.

The project includes a Tkinter graphical interface that allows the user to:

- Load a color image
- Choose the number of pyramid levels
- Select the Gaussian kernel size and sigma
- Compress and save the image into a custom archive
- Open a previously compressed archive
- Reconstruct the image
- Save the reconstructed image

## Pipeline

1. Load the RGB image.
2. Generate a Gaussian kernel.
3. Blur the image to reduce aliasing.
4. Downsample the image by a factor of two.
5. Expand the smaller image back to its previous resolution.
6. Compute the Laplacian residual (difference image).
7. Repeat for the requested number of levels.
8. Quantize residuals and save them together with the smallest pyramid image.

For reconstruction:

1. Load the smallest stored image.
2. Expand it to the next pyramid level.
3. Add the stored residual.
4. Repeat until the original resolution is reached.

## Project Structure

```
src/
    main.py
    gui.py
    pyramid_codec.py
    filters.py
    image_utils.py
```

## Technologies

- Python
- NumPy
- SciPy
- Pillow
- scikit-image
- Tkinter

## Running

```bash
pip install numpy scipy pillow scikit-image
python main.py
```

The project demonstrates multiscale image compression using Gaussian filtering, Laplacian residuals, quantization, and image reconstruction.
