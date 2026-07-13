"""
Image loading, conversion, preview, and saving utilities.

Images are represented internally as float32 NumPy arrays with values in
[0, 1].
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import io


def load_image(filepath):
    """
    Load an image from disk as an RGB floating-point array.

    Grayscale images are converted to RGB, and alpha channels are removed.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        RGB image with shape (height, width, 3), dtype float32, and values
        in the range [0, 1].
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No such image file: {filepath}")

    image = io.imread(filepath)

    # Normalize ndim to 3 and turn grayscale images to RGB.
    if image.ndim == 2: # grayscale image
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2) # add a new axis and repeat 3 times
    elif image.ndim == 3: # color image
        if image.shape[2] == 4:
            image = image[:, :, :3] # remove alpha channel
        elif image.shape[2] != 3:
            raise ValueError("The image must have 1, 3, or 4 channels.")
    else:
        raise ValueError("Unsupported image dimensions.")

    # Normalize RGB values to lie in [0,1]
    if np.issubdtype(image.dtype, np.integer): # Handle integer image formats such as uint8 and uint16.
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else: # float32, float64
        image = np.clip(image.astype(np.float32), 0.0, 1.0)

    return image


def to_uint8(image):
    """
    Convert a floating-point image to unsigned 8-bit format.

    Parameters
    ----------
    image : numpy.ndarray
        Input image whose values are expected to be in or near [0, 1].

    Returns
    -------
    numpy.ndarray
        Clipped uint8 image with values in [0, 255].
    """
    image = np.asarray(image, dtype=np.float32)
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def make_preview(image, max_size=(600, 450)):
    """
    Create a resized PIL image for display in the Tkinter GUI.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image represented as a NumPy array.

    max_size : tuple of int, optional
        Maximum preview width and height in pixels.

    Returns
    -------
    PIL.Image.Image
        Resized preview image with preserved aspect ratio.
    """
    preview = Image.fromarray(to_uint8(image))
    # Resize the preview using high-quality Lanczos resampling.
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    return preview


def save_image(filepath, image):
    """
    Save an RGB image to disk.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Destination path. The format is inferred from the extension.

    image : numpy.ndarray
        Floating-point image to save.
    """
    io.imsave(Path(filepath), to_uint8(image))
