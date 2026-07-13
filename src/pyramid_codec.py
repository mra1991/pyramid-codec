"""
Laplacian pyramid compression and reconstruction algorithms.

The module repeatedly applies Gaussian filtering and factor-of-two
subsampling. At each level it stores a Laplacian residual containing the
detail lost during downsampling. Residuals and the final small image are
saved in a compressed NumPy archive.
"""

from pathlib import Path

import numpy as np

from filters import gaussian_kernel, fast_filter2d


ARCHIVE_VERSION = 1
RESIDUAL_MAX = np.iinfo(np.int16).max


def downsample_image(image, kernel):
    """
    Blur an image and downsample it by a factor of two.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or color floating-point image.

    kernel : numpy.ndarray
        Normalized 2D Gaussian kernel.

    Returns
    -------
    numpy.ndarray
        Anti-aliased image sampled at every second row and column.
    """
    blurred = fast_filter2d(image, kernel)
    return blurred[::2, ::2, ...] # skip every second row and column.


def expand_image(image, target_shape, kernel):
    """
    Expand an image to a specified shape using zero insertion and filtering.

    Parameters
    ----------
    image : numpy.ndarray
        Smaller image to expand.

    target_shape : tuple of int
        Desired output shape.

    kernel : numpy.ndarray
        Gaussian kernel used during expansion.

    Returns
    -------
    numpy.ndarray
        Expanded image with exactly target_shape.
    """
    target_shape = tuple(int(value) for value in target_shape) # convert target shape to a regular tuple
    if image.ndim != len(target_shape): # validate target dimensions
        raise ValueError("Input image and target shape must have equal rank.")

    expanded = np.zeros(target_shape, dtype=np.float32)
    expected = expanded[::2, ::2, ...].shape

    if image.shape != expected:
        raise ValueError(
            f"Input shape {image.shape} is incompatible with target {target_shape}."
        )

    expanded[::2, ::2, ...] = image
    
    # Gaussian filtering interpolates the inserted zeros. Multiplying by four
    # compensates for the reduced two-dimensional sample density.
    return fast_filter2d(expanded, kernel) * 4.0 


def build_laplacian_pyramid(image, levels, kernel):
    """
    Build a Laplacian pyramid from an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image convertible to float32.

    levels : int
        Number of downsampling levels.

    kernel : numpy.ndarray
        Gaussian kernel used for reduction and expansion.

    Returns
    -------
    residuals : list of numpy.ndarray
        Residual images ordered from largest to smallest.

    smallest : numpy.ndarray
        Final low-resolution pyramid image.
    """
    if levels < 1:
        raise ValueError("The number of levels must be at least 1.")

    current = np.asarray(image, dtype=np.float32)
    residuals = []

    for _ in range(levels):
        if min(current.shape[:2]) < 2:
            raise ValueError("Too many levels for this image.")

        # smooth and downsample image
        smaller = downsample_image(current, kernel)
        # Expand the smaller image and filter it to form an approximation
        # of the current pyramid level.
        approximation = expand_image(smaller, current.shape, kernel)
        # the residual image is the difference between the current image and the approximation
        residuals.append(current - approximation)
        # do the same for the downsampled image until you reach the specified level
        current = smaller

    return residuals, current


def reconstruct_image(residuals, smallest, kernel):
    """
    Reconstruct an image from a Laplacian pyramid.

    Parameters
    ----------
    residuals : sequence of numpy.ndarray
        Residual images ordered from largest to smallest.

    smallest : numpy.ndarray
        Final low-resolution pyramid image.

    kernel : numpy.ndarray
        Gaussian kernel used during pyramid construction.

    Returns
    -------
    numpy.ndarray
        Reconstructed image clipped to [0, 1].
    """
    current = np.asarray(smallest, dtype=np.float32)

    # for i-th level image: G(i) = Expand(G(i+1)) + L(i)
    for residual in reversed(residuals):
        residual = np.asarray(residual, dtype=np.float32)
        current = expand_image(current, residual.shape, kernel) + residual

    return np.clip(current, 0.0, 1.0)


def quantize_residual(residual):
    """
    Quantize a residual image to signed 16-bit integers.

    Parameters
    ----------
    residual : numpy.ndarray
        Floating-point Laplacian residual.

    Returns
    -------
    quantized : numpy.ndarray
        Residual encoded as int16.

    scale : numpy.float32
        Scale required for dequantization.
    """
    residual = np.asarray(residual, dtype=np.float32)
    # scale is the largest intensity magnitude in residual image 
    scale = np.float32(np.max(np.abs(residual)))

    if scale == 0:
        return np.zeros_like(residual, dtype=np.int16), np.float32(0.0)

    quantized = np.round(residual / scale * RESIDUAL_MAX)
    quantized = np.clip(quantized, -RESIDUAL_MAX, RESIDUAL_MAX).astype(np.int16)
    return quantized, scale


def dequantize_residual(quantized, scale):
    """
    Restore a floating-point residual from int16 data.

    Parameters
    ----------
    quantized : numpy.ndarray
        Residual stored as int16.

    scale : float
        Scale recorded during quantization.

    Returns
    -------
    numpy.ndarray
        Dequantized float32 residual.
    """
    scale = np.float32(scale)
    if scale == 0:
        return np.zeros_like(quantized, dtype=np.float32)
    return quantized.astype(np.float32) * (scale / RESIDUAL_MAX)


def compress_image(image, filepath, levels=5, kernel_size=5, sigma=1.0):
    """
    Compress an image and save a Pyramid Codec archive.

    Parameters
    ----------
    image : numpy.ndarray
        RGB floating-point image in [0, 1].

    filepath : str or pathlib.Path
        Destination archive path. A custom .pyr extension may be used.

    levels : int, optional
        Number of pyramid levels.

    kernel_size : int, optional
        Gaussian kernel size.

    sigma : float, optional
        Gaussian standard deviation.

    Returns
    -------
    dict
        Archive information and an approximate array-data compression ratio.
    """
    filepath = Path(filepath)
    # build the Gaussian kernel and use it to build the Laplacian pyramid  
    kernel = gaussian_kernel(kernel_size, sigma)
    residuals, smallest = build_laplacian_pyramid(image, levels, kernel)

    # save format version, number of levels in the Laplacian pyramid,
    # Gaussian kernel Parameters, and the smallest downsampled image,
    # all as np-arrays, into an archive 
    archive = {
        "version": np.array(ARCHIVE_VERSION, dtype=np.int16), # version for our new archive format
        "levels": np.array(levels, dtype=np.int16),
        "kernel_size": np.array(kernel_size, dtype=np.int16),
        "sigma": np.array(sigma, dtype=np.float32),
        "original_shape": np.array(image.shape, dtype=np.int32),
        "smallest": np.round(np.clip(smallest, 0.0, 1.0) * 255.0).astype(np.uint8), # use one byte per pixel to save space
    }

    # add quantized residuals of the Laplacian pyramid and their scale to the archive
    for index, residual in enumerate(residuals):
        quantized, scale = quantize_residual(residual)
        archive[f"residual_{index}"] = quantized
        archive[f"scale_{index}"] = np.array(scale, dtype=np.float32)

    # write the archive into a binary output file )
    with filepath.open("wb") as output_file:
        # Dictionary unpacking turns each key into a named array in the archive.
        np.savez_compressed(output_file, **archive) 

    # get the original image's array size
    original_bytes = np.asarray(image, dtype=np.float32).nbytes
    # get the size of our newly saved archive  
    archive_bytes = filepath.stat().st_size

    # compression_ratio = original_bytes / archive_bytes
    return {
        "path": filepath,
        "original_bytes": original_bytes,
        "archive_bytes": archive_bytes,
        "compression_ratio": original_bytes / archive_bytes if archive_bytes else 0.0,
    }


def load_compressed_image(filepath):
    """
    Load and decode a Pyramid Codec archive.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to a previously saved archive.

    Returns
    -------
    residuals : list of numpy.ndarray
        Dequantized residual images.

    smallest : numpy.ndarray
        Smallest image as float32 in [0, 1].

    kernel : numpy.ndarray
        Gaussian kernel rebuilt from metadata.

    metadata : dict
        Archive settings and original image shape.
    """
    filepath = Path(filepath)

    with np.load(filepath, allow_pickle=False) as archive: # open archive of np arrays from filepath
        version = int(archive["version"])
        if version != ARCHIVE_VERSION: # verify archive version
            raise ValueError(f"Unsupported archive version: {version}")

        # get the number of levels in Laplacian pyramid
        levels = int(archive["levels"]) 
        # get the Gaussian kernel parameters
        kernel_size = int(archive["kernel_size"])
        sigma = float(archive["sigma"])
        # get original image dimensions 
        original_shape = tuple(int(v) for v in archive["original_shape"])

        # dequantize residual images
        residuals = []
        for index in range(levels):
            quantized = archive[f"residual_{index}"]
            scale = float(archive[f"scale_{index}"])
            residuals.append(dequantize_residual(quantized, scale))

        # get the smallest image
        smallest = archive["smallest"].astype(np.float32) / 255.0

    # build the Guassian kernel of specified parameters to return
    kernel = gaussian_kernel(kernel_size, sigma)
    # pack metadata in a dictionary to return
    metadata = {
        "version": version,
        "levels": levels,
        "kernel_size": kernel_size,
        "sigma": sigma,
        "original_shape": original_shape,
        "path": filepath,
    }
    return residuals, smallest, kernel, metadata


def decompress_image(filepath):
    """
    Load a Pyramid Codec archive and reconstruct its image.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the compressed archive.

    Returns
    -------
    image : numpy.ndarray
        Reconstructed RGB image in [0, 1].

    metadata : dict
        Metadata read from the archive.
    """
    residuals, smallest, kernel, metadata = load_compressed_image(filepath)
    return reconstruct_image(residuals, smallest, kernel), metadata
