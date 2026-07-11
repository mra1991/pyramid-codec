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
    return blurred[::2, ::2, ...]


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
    target_shape = tuple(int(value) for value in target_shape)
    if image.ndim != len(target_shape):
        raise ValueError("Input image and target shape must have equal rank.")

    expanded = np.zeros(target_shape, dtype=np.float32)
    expected = expanded[::2, ::2, ...].shape

    if image.shape != expected:
        raise ValueError(
            f"Input shape {image.shape} is incompatible with target {target_shape}."
        )

    expanded[::2, ::2, ...] = image
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

        smaller = downsample_image(current, kernel)
        approximation = expand_image(smaller, current.shape, kernel)
        residuals.append(current - approximation)
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
    kernel = gaussian_kernel(kernel_size, sigma)
    residuals, smallest = build_laplacian_pyramid(image, levels, kernel)

    archive = {
        "version": np.array(ARCHIVE_VERSION, dtype=np.int16),
        "levels": np.array(levels, dtype=np.int16),
        "kernel_size": np.array(kernel_size, dtype=np.int16),
        "sigma": np.array(sigma, dtype=np.float32),
        "original_shape": np.array(image.shape, dtype=np.int32),
        "smallest": np.round(np.clip(smallest, 0.0, 1.0) * 255.0).astype(np.uint8),
    }

    for index, residual in enumerate(residuals):
        quantized, scale = quantize_residual(residual)
        archive[f"residual_{index}"] = quantized
        archive[f"scale_{index}"] = np.array(scale, dtype=np.float32)

    with filepath.open("wb") as output_file:
        np.savez_compressed(output_file, **archive)

    original_bytes = np.asarray(image, dtype=np.float32).nbytes
    archive_bytes = filepath.stat().st_size

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

    with np.load(filepath, allow_pickle=False) as archive:
        version = int(archive["version"])
        if version != ARCHIVE_VERSION:
            raise ValueError(f"Unsupported archive version: {version}")

        levels = int(archive["levels"])
        kernel_size = int(archive["kernel_size"])
        sigma = float(archive["sigma"])
        original_shape = tuple(int(v) for v in archive["original_shape"])

        residuals = []
        for index in range(levels):
            quantized = archive[f"residual_{index}"]
            scale = float(archive[f"scale_{index}"])
            residuals.append(dequantize_residual(quantized, scale))

        smallest = archive["smallest"].astype(np.float32) / 255.0

    kernel = gaussian_kernel(kernel_size, sigma)
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
