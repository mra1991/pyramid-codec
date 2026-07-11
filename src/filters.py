"""
Gaussian filtering utilities for the pyramid codec.

This module provides functions for generating normalized Gaussian kernels
and applying fast 2D cross-correlation to grayscale or color images using
SciPy's optimized filtering routines.
"""

import numpy as np
from scipy.ndimage import correlate


def gaussian_kernel(size=5, sigma=1.0):
    """
    Generate a normalized two-dimensional Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Width and height of the kernel. The value must be a positive odd
        integer so that the kernel has a well-defined center.

    sigma : float, optional
        Standard deviation of the Gaussian distribution. Must be greater
        than zero.

    Returns
    -------
    numpy.ndarray
        Normalized 2D Gaussian kernel whose values sum to one.

    Raises
    ------
    ValueError
        If ``size`` is not a positive odd integer or if ``sigma`` is not
        greater than zero.
    """
    if size <= 0 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    axis = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    gaussian_1d = np.exp(-(axis**2) / (2.0 * sigma**2))

    kernel = np.outer(gaussian_1d, gaussian_1d)
    kernel /= kernel.sum()

    return kernel.astype(np.float32)


def fast_filter2d(image, kernel):
    """
    Apply fast 2D cross-correlation to a grayscale or color image.

    For grayscale images, the kernel is applied directly to the 2D array.
    For color images, the same kernel is applied independently to each
    channel without mixing color information.

    The function uses SciPy's optimized compiled implementation and reflect
    padding, which reduces artificial dark borders compared with zero
    padding.

    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape ``(height, width)`` for grayscale images or
        ``(height, width, channels)`` for color images.

    kernel : numpy.ndarray
        Two-dimensional filter kernel.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape as the input image.

    Raises
    ------
    ValueError
        If the input image is not 2D or 3D, or if the kernel is not 2D.
    """
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    if kernel.ndim != 2:
        raise ValueError("The kernel must be a 2D array.")

    if image.ndim == 2:
        return correlate(
            image,
            kernel,
            mode="reflect"
        )

    if image.ndim == 3:
        kernel_3d = kernel[:, :, np.newaxis]

        return correlate(
            image,
            kernel_3d,
            mode="reflect"
        )

    raise ValueError(
        "The input image must be a 2D grayscale image or a 3D color image."
    )