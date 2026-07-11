"""
Entry point for the Pyramid Codec application.

This module launches the graphical user interface for compressing and
reconstructing color images with a Laplacian image pyramid.
"""

from gui import start_gui


if __name__ == "__main__":
    start_gui()
