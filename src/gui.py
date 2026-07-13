"""
Graphical user interface for the Pyramid Codec application.

The interface lets the user select a color image, configure pyramid
parameters, save a compressed archive, open an existing archive, reconstruct
it, preview the result, and save the decompressed image.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from image_utils import load_image, make_preview, save_image
from pyramid_codec import compress_image, decompress_image

# the root folter is the grandparent of the current file
PROJECT_ROOT = Path(__file__).resolve().parent.parent # resolve() gives the absolute path
# folder path for input images
IMAGE_DIR = PROJECT_ROOT / "sample_images"
# folder path for storing reconstructed images
OUTPUT_DIR = PROJECT_ROOT / "output"
# folder path for storing pyramid-codec archive files
COMPRESSED_DIR = PROJECT_ROOT / "compressed"


class PyramidCodecGUI:
    """
    Tkinter GUI for image-pyramid compression and reconstruction.

    The class owns the selected source image, compressed archive path,
    reconstructed image, widgets, and preview references.
    """

    def __init__(self, root):
        """
        Initialize the Pyramid Codec graphical interface.

        Parameters
        ----------
        root : tkinter.Tk
            Main application window.
        """
        self.root = root # the root Tk window
        self.root.title("Pyramid Codec") # window title
        self.root.resizable(False, False) # prevent resizing

        # Create the default input and output directories if they do not exist.
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

        self.source_image = None
        self.source_path = None
        self.archive_path = None
        self.reconstructed_image = None

        # Tkinter needs persistent references to PhotoImage objects.
        self.source_tk_image = None
        self.output_tk_image = None

        self._build_controls()
        self._build_previews()
        self._update_button_states()

    def _build_controls(self):
        """Create the parameter controls and action buttons."""
        
        # Create a labeled frame for the compression settings.
        # It spans both preview columns and stretches horizontally.
        controls = ttk.LabelFrame(self.root, text="Compression Settings")
        controls.grid(row=0, column=0, columnspan=2, padx=12, pady=10, sticky="ew")

        self.levels_slider = tk.Scale(
            controls, # the control frame is the parent of the slider, not the root window
            from_=1, # min slider amount
            to=8, # max slider amount
            resolution=1, # step 
            orient="horizontal", # alignment
            label="Pyramid levels", # caption displayed above slider
            length=180, # slider length in pixels
        )
        self.levels_slider.set(5) # set slider's initial value
        self.levels_slider.grid(row=0, column=0, padx=8, pady=4)

        self.kernel_slider = tk.Scale(
            controls,
            from_=3,
            to=11,
            resolution=2,
            orient="horizontal",
            label="Gaussian kernel size",
            length=180,
        )
        self.kernel_slider.set(5)
        self.kernel_slider.grid(row=0, column=1, padx=8, pady=4)

        self.sigma_slider = tk.Scale(
            controls,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            label="Gaussian sigma",
            length=180,
        )
        self.sigma_slider.set(1.0)
        self.sigma_slider.grid(row=0, column=2, padx=8, pady=4)

        # Create an unlabeled frame for the action buttons.
        # It spans both preview columns below the settings frame.        
        actions = ttk.Frame(self.root)
        actions.grid(row=1, column=0, columnspan=2, padx=12, pady=4)

        self.browse_button = ttk.Button(
            actions, # actions frame is the parent of every button
            text="Browse Image", # button caption
            command=self.browse_image # name of the method called once the button is pressed
        )
        self.browse_button.grid(row=0, column=0, padx=5)

        self.compress_button = ttk.Button(
            actions, text="Compress and Save", command=self.compress_selected_image
        )
        self.compress_button.grid(row=0, column=1, padx=5)

        self.open_archive_button = ttk.Button(
            actions, text="Open Compressed", command=self.open_compressed_image
        )
        self.open_archive_button.grid(row=0, column=2, padx=5)

        self.decompress_button = ttk.Button(
            actions, text="Decompress", command=self.decompress_archive
        )
        self.decompress_button.grid(row=0, column=3, padx=5)

        self.save_output_button = ttk.Button(
            actions, text="Save Decompressed", command=self.save_decompressed_image
        )
        self.save_output_button.grid(row=0, column=4, padx=5)

        # set status text to an initial message asking the user to select an input file
        self.status_var = tk.StringVar(
            value="Select an image or a compressed archive."
        )
        # place the status message 
        ttk.Label(self.root, textvariable=self.status_var, anchor="center").grid(
            row=2, column=0, columnspan=2, padx=12, pady=(4, 8), sticky="ew"
        )

    def _build_previews(self):
        """Create source and reconstructed image preview areas."""
        source_frame = ttk.LabelFrame(self.root, text="Source Image")
        source_frame.grid(row=3, column=0, padx=(12, 6), pady=(0, 12), sticky="n")

        output_frame = ttk.LabelFrame(self.root, text="Reconstructed Image")
        output_frame.grid(row=3, column=1, padx=(6, 12), pady=(0, 12), sticky="n") # padding on 3 sides, aligned north

        self.source_label = ttk.Label(
            source_frame, text="No source image selected", anchor="center", width=50
        ) # 50 character long label
        self.source_label.pack(padx=10, pady=10)

        self.output_label = ttk.Label(
            output_frame, text="No reconstructed image", anchor="center", width=50
        )
        self.output_label.pack(padx=10, pady=10)

    def _update_button_states(self):
        """Enable or disable actions based on the current application state."""
        self.compress_button.config(
            state="normal" if self.source_image is not None else "disabled"
        )
        self.decompress_button.config(
            state="normal" if self.archive_path is not None else "disabled"
        )
        self.save_output_button.config(
            state="normal" if self.reconstructed_image is not None else "disabled"
        )

    def _show_preview(self, image, label, reference_name):
        """
        Display an image inside a Tkinter label.

        Parameters
        ----------
        image : numpy.ndarray
            RGB floating-point image.

        label : tkinter.ttk.Label
            Label that will display the image.

        reference_name : str
            Instance attribute used to keep the PhotoImage alive.
        """
        pil_image = make_preview(image, max_size=(480, 380))
        tk_image = ImageTk.PhotoImage(pil_image)
        # dynamically set attribute for source/output image
        setattr(self, reference_name, tk_image)
        # assign image and remove placeholder text
        label.config(image=tk_image, text="") 
        
    def browse_image(self):
        """Open a file dialog, load a color image, and show its preview."""
        filepath = filedialog.askopenfilename(
            initialdir=IMAGE_DIR,
            title="Select an image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            self.source_image = load_image(filepath)
        except (OSError, ValueError) as error:
            messagebox.showerror("Unable to Load Image", str(error))
            return

        self.source_path = Path(filepath)
        self._show_preview(self.source_image, self.source_label, "source_tk_image")
        
        # set status_var, which was passed to tk.Label object as textvariable
        self.status_var.set( 
            f"Loaded {self.source_path.name} "
            f"({self.source_image.shape[1]} x {self.source_image.shape[0]})." # width x height
        )
        self._update_button_states()

    def compress_selected_image(self):
        """Compress the selected image and save a .pyr archive."""
        if self.source_image is None:
            return

        default_name = (
            f"{self.source_path.stem}.pyr"
            if self.source_path is not None
            else "compressed_image.pyr"
        )

        filepath = filedialog.asksaveasfilename(
            initialdir=COMPRESSED_DIR,
            initialfile=default_name,
            title="Save compressed image",
            defaultextension=".pyr",
            filetypes=[
                ("Pyramid Codec Archive", "*.pyr"),
                ("NumPy Compressed Archive", "*.npz"),
                ("All Files", "*.*"),
            ],
        )
        if not filepath:
            return

        try: # call to codec
            information = compress_image(
                self.source_image,
                filepath,
                levels=self.levels_slider.get(),
                kernel_size=self.kernel_slider.get(),
                sigma=self.sigma_slider.get(),
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Compression Failed", str(error))
            return

        self.archive_path = Path(filepath)
        self.status_var.set(
            f"Saved {self.archive_path.name}. Approximate array-data ratio: "
            f"{information['compression_ratio']:.2f}:1."
        )
        self._update_button_states()

    def open_compressed_image(self):
        """Select a previously saved Pyramid Codec archive."""
        filepath = filedialog.askopenfilename(
            initialdir=COMPRESSED_DIR,
            title="Open compressed image",
            filetypes=[
                ("Pyramid Codec Archive", "*.pyr *.npz"),
                ("All Files", "*.*"),
            ],
        )
        if not filepath:
            return

        self.archive_path = Path(filepath)
        self.reconstructed_image = None
        self.output_tk_image = None
        self.output_label.config(image="", text="Ready to decompress")
        self.status_var.set(f"Selected archive: {self.archive_path.name}")
        self._update_button_states()

    def decompress_archive(self):
        """Reconstruct the selected archive and display the result."""
        if self.archive_path is None:
            return

        try: # call to codec
            image, metadata = decompress_image(self.archive_path)
        except (OSError, KeyError, ValueError) as error:
            messagebox.showerror("Decompression Failed", str(error))
            return

        self.reconstructed_image = image
        self._show_preview(
            self.reconstructed_image, self.output_label, "output_tk_image"
        )
        self.status_var.set(
            f"Reconstructed {metadata['levels']} levels using a "
            f"{metadata['kernel_size']} x {metadata['kernel_size']} kernel."
        )
        self._update_button_states()

    def save_decompressed_image(self):
        """Save the reconstructed image in a standard image format."""
        if self.reconstructed_image is None:
            return

        default_name = (
            f"{self.archive_path.stem}_reconstructed.png"
            if self.archive_path is not None
            else "reconstructed.png"
        )

        filepath = filedialog.asksaveasfilename(
            initialdir=OUTPUT_DIR,
            initialfile=default_name,
            title="Save reconstructed image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg *.jpeg"),
                ("Bitmap Image", "*.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not filepath:
            return

        try: # call to image utils
            save_image(filepath, self.reconstructed_image)
        except OSError as error:
            messagebox.showerror("Unable to Save Image", str(error))
            return

        self.status_var.set(f"Saved reconstructed image: {Path(filepath).name}")


def start_gui():
    """
    Create and start the Pyramid Codec graphical interface.

    This function creates the main Tkinter window, initializes the GUI, and
    starts the Tk event loop.
    """
    root = tk.Tk()
    PyramidCodecGUI(root)
    root.mainloop()
