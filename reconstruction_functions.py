"""
Functions and classes for MRI reconstruction and visualization.

@author: Julia Marcolan
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import ScalarFormatter

class MRIGeometry:
    """
    Stores MRI acquisition geometry and computes voxel dimensions.

    Args:
        FOVr (float): Field of view in readout direction (mm).
        FOVp2 (float): Field of view in phase-encoding direction 2 (mm).
        FOVp3 (float): Field of view in phase-encoding direction 3 (mm).

        Nr (int): Number of samples in readout direction.
        Np2 (int): Number of samples in phase-encoding direction 2.
        Np3 (int): Number of samples in phase-encoding direction 3.
    """

    def __init__(self, FOVr, FOVp2, FOVp3, Nr, Np2, Np3):

        self.FOVr = FOVr
        self.FOVp2 = FOVp2
        self.FOVp3 = FOVp3

        self.Nr = Nr
        self.Np2 = Np2
        self.Np3 = Np3

        # Voxel dimensions
        # Readout corresponds to the Y direction
        # Phase encoding 2 corresponds to the X direction
        self.dy = FOVr / Nr
        self.dx = FOVp2 / Np2
        self.dz = FOVp3 / Np3

def apply_FFT_2D(data):
    """
    Apply 2D Fast Fourier Transform (FFT) to input data.

    Args:
        data (array_like): 2D input data.

    Returns:
        numpy.ndarray: Transformed data after applying 2D FFT.
    """

    shifted_data = np.fft.fftshift(data, axes=(0, 1))
    transf_data = np.fft.fft(shifted_data, axis=0)
    transf_data = np.fft.fft(transf_data, axis=1)
    transf_data = np.fft.ifftshift(transf_data, axes=(0, 1))

    return transf_data

def apply_FFT_3D(data):
    """
    Applies centered N-dimensional FFT (k-space → image space).

    Args:
        data (np.ndarray): Input k-space array.

    Returns:
        np.ndarray: Complex image in spatial domain.
    """

    return np.fft.fftshift( np.fft.fftn( np.fft.ifftshift(data)))

def reconstruct_rss2D(data):
    """
    Reconstructs a 2D MRI image from multi-coil k-space data using
    Root Sum of Squares (RSS).

    Args:
        data (np.ndarray): K-space data with shape
                           (nRcvrs, nPhase, nReadout).

    Returns:
        np.ndarray: RSS reconstructed image with shape (height, width).
    """

    nRcvrs = data.shape[0]

    recon = []

    # --- FFT reconstruction for each receiver ---
    for r in range(nRcvrs):

        img = apply_FFT_2D(data[r])
        recon.append(img)

    recon = np.array(recon)

    # --- Root Sum of Squares reconstruction ---
    RSS = np.sqrt(np.sum(np.abs(recon) ** 2, axis=0))

    # --- Orientation correction ---
    RSS = np.rot90(RSS)
    RSS = np.flip(RSS, axis=1)

    return RSS

def reconstruct_rss3D(data):
    """
    Reconstructs MRI image from multi-coil k-space using RSS.

    Args:
        data (np.ndarray): K-space data with shape (nRcvrs, nSlices, nPhase, nReadout).

    Returns:
        np.ndarray: RSS reconstructed image with shape (nSlices, height, width).
    """

    nRcvrs = data.shape[0]

    recon = []

    for r in range(nRcvrs):
        img = apply_FFT_3D(data[r])
        recon.append(img)

    recon = np.array(recon)

    # --- Root Sum of Squares reconstruction ---
    RSS = np.sqrt( np.sum(np.abs(recon) ** 2, axis=0) )

    # --- Orientation correction --- 
    RSS = np.rot90(RSS,axes=(1, 2))
    
    RSS = np.flip(RSS,axis=2)

    # --- Reverse slice order ---
    RSS = np.flip(RSS,axis=0)

    # TO  GRE 3D TWF
    #RSS = np.flip(RSS, axis=1)

    return RSS

def reconstruct_phase(data):
    """
    Reconstructs MRI phase image from multi-coil k-space.

    Args:
        data (np.ndarray): K-space data with shape (nRcvrs, nSlices, nPhase, nReadout).

    Returns:
        np.ndarray: Phase image with shape (nSlices, height, width).
    """

    nRcvrs = data.shape[0]

    recon = []

    for r in range(nRcvrs):
        img = apply_FFT_3D(data[r])
        recon.append(img)

    recon = np.array(recon)

    # --- Extract phase from complex reconstructed images ---
    phase = np.angle(recon)

    # --- Combine coils ---
    # Simple mean of phase across receivers
    phase = np.mean(phase, axis=0)

    # --- Orientation correction ---
    phase = np.rot90(phase, axes=(1, 2))

    phase = np.flip(phase, axis=2)

    # --- Reverse slice order ---
    phase = np.flip(phase, axis=0)

    # TO  GRE 3D TWF
    phase = np.flip(phase, axis=1)

    return phase

def apply_spatial_shift(image, shift=5, axis=1):
    """
    Applies a circular spatial shift to an MRI image or image volume.

    Args:
        image (np.ndarray): Two-dimensional image or three-dimensional image volume.

        shift (int): Number of pixels to shift. Positive values shift toward higher indices.

        axis (int): Axis along which the shift is applied.

    Returns:
        np.ndarray: Spatially shifted image or image volume.
    """

    return np.roll(image, shift=shift, axis=axis)

def show_mosaic(data, FOV_x, FOV_y, mosaic_size=None, save_image=False, output_filename="image.png", first_slice_blank=False):
    """
    Displays a 3D volume as a mosaic of slices with correct physical aspect ratio
    and interactive intensity window.

    Args:
        data (np.ndarray): 3D array (nSlices, height, width).

        FOV_x (float): Field of view in readout direction (mm).

        FOV_y (float): Field of view in phase direction (mm).

        mosaic_size (int, optional): Number of slices around center.

        save_image (bool): If True, the image is saved automatically when the
            window is closed.

        output_filename (str, optional): Name of the output image file.

        first_slice_blank (bool, optional): If True, the first position of the
            mosaic (top-left corner) is displayed as white instead of the
            first slice. Default is False.
    """

    # --- Select slices ---
    n_slices = data.shape[0]

    if mosaic_size is None:
        slices = data
    else:
        center = n_slices // 2

        slices = data[
            max(center - mosaic_size, 0):
            min(center + mosaic_size, n_slices)
        ]

    # --- Mosaic dimensions ---
    n = len(slices)

    ncols = math.ceil(np.sqrt(n))
    nrows = math.ceil(n / ncols)

    h, w = slices[0].shape

    mosaic = np.zeros((nrows * h, ncols * w))

    for i in range(n):

        r = i // ncols
        c = i % ncols

        mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = np.abs(slices[i])

    # --- Make first position blank ---
    if first_slice_blank:
        mosaic[0:h, 0:w] = mosaic[0:h, 0:w] = 0

    # --- Brightness ---
    vmin = 0
    vmax = np.max(mosaic)

    # --- Physical aspect ratio ---
    dx = FOV_x / w
    dy = FOV_y / h

    aspect = dy / dx

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    plt.subplots_adjust(bottom=0.25)

    im = ax.imshow(
        mosaic,
        cmap='gray',
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation='nearest'
    )

    ax.axis('off')

    # --- Sliders ---
    ax_vmin = plt.axes([0.25, 0.10, 0.5, 0.03])
    ax_vmax = plt.axes([0.25, 0.05, 0.5, 0.03])

    s_vmin = Slider(
        ax_vmin,
        'Min',
        0,
        vmax,
        valinit=vmin
    )

    s_vmax = Slider(
        ax_vmax,
        'Max',
        0,
        vmax,
        valinit=vmax
    )

    def update(val):

        im.set_clim(
            vmin=s_vmin.val,
            vmax=s_vmax.val
        )

        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    # --- Save image when window is closed ---

    def save_when_closed(event):

        if not save_image:
            return

        current_vmin = s_vmin.val
        current_vmax = s_vmax.val

        print("\nSaving image...")

        im.set_clim(
            vmin=current_vmin,
            vmax=current_vmax
        )

        ax_vmin.set_visible(False)
        ax_vmax.set_visible(False)

        fig.savefig(
            output_filename,
            dpi=1200,
            bbox_inches=None,
            pad_inches=0
        )

        print(f"Image saved as: {output_filename}")

    fig.canvas.mpl_connect(
        'close_event',
        save_when_closed
    )

    plt.show()

def apply_spatial_shift(image, shift=5, axis=1):
    """
    Applies a circular spatial shift to an MRI image or volume.

    Args:
        image (np.ndarray):
            Two-dimensional image or three-dimensional volume.

        shift (int):
            Number of pixels to shift.

        axis (int):
            Axis along which the shift is applied.

    Returns:
        np.ndarray:
            Spatially shifted image or volume.
    """

    return np.roll(
        image,
        shift=shift,
        axis=axis
    )

def get_slices(volume, plane):
    """
    Reorients a 3D MRI volume into a selected anatomical plane.

    Args: volume (np.ndarray): 3D image volume with shape (Z, Y, X).

        plane (str): Anatomical plane ('axial', 'coronal', or 'sagittal').

    Returns:
        np.ndarray: Reoriented 3D volume.

    Raises:
        ValueError:
            If plane is not axial, coronal, or sagittal.
    """

    if plane == 'axial':
        return volume

    if plane == 'coronal':
        return np.transpose(
            volume,
            (1, 0, 2)
        )

    if plane == 'sagittal':
        return np.transpose(
            volume,
            (2, 0, 1)
        )

    raise ValueError(
        "plane must be axial, coronal or sagittal"
    )

def get_aspect(geom, plane):
    """
    Computes the correct physical aspect ratio for MRI display.

    Args:
        geom (MRIGeometry): MRI geometry object containing voxel dimensions.

        plane (str): Anatomical plane ('axial', 'coronal', or 'sagittal').

    Returns:
        float: Physical aspect ratio.

    Raises:
        ValueError: If plane is not axial, coronal, or sagittal.
    """

    if plane == 'axial':
        return geom.dy / geom.dx

    if plane == 'coronal':
        return geom.dz / geom.dx

    if plane == 'sagittal':
        return geom.dz / geom.dy

    raise ValueError(
        "plane must be axial, coronal or sagittal"
    )

def show_plane( volume, geom, plane='axial', slice_number=None, save_image=False, output_filename="mosaic.png"):
    """
    Displays an MRI volume either as a mosaic of all slices or as a single selected slice.

    The brightness can be adjusted interactively using the intensity sliders.

    If save_image=True, the image is automatically saved when the window is closed, using the final intensity values.

    It is possible to choose each of the anatomical planes ('axial', 'coronal', or 'sagittal' ) will be shown. 

    Args:
        volume (np.ndarray): 3D MRI volume.

        geom (MRIGeometry): MRI acquisition geometry.

        plane (str): Anatomical plane ('axial', 'coronal', or 'sagittal').

        slice_number (int, optional): Index of the slice to display.

            If None:
                All slices are displayed as a mosaic.

            If an integer:
                Only the selected slice is displayed.

            Default is None.

        save_image (bool): If True, saves the image when the window is closed. Default is False.

        output_filename (str): Output image filename.
    """

    # --- Reorient volume --- 

    slices = get_slices( volume, plane )

    # --- Select mosaic or single slice --- 

    if slice_number is None:

        # Show all slices
        display_data = slices

        nslices = display_data.shape[0]

        # Automatically determine mosaic dimensions
        ncols = math.ceil(np.sqrt(nslices))
        nrows = math.ceil(nslices / ncols)

        h = display_data.shape[1]
        w = display_data.shape[2]

        mosaic = np.zeros( (nrows * h, ncols * w), dtype=np.float32)

        # Build mosaic
        for i in range(nslices):

            r = i // ncols
            c = i % ncols

            mosaic[
                r*h:(r+1)*h,
                c*w:(c+1)*w
            ] = np.abs(display_data[i])

        image = mosaic

    else:

        # Check slice index
        if slice_number < 0 or slice_number >= slices.shape[0]:
            raise IndexError(
                f"slice_number must be between 0 and "
                f"{slices.shape[0] - 1}"
            )

        # Show only selected slice
        image = np.abs( slices[slice_number])

    # --- Physical aspect ratio ---

    aspect = get_aspect(geom, plane)

    # --- Initial brightness --- 

    vmin = 0
    vmax = np.max(image)

    # --- Figure --- 

    fig, ax = plt.subplots( figsize=(10, 10), dpi=100)

    plt.subplots_adjust( bottom=0.25)

    im = ax.imshow(
        image,
        cmap='gray',
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation='nearest'
    )

    ax.axis('off')

    # --- Sliders ---

    ax_vmin = plt.axes([0.25, 0.10, 0.50, 0.03])

    ax_vmax = plt.axes([0.25, 0.05, 0.50, 0.03])

    s_vmin = Slider(
        ax_vmin,
        'Min',
        0,
        vmax,
        valinit=vmin
    )

    s_vmax = Slider(
        ax_vmax,
        'Max',
        0,
        vmax,
        valinit=vmax
    )

    def update(val):

        im.set_clim(
            vmin=s_vmin.val,
            vmax=s_vmax.val
        )

        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    # --- Save when window is closed --- 

    def save_when_closed(event):

        if not save_image:
            return

        current_vmin = s_vmin.val
        current_vmax = s_vmax.val

        print("\nSaving image...")
        print(f"Plane = {plane}")

        # Apply final brightness
        im.set_clim(
            vmin=current_vmin,
            vmax=current_vmax
        )

        # Hide sliders
        ax_vmin.set_visible(False)
        ax_vmax.set_visible(False)

        # Save
        fig.savefig(
            output_filename,
            dpi=1200,
            bbox_inches=None,
            pad_inches=0
        )

        print(
            f"Image saved as: {output_filename}"
        )

    fig.canvas.mpl_connect(
        'close_event',
        save_when_closed
    )

    plt.show()

def show_single_slice( data, FOV_x, FOV_y, slice_number, save_image=False, output_filename=None):
    """
    Displays a selected slice of a 3D MRI volume with the correct physical aspect ratio and interactive intensity window.

    If save_image=True, the image is automatically saved when the window is closed, using the final intensity values selected
    with the sliders.

    Args:
        data (np.ndarray): Three-dimensional image volume with shape (nSlices, height, width).

        FOV_x (float): Field of view in the readout direction (mm).

        FOV_y (float): Field of view in the phase-encoding direction (mm).

        slice_number (int): Index of the slice to display.

        save_image (bool):
            If True, the image is saved automatically when the
            window is closed.
            Default is False.

        output_filename (str, optional):
            Name of the output image file.

    Displays:
        Interactive view of the selected slice with minimum and maximum intensity sliders.
    """

    # --- Select slice --- 

    image = np.abs(data[slice_number])

    h, w = image.shape

    # --- Initial intensity limits --- 

    vmin = 0
    vmax = np.max(image)

    # --- Physical aspect ratio --- 

    dx = FOV_x / w
    dy = FOV_y / h

    aspect = dy / dx

    # --- Output filename --- 
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    plt.subplots_adjust(bottom=0.25)

    im = ax.imshow(
        image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation="nearest"
    )

    ax.axis("off")

    # --- Intensity sliders --- 

    ax_vmin = plt.axes( [0.25, 0.10, 0.50, 0.03] )

    ax_vmax = plt.axes([0.25, 0.05, 0.50, 0.03])

    s_vmin = Slider(
        ax_vmin,
        "Min",
        0,
        vmax,
        valinit=vmin
    )

    s_vmax = Slider(
        ax_vmax,
        "Max",
        0,
        vmax,
        valinit=vmax
    )

    def update(val):

        im.set_clim(
            vmin=s_vmin.val,
            vmax=s_vmax.val
        )

        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    # --- Save when window is closed --- 

    def save_when_closed(event):

        # Do not save if save_image is False
        if not save_image:
            return

        current_vmin = s_vmin.val
        current_vmax = s_vmax.val

        print("\nSaving image...")
        print(f"Slice = {slice_number}")

        # Apply final intensity window
        im.set_clim(
            vmin=current_vmin,
            vmax=current_vmax
        )

        # Hide sliders before saving
        ax_vmin.set_visible(False)
        ax_vmax.set_visible(False)

        # Save high-resolution image
        fig.savefig(
            output_filename,
            dpi=1200,
            bbox_inches=None,
            pad_inches=0
        )

        print(f"Image saved as: {output_filename}")

    fig.canvas.mpl_connect(
        "close_event",
        save_when_closed
    )

    plt.show()

def show_diff_map(data, FOV_x, FOV_y, mosaic_size=None, cmap="gray", first_slice_blank=False, scale=5):
    """
    Displays a 3D volume as a mosaic of slices with the correct physical
    aspect ratio from the difference between two MRI sets.

    Args:
        data (np.ndarray): 3D image volume with shape (nSlices, height, width).

        FOV_x (float): Field of view in the readout direction (mm).

        FOV_y (float): Field of view in the phase-encoding direction (mm).

        mosaic_size (int, optional):
            Number of slices to take on each side of the center.
            For example, mosaic_size=10 displays approximately 20 slices
            centered around the middle.
            If None, all slices are shown.

        cmap (str, optional): Matplotlib colormap used to display the mosaic. Default is "gray".

        first_slice_blank (bool, optional):
            If True, the first position of the mosaic (top-left corner)
            is left blank, using the same background as the other slices.
            Default is False.

        scale (float, optional):
            Multiplicative scale factor applied to the difference map for visualization. Default is 5.

    Displays:
        Mosaic of image slices.
    """

    # --- Select slices ---
    n_slices = data.shape[0]

    if mosaic_size is None:

        slices = data

    else:

        center = n_slices // 2

        start = max(center - mosaic_size, 0)
        end = min(center + mosaic_size, n_slices)

        slices = data[start:end]

    # --- Mosaic dimensions ---
    n = len(slices)

    ncols = math.ceil(np.sqrt(n))
    nrows = math.ceil(n / ncols)

    h, w = slices[0].shape

    mosaic = np.zeros(
        (nrows * h, ncols * w)
    )

    # --- Fill mosaic ---
    for i in range(n):

        r = i // ncols
        c = i % ncols

        mosaic[
            r*h:(r+1)*h,
            c*w:(c+1)*w
        ] = np.abs(slices[i])

    # --- Scale difference ---
    mosaic = mosaic.astype(np.float64)
    mosaic *= scale

    # --- First slice blank ---
    if first_slice_blank:
        mosaic[0:h, 0:w] = 0

    # --- Physical aspect ratio ---
    dx = FOV_x / w
    dy = FOV_y / h

    aspect = dy / dx

    # --- Intensity range ---
    vmin = 0
    vmax = np.nanpercentile(mosaic, 99)

    # --- Figure ---
    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    im = ax.imshow(
        mosaic,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        origin="upper",
    )

    ax.axis("off")

    # --- Colorbar ---
    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        f"Signal Difference × {scale}"
    )

    # --- Force scientific notation ---
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    cbar.update_ticks()

    plt.tight_layout()
    plt.show()

def save_slice_as_npy(data, slice_index, filename, axis=0):
    """
    Extracts a specific slice from a NumPy array and saves it as a .npy file.

    Args:
        data (np.ndarray): Input data array.

        slice_index (int): Index of the slice to extract.

        filename (str): Name or path of the output .npy file.

        axis (int, optional): Axis along which the slice is selected. Default is 0.

    Returns:
        np.ndarray: The extracted slice.
    """

    # Extract the requested slice
    slice_data = np.take(data, slice_index, axis=axis)

    # Save the slice
    np.save(filename, slice_data)

    print(f"Slice {slice_index} saved as: {filename}")
    print(f"shape  {slice_data.shape}")

    return slice_data

def show_single_slice_grid(data_dict, slice_index, FOV_x, FOV_y, save_image=False, output_filename="grid.png"):
        """
        Displays a grid of 2D slices from multiple 3D image volumes, using the same slice index for each dataset. 
        The images are arranged into a single compact mosaic without spacing,  preserving the correct physical aspect 
        ratio based on the provided field of view (FOV).  
        
        The visualization is shown in grayscale with interactive contrast adjustment via sliders.

        Args: 

            - data_dict (dict): A dictionary where keys are dataset names and values are 3D NumPyarrays with shape 
            (n_slices, height, width), typically corresponding to  (slice, phase, readout).

            - slice_index (int): The index of the slice to be extracted and displayed from each  dataset.

            - FOV_x (float): Field of view in the readout direction, in millimeters.

            - FOV_y (float): Field of view in the phase encoding direction, in millimeters.

            - save_image (bool): If True, saves the figure to the specified output filename. Default is False.

            - output_filename (str): Name of the output image file. Default is "grid.png".

        Displays:

            A single figure containing a grid of images, where each image corresponds to the same 
            slice index from a different dataset. The images are shown in grayscale with a global 
            intensity scale and correct physical aspect ratio. Two sliders allow interactive 
            adjustment of the minimum and maximum intensity (contrast windowing).
    """

        names = list(data_dict.keys())
        images = []

        # --- extract the correct slice from each dataset ---
        for name in names:
            data = data_dict[name]

            img = np.abs(data[slice_index, :, :])
            images.append(img)

        # --- grid ---
        n = len(images)
        ncols = math.ceil(np.sqrt(n))
        nrows = math.ceil(n / ncols)

        h, w = images[0].shape

        full = np.zeros((nrows * h, ncols * w))

        for i in range(n):
            r = i // ncols
            c = i % ncols

            full[r*h:(r+1)*h, c*w:(c+1)*w] = images[i]

    # --- Global intensity ---
        vmin = 0
        vmax = np.percentile(full, 99)

        # --- Physical aspect ratio ---
        dx = FOV_x / w
        dy = FOV_y / h
        aspect = dy / dx

        # --- plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)

        im = ax.imshow(full, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)

        ax.axis("off")

        # --- sliders ---
        ax_vmin = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_vmax = plt.axes([0.2, 0.05, 0.6, 0.03])

        s_vmin = Slider(ax_vmin, 'Min', 0, vmax, valinit=vmin)
        s_vmax = Slider(ax_vmax, 'Max', 0, vmax, valinit=vmax)

        def update(val):
            im.set_clim(vmin=s_vmin.val, vmax=s_vmax.val)
            fig.canvas.draw_idle()

        s_vmin.on_changed(update)
        s_vmax.on_changed(update)

        # --- Save without sliders ---
        if save_image:
            ax_vmin.set_visible(False)
            ax_vmax.set_visible(False)

            fig.savefig(output_filename, bbox_inches="tight", dpi=300)

            ax_vmin.set_visible(True)
            ax_vmax.set_visible(True)

            print(f"Image saved to: {output_filename}")

        plt.show()

def show_two_mosaics(data1, data2, FOV_x, FOV_y, title1="SE", title2="STE", mosaic_size=None):
    """
    Displays two slice mosaics side by side using a shared intensity scale.

    Both mosaics are displayed with the same grayscale limits, allowing direct visual comparison. 
    Interactive sliders control the minimum and maximum display intensity simultaneously for both images.

    Args:

        data1 (np.ndarray): First 3D volume with shape (nSlices, height, width).

        data2 (np.ndarray): Second 3D volume with shape (nSlices, height, width).

        FOV_x (float): Field of view in the readout direction (mm).

        FOV_y (float): Field of view in the phase-encoding direction (mm).

        title1 (str, optional): Title displayed above the first mosaic.

        title2 (str, optional): Title displayed above the second mosaic.

        mosaic_size (int, optional): Number of slices around the center of each volume to include. If None, all slices are displayed.

    Displays:

        A figure containing two mosaics with synchronized intensity sliders.
    """
    # --- Create mosaics without opening separate figures ---
    def create_mosaic(data, mosaic_size):
        n_slices = data.shape[0]

        if mosaic_size is None:
            slices = data
        else:
            center = n_slices // 2
            slices = data[
                max(center - mosaic_size, 0):
                min(center + mosaic_size, n_slices)
            ]

        n = len(slices)
        ncols = math.ceil(np.sqrt(n))
        nrows = math.ceil(n / ncols)

        h, w = slices[0].shape
        mosaic = np.zeros((nrows * h, ncols * w))

        for i in range(n):
            r = i // ncols
            c = i % ncols
            mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = np.abs(slices[i])

        return mosaic

    mosaic1 = create_mosaic(data1, mosaic_size)
    mosaic2 = create_mosaic(data2, mosaic_size)

    h, w = data1.shape[1:]

    dx = FOV_x/w
    dy = FOV_y/h
    aspect = dy/dx

    vmax = max(np.max(mosaic1), np.max(mosaic2))
    vmin = 0

    fig, axes = plt.subplots(1, 2, figsize=(14,7))
    plt.subplots_adjust(bottom=0.22)

    im1 = axes[0].imshow(
        mosaic1,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect=aspect
    )

    im2 = axes[1].imshow(
        mosaic2,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect=aspect
    )

    axes[0].set_title(title1)
    axes[1].set_title(title2)

    axes[0].axis("off")
    axes[1].axis("off")

    # Sliders
    ax_vmin = plt.axes([0.25, 0.10, 0.5, 0.03])
    ax_vmax = plt.axes([0.25, 0.05, 0.5, 0.03])

    s_vmin = Slider(ax_vmin, "Min", 0, vmax, valinit=vmin)
    s_vmax = Slider(ax_vmax, "Max", 0, vmax, valinit=vmax)

    def update(val):
        im1.set_clim(s_vmin.val, s_vmax.val)
        im2.set_clim(s_vmin.val, s_vmax.val)
        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    # --- Save without sliders ---
    original_savefig = fig.savefig

    def save_without_sliders(*args, **kwargs):
            ax_vmin.set_visible(False)
            ax_vmax.set_visible(False)
            result = original_savefig(*args, **kwargs)
            ax_vmin.set_visible(True)
            ax_vmax.set_visible(True)
            return result

    fig.savefig = save_without_sliders

    plt.show()

def show_ratio_mosaic(data, FOV_x, FOV_y, mosaic_size=None, cmap='jet', label = 'ratio'):
    """
        Display a 3D B1 map based in the DAM method.
        Arranges them into a mosaic, and displays the full image with correct physical 
        aspect ratio and color scaling.

    Args: 

        data (np.ndarray): 3D array (nSlices, height, width)

        FOV_x (float): Field of view in readout direction (mm)

        FOV_y (float): Field of view in phase direction (mm)

        mosaic_size (int, optional): Number of slices around center

        cmap (string): matplotlib color map 

    Displays:

        B1 map in KHz. 

    """

    n_slices = data.shape[0]

    if mosaic_size is None:
        slices = data
    else:
        center = n_slices // 2
        slices = data[max(center-mosaic_size, 0):
                      min(center+mosaic_size, n_slices)]

    n = len(slices)

    ncols = math.ceil(np.sqrt(n))
    nrows = math.ceil(n / ncols)

    h, w = slices[0].shape
    mosaic = np.full((nrows * h, ncols * w), np.nan)

    for i in range(n):
        r = i // ncols
        c = i % ncols
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = slices[i]
   
    # Display the ratio directly
    ratio = mosaic

    # Define the color scale from the data
    vmin = np.nanmin(ratio)
    vmax = np.nanmax(ratio)

    # ou, se preferir uma escala fixa:
    # vmin = 0
    # vmax = 2

    dx = FOV_x / w
    dy = FOV_y / h
    aspect = dy / dx

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        ratio,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect
    )

    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)
    plt.show()

def data_mask(data, threshold):
    """
    Generates a binary mask separating signal from background based on intensity.

    The input data is normalized by its global maximum, allowing the threshold
    to be defined in relative terms (0–1).

    Args: 

        data (np.ndarray): Input magnitude image (2D or 3D).

        threshold (float): Threshold in normalized scale (0–1).
                           Values above threshold → signal (True).

    Returns:

        np.ndarray: Boolean mask with same shape as `data`:
                    True → signal
                    False → background
    """

    data_norm = data / np.max(data)

    mask = data_norm > threshold

    return mask

def zero_fill_1D(data, dim):
    """
    Applies zero-filling to a 1D acquired signal.

    Args:
        data (ndarray): Input 1D acquired signal.
        dim (int): Desired size of the zero-filled signal.

    Returns:
        ndarray: Zero-filled 1D signal.

    Raises:
        ValueError: If the desired dimension is smaller than the
                    original signal length.
    """

    data = np.asarray(data).flatten()

    # --- Original data size ---
    N = len(data)

    # --- Check desired dimension ---
    if dim < N:
        raise ValueError(
            f"Desired dimension ({dim}) must be greater than "
            f"or equal to the original data size ({N})."
        )

    # --- Calculate zero-padding ---
    padding = dim - N

    # --- Apply zero-filling symmetrically ---
    padding_left = pad
    ding // 2
    padding_right = padding - padding_left

    zero_filled_data = np.pad(data, (padding_left, padding_right), mode='constant')

    return zero_filled_data
