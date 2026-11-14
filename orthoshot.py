"""
This script uses the CIERMag managers to read the HDF5 file from orthoshot Method and extract the data 
and the execution protocol. 

This file, read the data from a 2D  image and plot the image.

@author: Julia Marcolan 
"""

import sys
import os
import numpy as np
import scipy as sc
import pyqtgraph as pg
import matplotlib.pyplot as plt
import reconstruction_functions as mri_recon
import nibabel as nib


from pymr import config
from pymr.core.data import DataManager
from pymr.core.protocol import ProtocolManager
from pymr.gui.mw_processing import WidgetManager
from pymr.gui.mw_processing import InteractiveWidget
from ciermag_spec.functions import logger


# --- Read filename from the user ---
print("\nEnter the name of the file (with or without the .h5 extension):")

data_path = input("Dataset: ").strip()

# Remove quotation marks if present
data_path = data_path.strip('"').strip("'")

# Add the extension automatically if omitted
if not data_path.lower().endswith(".h5"):
    data_path += ".h5"

print("\nDataset to be loaded:")
print(f"  - {data_path}")


# --- Initialize managers ---
prot_mng = ProtocolManager()
data_mng = DataManager()
widget_mng = WidgetManager()

logger.info(f"Paths: {data_path}")
logger.info(f"Current CWD: {os.getcwd()}")

prot_mng.open_protocols(data_path=data_path)
dataset = data_mng.read_data(data_path)

logger.info(f"Original dataset shape: {dataset.shape}")


# reorganize data in the desired dimensions
type_indexer = 'type'
type_order_list = ['receiver','slice', 'phEnc', 'readout']

data, labels, types = data_mng.get_data_from_indexer(type_indexer, type_order_list)

logger.info(f"Labels and types: {labels}, {types}")
logger.info(f"New data shape: {data.shape}")

# --- Reading parameters from the execution protocol --- 
num_planes = prot_mng.get_parameter("C_Numplanes")
logger.info(f"Num planes: {num_planes}")

FOV_x = prot_mng.get_parameter("FOVr", "system")
FOV_y = prot_mng.get_parameter("FOVp2","system")



if num_planes in [1, 3]:

    if num_planes == 1:

        data_2D = data[:, 0, :, :]

        logger.info(f"2D k-space shape: {data_2D.shape}")

        # --- Display k-space ---

        kspace_rss = np.sqrt(np.sum(np.abs(data_2D)**2, axis=0))

        mri_recon.show_k_space(
            kspace_rss,
            FOV_x=FOV_x,
            FOV_y=FOV_y
        )

        # --- Apply 2D multi-coil RSS reconstruction ---

        data_img_original = mri_recon.reconstruct_rss2D(data_2D)

        logger.info(f"Reconstructed image shape: {data_img_original.shape}")

        # --- Display reconstructed image ---

        mri_recon.show_single_slice(
            data_img_original[np.newaxis, :, :],
            FOV_x=FOV_x,
            FOV_y=FOV_y,
            slice_number=0,
            save_image=False,
            output_filename=None
        )

        # --- Save K-space and reconstructed image as NIfTI ---
        #mri_recon.save_nifti_2D(kspace_rss[np.newaxis, :, :], data_img_original[np.newaxis, :, :], output_folder='NIfTI')

    elif num_planes == 3:

        logger.info("Reconstructing three planes.")

        reconstructed_images = []
        kspace_planes = []

        plane_names = ['Axial', 'Sagittal', 'Coronal']

        for i in range(num_planes):

            logger.info(f"Reconstructing {plane_names[i]} plane.")

            # Select current plane from all receivers
            # Expected shape: (nRcvrs, nPhase, nReadout)

            data_2D = data[:, i, :, :]

            logger.info(f"{plane_names[i]} k-space shape: {data_2D.shape}")

            # --- Calculate RSS k-space ---

            kspace_rss = np.sqrt(np.sum(np.abs(data_2D)**2, axis=0))

            kspace_planes.append(kspace_rss)

            # --- Apply RSS reconstruction ---

            image_2D = mri_recon.reconstruct_rss2D(data_2D)

            logger.info(f"{plane_names[i]} image shape: {image_2D.shape}")

            reconstructed_images.append(image_2D)

        # --- Stack k-space planes ---

        kspace_data = np.stack(kspace_planes, axis=0)

        # --- Display k-space mosaic ---

        mri_recon.show_k_space_grid(
            kspace_data,
            3,
            FOV_x=FOV_x,
            FOV_y=FOV_y
        )
        
        # --- Save K-space and reconstructed image as NIfTI ---
        #kspace_data = np.stack(kspace_planes, axis=0)
        #mri_recon.save_nifti_2D(kspace_data, output_folder='NIfTI')

        # --- Stack reconstructed images ---

        data_img_original = np.stack(reconstructed_images, axis=0)

        logger.info(f"Final reconstructed data shape: {data_img_original.shape}")

        # --- Display reconstructed images as a mosaic ---

        mri_recon.show_mosaic(
            data_img_original,
            FOV_x=FOV_x,
            FOV_y=FOV_y,
            mosaic_size=None,
            save_image=False,
            output_filename=None,
            first_slice_blank=False
        )

        # --- Save K-space and reconstructed image as NIfTI ---
        mri_recon.save_nifti_2D(kspace_rss, data_img_original, output_folder='NIfTI')



    else:

        logger.warning("It is only possible to choose 1 or 3 planes.")
