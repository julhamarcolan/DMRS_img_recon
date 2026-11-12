
"""
Script for comparing MRI projections acquired using the Position Method.

The script loads two sets of projection data saved as NumPy (.npy) files:
one acquired with the sample positioned outside the center and another with the sample positioned at the center.

A single frequency axis file (freq.npy) is used for all projections.

The projections (X, Y, Z) are plotted in a 3x2 figure, allowing direct comparison between the two acquisition
 positions.

The first column shows the projections acquired outside the center in red, while the second column shows the projections 
acquired at the center in blue.

@author: Julia Marcolan
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# --- Read frequency file path from the user ---

freq_path = input("\nEnter the path to the frequency file: ").strip().strip('"').strip("'")

freq = np.load(freq_path)


# --- Read file paths from the user ---

print("\n--- Position Outside the Center ---")

path_out_x = input("\nEnter the path to the outside-center X projection file: ").strip().strip('"').strip("'")
path_out_y = input("Enter the path to the outside-center Y projection file: ").strip().strip('"').strip("'")
path_out_z = input("Enter the path to the outside-center Z projection file: ").strip().strip('"').strip("'")


print("\n--- Position at the Center ---")

path_center_x = input("\nEnter the path to the center X projection file: ").strip().strip('"').strip("'")
path_center_y = input("Enter the path to the center Y projection file: ").strip().strip('"').strip("'")
path_center_z = input("Enter the path to the center Z projection file: ").strip().strip('"').strip("'")


# --- Load projection data ---

projection_out_x = np.load(path_out_x)
projection_out_y = np.load(path_out_y)
projection_out_z = np.load(path_out_z)

projection_center_x = np.load(path_center_x)
projection_center_y = np.load(path_center_y)
projection_center_z = np.load(path_center_z)


# --- Reorganize data ---

projections_out = [projection_out_x, projection_out_y, projection_out_z]

projections_center = [projection_center_x, projection_center_y, projection_center_z]

axis_names = ['x', 'y', 'z']


# --- Plot projections ---

fig, axs = plt.subplots(3, 2, figsize=(15, 10))

for row, (projection_out, projection_center, axis_name) in enumerate(
    zip(projections_out, projections_center, axis_names)
):

    # Position outside the center
    axs[row, 0].plot(freq, projection_out, color='red', linewidth=2)
    axs[row, 0].set_title(f'Position Outside the Center - {axis_name} Axis')
    axs[row, 0].set_xlabel('Frequency (kHz)')
    axs[row, 0].set_ylabel('Signal Intensity')
    axs[row, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    axs[row, 0].grid(True)

    # Position at the center
    axs[row, 1].plot(freq, projection_center, color='#003c64ff', linewidth=2)
    axs[row, 1].set_title( f'Position at the Center - {axis_name} Axis')
    axs[row, 1].set_xlabel('Frequency (kHz)')
    axs[row, 1].set_ylabel('Signal Intensity')
    axs[row, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    axs[row, 1].grid(True)


plt.tight_layout()
#plt.savefig("position_comparison.png", dpi=300)
plt.show()
