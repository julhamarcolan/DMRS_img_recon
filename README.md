# MRI Reconstruction and Visualization 🧲

The repository provides routines for loading MRI data, performing multi-coil root-sum-of-squares (RSS) reconstruction, when applicable, applying spatial corrections, reorienting volumes into anatomical planes, and interactively visualizing slices and mosaics with adjustable intensity windows.

## Features

- Load MRI datasets acquired with the PyMR framework.
- Organize raw data according to receiver, slice, phase-encoding, and readout dimensions.
- Perform centered  Fourier transform reconstruction.
- Perform multi-coil Root Sum of Squares (RSS) reconstruction.
- Apply optional spatial shifts for image correction.
- Reorient reconstructed volumes into:
  - Axial
  - Coronal
  - Sagittal
- Calculate physical display aspect ratios from the acquisition geometry.
- Interactively adjust image intensity using minimum and maximum intensity sliders.
- Optionally save figures automatically when the visualization window is closed.
- Save visualization figures at high resolution.
