# MRI Reconstruction and Visualization 🧲

The repository provides routines for loading MRI data, performing multi-coil root-sum-of-squares (RSS) reconstruction, when applicable, applying spatial corrections, reorienting volumes into anatomical planes, and interactively visualizing slices and mosaics with adjustable intensity windows.

## Features

\- `reconstruction_functions.py` – Functions and classes for MRI and NMR reconstruction and visualization.

\- `read_param_from_protocol.py` – Script for retrieving and displaying a specific parameter value from a DMRS  acquisition protocol. 

\- `position.py` – Processes Position Method acquisition data, with optional 1D zero-filling, FFT calculation, data saving, and visualization of x, y, and z projections.

\- `position_center_comparison.py` – Loads and compares x, y, and z projections acquired at the center and outside the center, using a shared frequency axis and side-by-side plots.

\- `orthoshot.py` –  This script uses the CIERMag managers to read the HDF5 file from orthoshot Method and extract the data  and the execution protocol. 
