# MRI Reconstruction and Visualization 🧲

The repository provides routines for loading MRI data, performing multi-coil root-sum-of-squares (RSS) reconstruction, when applicable, applying spatial corrections, reorienting volumes into anatomical planes, and interactively visualizing slices and mosaics with adjustable intensity windows.

## Features

\- `Position.py` – Processes Position Method acquisition data, with optional 1D zero-filling, FFT calculation, data saving, and visualization of x, y, and z projections.

\- `Position_Projection_Comparison.py` – Loads and compares x, y, and z projections acquired at the center and outside the center, using a shared frequency axis and side-by-side plots.
