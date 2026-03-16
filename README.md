# MRAI-Tumor-Segmentation-and-3D-Export Engine

**Developer:** Tair Fridman | B.Sc. Biomedical Engineering, Technion

## Overview<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5d9e7c40-4e3d-4645-8c83-6e9b8edcc042" />

An end-to-end Python application designed for the processing, analysis, and visualization of neuroimaging data. This tool integrates a custom-trained 3D Fully Convolutional Network (FCN) to automatically detect and segment brain tumors from MRI scans, featuring real-time clinical geometry filtering to eliminate false positives.

## Key Features
* **Multi-Format Data Pipeline:** Robust parsing and physical calibration of raw DICOM series and NIfTI volumes.
* **True-Scale AI Inference:** Utilizes a 3D U-Net architecture running at a 1:1 scale (no compression) for pixel-perfect segmentation.
* **Advanced DSP & Computer Vision:** Includes a Gaussian High-Pass Filter (FFT) module and geometric/morphological filtering (Solidity & Area thresholds) for noise reduction.
* **Interactive Orthogonal UI:** Built with Tkinter and Matplotlib, allowing synchronized navigation across Axial, Coronal, and Sagittal planes.
* **Clinical Export & Volumetry:** Automatically generates printable 3D STL meshes of the segmented regions and outputs comprehensive clinical PDF reports.

## Tech Stack
* **Deep Learning:** PyTorch
* **Image Processing:** Scikit-Image, SciPy, NumPy, Nilearn
* **Medical Formats:** NiBabel, PyDICOM
* **GUI & Visualization:** Matplotlib, Tkinter, stl-mesh

## How to Run

This project requires Python 3.8+ and a standard 64-bit OS. 

1. **Download the project**
   Clone this repository or download it as a ZIP file and extract it.
   
2. **Install dependencies**
   Open a terminal in the project folder and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Viewer**
   Ensure the pre-trained model weights file (`brats_3d_unet_full.pth`) is in the same folder, then run the main application:
   ```bash
   python MriFINAL.py
   ```
