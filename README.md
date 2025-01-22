# Photogrammetry Laser Point Analysis

This project implements a photogrammetry-based approach for identifying laser points in images and reconstructing their 3D structure. The process involves noise reduction, laser point center detection using various fitting methods, and 3D structure reconstruction.

## Table of Contents
1. [Overview](#overview)
2. [Workflow](#workflow)
    - [Step 1: Noise Reduction](#step-1-noise-reduction)
    - [Step 2: Laser Point Fitting](#step-2-laser-point-fitting)
    - [Step 3: 3D Structure Reconstruction](#step-3-3d-structure-reconstruction)
3. [File Outputs](#file-outputs)
4. [Installation and Dependencies](#installation-and-dependencies)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

---

## Overview

The goal of this project is to identify laser points in an image, reduce noise, estimate the laser point centers, and reconstruct the 3D structure using known camera setup parameters. Intermediate results are saved as `.npy` and `.png` files.

---

## Workflow

### Step 1: Noise Reduction
- Rationale: Remove background noise and highlight high-intensity laser points.
- Method: Averaging pixel values to estimate high-intensity regions.

### Step 2: Laser Point Fitting
- Isolate areas around detected high points.
- Apply various fitting methods to determine the precise laser point centers.

### Step 3: 3D Structure Reconstruction
- Reconstruct the 3D structure using camera geometry and the determined laser point centers.

---

## File Outputs

- **Intermediate Results**:
    - `.npy`: Raw data saved at various stages of processing.
    - `.png`: Visualization of laser point identification and fitting.

- **Final Output**:
    - 3D structure data in the desired format.

---

## Installation and Dependencies

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>