# Photogrammetry Laser Point Photogrammetry

This project implements a photogrammetry-based approach to recreate a 3D structure. This project was designed to be used on wind power plants but it should be possible to use it for other 3D structures.
The used measuring setup is generally comprised of two cameras, a laser and a differential optical element (DOE) that divides the laser beam into a cross grid. The laser spots are used as measuring marks on the otherwise
smooth surface.

Disclaimer: This project is at the moment part of a master thesis and still a work in progress. In the current state it is barely useable for other users and might have several bugs.

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

The goal of this project is to identify laser points in an image, estimate the laser point centers, and reconstruct the 3D structure using known camera setup parameters. Intermediate results are saved as `.npy` and `.png` files.

---

## Workflow

### Step 1: Laser spot identification
The base input is a pair of bmp files. These images lag rgb values, these files are therefore limited to an intensity value between 0-255. In the first step of the evaluation these files are read by the programm and saved
in form of `.npy` files in an 8-bit format, which increases the speed for the following steps. The files usually have low signal noise relation, to identify laser spots the array is averaged and the averaged values are
compared to each other to find maximums. The number of found laser has to be equal to continue with the evaluation. The coordinates of each found laser spot is saved in an `.npy` file and also plottet in a `.png` file.
The plot allows for easy identificationof missing or superfluous laser spots.

If to many laser spots were identified several actions can be taken.
- There are build in filters with in the programm. If the number of laser spots doesn't match, the programm asks the user to apply them.
- Laser spots are identified via 3 criterias.
	- The value of a maximum has to be larger than a certain threshold, this threshold is automatically set by the intensity distribution of the input files, but can be adjusted manually
	- The value of a maximum has to be larger than the adjacent values with in a nxn subarray, by default 7x7, but can be adjusted manually
	- The given (DOE) creates an almost evenly spaced cross grid on a sufficiently smooth surface. The average distance between adjacent maximums is compared and mis matched disqualify as maximum.

If not enough laser spots were identified similar actions can be taken.
- The most limiting factor is the set threshold setting it to a lower value usually results in the detection of all laser spots, which is the easiest fix that should be tried first. 

If the programm can't find the laser spots the given images most likely have a very low signal noise relation, which will probably effect further evaluation, ideally the intensity of the laser spots fills the entire range of
the camera from 0-255. In current test measurements background noise was usually below 50

### Step 2: Laser Point Fitting
Around each detected laser spot the main array is cut into a subarray. The size of each subarray is equal and dependend on the distance between laser spots. These subarrays are stacked into a 3D array (k,n,m), where k is the 
number laser spots found before and n and m the height and width of each subarray. This shape is choosen to calculate the laser point center (LPC) in a batcch calculation looping through the 3D array.
At the moment 5 fitting routines are available:
-Center of Mass
-Gauss Fit
-Skewed Gauss Fit, which expands the regular gaussian function with two skeweness function in x and y direction
-non linear Center of Mass, which applies an exponent >1 before calculating the center of mass to shift the focus towards larger values
-Center of Mass with threshold, which sets every value below a certain threshold to zero. The value is calculate using the mean value and deviation of the back ground noise.

The fitting routines return an array equal in size and shape to the laser spot array it is again saved in an `.npy` file and also plottet in a `.png` file. Which allows for a manuall check up. 

Other fitting routines can be easily added in the intensity_analysis package.

### Step 3: 3D Structure Reconstruction
The calculate LPCs for each pair of laser points allow for the rekonstruktion of a 3D structure. All previous coordinates were with in the pixel coordinate system of the cameras, a 2D plane equal in size to the resolution
of each camera the position and orientation of the camera allows to calculate the position of the laser spots within the real world coordinate system from there.
By default the programm assumes that the two cameras are placed at a distance of 40cm in x-direction both looking in z-direction with y as the height coordinate in a right turned coordinate system. Determining the camera 
position by measuring the distance is not sufficient, a calibration measurement is necessary as is done by an other programm, which is not ready at the moment. This programm is supposed to return a translation vector and
a rotation matrix to describe the position of the second camera, one camera is used to base the coordinate system. Distortion corrections might also be necessary which is a point of discussion in the calibration programm
but at the moment not taken into account in this programm. 

To calculate the correct 3D structure the LPCs have to be paired. At the moment the used DOE creates a central beam and a cross grid of 16x16+1 points, the additional point is used as a reference to pair all other 
laser spots. A crossgrid is constructed using the average distance between the points and all points are sorted in this cross grid and assigned a pair indices from (-8,-8) to (8,8) and paired using these.
This approach is not perfectly reliable at the moment and might be approved in the future or changed to a new method.

From this point the a pair of lines is constructed using the camera position and the converted pixel coordinates and the intersection point is calculated. With real data and the limitations in accuracy a real intersection
usually does not exist, therefore the closest point of the lines is calculated. This intersection is comprised of 3 coordinates (x,y,z) within the real coordinate system and saved  in an `.npy` file and also plottet in
a `.png` file.

Since this programm was designed to evaluate the 3D structure of wind turbine blades some plotting function might be added, that introduce smoothing of the 3D structure. 

---

## File input
	- 2 `.bmp` files
	- camera stat file (not included at the moment)
	- calibration file (not included at the moment)

## File Outputs

- **Intermediate Results**:
    - `.npy`: First estimate of the laser spot postion.
    - `.png`: Visualization of laser spot position.
	- `.npy`: Calculated LPCs.
    - `.png`: Visualization of LPCs.

- **Final Output**:
    - `.npy`: 3D structure data.
	- `.png`:  Visualization of 3D data

---

## Installation and Dependencies

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>