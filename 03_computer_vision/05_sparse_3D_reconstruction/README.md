This an academic project for the MAI644 Lab course, focusing on Sparse 3D Reconstruction from a stereo pair of images. The project walks through the fundamental steps of recovering 3D structure from 2D images, including feature matching, fundamental matrix estimation, and triangulation.

This repository contains the Python implementation for the project, divided into several key parts as outlined in the assignment. It includes implementations for estimating the fundamental matrix using the RANSAC algorithm, computing epipolar lines, normalizing points for better stability, and finally triangulating 2D point correspondences to reconstruct 3D points.

## Features

*   **Fundamental Matrix Estimation:** Calculation of the fundamental matrix from point correspondences using a linear system.
*   **Epipolar Line Computation:** Visualization of the epipolar geometry by computing and drawing epipolar lines on the stereo image pair.
*   **RANSAC for Robust Estimation:** A from-scratch implementation of the RANSAC algorithm to robustly estimate the fundamental matrix in the presence of outliers.
*   **Point Normalization:** A technique to precondition the input points to improve the accuracy and stability of the fundamental matrix estimation.
*   **3D Triangulation:** Reconstruction of 3D points from 2D correspondences given the camera projection matrices.
*   **Comparative Analysis:** The results of the custom implementation are compared against OpenCV's built-in functions for validation.

## Getting Started

### Prerequisites

*   Python 3.x
*   NumPy
*   OpenCV (`opencv-python`)
*   Matplotlib

### Installation

1.  Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2.  Install the required packages:
    ```sh
    pip install numpy opencv-python matplotlib
    ```

## Usage

To run the full pipeline for 3D reconstruction, execute the `code.py` script. Make sure the `data` directory is in the same root folder and contains the necessary files (`image1.png`, `image2.png`, `intrinsics.npz`, `good_correspondences.npz`).

```sh
python code.py
```

The script will perform the following steps:
1.  Load the stereo images and find point correspondences using SIFT and the ratio test.
2.  Estimate the fundamental matrix using both the custom RANSAC implementation and OpenCV's `findFundamentalMat`.
3.  Visualize the inlier correspondences found by both methods.
4.  Compute and display the epipolar lines on both images for both fundamental matrices.
5.  Load camera intrinsic parameters and the set of good correspondences.
6.  Estimate the essential matrix from the fundamental matrix.
7.  Decompose the essential matrix to find the rotation and translation between the cameras.
8.  Triangulate the 2D points to obtain the 3D structure of the scene.
9.  Visualize the reconstructed 3D point clouds from both the custom and OpenCV's methods.

## Visualizations

The script will generate several plots to visualize the intermediate and final results:

*   **Initial Image Pair and Correspondences:** The original stereo images and the detected SIFT feature correspondences.
*   **RANSAC Inliers:** A comparison of the inlier points identified by the custom RANSAC implementation versus OpenCV's RANSAC.
*   **Epipolar Lines:** Visualization of epipolar lines on both images, demonstrating the geometric constraints.
*   **3D Point Cloud:** A 3D plot of the reconstructed sparse point cloud of the scene.

## Acknowledgements

This project was completed as part of the MAI644 Lab course. The assignment instructions and provided data files were instrumental in the development of this solution.
