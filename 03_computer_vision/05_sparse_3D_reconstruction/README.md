# Sparse 3D Reconstruction from a Stereo Pair of Images

This project implements a pipeline for sparse 3D reconstruction from two images. The process involves feature matching, estimating the fundamental matrix using RANSAC, computing epipolar lines, and triangulating 2D point correspondences to obtain 3D points. This was completed as an assignment for the MAI644 Lab 2024/2025 course.

## About The Project

The core of this project is to reconstruct a 3D scene from a pair of 2D images. This is achieved through the following key steps:

1.  **Feature Detection and Matching:** SIFT (Scale-Invariant Feature Transform) is used to detect keypoints and their descriptors in both images. These descriptors are then matched to find initial correspondences.
2.  **Fundamental Matrix Estimation:** The RANSAC (Random Sample Consensus) algorithm is employed to robustly estimate the fundamental matrix from the noisy correspondences. This step also filters out incorrect matches (outliers).
3.  **Point Normalization:** To improve the accuracy of the fundamental matrix estimation, the corresponding points are normalized before being used in the RANSAC algorithm.
4.  **Epipolar Lines Computation:** Based on the estimated fundamental matrix, epipolar lines are computed and visualized to demonstrate the epipolar constraint.
5.  **Essential Matrix Estimation:** Using the camera intrinsic parameters, the essential matrix is derived from the fundamental matrix.
6.  **Camera Pose Recovery:** The essential matrix is decomposed to retrieve the relative rotation and translation between the two cameras.
7.  **Triangulation:** The 2D image points are triangulated using the camera projection matrices to reconstruct the corresponding 3D points in the scene.
8.  **Visualization:** The results at various stages, including feature correspondences, inliers after RANSAC, epipolar lines, and the final 3D point cloud, are visualized.

## File Structure

```
├── data/
│   ├── image1.png
│   ├── image2.png
│   ├── intrinsics.npz
│   └── good_correspondences.npz
├── code.py
├── helper.py
└── README.md
```

*   `data/`: Contains the input images and camera intrinsic parameters.
*   `code.py`: The main Python script that implements the entire 3D reconstruction pipeline.
*   `helper.py`: A utility script with helper functions for visualization.
*   `README.md`: This file.

## Prerequisites

This project requires Python and several libraries.

*   Python 3.x
*   NumPy
*   OpenCV
*   Matplotlib

## Installation

1.  Clone the repository:
    ```sh
    git clone https://your-repository-url.com/your-repo.git
    ```
2.  Install the required packages:
    ```sh
    pip install numpy opencv-python matplotlib
    ```

## Running the Code

To run the project, simply execute the `code.py` script from your terminal:

```sh
python code.py
```

This will run the entire pipeline and display the visualizations for each step.

## Functionality

The `code.py` script is divided into several parts, each implementing a specific functionality:

### Part 1: Fundamental Matrix Linear System

*   `fundamental_matrix_linear_system(pts1, pts2)`: This function sets up the linear system of equations (Ax = b) to solve for the fundamental matrix. It takes in corresponding points from two images and constructs the 'A' matrix and 'b' vector.

### Part 2: Compute Epipolar Lines

*   `compute_correspond_epilines(points, which_image, F)`: This function computes the epipolar lines in one image corresponding to points in the other image, given the fundamental matrix.

### Part 3: Estimating Fundamental Matrix using RANSAC

*   `ransac(src_points, dst_points, ...)`: This function implements the RANSAC algorithm to robustly estimate the fundamental matrix. It iteratively samples a minimal set of correspondences, computes a candidate fundamental matrix, and then determines the set of inlier correspondences. The final fundamental matrix is refined using all the inliers.

### Part 4: Normalizing Points

*   `points_normalization(pts1, pts2)`: This function normalizes the image points before estimating the fundamental matrix. This pre-conditioning step is crucial for improving the accuracy and stability of the estimation.

### Part 5: 2D Points Triangulation

*   `triangulation(P1, pts1, P2, pts2)`: Given the projection matrices of the two cameras and the 2D point correspondences, this function reconstructs the 3D world points.
*   `choose_correct_RT(R1, R2, t, ...)`: After decomposing the essential matrix, there are four possible solutions for the camera's rotation and translation. This function selects the correct one by ensuring that the reconstructed 3D points are in front of both cameras.

### Visualization

The script generates several plots to visualize the intermediate and final results:

*   The original pair of images.
*   All SIFT feature correspondences after the ratio test.
*   A comparison of inlier correspondences found by the custom RANSAC implementation and OpenCV's RANSAC.
*   Epipolar lines in both images, generated from both the custom and OpenCV's fundamental matrices.
*   A 3D plot of the reconstructed sparse point cloud.
