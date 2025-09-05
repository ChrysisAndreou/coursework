# Computer Vision: Harris Corner Detector and Panorama Stitching

This project implements two fundamental computer vision algorithms from scratch as part of the MAI644 Lab coursework. The first part focuses on feature detection with a custom Harris Corner Detector. The second part involves creating a panoramic image by detecting, matching, and filtering features using a custom RANSAC algorithm to compute homographies.

The results of the custom implementations are validated against their counterparts in the OpenCV library.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Implementation Details](#implementation-details)
  - [Part 1: Harris Corner Detector](#part-1-harris-corner-detector)
  - [Part 2: Image Stitching with RANSAC](#part-2-image-stitching-with-ransac)
- [Results](#results)

## Features

- **Harris Corner Detector**: A from-scratch implementation of the Harris algorithm to detect corners in images.
- **Non-Maximum Suppression**: Uses a k-d tree to efficiently suppress corners that are too close to each other, ensuring a good spatial distribution of features.
- **SIFT Feature Matching**: Utilizes OpenCV's SIFT to detect keypoints and a Brute-Force Matcher with Lowe's ratio test to find good correspondences between images.
- **RANSAC Algorithm**: A custom implementation of the Random Sample Consensus (RANSAC) algorithm to robustly estimate the homography matrix and filter outliers.
- **Image Stitching**: Combines multiple images into a seamless panorama using the estimated homographies.
- **Comparative Analysis**: Visually compares the results of the custom Harris and RANSAC implementations with OpenCV's built-in functions.

## Project Structure

The project requires a specific directory structure for the input images:

```
.
├── code.py                 # Main Python script with all implementations
├── data/
│   ├── corners/            # Contains images for corner detection
│   │   ├── image1.png
│   │   └── ... (6 images total)
│   └── panoramas/          # Contains images for panorama stitching
│       ├── image1.jpg
│       └── ... (5 images total)
└── README.md               # This file
```

## Dependencies

The project is written in Python 3 and requires the following libraries. You can install them using pip:

```bash
pip install numpy opencv-python matplotlib scipy
```

- **NumPy**: For numerical operations.
- **OpenCV (`cv2`)**: For image processing tasks and SIFT feature detection.
- **Matplotlib**: For plotting and displaying images.
- **SciPy**: For the `KDTree` data structure used in non-maximum suppression.

## How to Run

1.  Ensure you have the required dependencies installed.
2.  Make sure the `data` directory is populated with the necessary images as described in the [Project Structure](#project-structure).
3.  Execute the script from your terminal:

    ```bash
    python code.py
    ```

The script will run both parts of the assignment sequentially and display the results in a series of Matplotlib windows.

## Implementation Details

### Part 1: Harris Corner Detector

The `detect_corners` function implements the Harris Corner Detector with the following steps:

1.  **Image Derivatives**: Computes the first-order partial derivatives (Ix, Iy) of the input image using the `cv2.Sobel()` function.
2.  **Structure Tensor**: Calculates the components of the structure tensor (I_x^2, I_y^2, I_xy) for each pixel.
3.  **Window Summation**: Applies a box filter (summation window) to the tensor components, equivalent to summing them over a `block_size` neighborhood.
4.  **Harris Response (R-score)**: Computes the Harris response score `R = det(M) - k * (trace(M))^2` for each pixel.
5.  **Thresholding**: Identifies candidate corners by thresholding the R-scores. Only pixels with a score greater than `quality_level * max(R)` are considered.
6.  **Non-Maximum Suppression (NMS)**:
    - The candidate corners are sorted in descending order of their R-scores.
    - A **k-d tree** is used to efficiently filter out corners. For each corner, we check if it is within `min_distance` of any previously selected strong corner. This ensures that the final detected corners are well-separated.

### Part 2: Image Stitching with RANSAC

This part stitches five images into a single panorama.

1.  **Feature Detection and Matching**:
    - **SIFT**: SIFT features (keypoints and descriptors) are extracted from each of the five grayscale images.
    - **Matching**: A `cv2.BFMatcher` is used to find the two nearest neighbors for each descriptor in adjacent image pairs.
    - **Ratio Test**: Lowe's ratio test is applied to filter for good matches, keeping only those where the distance to the best match is significantly smaller than the distance to the second-best match (ratio < 0.75).

2.  **RANSAC for Homography Estimation**:
    The `ransac` function estimates the homography matrix (H) that maps points from one image to another.
    - **Initialization**: The algorithm iterates for a maximum number of iterations (`max_iters`).
    - **Random Sampling**: In each iteration, it randomly selects 4 pairs of corresponding points.
    - **Homography Estimation**: It computes a candidate homography matrix `H_sample` from these 4 pairs.
    - **Inlier Counting**: The `H_sample` is used to project all source points to the destination image's coordinate system. The reprojection error (Euclidean distance between the projected point and the actual destination point) is calculated. Points with an error below `ransac_reproj_threshold` are classified as **inliers**.
    - **Model Update**: If the current `H_sample` produces the largest set of inliers found so far, it is stored as the best model.
    - **Final Homography**: After all iterations, the final homography `H` is re-estimated using a linear least-squares method (via SVD) on **all inliers** from the best model to get a more accurate result.

3.  **Panorama Creation**:
    - The homographies are chained together to map each image to the coordinate system of the first image (e.g., `H13 = H23 @ H12`).
    - Images 2 through 5 are warped using their respective homographies.
    - The warped images are stitched onto a progressively growing canvas, starting with the first image. A simple averaging blend is used in overlapping regions to create a seamless panorama.

## Results

When the script is executed, it will produce several plots:

1.  **Original Images**: The input images for both the corner detection and panorama tasks.
2.  **Corner Detection**: A side-by-side comparison of corners detected by the custom `detect_corners` function and OpenCV's `goodFeaturesToTrack` function.
3.  **SIFT Features**: The source images with detected SIFT keypoints overlaid.
4.  **Feature Matches**: Visualizations of the "good" feature correspondences found between adjacent image pairs after the ratio test.
5.  **RANSAC Inliers**: A comparison of the inlier correspondences identified by the custom `ransac` function versus OpenCV's `findHomography` with RANSAC.
6.  **Final Panoramas**: The final stitched panoramas created using homographies from both the custom implementation and OpenCV, allowing for a direct visual comparison of the results.
