# MAI644 Lab 2024/2025: Image Segmentation

This project, part of the MAI644 Lab for the 2024/2025 academic year, focuses on the implementation and application of two fundamental image segmentation techniques: K-Means Clustering and Efficient Graph-Based Image Segmentation.

## Project Overview

The primary goal of this assignment is to gain hands-on experience with popular clustering and segmentation algorithms. The project is divided into two main parts:

1.  **K-Means Clustering:** This section involves implementing the K-Means algorithm to segment an image by clustering its pixel values. The results are then compared with the implementation provided by the OpenCV library.

2.  **Efficient Graph-Based Image Segmentation:** This part requires the implementation of a more advanced segmentation technique based on graph theory. This method treats the image as a graph and partitions it into segments based on the weights of the edges connecting pixels.

## Features

*   **K-Means Clustering:**
    *   Custom implementation of the K-Means clustering algorithm.
    *   Application of K-Means using two different feature sets:
        *   (R, G, B) color values.
        *   (i, j, R, G, B) pixel coordinates and color values.
    *   Side-by-side comparison with OpenCV's built-in K-Means function.

*   **Efficient Graph-Based Image Segmentation:**
    *   Image smoothing using Gaussian blurring as a preprocessing step.
    *   Construction of a k-Nearest Neighbors (k-NN) graph from the image pixels.
    *   Implementation of the graph-based segmentation algorithm.
    *   Post-processing step to merge small, adjacent segments for a cleaner result.
    *   Visualization of the final segmented image with distinct colors for each segment.

## Prerequisites

Before running the project, you need to have Python installed along with the following libraries:

*   OpenCV
*   NumPy
*   Matplotlib
*   Scikit-learn

## Getting Started

### Installation

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install opencv-python numpy matplotlib scikit-learn
    ```

### Directory Structure

Ensure your project directory is set up as follows:

```
.
├── code.py
└── data/
    ├── home.jpg
    └── eiffel_tower.jpg
```

## How to Run

To execute the project and see the segmentation results, run the `code.py` script from your terminal:

```bash
python code.py
```

## Expected Output

The script will generate and display several plots using Matplotlib:

1.  **Original Image (`home.jpg`):** The initial image for the K-Means clustering part.
2.  **K-Means with (R, G, B) Features:** A comparison of your K-Means implementation and OpenCV's.
3.  **K-Means with (i, j, R, G, B) Features:** A second comparison using both color and coordinate information.
4.  **Graph-Based Segmentation Results:**
    *   The original `eiffel_tower.jpg` image.
    *   The segmented image without the post-processing step.
    *   The final segmented image after merging small clusters.

## Code Overview

### Part 1: K-Means Clustering

*   `kmeans(data, K, thresh, n_iter, n_attempts)`: This function contains the custom implementation of the K-Means algorithm. It takes the data, number of clusters, convergence threshold, and other parameters as input and returns the compactness, labels, and centers of the clusters.
*   **Data Preparation:** The script first reshapes the input image into a 2D array of pixels. It creates two versions of this data: one with just RGB values and another that includes the (i, j) coordinates of each pixel.
*   **Comparison:** Both the custom `kmeans` function and `cv2.kmeans` are run on both datasets.
*   **Visualization:** The resulting segmented images are reshaped to their original dimensions and displayed.

### Part 2: Efficient Graph-Based Image Segmentation

*   `nn_graph(input_image, k)`: This function builds a k-Nearest Neighbors graph from the input image. It uses `sklearn.neighbors.NearestNeighbors` to find the closest pixels in the feature space (i, j, r, g, b).
*   `segmentation(G, k_param, min_size, post_process=True)`: This is the core of the graph-based segmentation algorithm. It sorts the graph edges by weight and iteratively merges clusters based on a defined criterion. It also includes an optional post-processing step to merge smaller segments.
*   `map_clusters_to_colors(clusters, image_shape)`: A utility function to assign a unique color to each segment for visualization.
*   `plot_results(...)`: A helper function to display the original and segmented images side-by-side.

## Adjustable Parameters

You can modify the following parameters within the `code.py` script to experiment with the results:

*   **For K-Means:**
    *   `K`: The number of clusters.
    *   `thresh`: The convergence threshold for the centroids.
    *   `n_iter`: The maximum number of iterations for each attempt.
    *   `n_attempts`: The number of times the algorithm is run with different initializations.

*   **For Graph-Based Segmentation:**
    *   `k_nn`: The number of nearest neighbors to consider when building the graph.
    *   `k_param`: A parameter that influences the merging of segments. Larger values lead to larger segments.
    *   `min_size`: The minimum size for a segment in the post-processing step.
