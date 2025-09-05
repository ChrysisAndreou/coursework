# Canny Edge Detector from Scratch

## Project Overview

This project is a Python implementation of the Canny Edge Detector algorithm, completed as part of the MAI644 Lab course. The implementation is built from the ground up using NumPy for numerical operations, OpenCV for image I/O and padding, and Matplotlib for visualizing results. It avoids using high-level, built-in functions for core operations like convolution or edge detection to demonstrate a fundamental understanding of the underlying computer vision concepts.

The implementation follows the canonical steps of the Canny algorithm:
1.  **Noise Reduction**: Applying a Gaussian blur to the image to reduce noise.
2.  **Image Gradient Calculation**: Using Sobel operators to find the intensity gradients of the image.
3.  **Non-maximum Suppression**: Thinning the edges to 1-pixel width.
4.  **Hysteresis Thresholding**: Identifying strong and weak edges and linking them to form the final edge map.

An extra credit "flattened" convolution method is also included.

## File Structure

-   `code.py`: The main Python script containing all function implementations and the execution pipeline.
-   `building.jpg`: The input image used for edge detection.
-   `README.md`: This file.

## Dependencies

The project requires the following Python libraries. You can install them using pip:

```bash
pip install opencv-python numpy matplotlib
```

-   **OpenCV (`cv2`)**: Used for reading the image, color space conversion, and border padding.
-   **NumPy**: Used for all numerical computations, including array manipulation and mathematical operations.
-   **Matplotlib**: Used for plotting and visualizing the original image, intermediate steps, and the final results.

## How to Run the Code

1.  Make sure you have Python and all the required libraries installed.
2.  Place the `code.py` script and the `building.jpg` image in the same directory.
3.  Run the script from your terminal:
    ```bash
    python code.py
    ```
4.  The script will execute the entire Canny edge detection pipeline and display several plot windows showing the results of each major step.

## Functionality Breakdown

The `code.py` script is organized into sections corresponding to the assignment questions.

### 1. 2D Convolution (`convolution_2D`)

-   A generic 2D convolution function that works for both single-channel (grayscale) and multi-channel (color) images.
-   It first flips the kernel (a requirement for the convolution operation).
-   It uses `cv2.copyMakeBorder` for padding to ensure the output image has the same dimensions as the input.
-   The core convolution is performed efficiently using NumPy's `stride_tricks` and `einsum` to avoid explicit loops, significantly speeding up the process.

### 2. Noise Reduction (`gaussian_kernel_2D`)

-   `gaussian_kernel_2D(ksize, sigma)`: Generates a 2D Gaussian kernel of a given size and standard deviation from scratch. It constructs the kernel by taking the outer product of a 1D Gaussian kernel with itself. The resulting kernel is normalized to sum to 1.
-   The script then uses the custom `convolution_2D` function with this Gaussian kernel to blur the input image, reducing high-frequency noise.

### 3. Image Gradient (`sobel_x`, `sobel_y`)

-   `sobel_x(arr)` and `sobel_y(arr)`: These functions calculate the first-order partial derivatives along the x and y axes, respectively. They use hardcoded Sobel kernels and the custom `convolution_2D` function. The results are normalized to the range `[-1, 1]`.
-   The script also calculates and visualizes the gradient magnitude (edge strength) and direction. The gradient direction is color-coded using the HSV color space for intuitive visualization.

### 4. Non-maximum Suppression (`non_maximum_suppression`)

-   This function is responsible for thinning the edges. It works by:
    1.  Quantizing the gradient direction into one of 8 principal directions (horizontal, vertical, and two diagonals).
    2.  For each pixel, comparing its gradient magnitude with the magnitudes of its two neighbors along the gradient direction.
    3.  If the pixel's magnitude is not the local maximum, it is suppressed (set to zero). This process results in sharp, 1-pixel-wide edges.

### 5. Hysteresis Thresholding (`hysteresis_thresholding`)

-   The final step in the Canny algorithm. This function uses two thresholds (a low and a high threshold) calculated as ratios of the maximum gradient magnitude.
-   Pixels with magnitudes above the `high_threshold` are marked as **strong edges**.
-   Pixels with magnitudes between the `low_threshold` and `high_threshold` are marked as **weak edges**.
-   The algorithm then performs edge linking: weak edges that are connected to strong edges (in their 8-pixel neighborhood) are promoted to strong edges. This process is repeated until all connected weak edges are resolved.
-   The final output is a binary image where `255` represents the detected edges.

### 6. Extra Credit: "Flattened" Convolution (`flattened_convolution_2D`)

-   An alternative implementation of 2D convolution.
-   Instead of using optimized NumPy operations, this version uses nested loops to iterate over each pixel of the input image.
-   At each position, it extracts the underlying image patch, flattens both the patch and the kernel into 1D arrays, and computes their dot product.
-   This method is more intuitive but significantly less efficient than the `einsum`-based approach.

## Expected Output

When you run the script, several Matplotlib windows will appear, showing:
1.  The original image, the custom blurred image, and the OpenCV blurred image for comparison.
2.  The Sobel X and Sobel Y derivative images.
3.  The gradient magnitude and the color-coded gradient direction images.
4.  The gradient magnitude before and after non-maximum suppression.
5.  The final edge map from the custom Canny implementation alongside the result from OpenCV's `cv2.Canny` function for comparison.
6.  A comparison of blurring using the "flattened" convolution implementation.
