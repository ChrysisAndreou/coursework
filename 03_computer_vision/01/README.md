# Assignment 1: Python Introduction (MAI644)

## Project Overview

This project is a submission for "Assignment 1: Python Introduction" for the MAI644 Lab course. The assignment is designed to build familiarity with fundamental Python libraries for scientific computing and computer vision, namely **NumPy**, **Matplotlib**, and **OpenCV**.

The project is divided into three main parts:
1.  **Basic Matrix/Vector Manipulation:** Performing various linear algebra operations using NumPy.
2.  **Basic Image Manipulations:** Loading, processing, and combining images using OpenCV and NumPy.
3.  **Bilinear Interpolation:** Implementing image resizing from scratch and comparing it with OpenCV's built-in function.

---

## File Structure

-   `01_code.ipynb`: The Jupyter Notebook containing all the code and solutions for the assignment.
-   `image1.jpg`: Input image used in Part 2.
-   `image2.jpg`: Input image used in Part 2.
-   `image3.png`: Input image used in Part 3 for resizing.
-   `README.md`: This documentation file.

---

## Requirements

The project requires Python 3 and the following libraries:

-   `numpy`
-   `matplotlib`
-   `opencv-python`

You can install these dependencies using `pip`. It is recommended to use a virtual environment to manage project dependencies.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows, use: venv\Scripts\activate
# On macOS/Linux, use: source venv/bin/activate

# Install the required packages
pip install numpy matplotlib opencv-python
```

---

## How to Run

1.  **Clone the repository or download the files.** Ensure all files (`01_code.ipynb`, `image1.jpg`, `image2.jpg`, `image3.png`) are in the same directory.
2.  **Install the dependencies** as described in the "Requirements" section.
3.  **Launch Jupyter Notebook or JupyterLab** and open the `01_code.ipynb` file.
    ```bash
    jupyter notebook 01_code.ipynb
    ```
4.  **Run the cells** in the notebook sequentially. The outputs, including printed values for Part 1 and image plots for Parts 2 and 3, will be displayed directly in the notebook.

---

## Code Implementation Details

The Jupyter notebook `01_code.ipynb` is structured into three cells, corresponding to the three main parts of the assignment.

### Part 1: Basic Matrix/Vector Manipulation

This section covers questions 1(a) to 1(i) from the assignment.
-   **1(a):** Defines a 4x3 matrix `M` and three 1D vectors `a`, `b`, `c`.
-   **1(b-d):** Calculates the dot product, element-wise product, and the expression `Ma(c^T b)`.
-   **1(e-f):** Finds the magnitude of vectors `a` and `b` using a custom-defined function `manual_magnitude` (as required by the assignment) and then normalizes them.
-   **1(g):** Computes the angle between vectors `a` and `c` in degrees.
-   **1(h):** Finds a vector `d` that is perpendicular to both `a` and `b` using `np.cross()`.
-   **1(i):** Creates an orthonormal basis from the normalized versions of `a`, `b`, and `d`. It verifies the result by checking if `Q @ Q.T` is close to the identity matrix, where `Q` is the matrix formed by the basis vectors.

### Part 2: Basic Image Manipulations

This section covers questions 2(a) to 2(e).
-   **2(a):** Loads `image1.jpg` and `image2.jpg` and displays them side-by-side after converting their color space from BGR to RGB for Matplotlib.
-   **2(b):** Converts the images to `float64` (double precision) and rescales their pixel values to the range [0, 1].
-   **2(c):** Adds the two rescaled images and re-normalizes the resulting image to the range [0, 1] using min-max normalization.
-   **2(d):** Creates a composite image by combining the left half of `image2` with the right half of `image1`.
-   **2(e):** Creates a new image where even-numbered rows are taken from `image1` and odd-numbered rows are from `image2`, using NumPy array slicing.

### Part 3: Bilinear Interpolation & Extra Credit

This section covers question 2(f) and the extra credit task.
-   **`bilinear_interpolation_resize` function:** A custom function is implemented to resize an image. It calculates the corresponding coordinates in the source image for each pixel in the target image and performs a weighted average of the four nearest neighboring pixels.
-   **Task 2(f):** Resizes `image3.png` by a factor of 5 using the custom function and compares the output with OpenCV's `cv2.resize` (with `interpolation=cv2.INTER_LINEAR`).
-   **Extra Credit:** The same custom function is used to perform a non-symmetric resize of `image3.png` to 160x90 pixels.
-   **Verification:** The provided `check_resizing` function is used in both cases to verify that the custom implementation's pixel values are in 100% agreement with OpenCV's results within a tolerance of 1.
