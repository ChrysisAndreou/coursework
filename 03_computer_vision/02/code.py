import cv2
import numpy as np
import matplotlib.pyplot as plt

# part 1 : convolution_2D
def convolution_2D(arr, kernel, border_type):
    """Calculate the 2D convolution kernel*arr
    :param arr: numpy.array(float), input array
    :param kernel: numpy.array(float), convolution kernel of nxn size (only odd dimensions are allowed)
    :param border_type: int, padding method (OpenCV)
    :return: conv_arr: numpy.array(float), convolution output
    """
    # Check if kernel is square and odd-sized
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 != 0, "Kernel must be square with odd dimensions"

    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Calculate padding half the size of the kernel to have the same size as the input array 
    pad = kernel.shape[0] // 2

    def convolve_single_channel(padded_arr, kernel):
        # Get shapes
        k_h, k_w = kernel.shape
        p_h, p_w = padded_arr.shape

        '''view_shape is a tuple that describes a 4D array where the first two dimensions 
        correspond to the number of positions the kernel can be applied to, 
        and the last two dimensions correspond to the size of the kernel itself.'''
        view_shape = tuple(np.subtract(padded_arr.shape, kernel.shape) + 1) + kernel.shape
        # Strides are the number of bytes to step in each dimension when traversing an array.
        strides = padded_arr.strides + padded_arr.strides

        ''' a collection of sub-matrices, each of which is the same size as the kernel. 
        These sub-matrices represent all the possible positions the kernel can be applied to the padded array.'''
        sub_matrices = np.lib.stride_tricks.as_strided(padded_arr, view_shape, strides)

        # Perform convolution using matrix multiplication
        '''The notation 'ijkl,kl->ij' directs einsum to take a tensor A of shape (i, j, k, l) 
        and a matrix B of shape (k, l).
        It multiplies A and B over the shared dimensions k and l, and sums over these dimensions.
        The result has shape (i, j), where dimensions k and l have been reduced.'''
        return np.einsum('ijkl,kl->ij', sub_matrices, kernel)

    # add padding and convolve over 3D array or 2D array
    if len(arr.shape) == 3:
        result = np.zeros_like(arr)
        for channel in range(arr.shape[2]):
            padded_channel = cv2.copyMakeBorder(arr[:,:,channel], pad, pad, pad, pad, border_type) # same pad top bottom left right
            result[:,:,channel] = convolve_single_channel(padded_channel, kernel)
        return result
    else:
        padded_arr = cv2.copyMakeBorder(arr, pad, pad, pad, pad, border_type)
        return convolve_single_channel(padded_arr, kernel)

# part 2 : Noise Reduction
def gaussian_kernel_2D(ksize, sigma):
    """
        Calculate a 2D Gaussian kernel using the outer product of a 1D Gaussian kernel
    :param ksize: int, size of 2d kernel, always needs to be an odd number
    :param sigma: float, standard deviation of gaussian
        :return: numpy.array(float), ksize x ksize gaussian kernel with mean=0
    """
    assert ksize % 2 != 0 and ksize > 0, "ksize must be an odd positive non-zero number"
    assert sigma > 0, "sigma must be a positive non-zero number"

    # Create a 1D coordinate array centered on zero, e.g., [-1, 0, 1] for ksize=3
    coord = np.arange(ksize) - (ksize - 1) / 2

    # Calculate the 1D Gaussian kernel
    kernel_1d = np.exp(-(coord**2) / (2 * sigma**2))
    
    # In mathematical terms, a Gaussian function includes a normalization factor of 1/(sqrt(2πσ^2))
    # to ensure the total area under the curve is 1. In code, we achieve this normalization by
    # dividing the kernel by its sum, which ensures the kernel sums to 1 for discrete convolution.
    kernel_1d = kernel_1d / kernel_1d.sum() #

    # Calculate the 2D Gaussian kernel using the outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    # Check if the kernel sums to 1 (within floating-point precision)
    kernel_sum = np.sum(kernel_2d)
    assert np.isclose(kernel_sum, 1.0), f"Warning: Kernel sum is {kernel_sum}, which is not exactly 1."

    return kernel_2d


# Example usage for noise reduction
image = cv2.imread('building.jpg')
# Convert BGR to RGB for correct color display in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a Gaussian kernel
ksize = 3
sigma = 1.0
gaussian_kernel = gaussian_kernel_2D(ksize, sigma)

# Apply noise reduction using the manual Gaussian kernel and manual convolution_2D
blurred_image = convolution_2D(image, gaussian_kernel, cv2.BORDER_REPLICATE)

# OpenCV's GaussianBlur for comparison
opencv_blurred = cv2.GaussianBlur(image, (3, 3), 1, borderType=cv2.BORDER_REPLICATE)

# Plot the results
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(blurred_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Manual Gaussian Blur')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(opencv_blurred, cv2.COLOR_BGR2RGB))
plt.title('OpenCV GaussianBlur')
plt.axis('off')

plt.tight_layout()
plt.show()

# Part 3 : image Gradient
def sobel_x(arr):
    """
        Calculate the 1st order partial derivatives along x-axis
    :param arr: numpy.array(float), input image
    :return: 
        dx: numpy.array(float), output partial derivative
    """
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    dx = convolution_2D(arr, sobel_x_kernel, cv2.BORDER_REPLICATE)
    # Normalize the output to the range (-1, 1] to retain relative edge strength
    return dx / np.max(np.abs(dx))

def sobel_y(arr):
    """
        Calculate the 1st order partial derivatives along y-axis
    :param arr: numpy.array(float), input image
    :return: 
        dy: numpy.array(float), output partial derivatives
    """
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    dy = convolution_2D(arr, sobel_y_kernel, cv2.BORDER_REPLICATE)
    return dy / np.max(np.abs(dy))

def calculate_gradient(dx, dy):
    """Calculate gradient magnitude and direction
    :param dx: numpy.array(float), x-derivative
    :param dy: numpy.array(float), y-derivative
    :return: magnitude, direction
    """
    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude = magnitude / np.max(magnitude)  # Normalize to [0, 1]
    direction = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    return magnitude, direction

def direction_to_hsv(magnitude, direction):
    """Convert gradient magnitude and direction to HSV image with white background
    :param magnitude: numpy.array(float), gradient magnitude
    :param direction: numpy.array(float), gradient direction in degrees
    :return: hsv_image
    """
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)  # Shape: (height, width, 3) for HSV channels
    hsv[..., 0] = (direction + 180) * 179 / 360  # hue : Convert direction (-180 to 180) to hue (0 to 179)
    hsv[..., 1] = (magnitude * 255).astype(np.uint8)  # Saturation scaling the normalized magnitude (0 to 1) to the range 0 to 255
    hsv[..., 2] = 255  # Value (always white background) 
    return hsv

# sobel x and y
image = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

# Apply noise reduction using the manual Gaussian kernel and manual convolution_2D
blurred_image = convolution_2D(image, gaussian_kernel, cv2.BORDER_REPLICATE)

# Calculate Sobel derivatives
dx = sobel_x(blurred_image)
dy = sobel_y(blurred_image)

# Plot the Sobel X and Y images in a 1x2 grid
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(dx, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(122)
plt.imshow(dy, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate gradient magnitude and direction
magnitude, direction = calculate_gradient(dx, dy)

# Convert direction to HSV
hsv_direction = direction_to_hsv(magnitude, direction)
rgb_direction = cv2.cvtColor(hsv_direction, cv2.COLOR_HSV2RGB)

# Plot the Gradient Magnitude and Direction in a 1x2 grid
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(122)
plt.imshow(rgb_direction)
plt.title('Gradient Direction')
plt.axis('off')

plt.tight_layout()
plt.show()

# part 4 : non-maximum suppression 
def non_maximum_suppression(arr_mag, arr_dir):
    """
        Find all local maxima along image gradient direction
    :param arr_mag: numpy.array(float), input image gradient magnitude
    :param arr_dir: numpy.array(float), input image gradient direction
    :return:
        arr_local_maxima: numpy.array(float)
    """

    h, w = arr_mag.shape
    arr_local_maxima = np.zeros((h, w), dtype=arr_mag.dtype)

    # Quantize the directions into 8 bins (0-7), each covering 45 degrees
    # Adjust angles to range [0, 360) and shift by 22.5 to center bins
    # Use modulo 360 to wrap angles within [0, 360) and modulo 8 to cycle through 8 direction bins
    direction_quantized = (((arr_dir + 22.5) % 360) // 45).astype(int) % 8  

    for i in range(8):
        # Create a mask for pixels in the current direction bin
        mask = (direction_quantized == i)

        # Initialize shifted arrays for neighbor comparisons
        shifted_plus = np.zeros_like(arr_mag)
        shifted_minus = np.zeros_like(arr_mag)

        if i == 0:
            # Direction 0 (0 degrees): Horizontal (left-right)
            shifted_plus[:, :-1] = arr_mag[:, 1:]    # Right neighbor
            shifted_minus[:, 1:] = arr_mag[:, :-1]   # Left neighbor
        elif i == 1:
            # Direction 1 (45 degrees): Diagonal (upper-right to lower-left)
            shifted_plus[:-1, :-1] = arr_mag[1:, 1:]     # Lower-right neighbor
            shifted_minus[1:, 1:] = arr_mag[:-1, :-1]    # Upper-left neighbor
        elif i == 2:
            # Direction 2 (90 degrees): Vertical (up-down)
            shifted_plus[:-1, :] = arr_mag[1:, :]    # Bottom neighbor
            shifted_minus[1:, :] = arr_mag[:-1, :]   # Top neighbor
        elif i == 3:
            # Direction 3 (135 degrees): Diagonal (upper-left to lower-right)
            shifted_plus[:-1, 1:] = arr_mag[1:, :-1]     # Lower-left neighbor
            shifted_minus[1:, :-1] = arr_mag[:-1, 1:]    # Upper-right neighbor
        elif i == 4:
            # Direction 4 (180 degrees): Horizontal (left-right)
            shifted_plus[:, 1:] = arr_mag[:, :-1]    # Left neighbor
            shifted_minus[:, :-1] = arr_mag[:, 1:]   # Right neighbor
        elif i == 5:
            # Direction 5 (225 degrees): Diagonal (lower-left to upper-right)
            shifted_plus[1:, 1:] = arr_mag[:-1, :-1]     # Upper-left neighbor
            shifted_minus[:-1, :-1] = arr_mag[1:, 1:]    # Lower-right neighbor
        elif i == 6:
            # Direction 6 (270 degrees): Vertical (up-down)
            shifted_plus[1:, :] = arr_mag[:-1, :]    # Top neighbor
            shifted_minus[:-1, :] = arr_mag[1:, :]   # Bottom neighbor
        elif i == 7:
            # Direction 7 (315 degrees): Diagonal (lower-right to upper-left)
            shifted_plus[1:, :-1] = arr_mag[:-1, 1:]     # Upper-right neighbor
            shifted_minus[:-1, 1:] = arr_mag[1:, :-1]    # Lower-left neighbor

        # Suppress non-maxima by keeping only the local maxima along the gradient direction
        # cond is a boolean array indicating local maxima 
        cond = (arr_mag >= shifted_plus) & (arr_mag >= shifted_minus) & mask 
        arr_local_maxima[cond] = arr_mag[cond]

    return arr_local_maxima

# Apply non-maximum suppression
suppressed = non_maximum_suppression(magnitude, direction)

# Plot the results
plt.figure(figsize=(10, 5))  # Adjusted figure size for two plots

plt.subplot(121)
plt.imshow(magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(122)
plt.imshow(suppressed, cmap='gray')
plt.title('Non-Maximum Suppression')
plt.axis('off')

plt.tight_layout()
plt.show()

# part 5: hysteresis_thresholding
def hysteresis_thresholding(arr, low_ratio, high_ratio):
    """
        Use the low and high ratios to threshold the non-maximum suppression image and then link
        non-weak edges
    :param arr: numpy.array(float), input non-maximum suppression image
    :param low_ratio: float, low threshold ratio
    :param high_ratio: float, high threshold ratio
    :return: 
        edges: numpy.array(uint8), output edges
    """
    # Calculate thresholds
    high_threshold = arr.max() * high_ratio
    low_threshold = arr.max() * low_ratio

    # Initialize output
    strong_edges = (arr > high_threshold)
    weak_edges = (arr > low_threshold) & (arr <= high_threshold)
    edges = np.zeros(arr.shape, dtype=np.uint8)
    edges[strong_edges] = 255

    # Recursive function to link edges
    def link_edges(i, j):
        # If the current pixel is already marked as a strong edge, return
        if edges[i, j] == 255:
            return
        # Mark the current pixel as a strong edge
        edges[i, j] = 255
        # Get the dimensions of the edges array
        h, w = edges.shape
        # Iterate over the 3x3 neighborhood of the current pixel
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                # Calculate the neighbor's indices
                ni, nj = i + di, j + dj
                # Check if the neighbor is within the image boundaries
                if 0 <= ni < h and 0 <= nj < w and weak_edges[ni, nj]:
                    # If the neighbor is a weak edge, recursively link it
                    link_edges(ni, nj)

    # Iterate over each pixel in the edges array
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            # If the current pixel is a strong edge
            if strong_edges[i, j]:
                # Start linking process from this strong edge
                link_edges(i, j)

    return edges

# Apply hysteresis thresholding
low_ratio = 0.1  
high_ratio = 0.2  
edges = hysteresis_thresholding(suppressed, low_ratio, high_ratio)

# OpenCV's Canny Edge Detector
opencv_edges = cv2.Canny(image=blurred_image.astype(np.uint8), threshold1=100, threshold2=200)

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.imshow(edges, cmap='gray')
plt.title('Custom Canny Edge Detector')
plt.axis('off')

plt.subplot(122)
plt.imshow(opencv_edges, cmap='gray')
plt.title('OpenCV Canny Edge Detector')
plt.axis('off')

plt.tight_layout()
plt.show()

# part 6: fltatened_convolution_2D
def flattened_convolution_2D(arr, kernel, border_type):
    """Calculate the 2D convolution kernel*arr using flattened convolution operation
    :param arr: numpy.array(float), input array
    :param kernel: numpy.array(float), convolution kernel of nxn size (only odd dimensions are allowed)
    :param border_type: int, padding method (OpenCV)
    :return: conv_arr: numpy.array(float), convolution output
    """
    # Check if kernel is square and odd-sized
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 != 0, "Kernel must be square with odd dimensions"

    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Calculate padding: half the size of the kernel to ensure the output has the same size as the input array
    pad = kernel.shape[0] // 2

    def convolve_single_channel(padded_arr, kernel):
        # Get kernel height and width
        k_h, k_w = kernel.shape

        # Initialize the result array
        # result_h and result_w are set to match the original image dimensions
        result_h, result_w = padded_arr.shape[0] - 2 * pad, padded_arr.shape[1] - 2 * pad
        result = np.zeros((result_h, result_w))

        # Flatten the kernel into a 1D array
        kernel_flat = kernel.flatten()

        # Perform flattened convolution by sliding the kernel over the input array
        for i in range(result_h):
            for j in range(result_w):
                # Extract the sub-matrix corresponding to the current position of the kernel
                sub_matrix = padded_arr[i:i + k_h, j:j + k_w]

                # Flatten the sub-matrix and compute the dot product with the flattened kernel
                sub_matrix_flat = sub_matrix.flatten()
                result[i, j] = np.dot(sub_matrix_flat, kernel_flat)

        return result

    # Add padding and convolve over a 3D array (e.g., color image) or 2D array (e.g., grayscale image)
    if len(arr.shape) == 3:
        result = np.zeros_like(arr)
        for channel in range(arr.shape[2]):
            # Pad the channel with the specified border type
            padded_channel = cv2.copyMakeBorder(arr[:, :, channel], pad, pad, pad, pad, border_type)
            # Convolve the padded channel with the kernel
            result[:, :, channel] = convolve_single_channel(padded_channel, kernel)
        return result
    else:
        # For 2D arrays (grayscale images)
        padded_arr = cv2.copyMakeBorder(arr, pad, pad, pad, pad, border_type)
        return convolve_single_channel(padded_arr, kernel)

image = cv2.imread('building.jpg')
# Convert BGR to RGB for correct color display in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a Gaussian kernel
ksize = 3
sigma = 1.0
gaussian_kernel = gaussian_kernel_2D(ksize, sigma)

# Apply noise reduction using the manual Gaussian kernel and manual convolution_2D
blurred_image_with_flattened = flattened_convolution_2D(image, gaussian_kernel, cv2.BORDER_REPLICATE)

# OpenCV's GaussianBlur for comparison
opencv_blurred = cv2.GaussianBlur(image, (3, 3), 1, borderType=cv2.BORDER_REPLICATE)

# Plot the results
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(blurred_image_with_flattened.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Manual Gaussian Blur with flattened convolution')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(opencv_blurred, cv2.COLOR_BGR2RGB))
plt.title('OpenCV GaussianBlur')
plt.axis('off')

plt.tight_layout()
plt.show()
