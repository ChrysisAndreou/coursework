import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import glob

def detect_corners(input_image, max_corners=0, quality_level=0.01, min_distance=10, block_size=5, k=0.05):
    """
        Detect corners using Harris Corner Detector
    :param input_image: numpy.array(uint8 or float), input 8-bit or foating-point 32-bit, single-channel
                image
    :param max_corners: int, maximum number of corners to return, if 0 then return all
    :param quality_level: float, parameter characterizing the minimal accepted quality of image corners
    :param min_distance: float, minimum possible Euclidean distance between the returned corners
    :param block_size: int, size of an average block for computing a derivative covariation matrix
                    over each pixel neighborhood.
    :param k: float, free parameter of the Harris detector
    :return:
        corners: numpy.array(uint8)), corner coordinates for each input image
    """
    # input validation for image
    assert isinstance(input_image, np.ndarray), "input_image must be a numpy array"
    assert input_image.ndim == 2, "input_image must be single-channel"
    assert input_image.dtype in [np.uint8, np.float32], "input_image must be uint8 or float32"
    
    # Input validation
    assert isinstance(max_corners, int) and max_corners >= 0, "max_corners must be a non-negative integer"
    assert 0 <= quality_level <= 1, "quality_level must be between 0 and 1"
    assert min_distance >= 0, "min_distance must be non-negative"
    assert isinstance(block_size, int) and block_size > 0 and block_size % 2 == 1, "block_size must be positive odd integer"
    assert 0.04 <= k <= 0.06, "k must be between 0.04 and 0.06"

    # Convert image to float32
    img = input_image.astype(np.float32)

    # Calculate derivatives
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Compute components of the structure tensor
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # Apply block averaging
    window_size = block_size
    dx2_sum = cv2.boxFilter(dx2, -1, (window_size, window_size))
    dy2_sum = cv2.boxFilter(dy2, -1, (window_size, window_size))
    dxy_sum = cv2.boxFilter(dxy, -1, (window_size, window_size))

    # Calculate Harris response
    det = dx2_sum * dy2_sum - dxy_sum * dxy_sum
    trace = dx2_sum + dy2_sum
    harris_response = det - k * trace * trace

    # Threshold and get candidate corners
    max_response = harris_response.max()
    threshold = quality_level * max_response
    corner_mask = harris_response > threshold
    
    # Get coordinates and scores of candidate corners
    corner_coords = np.column_stack(np.where(corner_mask))
    corner_scores = harris_response[corner_mask]
    
    # Sort corners by score
    sorted_indices = np.argsort(-corner_scores)
    corner_coords = corner_coords[sorted_indices]
    corner_scores = corner_scores[sorted_indices]

    # Modify the coordinates to return (x,y) format instead of (y,x)
    corner_coords = np.fliplr(corner_coords)

    # Apply non-maximum suppression using KD-tree
    if len(corner_coords) > 0:
        selected_corners = []
        kdtree = KDTree(corner_coords)
        
        for i in range(len(corner_coords)):
            if len(selected_corners) == max_corners and max_corners > 0:
                break
                
            point = corner_coords[i]
            if not selected_corners or all(np.linalg.norm(point - sc) >= min_distance for sc in selected_corners):
                selected_corners.append(point)
        
        corners = np.array(selected_corners)
    else:
        corners = np.array([])

    return corners



# Load and process images
image_paths = sorted(glob.glob('data/corners/*.png'))
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Parameters
params = {
    'max_corners': 0,
    'quality_level': 0.01,
    'min_distance': 10.0,
    'block_size': 5,
    'k': 0.05
}

# Process each image
opencv_results = []
our_results = []

for img in images:
    # OpenCV implementation
    opencv_corners = cv2.goodFeaturesToTrack(
        image=img,
        maxCorners=params['max_corners'],
        qualityLevel=params['quality_level'],
        minDistance=params['min_distance'],
        blockSize=params['block_size'],
        useHarrisDetector=True,
        k=params['k']
    )
    
    # Our implementation
    our_corners = detect_corners(img, **params)
    
    # Draw results
    opencv_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    our_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if opencv_corners is not None:
        opencv_corners = opencv_corners.reshape(-1, 2)
        for x, y in opencv_corners:
            cv2.circle(opencv_img, (int(x), int(y)), 3, (0, 255, 0), -1)
            
    for x, y in our_corners:
        cv2.circle(our_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        
    opencv_results.append(opencv_img)
    our_results.append(our_img)

# Plot results
def plot_corners(images, titles, num_rows=2, num_cols=3):
    """Helper function to plot images with detected corners"""
    plt.figure(figsize=(15, 10))
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_corners(images, ['Original Image ' + str(i+1) for i in range(len(images))])
plot_corners(opencv_results, ['OpenCV Harris Corners ' + str(i+1) for i in range(len(images))])
plot_corners(our_results, ['Our Harris Corners ' + str(i+1) for i in range(len(images))])


# part 2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Helper function to display images in a grid
def plot_images(images, titles=None, cols=5, figsize=(20, 5)):
    rows = len(images) // cols + int(len(images) % cols > 0)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load images
image_paths = sorted(glob.glob('data/panoramas/*.jpg'))
images_color = [cv2.imread(path) for path in image_paths]
images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_color]

# Display images in a 1x5 grid
plot_images(images_color, titles=['Image {}'.format(i+1) for i in range(len(images_color))], cols=5)

# Compute SIFT features
sift = cv2.SIFT_create()
keypoints_list = []
descriptors_list = []

for img in images_gray:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Plot detected SIFT features in a 1x5 grid
images_with_keypoints = []
for img, keypoints in zip(images_color, keypoints_list):
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    images_with_keypoints.append(img_with_keypoints)

plot_images(images_with_keypoints, titles=['SIFT Features Image {}'.format(i+1) for i in range(len(images_with_keypoints))], cols=5)

# Match features between adjacent image pairs
bf = cv2.BFMatcher(cv2.NORM_L2)
pairs = [(i, i+1) for i in range(len(images_gray)-1)]
matches_list = []

for i, j in pairs:
    descriptors1 = descriptors_list[i]
    descriptors2 = descriptors_list[j]
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches_list.append(matches)

# Apply ratio test to identify good matches
good_matches_list = []

for matches in matches_list:
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches_list.append(good_matches)

# Plot good correspondences in a 2x2 grid
correspondence_images = []
for idx, (i, j) in enumerate(pairs):
    img1 = images_color[i]
    img2 = images_color[j]
    keypoints1 = keypoints_list[i]
    keypoints2 = keypoints_list[j]
    good_matches = good_matches_list[idx]
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    correspondence_images.append(img_matches)

plot_images(correspondence_images, titles=['Good Matches between Image {} and {}'.format(i+1, j+1) for i, j in pairs], cols=2, figsize=(15, 10))

# Extract source and destination points for RANSAC
all_src_points = []
all_dst_points = []

for idx, (i, j) in enumerate(pairs):
    good_matches = good_matches_list[idx]
    src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in good_matches]) # Extracts (x, y) coordinates of keypoints from the first image in the pair
    dst_pts = np.float32([keypoints_list[j][m.trainIdx].pt for m in good_matches]) # Extracts (x, y) coordinates of keypoints from the second image in the pair
    all_src_points.append(src_pts)
    all_dst_points.append(dst_pts)

# Implement the ransac function
def ransac(src_points, dst_points, ransac_reproj_threshold=2, max_iters=500, inlier_ratio=0.8):
    """
        Estimate the homography transformation using the RANSAC algorithm, while
    identifying inlier correspondences.
    :param src_points: numpy.array(float), coordinates of points in the source image
    :param dst_points: numpy.array(float), coordinates of points in the destination image
    :param ransac_reproj_threshold: float, maximum reprojection error allowed to classify a point pair as an inlier
    :param max_iters: int, maximum number of RANSAC iterations
    :param inlier_ratio: float, the desired ratio of inliers to total correspondences
    return:
        H: numpy.array(float), the estimated homography matrix using linear least-squares
        mask: numpy.array(uint8), mask indicating the inlier correspondences
    """
    # Input validation
    assert src_points.shape == dst_points.shape, f"src_points shape {src_points.shape} does not match dst_points shape {dst_points.shape}"
    assert ransac_reproj_threshold >= 0, f"ransac_reproj_threshold must be non-negative, got {ransac_reproj_threshold}"
    assert isinstance(max_iters, int) and max_iters > 0, "max_iters must be a positive, non-zero integer."
    assert 0 <= inlier_ratio <= 1, "inlier_ratio must be within the range [0,1]."

    num_points = src_points.shape[0]
    if num_points < 4:
        raise ValueError("At least 4 point correspondences are required to compute homography.")

    best_H = None
    best_inliers = []
    best_num_inliers = 0

    for iteration in range(max_iters):
        # Randomly sample 4 correspondences
        idx = np.random.choice(num_points, 4, replace=False)
        src_sample = src_points[idx]
        dst_sample = dst_points[idx]

        # Estimate homography from 4 correspondences
        # Using cv2.findHomography() for robustness; it can handle more than four points if in the future we want to change the settings 
        # and provides a least-squares solution that minimizes reprojection error,
        # making it more reliable in cases with potential outliers.
        # In contrast, cv2.getPerspectiveTransform() requires exactly four points
        # and may not perform well if those points are collinear or poorly distributed.
        H_sample, status = cv2.findHomography(src_sample, dst_sample, method=0)  # 0 means regular method (no RANSAC)
        if H_sample is None:
            continue

        # Compute reprojection errors
        src_pts_hom = np.concatenate([src_points, np.ones((num_points,1))], axis=1)  # Convert to homogeneous coordinates
        dst_pts_proj = (H_sample @ src_pts_hom.T).T  # Apply homography
        dst_pts_proj /= dst_pts_proj[:, [2]]  # Normalize by dividing by the third coordinate to return to 2D
        reproj_errors = np.linalg.norm(dst_points - dst_pts_proj[:, :2], axis=1)

        # Identify inliers
        inliers = reproj_errors <= ransac_reproj_threshold
        num_inliers = np.sum(inliers)

        # Update best model if current one has more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_H = H_sample

            # Early exit if desired inlier ratio is reached
            if num_inliers / num_points >= inlier_ratio:
                break

    if best_H is None:
        raise ValueError("RANSAC failed to find a valid homography.")

    # Re-estimate homography using all inliers
    src_inliers = src_points[best_inliers]
    dst_inliers = dst_points[best_inliers]

    # # simple way for computing homography 
    # H, status = cv2.findHomography(src_inliers, dst_inliers, method=0)  # Regular method
    # Manual implementation of findHomography using linear least squares with pseudo-inverse
    A = []
    for src, dst in zip(src_inliers, dst_inliers):
        x, y = src
        u, v = dst
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize the homography matrix
    H /= H[2, 2]

    mask = np.zeros(num_points, dtype=np.uint8)
    mask[best_inliers] = 1

    return H, mask

# Compute homographies using OpenCV and custom RANSAC implementation
H_cv_list = []
mask_cv_list = []
H_our_list = []
mask_our_list = []

for idx, (i, j) in enumerate(pairs):
    src_pts = all_src_points[idx]
    dst_pts = all_dst_points[idx]

    # OpenCV's method
    H_cv, mask_cv = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1, maxIters=1000)
    H_cv_list.append(H_cv)
    mask_cv_list.append(mask_cv.ravel().tolist())

    # Our method
    H_our, mask_our = ransac(src_pts, dst_pts, ransac_reproj_threshold=1, max_iters=1000, inlier_ratio=0.8)
    H_our_list.append(H_our)
    mask_our_list.append(mask_our.tolist())

# Plot inlier correspondences for both methods
def draw_inliers(img1, img2, kp1, kp2, matches, mask, title):
    inlier_matches = [matches[i] for i, m in enumerate(mask) if m]
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

# plot two pairs per figure
for plot_idx in range(0, len(pairs), 2):
    plt.figure(figsize=(15, 20))  

    for subplot_idx in range(2):  # Two pairs per plot
        if plot_idx + subplot_idx < len(pairs):
            i, j = pairs[plot_idx + subplot_idx]

            plt.subplot(4, 2, subplot_idx * 2 + 1)
            draw_inliers(images_color[i], images_color[j], keypoints_list[i], keypoints_list[j], good_matches_list[plot_idx + subplot_idx], mask_cv_list[plot_idx + subplot_idx], 'OpenCV Inliers between Image {} and {}'.format(i+1, j+1))

            plt.subplot(4, 2, subplot_idx * 2 + 2)
            draw_inliers(images_color[i], images_color[j], keypoints_list[i], keypoints_list[j], good_matches_list[plot_idx + subplot_idx], mask_our_list[plot_idx + subplot_idx], 'Our Inliers between Image {} and {}'.format(i+1, j+1))

    plt.tight_layout()
    plt.show()

# panorama 
def compute_homographies(homographies):
    H12 = homographies[0]
    H23 = homographies[1]
    H34 = homographies[2]
    H45 = homographies[3]

    H13 = H23 @ H12
    H14 = H34 @ H13
    H15 = H45 @ H14

    return [H12, H13, H14, H15]

def stitch_images(images, homographies):
    panorama = images[0]
    for i in range(1, len(images)):
        H = homographies[i-1]
        panorama_height = max(panorama.shape[0], images[i].shape[0])
        panorama_width = panorama.shape[1] + images[i].shape[1]
        panorama = cv2.copyMakeBorder(panorama, 0, 0, 0, images[i].shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        warped_img = cv2.warpPerspective(images[i], H, (panorama_width, panorama_height), flags=cv2.WARP_INVERSE_MAP)
        
        temp_panorama = np.round(0.5 * panorama + 0.5 * warped_img).astype(np.uint8)
        temp_panorama[warped_img == [0, 0, 0]] = panorama[warped_img == [0, 0, 0]]
        temp_panorama[panorama == [0, 0, 0]] = warped_img[panorama == [0, 0, 0]]
        panorama = temp_panorama.copy()
    
    return panorama

# Extract the first five images
images = images_color[:5]

# Compute homographies for both methods
homographies_our = compute_homographies(H_our_list)
homographies_cv = compute_homographies(H_cv_list)

# Stitch panoramas
panorama_our = stitch_images(images, homographies_our)
panorama_cv = stitch_images(images, homographies_cv)

# Display both panoramas
plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(panorama_our, cv2.COLOR_BGR2RGB))
plt.title("Stitched Panorama using Custom Homographies")
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(panorama_cv, cv2.COLOR_BGR2RGB))
plt.title("Stitched Panorama using OpenCV Homographies")
plt.axis('off')

plt.tight_layout()
plt.show()
