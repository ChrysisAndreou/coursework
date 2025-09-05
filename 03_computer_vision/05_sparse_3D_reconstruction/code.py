import numpy as np
import cv2
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(123)


############################################################
# Part 1: Fundamental Matrix Linear System
############################################################
def fundamental_matrix_linear_system(pts1, pts2):
    """
        Create linear equations for estimating the fundamental matrix in matrix form
    :param pts1: numpy.array(float), an array Nx2 that holds the source image points
    :param pts2: numpy.array(float), an array Nx2 that holds the destination image points
    :return:
        A: numpy.array(float), an array Nx8 that holds the left side coefficients of the linear equations
        b: numpy.array(float), an array Nx1 that holds the right side coefficients of the linear equations
    """
    assert pts1.ndim == 2 and pts2.ndim == 2, "pts1 and pts2 must be two-dimensional arrays."
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2, "Each point array must have shape (N, 2)."
    assert pts1.shape[0] == pts2.shape[0], "pts1 and pts2 must have the same number of points."
    N = pts1.shape[0]
    assert N >= 8, "At least 8 points are required."

    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    # Construct A and b according to the epipolar constraint x'^T F x = 0 with f33 set to 1
    A = np.column_stack((x2 * x1, x2 * y1, x2,
                         y2 * x1, y2 * y1, y2,
                         x1, y1))
    b = -1 * np.ones((N, 1))

    return A, b


############################################################
# Part 2: Compute Epipolar Lines
############################################################
def compute_correspond_epilines(points, which_image, F):
    """
        For points in an image of a stereo pair, computes the corresponding epilines in the other image
    :param points: numpy.array(float), an array Nx2 that holds the image points
    :param which_image: int, index of the image (1 or 2) that contains the points
    :param F: numpy.array(float), fundamental matrix between the stereo pair
    :return:
        epilines: numpy.array(float): an array Nx3 that holds the coefficients of the corresponding
                                epipolar lines
    """
    assert points.ndim == 2 and points.shape[1] == 2, "points must be Nx2 array."
    assert which_image in [1, 2], "which_image must be either 1 or 2."
    assert F.shape == (3,3), "F must be a 3x3 matrix."

    ones = np.ones((points.shape[0], 1))
    # Add ones to the points to make them homogeneous
    pts_hom = np.hstack((points, ones))

    if which_image == 1:
        # Points in image 1 -> lines in image 2: l = F * p
        lines = (F @ pts_hom.T).T
    else:
        # Points in image 2 -> lines in image 1: l = F^T * p
        lines = (F.T @ pts_hom.T).T

    # Normalize lines so that a^2 + b^2 = 1
    a = lines[:,0]
    b = lines[:,1]
    c = lines[:,2]
    norm_factor = np.sqrt(a**2 + b**2).reshape(-1,1)
    epilines = lines / norm_factor

    return epilines


############################################################
# Helper functions for visualization
############################################################
def drawlines(ax1, ax2, img1, img2, lines, pts1, pts2, title='Epilines'):
    """
        Draw epilines
    :param img1: numpy.array(), draw epilines on
    :param img2: numpy.array(), draw points on
    :param lines: numpy.array(), epilines
    :param pts1: numpy.array(), corresponding points on img1
    :param pts2: numpy.array(), corresponding points on img2
    :return:
        img1, img2: numpy.array()
    """
    H, W, _ = img1.shape
    img1 = np.copy(img1)
    img2 = np.copy(img2)

    for coeff, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = 0, int(-coeff[2] / coeff[1])
        x1, y1 = W, int(-(coeff[2] + coeff[0] * W) / coeff[1])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        img1 = cv2.circle(img1, tuple(np.int32(pt1)), 5, color, -1, cv2.LINE_AA)
        img2 = cv2.circle(img2, tuple(np.int32(pt2)), 5, color, -1, cv2.LINE_AA)


    ax1.imshow(img1)
    ax1.set_title(title)
    ax1.axis('off')
    ax2.imshow(img2)
    ax2.set_title(title)
    ax2.axis('off')

def drawcorrespondences(ax1, ax2, img1, img2, pts1, pts2, title='Correspondences'):
    """
        Draw correspondences
    :param img1: numpy.array(), draw points on
    :param img2: numpy.array(), draw points on
    :param pts1: numpy.array(), corresponding points on img1
    :param pts2: numpy.array(), corresponding points on img2
    :return:
        img1, img2: numpy.array()
    """

    img1 = np.copy(img1)
    img2 = np.copy(img2)

    for pt1, pt2 in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.circle(img1, tuple(np.int32(pt1)), 5, color, -1, cv2.LINE_AA)
        img2 = cv2.circle(img2, tuple(np.int32(pt2)), 5, color, -1, cv2.LINE_AA)

    ax1.imshow(img1)
    ax1.set_title(title + " - Image 1")
    ax1.axis('off')

    ax2.imshow(img2)
    ax2.set_title(title + " - Image 2")
    ax2.axis('off')



def plot_3d_points(pts3d, fig_title=None, xlim=(-1, 1), ylim=(-1, 1), zlim=(2, 7), elev=-170, azim=20, vertical_axis='y',
                   show=True, ax=None):
    """
    Plot 3D points using matplotlib
    :param pts3d: numpy.array(float), an array Nx3 that holds the input 3D points
    :param fig_title: str, figure title
    :param xlim: tuple, x-axis view limits
    :param ylim: tuple, y-axis view limits
    :param zlim: tuple, z-axis view limits
    :param elev: float, the elevation angle in degrees rotates the camera above the plane pierced by the vertical axis
    :param azim: float, the azimuthal angle in degrees rotates the camera about the vertical axis
    :param vertical_axis: str, the axis to align vertically
    :param show: bool, show or draw figure
    :param ax: matplotlib axis, optional axis to plot on
    :return:
        None
    """
    if ax is None:
        fig = plt.figure(fig_title)
        ax = fig.add_subplot(projection='3d')

    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)

    if show and ax is None:
        plt.show()
    elif ax is None:
        plt.draw()

############################################################
# Part 4: Normalizing Points
############################################################
def points_normalization(pts1, pts2):
    """
        Normalize points so that each coordinate system is located at the centroid of the image points and
        the mean square distance of the transformed image points from the origin should be 2 pixels
    :param pts1: numpy.array(float), an Nx2 array that holds the source image points
    :param pts2: numpy.array(float), an Nx2 array that holds the destination image point
    :return:
        pts1_normalized: numpy.array(float), an Nx2 array with the transformed source image points
        pts2_normalized: numpy.array(float), an Nx2 array with the transformed destination image points
        M1: numpy.array(float), an 3x3 array - transformation for source image
        M2: numpy.array(float), an 3x3 array - transformation for destination image
    """

    # Checks
    assert pts1.ndim == 2 and pts1.shape[1] == 2, "pts1 must be Nx2."
    assert pts2.ndim == 2 and pts2.shape[1] == 2, "pts2 must be Nx2."
    assert pts1.shape[0] == pts2.shape[0], "pts1 and pts2 must have the same number of points."

    # Normalize pts1
    mean1 = np.mean(pts1, axis=0)
    pts1_centered = pts1 - mean1
    avg_dist1 = np.mean(np.sqrt(np.sum(pts1_centered**2, axis=1)))
    scale1 = np.sqrt(2) / avg_dist1 if avg_dist1 > 0 else 1.0
    M1 = np.array([[scale1,   0,    -scale1 * mean1[0]],
                   [0,      scale1,  -scale1 * mean1[1]],
                   [0,         0,                1      ]])
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    pts1_norm_hom = (M1 @ pts1_hom.T).T
    pts1_normalized = pts1_norm_hom[:, :2] / pts1_norm_hom[:, [2]]

    # Normalize pts2
    mean2 = np.mean(pts2, axis=0)
    pts2_centered = pts2 - mean2
    avg_dist2 = np.mean(np.sqrt(np.sum(pts2_centered**2, axis=1)))
    scale2 = np.sqrt(2) / avg_dist2 if avg_dist2 > 0 else 1.0
    M2 = np.array([[scale2,   0,    -scale2 * mean2[0]],
                   [0,      scale2,  -scale2 * mean2[1]],
                   [0,         0,               1       ]])
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0],1))))
    pts2_norm_hom = (M2 @ pts2_hom.T).T
    pts2_normalized = pts2_norm_hom[:, :2] / pts2_norm_hom[:, [2]]

    return pts1_normalized, pts2_normalized, M1, M2

############################################################
# Part 3: Estimating Fundamental Matrix using RANSAC
############################################################
def ransac(src_points, dst_points, ransac_reproj_threshold=2, max_iters=500, inlier_ratio=0.8, normalize=False):
    """
        Calculate the set of inlier correspondences w.r.t. fundamental matrix, using the RANSAC method.
    :param src_points: numpy.array(float), an Nx2 array that holds the coordinates of the points in the
                                        source image
    :param dst_points: numpy.array(float), an Nx2 array that holds the coordinates of the points in the
    destination image
    :param ransac_reproj_threshold: float, maximum allowed reprojection error to treat a point-epiline pair
    as an inlier
    :param max_iters: int, the maximum number of RANSAC iterations
    :param inlier_ratio: float, ratio of inliers w.r.t. total number of correspondences
    :return:
        F: numpy.array(float), the estimated fundamental matrix using linear least-squares
    mask: numpy.array(uint8), mask that denotes the inlier correspondences
    """
    assert src_points.shape == dst_points.shape, "src_points and dst_points must have same shape."
    assert ransac_reproj_threshold >= 0, "ransac_reproj_threshold must be >= 0"
    assert max_iters > 0, "max_iters must be > 0"
    assert 0 <= inlier_ratio <= 1, "inlier_ratio must be in [0, 1]"

    N = src_points.shape[0]
    best_inliers_count = 0
    best_F = None
    best_mask = None

    for _ in range(max_iters):
        # Sample 8 points
        idx = np.random.choice(N, 8, replace=False)
        pts1_8 = src_points[idx]
        pts2_8 = dst_points[idx]

        # Normalize if requested
        if normalize:
            pts1_8_norm, pts2_8_norm, M1_8, M2_8 = points_normalization(pts1_8, pts2_8)
            try:
                A, b = fundamental_matrix_linear_system(pts1_8_norm, pts2_8_norm)
                x = np.linalg.inv(A.T @ A) @ A.T @ b
            except np.linalg.LinAlgError as e:
                if "Singular matrix" in str(e):
                    continue
                else:
                    raise e

            F_norm = np.array([[x[0,0], x[1,0], x[2,0]],
                               [x[3,0], x[4,0], x[5,0]],
                               [x[6,0], x[7,0],       1]])

            # De-normalize
            F = M2_8.T @ F_norm @ M1_8
        else:
            # Without normalization
            try:
                A, b = fundamental_matrix_linear_system(pts1_8, pts2_8)
                x = np.linalg.inv(A.T @ A) @ A.T @ b
            except np.linalg.LinAlgError as e:
                if "Singular matrix" in str(e):
                    continue
                else:
                    raise e

            F = np.array([[x[0,0], x[1,0], x[2,0]],
                          [x[3,0], x[4,0], x[5,0]],
                          [x[6,0], x[7,0],       1]])

        # Evaluate inliers
        epilines = compute_correspond_epilines(src_points, which_image=1, F=F)
        
        # distance_point_to_line
        a = epilines[:, 0]
        b = epilines[:, 1]
        c = epilines[:, 2]
        x = dst_points[:, 0]
        y = dst_points[:, 1]
        distances = np.abs(a * x + b * y + c)
        
        inliers = distances < ransac_reproj_threshold
        inliers_count = np.sum(inliers)

        # Update best model if better
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_mask = inliers

            # Refine F using all inliers
            in_pts1 = src_points[inliers]
            in_pts2 = dst_points[inliers]

            if normalize:
                # Normalize all inliers
                in_pts1_norm, in_pts2_norm, M1_in, M2_in = points_normalization(in_pts1, in_pts2)
                try:
                    A_in, b_in = fundamental_matrix_linear_system(in_pts1_norm, in_pts2_norm)
                    x_in = np.linalg.inv(A_in.T @ A_in) @ A_in.T @ b_in
                except np.linalg.LinAlgError as e:
                    if "Singular matrix" in str(e):
                        # fallback
                        x_in = x
                        M1_in, M2_in = M1_8, M2_8
                    else:
                        raise e

                F_in_norm = np.array([[x_in[0,0], x_in[1,0], x_in[2,0]],
                                      [x_in[3,0], x_in[4,0], x_in[5,0]],
                                      [x_in[6,0], x_in[7,0],        1]])
                best_F = M2_in.T @ F_in_norm @ M1_in
            else:
                # Direct refinement without normalization
                try:
                    A_in, b_in = fundamental_matrix_linear_system(in_pts1, in_pts2)
                    x_in = np.linalg.inv(A_in.T @ A_in) @ A_in.T @ b_in
                except np.linalg.LinAlgError as e:
                    if "Singular matrix" in str(e):
                        # fallback
                        pass
                    else:
                        raise e

                best_F = np.array([[x_in[0,0], x_in[1,0], x_in[2,0]],
                                   [x_in[3,0], x_in[4,0], x_in[5,0]],
                                   [x_in[6,0], x_in[7,0],       1]])


    # If no model found
    if best_F is None:
        best_F = np.eye(3)
        best_mask = np.zeros((N,), dtype=np.uint8)
    else:
        best_mask = best_mask.astype(np.uint8)

    return best_F, best_mask


############################################################
# Main Execution of parts 1, 2, 3, 4 
############################################################

# Load images 
img1 = cv2.imread('data/image1.png')
img2 = cv2.imread('data/image2.png')
assert img1 is not None, "Could not load 'image1.png' from data/ directory."
assert img2 is not None, "Could not load 'image2.png' from data/ directory."

# Convert images to RGB format 
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT feature extraction
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test (0.75)
good = []
pts_src = []
pts_dst = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
        pts_src.append(kp1[m.queryIdx].pt)
        pts_dst.append(kp2[m.trainIdx].pt)

pts_src = np.array(pts_src)
pts_dst = np.array(pts_dst)

# Our RANSAC with normalization 
F_ours, mask_ours = ransac(pts_src, pts_dst, ransac_reproj_threshold=0.5, max_iters=5000, inlier_ratio=0.9, normalize=True)

# OpenCV's RANSAC
F_cv, mask_cv = cv2.findFundamentalMat(pts_src, pts_dst, cv2.FM_RANSAC, 0.5)

# Inliers for both methods
inlier_src_ours = pts_src[mask_ours.ravel()==1]
inlier_dst_ours = pts_dst[mask_ours.ravel()==1]

inlier_src_cv = pts_src[mask_cv.ravel()==1]
inlier_dst_cv = pts_dst[mask_cv.ravel()==1]

# Compute epilines
epilines_ours_img1 = compute_correspond_epilines(inlier_dst_ours, which_image=2, F=F_ours)
epilines_ours_img2 = compute_correspond_epilines(inlier_src_ours, which_image=1, F=F_ours)

lines1_cv = compute_correspond_epilines(inlier_dst_cv, which_image=2, F=F_cv)
lines2_cv = compute_correspond_epilines(inlier_src_cv, which_image=1, F=F_cv)

# #  OpenCV's function not used because instructions say to use our function 
# lines2_cv = cv2.computeCorrespondEpilines(inlier_src_cv.reshape(-1, 1, 2), 1, F_cv)
# lines2_cv = lines2_cv.reshape(-1, 3)

# lines1_cv = cv2.computeCorrespondEpilines(inlier_dst_cv.reshape(-1, 1, 2), 2, F_cv)
# lines1_cv = lines1_cv.reshape(-1, 3)

##############################
# Visualization
##############################

# 1. Plot original images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img1_rgb)
axes[0].set_title('Image 1')
axes[0].axis('off')

axes[1].imshow(img2_rgb)
axes[1].set_title('Image 2')
axes[1].axis('off')
plt.show()

# 2. Plot correspondences after ratio test
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
drawcorrespondences(ax1, ax2, img1_rgb, img2_rgb, pts_src, pts_dst, title='All Correspondences After Ratio Test')
plt.show()

# 3. Plot our RANSAC and CV RANSAC inliers
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Our RANSAC Inliers (Normalized)
drawcorrespondences(axes[0, 0], axes[0, 1], img1_rgb, img2_rgb, inlier_src_ours, inlier_dst_ours, title='Our RANSAC Inliers (Normalized)')

# OpenCV RANSAC Inliers
drawcorrespondences(axes[1, 0], axes[1, 1], img1_rgb, img2_rgb, inlier_src_cv, inlier_dst_cv, title='OpenCV RANSAC Inliers')

plt.show()

# 4. Plot epilines for our implementation and OpenCV’s
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Ours: Epilines in image 2 from image 1 points
drawlines(axes[0, 1], axes[0, 0], img2_rgb, img1_rgb, epilines_ours_img2, inlier_dst_ours, inlier_src_ours, 
          title="Our Epilines (Shown on Image 2) from Points in Image 1")

# OpenCV: Epilines in image 2 from image 1 points
drawlines(axes[1, 1], axes[1, 0], img2_rgb, img1_rgb, lines2_cv, inlier_dst_cv, inlier_src_cv, 
          title="CV Epilines (Shown on Image 2) from Points in Image 1")

plt.show()

# Plot epilines in image 1 from image 2 points
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Ours: Epilines in image 1 from image 2 points
drawlines(axes[0, 0], axes[0, 1], img1_rgb, img2_rgb, epilines_ours_img1, inlier_src_ours, inlier_dst_ours,
          title="Our Epilines (Shown on Image 1) from Points in Image 2")

# OpenCV: Epilines in image 1 from image 2 points
drawlines(axes[1, 0], axes[1, 1], img1_rgb, img2_rgb, lines1_cv, inlier_src_cv, inlier_dst_cv,
          title="CV Epilines (Shown on Image 1) from Points in Image 2")

plt.show()


############################################################
# Part 5: 2D Points Triangulation
############################################################

def triangulation(P1, pts1, P2, pts2):
    """
        Triangulate pairs of 2D points in the images to a set of 3D points
    :param P1: numpy.array(float), an array 3x4 that holds the projection matrix of camera 1
    :param pts1: numpy.array(float), an array Nx2 that holds the 2D points on image 1
    :param P2: numpy.array(float), an array 3x4 that holds the projection matrix of camera 2
    :param pts2: numpy.array(float), an array Nx2 that holds the 2D points on image 2
    :return:
        pts3d: numpy.array(float), an array Nx3 that holds the reconstructed 3D points
    """
    # Checks
    assert P1.shape == (3,4), "P1 must be a 3x4 matrix."
    assert P2.shape == (3,4), "P2 must be a 3x4 matrix."
    assert pts1.ndim == 2 and pts1.shape[1] == 2, "pts1 must be Nx2."
    assert pts2.ndim == 2 and pts2.shape[1] == 2, "pts2 must be Nx2."
    assert pts1.shape[0] == pts2.shape[0], "pts1 and pts2 must have the same number of points."

    N = pts1.shape[0]
    pts3d = []
    
    # For each point pair, perform triangulation
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        # For triangulation, we solve a linear system:
        A = np.zeros((4,4))
        A[0,:] = x1*P1[2,:] - P1[0,:]
        A[1,:] = y1*P1[2,:] - P1[1,:]
        A[2,:] = x2*P2[2,:] - P2[0,:]
        A[3,:] = y2*P2[2,:] - P2[1,:]

        # Solve for X (homogeneous)
        # Use SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X/X[-1]  # Convert to inhomogeneous coordinates
        pts3d.append(X[:3])
    
    pts3d = np.array(pts3d)
    return pts3d

# Load the intrinsic parameters
intrinsics = np.load('data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Load the good correspondences
good_data = np.load('data/good_correspondences.npz')
good_pts1 = good_data['pts1']
good_pts2 = good_data['pts2']

# We have two fundamental matrices: F_ours and F_cv
# Compute Essential matrices: E = K2^T * F * K1
E_ours = K2.T @ F_ours @ K1
E_cv = K2.T @ F_cv @ K1

# Decompose essential matrices into R, t
R1_ours, R2_ours, t_ours = cv2.decomposeEssentialMat(E_ours)
R1_cv, R2_cv, t_cv = cv2.decomposeEssentialMat(E_cv)

# Camera 1 projection matrix: P1 = K1 [I|0]
P1 = np.hstack((K1, np.zeros((3,1))))

def choose_correct_RT(R1, R2, t, K1, K2, pts1, pts2):
    """
    Among the four possible [R|t] configurations, choose the one that yields 
    the maximum number of points in front of both cameras.
    """
    # Four possible extrinsics for camera 2
    # {R1, t}, {R1, -t}, {R2, t}, {R2, -t}
    candidates = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    best_count = -1
    best_pts3d = None

    for R_candidate, t_candidate in candidates:
        P2_candidate = K2 @ np.hstack((R_candidate, t_candidate))
        pts3d = triangulation(P1, pts1, P2_candidate, pts2)

        # Check how many points are in front of both cameras.
        # A point X is in front of a camera if Z coordinate in camera's frame is > 0.
        # For the first camera: world and camera1 frame coincide, so just check Z > 0 in world frame.
        # For the second camera: transform pts3d into camera2 frame and check Z > 0.
        # Camera2 frame: X_c2 = R_candidate * X + t_candidate

        # In camera1 frame (which is also world frame):
        front1 = pts3d[:,2] > 0

        # In camera2 frame:
        pts3d_c2 = (R_candidate @ pts3d.T) + t_candidate
        pts3d_c2 = pts3d_c2.T
        front2 = pts3d_c2[:,2] > 0

        # Count how many are in front of both
        count_in_front = np.sum(front1 & front2)
        
        if count_in_front > best_count:
            best_count = count_in_front
            best_pts3d = pts3d

    return best_pts3d

# Choose correct RT for our essential matrix
pts3d_ours = choose_correct_RT(R1_ours, R2_ours, t_ours, K1, K2, good_pts1, good_pts2)

# Choose correct RT for CV essential matrix
pts3d_cv = choose_correct_RT(R1_cv, R2_cv, t_cv, K1, K2, good_pts1, good_pts2)

# Visualize the 3D reconstructed points 
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121, projection='3d')
plot_3d_points(pts3d_ours, ax=ax1)
ax1.set_title('Our 3D Points')

ax2 = fig.add_subplot(122, projection='3d')
plot_3d_points(pts3d_cv, ax=ax2)
ax2.set_title('CV 3D Points')

plt.show()
