# part 1 K-Means Clustering 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display the image
image = cv2.imread('data/home.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

def kmeans(data, K, thresh, n_iter, n_attempts):
    """
        Cluster data in K clusters using the K-Means algorithm
    :param data: numpy.array(float), the input data array with N (#data) x D (#feature_dimensions) dimensions
    :param K: int, number of clusters
    :param thresh: float, convergence threshold
    :param n_iter: int, #iterations of the K-Means algorithm
    :param n_attempts: int, #attempts to run the K-Means algorithm
    :return:
        compactness: float, the sum of squared distance from each point to their corresponding centers
        labels: numpy.array(int), the label array with Nx1 dimensions, where it denotes the corresponding cluster of
                            each data point
        centers : numpy.array(float), a KxD array with the final centroids
    """
    # Checks
    assert data.ndim == 2, "Data should be a 2D array"
    assert K > 0 and thresh > 0 and n_iter > 0 and n_attempts > 0, "K, thresh, n_iter, n_attempts must be positive non-zero numbers"
    N = data.shape[0]
    assert K <= N, "K must be less or equal to the number of data points"
    
    best_compactness = None
    best_labels = None
    best_centers = None
    
    for attempt in range(n_attempts):
        # Initialize centroids randomly from data
        indices = np.random.choice(N, K, replace=False)
        centers = data[indices]
        
        labels = np.zeros(N, dtype=np.int32)
        for iteration in range(n_iter):
            # Compute distances to centroids
            # Reshape data to (N, 1, D) for broadcasting with centers (K, D) 
            # This allows us to calculate the distance from each data point to each centroid
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)  # Resulting shape will be (N, K)
            # Assign labels based on the closest centroid
            # Use axis=1 to find the index of the centroid with the minimum distance for each data point
            new_labels = np.argmin(distances, axis=1)  # Resulting shape will be (N,)
            
            # Update centroids
            new_centers = []
            for k in range(K): # For each cluster
                assigned_data = data[new_labels == k] # Get all data points assigned to the current cluster 
                if assigned_data.size == 0: # If a cluster lost all its points, reinitialize its centroid randomly
                    new_center = data[np.random.choice(N)]
                else:
                    new_center = assigned_data.mean(axis=0) # Update the centroid to the mean of the assigned data points
                new_centers.append(new_center) 
            new_centers = np.array(new_centers) 
            
            # Check for convergence
            center_shift = np.linalg.norm(new_centers - centers, axis=1)
            if np.max(center_shift) < thresh:
                # Converged
                break
            centers = new_centers
            labels = new_labels
        
        # Compute compactness
        # Sum of squared distances from each point to their corresponding centers
        compactness = np.sum((data - centers[labels]) ** 2)
        
        if best_compactness is None or compactness < best_compactness: 
            best_compactness = compactness
            best_labels = labels.copy()
            best_centers = centers.copy()
    
    return best_compactness, best_labels, best_centers

# Set parameters
K, thresh, n_iter, n_attempts = 4, 1.0, 10, 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n_iter, thresh)

# Prepare data for (r, g, b) features
# Reshape the image from (Height, Width, 3) to (N, 3) where N is the number of pixels
data_rgb = image.reshape((-1, 3)).astype(np.float32)

# Run our K-Means implementation for (r, g, b) features
compactness_rgb, labels_rgb, centers_rgb = kmeans(data_rgb, K, thresh, n_iter, n_attempts)

# Run OpenCV's K-Means implementation for (r, g, b) features
compactness_cv_rgb, labels_cv_rgb, centers_cv_rgb = cv2.kmeans(data_rgb, K, None, criteria, n_attempts, cv2.KMEANS_RANDOM_CENTERS)

# Prepare data for (i, j, r, g, b) features
h, w = image.shape[:2] # Get the height and width of the image
i_coords, j_coords = np.indices((h, w)) # Get the coordinates of each pixel
# Stack the coordinates and reshape to (N, 2)
coords = np.stack((i_coords, j_coords), axis=2).reshape((-1, 2)) 
# Concatenate the coordinates and the RGB values 
# dimensions: (N, 5) where 5 = 2 (coords) + 3 (RGB)
data_coords_rgb = np.concatenate((coords, data_rgb), axis=1).astype(np.float32) 


# Run our K-Means implementation for (i, j, r, g, b) features
compactness_coords_rgb, labels_coords_rgb, centers_coords_rgb = kmeans(data_coords_rgb, K, thresh, n_iter, n_attempts)

# Run OpenCV's K-Means implementation for (i, j, r, g, b) features
compactness_cv_coords_rgb, labels_cv_coords_rgb, centers_cv_coords_rgb = cv2.kmeans(data_coords_rgb, K, None, criteria, n_attempts, cv2.KMEANS_RANDOM_CENTERS)

# Create clustered images for (r, g, b) features
# centers_rgb[labels_rgb] maps each pixel to its cluster centroid's RGB value
clustered_image_rgb = centers_rgb[labels_rgb].astype(np.uint8).reshape((h, w, 3))
labels_cv_rgb = labels_cv_rgb.flatten()
clustered_image_cv_rgb = centers_cv_rgb[labels_cv_rgb].astype(np.uint8).reshape((h, w, 3))

# Create clustered images for (i, j, r, g, b) features
centers_coords_rgb_colors = centers_coords_rgb[:, 2:]  # Ignore the coordinates
clustered_image_coords_rgb = centers_coords_rgb_colors[labels_coords_rgb].astype(np.uint8).reshape((h, w, 3))
centers_cv_coords_rgb_colors = centers_cv_coords_rgb[:, 2:]  # Ignore the coordinates
labels_cv_coords_rgb = labels_cv_coords_rgb.flatten()
clustered_image_cv_coords_rgb = centers_cv_coords_rgb_colors[labels_cv_coords_rgb].astype(np.uint8).reshape((h, w, 3))

# Plot the clustered images for (r, g, b) features
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(clustered_image_rgb)
axes[0].set_title('Our K-Means (RGB)')
axes[0].axis('off')

axes[1].imshow(clustered_image_cv_rgb)
axes[1].set_title('OpenCV K-Means (RGB)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Plot the clustered images for (i, j, r, g, b) features
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(clustered_image_coords_rgb)
axes[0].set_title('Our K-Means (Coords+RGB)')
axes[0].axis('off')

axes[1].imshow(clustered_image_cv_coords_rgb)
axes[1].set_title('OpenCV K-Means (Coords+RGB)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# part 2 Efficient Graph-Based Image Segmentation  

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import hsv_to_rgb

# Step 1: Load and Preprocess the Image
# Load the image
input_image = cv2.imread('data/eiffel_tower.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Apply Gaussian blurring
blurred_image = cv2.GaussianBlur(input_image, (3, 3), 0.8)

# Step 2: Create the Nearest Neighbor Graph (NN-Graph)
def nn_graph(input_image, k):
    """
        Create a graph based on the k-nearest neighbors of each pixel in the (i,j,r,g,b) feature space.
        Edge weights are calculated as the Euclidean distance of the node's features
        and its corresponding neighbors.
    :param input_image: numpy.array(uint8), input image of HxWx3 dimensions
    :param k: int, nearest neighbors for each node
    :return:
        graph: tuple(V: numpy.array(int), E: <graph connectivity representation>), the NN-graph where
            V is the set of pixel-nodes of (W*H)x2 dimensions and E is a representation of the graph's
            undirected edges along with their corresponding weight
    """
    # Validate inputs
    assert input_image.ndim == 3 and input_image.shape[2] == 3, "Input image must have 3 dimensions and 3 channels (RGB)."
    assert k > 0, "Parameter k must be a positive integer."

    H, W, _ = input_image.shape
    num_pixels = H * W

    # Construct feature vectors
    i_coords, j_coords = np.indices((H, W))
    i_coords = i_coords.flatten()
    j_coords = j_coords.flatten()
    rgb_values = input_image.reshape(-1, 3) # reshape to (H*W, 3) where H*W is the number of pixels
    features = np.column_stack((i_coords, j_coords, rgb_values)) 

    # Use NearestNeighbors to find k nearest neighbors and create adjacency matrix
    # +1 because the first neighbor is the point itself,
    # auto is the best algorithm for this case between ball_tree and KDTree 
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features) 
    adj_matrix = nbrs.kneighbors_graph(features, mode='distance') # distance not just binary connection

    # Remove self-loops
    adj_matrix.setdiag(0) 

    # Zero out the upper triangular part
    rows, cols = adj_matrix.nonzero()
    indices = np.where(rows <= cols)
    adj_matrix[rows[indices], cols[indices]] = 0
    adj_matrix.eliminate_zeros() # eliminate zeros to save memory in CSR format 

    # Convert adjacency matrix to edge list
    # COO format is a simple and efficient way to represent sparse matrices, 
    # where the non-zero entries are stored as three separate arrays: one for the row indices, 
    # one for the column indices, and one for the corresponding values (weights).
    coo_matrix = adj_matrix.tocoo() # convert CSR to COO format 
    # an edge is defined by the the source and target nodes and the weight 
    edges = np.column_stack((coo_matrix.row, coo_matrix.col, coo_matrix.data)) 

    V = np.column_stack((i_coords, j_coords)) # pixel coordinates 

    return V, edges

# Create NN-Graph
k_nn = 10  # Number of nearest neighbors for NN-Graph
V, E = nn_graph(blurred_image, k=k_nn)

# Step 3: Implement the Segmentation Algorithm
def segmentation(G, k_param, min_size, post_process=True):
    """
        Segment the image base on the Efficient Graph-Based Image Segmentation algorithm.
    :param G: tuple(V, E), the input graph
    :param k_param: int, sets the threshold k/|C|
    :param min_size: int, minimum size of clusters
    :return:
        clusters: numpy.array(int), a |V|x1 array where it denotes the cluster for each node v of the graph
    """
    V, E = G
    num_nodes = V.shape[0]

    # Validate inputs
    assert k_param > 0, "Parameters k_param must be positive integer."
    assert min_size >= 0, "min_size must be non-negative."

    # Initialize each node to its own cluster
    clusters = np.arange(num_nodes)
    cluster_sizes = np.ones(num_nodes, dtype=int)
    internal_diffs = np.zeros(num_nodes)

    # Sort edges by weight in ascending order to process the strongest connections first.
    # the ones with the smallest weights are the ones that are most likely to be merged
    # This allows the algorithm to effectively merge closely related clusters.
    E_sorted = E[E[:, 2].argsort()]

    # Union-Find data structure
    parent = np.arange(num_nodes) # each node is its own parent at the beginning 

    
    # The parent is the immediate predecessor of an element in the hierarchy, 
    # while the root is the representative element of a cluster that is its own parent.
    def find(u): # find the root of the cluster for node u 
        if parent[u] != u:
            parent[u] = find(parent[u])  # Path compression
        return parent[u]

    def union(u, v, w):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root: # if they are already in the same cluster, do nothing
            return
        MInt = min(internal_diffs[u_root] + k_param / cluster_sizes[u_root],
                   internal_diffs[v_root] + k_param / cluster_sizes[v_root])
        if w <= MInt:
            # Merge clusters
            parent[v_root] = u_root
            cluster_sizes[u_root] += cluster_sizes[v_root] 
            internal_diffs[u_root] = max(internal_diffs[u_root], internal_diffs[v_root], w)
            cluster_sizes[v_root] = 0  # Since v_root is merged, its size is now zero

    # Perform segmentation
    for edge in E_sorted:
        u, v, w = int(edge[0]), int(edge[1]), edge[2] # source, target, weight
        union(u, v, w)

    if post_process:
        # Post-processing to merge small clusters
        for edge in E_sorted:
            u, v, w = int(edge[0]), int(edge[1]), edge[2]
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                if cluster_sizes[u_root] < min_size or cluster_sizes[v_root] < min_size:
                    # Merge clusters
                    parent[v_root] = u_root
                    cluster_sizes[u_root] += cluster_sizes[v_root]
                    internal_diffs[u_root] = max(internal_diffs[u_root], internal_diffs[v_root], w)
                    cluster_sizes[v_root] = 0  # Since v_root is merged, its size is now zero

    # Assign cluster labels by finding the root of each node 
    clusters = np.array([find(i) for i in range(num_nodes)])

    return clusters

# Perform Segmentation without post-processing
k_param = 550  # Parameter for the segmentation algorithm
min_size = 300  # Minimum cluster size

clusters_no_pp = segmentation((V, E), k_param=k_param, min_size=min_size, post_process=False)
unique_clusters_no_pp = np.unique(clusters_no_pp)
num_clusters_no_pp = unique_clusters_no_pp.size
print(f"Number of clusters without post-processing: {num_clusters_no_pp}")

# Perform Segmentation with post-processing
clusters_with_pp = segmentation((V, E), k_param=k_param, min_size=min_size, post_process=True)
unique_clusters_with_pp = np.unique(clusters_with_pp)
num_clusters_with_pp = unique_clusters_with_pp.size
print(f"Number of clusters with post-processing: {num_clusters_with_pp}")

# Step 4: Visualize the Results
def map_clusters_to_colors(clusters, image_shape):
    """
    Map cluster labels to colors for visualization.

    :param clusters: numpy.array of shape (num_nodes,), cluster labels.
    :param image_shape: tuple, shape of the original image (H, W, 3).
    :return:
        segmented_image: numpy.array of shape (H, W, 3), colored image based on clusters.
    """
    num_nodes = clusters.shape[0]
    H, W, _ = image_shape

    # Normalize cluster labels
    unique_clusters = np.unique(clusters)
    num_clusters = unique_clusters.size
    cluster_map = {c: i for i, c in enumerate(unique_clusters)} 
    normalized_labels = np.array([cluster_map[c] for c in clusters]) 

    # Generate colors in HSV space
    hsv_colors = np.zeros((num_clusters, 3))
    hsv_colors[:, 0] = np.linspace(0, 1, num_clusters, endpoint=False)  # Hue
    hsv_colors[:, 1] = 1  # Saturation
    hsv_colors[:, 2] = 1  # Value

    rgb_colors = hsv_to_rgb(hsv_colors)

    # Assign colors to pixels
    segmented_image = rgb_colors[normalized_labels].reshape(H, W, 3)

    return segmented_image

def plot_results(input_image, segmented_image_no_pp, segmented_image_with_pp, num_clusters_no_pp, num_clusters_with_pp):
    """
    Plot the original and segmented images side by side.

    :param input_image: numpy.array, original image.
    :param segmented_image_no_pp: numpy.array, segmented image without post-processing.
    :param segmented_image_with_pp: numpy.array, segmented image with post-processing.
    :param num_clusters_no_pp: int, number of clusters without post-processing.
    :param num_clusters_with_pp: int, number of clusters with post-processing.
    """
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(input_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"k = {k_param}, no post-processing, |S| = {num_clusters_no_pp}")
    plt.imshow(segmented_image_no_pp)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"k = {k_param}, post-processing min size = {min_size}, |S| = {num_clusters_with_pp}")
    plt.imshow(segmented_image_with_pp)
    plt.axis('off')

    plt.show()

# Map clusters to colors
segmented_image_no_pp = map_clusters_to_colors(clusters_no_pp, input_image.shape)
segmented_image_with_pp = map_clusters_to_colors(clusters_with_pp, input_image.shape)

# Plot the results
plot_results(input_image, segmented_image_no_pp, segmented_image_with_pp, num_clusters_no_pp, num_clusters_with_pp)
