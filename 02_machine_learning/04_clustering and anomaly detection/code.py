import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



RANDOM_STATE = 2

# Task 1: Visualize the Sample Data
X, y_true = make_blobs(n_samples=3000, centers=5, cluster_std=0.45, random_state=RANDOM_STATE)

def visualize_data(X, y=None, title='Scatter Plot of Sample Data'):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

print("Task 1: Visualize the Sample Data")
visualize_data(X, y_true, title='Scatter Plot of Sample Data')

# Task 2: Implementing K-Means from Scratch
class MyKMeans:
    def __init__(self, n_clusters=5, max_iter=300, n_init=10, random_state=2, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        # Use K-Means++ initialization to ensure initial centroids are spread out for better clustering
        np.random.seed(self.random_state)
        centroids = [X[np.random.choice(X.shape[0])]]
        
        for _ in range(1, self.n_clusters): 
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            next_centroid = X[np.searchsorted(cumulative_probabilities, r)]
            centroids.append(next_centroid)
        
        return np.array(centroids)

    def fit(self, X):
        """
        Computes K-Means clustering with multiple initializations.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
        """
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for init in range(self.n_init):
            centroids = self._initialize_centroids(X)

            for i in range(self.max_iter):
                distances = self._compute_distances(X, centroids)
                labels = np.argmin(distances, axis=1)

                new_centroids = np.array([
                    X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else centroids[j]
                    for j in range(self.n_clusters)
                ])

                if np.linalg.norm(new_centroids - centroids) < self.tol:
                    break
                centroids = new_centroids

            inertia = self._compute_inertia(X, centroids, labels)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia

    def _compute_distances(self, X, centroids):
        """
        Computes the Euclidean distance between each point in X and each centroid.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
        - centroids: np.ndarray, shape (n_clusters, n_features)

        Returns:
        - distances: np.ndarray, shape (n_samples, n_clusters)
        """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances

    def _compute_inertia(self, X, centroids, labels):
        """
        Computes the inertia for given centroids and labels.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
        - centroids: np.ndarray, shape (n_clusters, n_features)
        - labels: np.ndarray, shape (n_samples,)

        Returns:
        - inertia: float
        """
        distances = np.linalg.norm(X - centroids[labels], axis=1)
        return np.sum(distances ** 2)

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)

        Returns:
        - labels: np.ndarray, shape (n_samples,)
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

print("Task 2: Implementing K-Means from Scratch")
optimal_K_initial = 5  
my_kmeans = MyKMeans(n_clusters=optimal_K_initial, random_state=RANDOM_STATE)
my_kmeans.fit(X)

print("Final Centroids:")
print(my_kmeans.centroids)
print(f"\nFinal Inertia: {my_kmeans.inertia_:.4f}\n")


print("Task 3: Comparing Silhouette Scores")
K_range = range(2, 16)
silhouette_scores_my = []
silhouette_scores_sk = []

for K in K_range:
    # MyKMeans
    my_kmeans = MyKMeans(n_clusters=K, random_state=RANDOM_STATE)
    my_kmeans.fit(X)
    labels_my = my_kmeans.labels_
    silhouette_my = silhouette_score(X, labels_my)
    silhouette_scores_my.append(silhouette_my)

    # Scikit-learn KMeans
    sk_kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    sk_kmeans.fit(X)
    labels_sk = sk_kmeans.labels_
    silhouette_sk = silhouette_score(X, labels_sk)
    silhouette_scores_sk.append(silhouette_sk)

    print(f"K={K}: MyKMeans Silhouette Score = {silhouette_my:.4f}, Scikit-learn KMeans Silhouette Score = {silhouette_sk:.4f}")

# Plotting the Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores_my, marker='o', label='MyKMeans')
plt.plot(K_range, silhouette_scores_sk, marker='s', label='Scikit-learn KMeans')
plt.title('Silhouette Scores for MyKMeans vs Scikit-learn KMeans')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)
plt.show()

print("Comparison of MyKMeans and Scikit-learn KMeans:")
print("1. Up to K=5, both implementations show similar silhouette scores.")
print("2. For K=6 to K=8, Scikit-learn's KMeans generally has slightly higher scores.")
print("3. For K=9 to K=13, MyKMeans achieves higher silhouette scores than Scikit-learn.")
print("4. The optimal number of clusters remains K=5 based on the highest silhouette score.")

print("\nReasons Why MyKMeans Output Differs from Scikit-learn's KMeans:")
print("1. Centroid Initialization: MyKMeans may not fully implement K-Means++ initialization, affecting initial centroid placement.")
print("2. Empty Cluster Handling: MyKMeans retains old centroids when clusters are empty, whereas Scikit-learn reinitializes them to new positions.\n")


print("Task 4: Comparing Inertia using the Elbow Method")

# Lists to store inertia
inertia_my = []
inertia_sk = []

for K in K_range:
    # MyKMeans
    my_kmeans = MyKMeans(n_clusters=K, random_state=RANDOM_STATE)
    my_kmeans.fit(X)
    inertia_my.append(my_kmeans.inertia_)

    # Scikit-learn KMeans
    sk_kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    sk_kmeans.fit(X)
    inertia_sk.append(sk_kmeans.inertia_)

    print(f"K={K}: MyKMeans Inertia = {my_kmeans.inertia_:.2f}, Scikit-learn KMeans Inertia = {sk_kmeans.inertia_:.2f}")

print("Comparison: Both implementations show similar inertia values, indicating effective clustering.")
print("Optimal Clusters: The elbow point at K=5 suggests this is the optimal number of clusters.")

# Plotting the Inertia
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_my, marker='o', label='MyKMeans', alpha=0.7,linewidth=2, color ='blue', linestyle='--')
plt.plot(K_range, inertia_sk, marker='s', label='Scikit-learn KMeans', alpha=0.7, linewidth=2, linestyle ='-', color ='orange')
plt.title('Inertia for MyKMeans vs Scikit-learn KMeans')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.legend()
plt.grid(True)
plt.show()




selected_K = 5
print(f"Task 5: Training MyKMeans with K={selected_K} and Plotting")
my_kmeans_optimal = MyKMeans(n_clusters=selected_K, random_state=RANDOM_STATE)
my_kmeans_optimal.fit(X)
labels_optimal = my_kmeans_optimal.labels_
centroids_optimal = my_kmeans_optimal.centroids

# Plotting the clusters
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown',
          'pink', 'gray', 'olive', 'lime', 'navy']

for i in range(selected_K):
    cluster_points = X[labels_optimal == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i % len(colors)],
                label=f'Cluster {i+1}', alpha=0.6)

# Plot centroids
plt.scatter(centroids_optimal[:, 0], centroids_optimal[:, 1], s=200, color='black',
            marker='X', label='Centroids')

plt.title(f'MyKMeans Clustering with K={selected_K}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
import time
from sklearn.preprocessing import StandardScaler

transactions = pd.read_csv('transactions_mini.csv')
validation = pd.read_csv('transactions_mini_validation.csv')

print("""Q1: Given that fraudulent transactions exist in the original/legitimate transactions,
       what kind of anomaly detection should we use and why? Outlier or Novelty?""")
print("""A1: We should use outlier detection because we are identifying rare instances
       (frauds) within the known data distribution.""")
print()


print("Q2: How many fraudulent and how many legitimate transactions exist in the original dataset?")
fraud_count = transactions['Class'].value_counts()[1]
legit_count = transactions['Class'].value_counts()[0]
print(f"A2: Number of fraudulent transactions: {fraud_count}")
print(f"Number of legitimate transactions: {legit_count}")
print()

print("Q3: What percentage of transactions is fraudulent in the original dataset?")
total_transactions = len(transactions)
fraud_percentage = (fraud_count / total_transactions) * 100
print(f"A3: Percentage of fraudulent transactions: {fraud_percentage:.2f}%")
print()

## Anomaly Detection 
# Separate features and labels for training data
X_train = transactions.drop('Class', axis=1)
y_train = transactions['Class']

# Separate features and labels for validation data
X_val = validation.drop('Class', axis=1)
y_val = validation['Class']

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train IsolationForest 
start_time = time.time()
isolation_forest = IsolationForest(
    n_estimators=100,  
    max_samples='auto', 
    contamination='auto',
    random_state=0
)
isolation_forest.fit(X_train_scaled)
iso_forest_training_time = time.time() - start_time

# Train OneClassSVM 
start_time = time.time()
one_class_svm = OneClassSVM(
    kernel='rbf', 
    gamma='scale',
    nu=0.5  
)
one_class_svm.fit(X_train_scaled)
ocsvm_training_time = time.time() - start_time


iso_train_pred = isolation_forest.predict(X_train_scaled)
ocsvm_train_pred = one_class_svm.predict(X_train_scaled)

print("Q4: How many fraudulent transactions were the IsolationForest and OneClassSVM able to capture separately?")
iso_train_detected_frauds = ((iso_train_pred == -1) & (y_train == 1)).sum()
ocsvm_train_detected_frauds = ((ocsvm_train_pred == -1) & (y_train == 1)).sum()
print(f"A4: Number of fraudulent transactions detected by IsolationForest in training data: {iso_train_detected_frauds} out of {fraud_count}")
print(f"Number of fraudulent transactions detected by OneClassSVM in training data: {ocsvm_train_detected_frauds} out of {fraud_count}")
print()

print("Q5: How many legitimate transactions were incorrectly classified as fraudulent by IsolationForest and OneClassSVM separately in training data?")
iso_train_false_positives = ((iso_train_pred == -1) & (y_train == 0)).sum()
ocsvm_train_false_positives = ((ocsvm_train_pred == -1) & (y_train == 0)).sum()
print(f"A5: Number of legitimate transactions incorrectly classified as fraudulent by IsolationForest in training data: {iso_train_false_positives}")
print(f"Number of legitimate transactions incorrectly classified as fraudulent by OneClassSVM in training data: {ocsvm_train_false_positives}")
print()

print("""Q6: Out of the total number of frauds in the original dataset, 
      what percentage of them were detected by IsolationForest and OneClassSVM respectively? 
      What is this metric called usually in machine learning?""")
iso_forest_recall = (iso_train_detected_frauds / fraud_count) * 100
ocsvm_recall = (ocsvm_train_detected_frauds / fraud_count) * 100
print(f"A6: Percentage of fraudulent transactions detected by IsolationForest: {iso_forest_recall:.2f}%")
print(f"Percentage of fraudulent transactions detected by OneClassSVM: {ocsvm_recall:.2f}%")
print("This metric is commonly referred to as 'recall' or 'sensitivity' in machine learning.")
print()

print("""Q7: How much time does it take to train IsolationForest and OneClassSVM for detecting fraudulent transactions?
       What do you notice? Why is this happening?""")
print(f"A7: Time taken to train IsolationForest: {iso_forest_training_time:.4f} seconds")
print(f"Time taken to train OneClassSVM: {ocsvm_training_time:.4f} seconds")
print("""Notice: OneClassSVM generally takes longer to train than IsolationForest due to its computational complexity. 
      IsolationForest isolates anomalies by constructing random trees, 
      which is efficient and scales well with data size. 
      In contrast, OneClassSVM uses kernel methods to transform data into higher dimensions 
      and solve complex optimization problems, leading to longer training times.""")
print()

print("Q8: Algorithmically speaking, how do we classify a sample as an anomaly when using IsolationForest and OneClassSVM?")
print("""A8: 
- IsolationForest: It isolates anomalies by randomly selecting a feature 
    and then randomly selecting a split value between the maximum and minimum values of that feature.
    Anomalies are isolated quickly in fewer random partitions, so they have shorter path lengths in the tree structure.
- OneClassSVM: It uses a hyperplane to separate the data points in a high-dimensional space. 
    Points that lie on one side of the hyperplane are considered normal, 
    while those on the other side are considered anomalies.
    It uses kernel methods to transform the data into higher dimensions to find this hyperplane.""")
print()


print("Q9: Can you spot any anomalies in questions 1-8. If yes, how many and why?")
# accuracy for IsolationForest
iso_train_accuracy = ((iso_train_pred == y_train).sum() / total_transactions) * 100
# accuracy for OneClassSVM
ocsvm_train_accuracy = ((ocsvm_train_pred == y_train).sum() / total_transactions) * 100
print(f"Accuracy of IsolationForest: {iso_train_accuracy:.2f}%")
print(f"Accuracy of OneClassSVM: {ocsvm_train_accuracy:.2f}%")

print(f"""Comment: 
- IsolationForest has a higher accuracy ({iso_train_accuracy:.2f}%) compared to OneClassSVM ({ocsvm_train_accuracy:.2f}%) in this dataset.
- OneClassSVM shows a higher recall ({ocsvm_recall:.2f}%), meaning it detects more fraudulent transactions ({ocsvm_train_detected_frauds} out of {fraud_count}), 
  but it also has a significantly higher false positive rate ({ocsvm_train_false_positives} false positives), leading to lower overall accuracy.
- This is an anomaly because while OneClassSVM seems to perform better in terms of recall, 
  its high false positive rate means it incorrectly classifies many legitimate transactions as frauds.
- The choice between these models depends on the specific requirements of the task, 
  such as whether minimizing false positives or maximizing recall is more important.
- The class imbalance affects accuracy because the dataset contains {legit_count} legitimate transactions and only {fraud_count} fraudulent transactions. 
  A model could achieve high accuracy by predicting all transactions as legitimate, but this would not be useful for detecting frauds.""")
print()

print("Q10: Using your two trained models, detect anomalies on the new data (transactions_mini_validation):")
print ("Q10A. What is this anomaly detection method called? Outlier or Novelty?")
print("""A. This anomaly detection method is called 'Novelty Detection' 
      because we are applying the trained model to new, unseen data to identify anomalies. 
      In contrast, question 1 referred to outlier detection, 
      which involves identifying anomalies within the known dataset.""")

# Predict using IsolationForest on validation data
iso_val_pred = isolation_forest.predict(X_val_scaled)
# Predict using OneClassSVM on validation data
ocsvm_val_pred = one_class_svm.predict(X_val_scaled)

# Calculate total predicted positives (anomalies) for IsolationForest
iso_val_total_positives = (iso_val_pred == -1).sum()

# Calculate true positives for IsolationForest
iso_val_true_positives = ((iso_val_pred == -1) & (y_val == 1)).sum()

# Calculate false positives for IsolationForest
iso_val_false_positives = ((iso_val_pred == -1) & (y_val == 0)).sum()

# Calculate false negatives for IsolationForest
iso_val_false_negatives = (y_val == 1).sum() - iso_val_true_positives

# Calculate total predicted positives (anomalies) for OneClassSVM
ocsvm_val_total_positives = (ocsvm_val_pred == -1).sum()

# Calculate true positives for OneClassSVM
ocsvm_val_true_positives = ((ocsvm_val_pred == -1) & (y_val == 1)).sum()

# Calculate false positives for OneClassSVM
ocsvm_val_false_positives = ((ocsvm_val_pred == -1) & (y_val == 0)).sum()

# Calculate false negatives for OneClassSVM
ocsvm_val_false_negatives = (y_val == 1).sum() - ocsvm_val_true_positives

print()
print("Q10B. How many anomalies are your trained models able to detect?")
print(f"IsolationForest detected {iso_val_total_positives} anomalies in total, "
      f"of which {iso_val_true_positives} were true positives (actual frauds), "
      f"{iso_val_false_positives} were false positives (legitimate transactions incorrectly flagged), "
      f"and {iso_val_false_negatives} were false negatives (frauds missed).")

print(f"OneClassSVM detected {ocsvm_val_total_positives} anomalies in total, "
      f"of which {ocsvm_val_true_positives} were true positives (actual frauds), "
      f"{ocsvm_val_false_positives} were false positives (legitimate transactions incorrectly flagged), "
      f"and {ocsvm_val_false_negatives} were false negatives (frauds missed).")

print("\nBased on the results, we would choose OneClassSVM for this task because it has a higher recall, "
      "meaning it detects more fraudulent transactions. Although it has a higher false positive rate, "
      "we can manually review and dismiss false alarms, ensuring that we do not miss potential frauds.")

