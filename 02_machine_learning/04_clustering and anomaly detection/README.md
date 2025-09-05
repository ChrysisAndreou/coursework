# MAI612 - Machine Learning: Assignment 4 - Clustering & Anomaly Detection

This repository contains the source code and report for Assignment 4 of the MAI612 Machine Learning course. The project focuses on two core unsupervised learning techniques: implementing the K-Means clustering algorithm from scratch and applying anomaly detection models to a credit card fraud dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Part A: K-Means Clustering from Scratch](#part-a-k-means-clustering-from-scratch)
- [Part B: Anomaly Detection on Credit Card Transactions](#part-b-anomaly-detection-on-credit-card-transactions)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Summary of Results](#summary-of-results)
- [Author](#author)
- [License](#license)

## Project Overview

This assignment is divided into two main parts:

1.  **Part A: Clustering**: This section involves implementing the K-Means clustering algorithm from the ground up. The custom implementation is then rigorously compared against Scikit-learn's standard `KMeans` model using metrics like the **Silhouette Score** and **Inertia (Elbow Method)**. The final clustered data is visualized.

2.  **Part B: Anomaly Detection**: This section tackles the real-world problem of fraud detection. Two popular anomaly detection algorithms, **Isolation Forest** and **One-Class SVM**, are trained on a dataset of credit card transactions to identify fraudulent activities. The models are evaluated on their ability to distinguish between legitimate and fraudulent transactions, both on the training data (outlier detection) and on new, unseen data (novelty detection).

## Part A: K-Means Clustering from Scratch

### Key Features
- **Custom `MyKMeans` Class**: A Python class built from scratch to encapsulate all K-Means functionality, including `fit()` and `predict()` methods.
- **K-Means++ Initialization**: Implemented a K-Means++ style initialization for smarter initial centroid placement, leading to more stable and accurate clustering.
- **Stopping Criterion**: The algorithm iteratively refines clusters and stops when the centroids' positions stabilize between iterations.
- **Comparative Analysis**: The performance of `MyKMeans` is benchmarked against `sklearn.cluster.KMeans` by plotting Silhouette Scores and Inertia values for a range of cluster numbers (K=2 to 15).
- **Visualization**: The final clusters and their corresponding centroids are visualized using Matplotlib, demonstrating the algorithm's effectiveness.

## Part B: Anomaly Detection on Credit Card Transactions

### Key Features
- **Models Used**: `IsolationForest` and `OneClassSVM` from Scikit-learn.
- **Dataset**: A toy dataset (`transactions_mini.csv`) containing legitimate (0) and fraudulent (1) transactions.
- **Outlier vs. Novelty Detection**: The project distinguishes between:
    - **Outlier Detection**: Identifying anomalies within the training dataset.
    - **Novelty Detection**: Using the trained models to detect anomalies in a new, unseen validation dataset (`transactions_mini_validation.csv`).
- **Performance Evaluation**: The models are compared based on:
    - **Recall (Sensitivity)**: The percentage of actual frauds correctly identified.
    - **False Positives**: The number of legitimate transactions incorrectly flagged as fraudulent.
    - **Training Time**: The computational efficiency of each model.

## Technologies Used
- **Language**: Python 3.x
- **Libraries**:
    - [NumPy](https://numpy.org/): For numerical operations and array manipulation.
    - [Pandas](https://pandas.pydata.org/): For data loading and manipulation from CSV files.
    - [Matplotlib](https://matplotlib.org/): for data visualization and plotting.
    - [Scikit-learn](https://scikit-learn.org/): For the reference `KMeans` implementation, anomaly detection models (`IsolationForest`, `OneClassSVM`), and evaluation metrics (`silhouette_score`).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MAI612-Assignment-4.git
    cd MAI612-Assignment-4
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file is included for easy setup.
    ```bash
    pip install -r requirements.txt
    ```
    If a `requirements.txt` is not available, you can install the packages manually:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

## Usage

To run the entire analysis, execute the `code.py` script from the root directory of the project. The script will perform all tasks for both Part A and Part B sequentially.

```bash
python code.py
```
All results, answers to the assignment questions, and generated plots will be printed to the console and displayed on the screen as the script runs.

## Project Structure

```
.
├── code.py                     # Main Python script with all implementations and analysis
├── transactions_mini.csv       # Training data for anomaly detection
├── transactions_mini_validation.csv # Validation data for anomaly detection
├── report.pdf                  # Final report summarizing all findings and plots
├── assignment.pdf              # The original assignment description
└── README.md                   # This file
```

## Summary of Results

### Part A: Clustering
- The custom `MyKMeans` implementation performed almost identically to Scikit-learn's `KMeans` for the given dataset, validating its correctness.
- Both the **Elbow Method** (based on inertia) and the **Silhouette Score** analysis clearly indicated that the optimal number of clusters is **K=5**, which matches the ground truth of the generated data.



### Part B: Anomaly Detection
- **OneClassSVM** achieved a significantly higher recall (**96.10%**) compared to **IsolationForest** (**77.41%**), meaning it was better at catching fraudulent transactions.
- However, this came at a cost: **OneClassSVM** produced a very high number of false positives (13,986), incorrectly flagging many legitimate transactions.
- **IsolationForest** was much faster to train (0.09s vs. 14.42s for OneClassSVM).
- **Conclusion**: For fraud detection, where missing a fraud (false negative) is more critical than investigating a false alarm (false positive), **OneClassSVM was deemed the better model for this task**, despite its lower precision.
