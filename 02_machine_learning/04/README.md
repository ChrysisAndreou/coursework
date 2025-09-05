# MAI612 - Machine Learning: Assignment 4 - Clustering & Anomaly Detection

This repository contains the solution for Assignment 4 of the MAI612 Machine Learning course. The project focuses on implementing the K-Means clustering algorithm from scratch and applying anomaly detection techniques to a credit card transaction dataset.

## Project Overview

This assignment is divided into two main parts:

*   **Part A: Clustering**: Implementation of the K-Means algorithm from the ground up and comparison with the Scikit-learn library's implementation. This part involves data visualization, silhouette score analysis, and the elbow method for determining the optimal number of clusters.
*   **Part B: Anomaly Detection**: Application of Isolation Forest and One-Class SVM to detect fraudulent credit card transactions. This includes data preprocessing, model training, and evaluation of the models' performance in identifying anomalies.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Project Structure](#project-structure)
*   [Part A: Clustering Results](#part-a-clustering-results)
*   [Part B: Anomaly Detection Results](#part-b-anomaly-detection-results)
*   [Contributing](#contributing)
*   [License](#license)

## Installation

To run the code in this repository, you need to have Python 3 and the following libraries installed:

*   NumPy
*   Matplotlib
*   Scikit-learn
*   Pandas

You can install these dependencies using pip:

```bash
pip install numpy matplotlib scikit-learn pandas
```

## Usage

To execute the project, run the `code.py` script from your terminal:

```bash
python code.py
```

The script will perform all the tasks outlined in the assignment, print the results to the console, and display the generated plots.

## Project Structure

*   `code.py`: The main Python script containing the implementation of the K-Means algorithm, anomaly detection models, and all the tasks from the assignment.
*   `transactions_mini.csv`: The training dataset for the anomaly detection part.
*   `transactions_mini_validation.csv`: The validation dataset for the anomaly detection part.
*   `Assignment 4 - Clustering & Anomaly Detection.pdf`: The original assignment description.
*   `MAI612 - MACHINE LEARNING Assignment 4 – Clustering & Anomaly Detection.pdf`: The final report in PDF format, containing all the results, plots, and explanations.
*   `README.md`: This file.

## Part A: Clustering Results

### K-Means Implementation

A custom `MyKMeans` class was implemented with the following features:

*   `fit()`: Trains the K-Means model on the input data.
*   `predict()`: Assigns data points to the closest cluster.
*   Inertia calculation.

### Comparison with Scikit-learn

*   **Silhouette Score**: Both `MyKMeans` and Scikit-learn's `KMeans` achieve the highest silhouette score at K=5, suggesting this is the optimal number of clusters. The scores are very similar, indicating a successful implementation of the custom K-Means algorithm.
*   **Elbow Method**: The inertia plots for both implementations show a distinct "elbow" at K=5, further confirming the optimal number of clusters.

![Silhouette Score Comparison](https://i.imgur.com/your-silhouette-score-image.png)
![Inertia Comparison](https://i.imgur.com/your-inertia-image.png)
![Final Clustering](https://i.imgur.com/your-clustering-image.png)

## Part B: Anomaly Detection Results

### Models

*   **Isolation Forest**: An ensemble-based method that isolates anomalies by randomly partitioning the data.
*   **One-Class SVM**: A non-linear method that learns a decision boundary around the normal data points.

### Key Findings

*   The dataset is highly imbalanced, with only 1.68% of transactions being fraudulent.
*   **Isolation Forest** achieved a higher accuracy but lower recall in detecting fraudulent transactions.
*   **One-Class SVM** had a significantly higher recall, identifying more of the actual frauds, but at the cost of a higher false positive rate.
*   For fraud detection, **One-Class SVM** is the preferred model as it is more crucial to identify as many fraudulent transactions as possible, even if it means some legitimate transactions are flagged for review.

| Metric | Isolation Forest | One-Class SVM |
| :--- | :---: | :---: |
| Fraudulent Transactions Detected (Training) | 377 out of 487 | 468 out of 487 |
| Legitimate Transactions Incorrectly Flagged (Training) | 746 | 13986 |
| Recall | 77.41% | 96.10% |
| Training Time | ~0.1 seconds | ~14.4 seconds |
| Anomalies Detected (Validation) | 2 | 11 |

