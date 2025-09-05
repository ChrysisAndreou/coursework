# MAI612 - Machine Learning - Assignment 2: Model Selection, Trees & Kernels

## Project Goal

This project focuses on model selection and improvement using the 1994 US Adult Census dataset. The primary objective is to build and evaluate several machine learning models to predict whether an individual's income is greater than $50K.

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Dependencies](#dependencies)

## Dataset

The project utilizes the 1994 US Adult Census dataset. The dataset is split into `adults.csv` for training and `adults_test.csv` for testing. The target variable is `Salary`, which is binarized into `0` for `<=50K` and `1` for `>50K`.

**Data Preprocessing Steps:**
- **Handling Missing Values:** Missing values in categorical columns (`Work Class`, `Occupation`, `Native Country`) are filled with the mode.
- **Encoding Categorical Features:**
    - One-Hot Encoding is applied to the `Native Country` column.
    - Ordinal Encoding is used for the `Education` column based on a predefined order of educational levels.
- **Feature Scaling:** Numerical features (`Age`, `Hours Per Week`) are standardized using `StandardScaler`.

## Implementation

The project is structured into two main parts:

**Part A: Preparation**
- **`finetune(clf, grid_param, rand_param, X, Y)`**: This function performs hyperparameter tuning for a given classifier using both `GridSearchCV` and `RandomizedSearchCV`. It returns the best parameters found by each search method.
- **`fit_and_evaluate(clf, X_train, y_train, X_test, y_test)`**: This function fits a given classifier on the training data, evaluates it on the test data, and returns the false positive rate (fpr), true positive rate (tpr), and the area under the ROC curve (AUC).

**Part B: Model Selection**
This part involves training and evaluating three different classifiers:
- Bagging Decision Tree (BDT)
- Random Forest (RF)
- XGBoost (XGB)

For each classifier, the following steps are performed:
1. **Hyperparameter Tuning:** The `finetune` function is used to find the best hyperparameters using both grid search and random search.
2. **Model Evaluation:** The `fit_and_evaluate` function is used to assess the performance (AUC) of the models with:
    - Default hyperparameters
    - Best hyperparameters from grid search
    - Best hyperparameters from random search
3. **Performance Comparison:** The AUC scores of the different model versions are compared, and ROC curves are plotted.
4. **Execution Time Analysis:** The training and evaluation times for the best performing versions of BDT, RF, XGB, and a Support Vector Classifier (SVC) with default settings are measured and compared.

## How to Run

1. **Prerequisites:** Ensure you have Python installed with the necessary libraries (see [Dependencies](#dependencies)).
2. **Dataset:** Place the `adults.csv` and `adults_test.csv` files in the same directory as the Python script.
3. **Execution:** Run the Python script `MAI612_ass2_candre15_python.py` from your terminal:
   ```bash
   python MAI612_ass2_candre15_python.py
   ```
The script will print the results of the hyperparameter tuning, model evaluations, and execution time comparisons to the console. It will also display plots of the ROC curves and a bar chart comparing the training and evaluation times.

## Results

### Hyperparameter Tuning
The best hyperparameters for each model were determined using both Grid Search and Randomized Search.

### Model Performance (AUC)
The performance of each model was evaluated with default parameters and with the best parameters found by the tuning methods. The results are summarized below:

| Model                   | Default AUC | Grid Search AUC | Random Search AUC |
| ----------------------- | ----------- | --------------- | ----------------- |
| **Random Forest**       | 0.7803      | 0.8163          | 0.8161            |
| **Bagging Decision Tree** | 0.7674      | 0.8101          | 0.8120            |
| **XGBoost**             | 0.8142      | 0.8193          | 0.8191            |

**Key Findings:**
- Hyperparameter tuning significantly improved the performance of all models compared to their default settings.
- **XGBoost** consistently performed the best, achieving the highest AUC score of **0.8193** with the parameters found by Grid Search.

### Execution Time
The training and evaluation times for the best performing models and a default SVC were as follows:

| Classifier              | Training Time (s) | Evaluation Time (s) |
| ----------------------- | ----------------- | ------------------- |
| **Random Forest**       | 0.61              | 0.06                |
| **Bagging Decision Tree** | 0.42              | 0.07                |
| **XGBoost**             | 0.12              | 0.01                |
| **SVC**                 | 117.51            | 15.88               |

**Key Findings:**
- **XGBoost** was the fastest in both training and evaluation.
- The tree-based ensemble methods (XGBoost, BDT, RF) were significantly faster than the Support Vector Classifier.
- **SVC** was considerably slower, making it less suitable for larger datasets or time-sensitive applications.

## Dependencies
- pandas
- scikit-learn
- xgboost
- numpy
- matplotlib
