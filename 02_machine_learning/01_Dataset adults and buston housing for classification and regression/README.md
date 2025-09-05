# MAI612 - Machine Learning: Assignment 1

This repository contains the solution for Assignment 1 of the MAI612 Machine Learning course. The project involves data preparation, visualization, feature engineering, classification, and regression tasks.

## Table of Contents
- [Project Goal](#project-goal)
- [Datasets](#datasets)
- [Installation](#installation)
- [Part A: Data Prep & Visualization | Classification](#part-a-data-prep--visualization--classification)
  - [Data Preparation and Initial Analysis](#data-preparation-and-initial-analysis)
  - [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
  - [Classification](#classification)
- [Part B: Feature Engineering | Regression](#part-b-feature-engineering--regression)
  - [Feature Engineering](#feature-engineering-1)
  - [Regression](#regression)
- [How to Run](#how-to-run)
- [Results](#results)

## Project Goal

The primary goal of this assignment is to practice and demonstrate proficiency in various machine learning tasks, including:
*   **Data Preparation:** Cleaning and preparing datasets for analysis.
*   **Data Visualization:** Creating insightful visualizations to understand data distributions.
*   **Feature Engineering:** Creating new features to improve model performance.
*   **Classification:** Building a logistic regression model to predict salary categories.
*   **Regression:** Building a linear regression model to predict Boston house prices.

## Datasets

This project utilizes two main datasets:

1.  **Adults Dataset:**
    *   `adults.csv`: Training data.
    *   `adults_test.csv`: Testing data.
    This dataset is used for the classification task to predict whether an individual's income exceeds $50K/yr.

2.  **Boston Housing Dataset:**
    *   `boston.csv`
    This dataset is used for the regression task to predict the median value of owner-occupied homes.

## Installation

To run the code in the `code.ipynb` notebook, you need to have Python 3 and the following libraries installed:

*   pandas
*   scikit-learn
*   plotly
*   matplotlib
*   seaborn

You can install these libraries using pip:

```bash
pip install pandas scikit-learn plotly matplotlib seaborn
```

## Part A: Data Prep & Visualization | Classification

### Data Preparation and Initial Analysis

1.  **Men from the United States:** The dataset contains **19,488** men from the United States.

2.  **Salary Guarantee for Bachelor's Degree Holders:** It is **false** that adults with at least a Bachelor's degree are guaranteed to receive more than 50K per year. In fact, 48.46% earn more than 50K, while 51.54% earn 50K or less.

3.  **Hours-per-week Statistics by Race and Gender:** The minimum, maximum, average, and standard deviation of hours-per-week for each race-gender pair are as follows:

| Race | Sex | min_hours | max_hours | avg_hours | std_hours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Amer-Indian-Eskimo| Female | 4 | 84 | 36.58 | 11.05 |
| Amer-Indian-Eskimo| Male | 3 | 84 | 42.20 | 11.60 |
| Asian-Pac-Islander| Female | 1 | 99 | 37.44 | 12.48 |
| Asian-Pac-Islander| Male | 1 | 99 | 41.47 | 12.39 |
| Black | Female | 2 | 99 | 36.83 | 9.42 |
| Black | Male | 1 | 99 | 39.99 | 10.91 |
| Other | Female | 6 | 65 | 35.93 | 10.30 |
| Other | Male | 5 | 98 | 41.85 | 11.08 |
| White | Female | 1 | 99 | 36.30 | 12.19 |
| White | Male | 1 | 99 | 42.67 | 12.19 |

### Data Preprocessing and Feature Engineering

1.  **Handling Missing Values and Duplicates:**
    *   **Train Data:** 2399 data points were removed due to missing values.
    *   **Test Data:** 1221 data points were removed due to missing values.

2.  **Ordinal Encoding for 'Education':** The 'Education' feature was converted to a numerical format using ordinal encoding.

3.  **Scaling the 'Age' Feature:** The 'Age' feature was scaled using `StandardScaler`.

4.  **Visualization of Native Country Distribution:** Pie charts were generated to visualize the distribution of adults based on their native country for both training and testing datasets. The "Others" category was used for countries with smaller representations.

5.  **Correlation Analysis:** A heatmap was used to visualize the correlation between 'Age', 'education\_ordinal', 'Hours Per Week', and 'Salary'. 'education\_ordinal' was found to be the most correlated feature with 'Salary' in both the training (0.34) and testing (0.33) datasets.

### Classification

A logistic regression model was trained to predict the salary class (`<=50K` or `>50K`).

*   **Features Used:** 'age\_scaled', 'education\_ordinal', 'Hours Per Week'.
*   **Classes:** `<=50K` is 0 and `>50K` is 1.

The model's performance on the test set is as follows:
*   **Accuracy:** 0.7837
*   **Precision:** 0.6086
*   **Recall:** 0.3354
*   **F1 Score:** 0.4325
*   **AUC:** 0.7884

The model shows good accuracy but struggles with identifying individuals earning >50K, as indicated by the low recall and F1 score. This suggests a potential class imbalance issue.

## Part B: Feature Engineering | Regression

### Feature Engineering

To improve the performance of the regression model, the following new features were created for the Boston Housing dataset:

1.  **Bivariate Feature:** `LSTAT_PTRATIO` (LSTAT * PTRATIO)
2.  **Polynomial Feature:** `RM^3` (RM ** 3)
3.  **Custom Feature:** `RM_LSTAT_ratio` (RM / LSTAT)

All three engineered features showed a stronger correlation with the target variable `MEDV` than their original constituent features.

### Regression

A linear regression model was trained to predict Boston house prices (`MEDV`).

*   **Train/Test Split:** 90% training, 10% testing.
*   **Mean Squared Error (MSE):** 36.93

A line chart was plotted to compare the real vs. predicted house prices on the test set.

## How to Run

1.  Clone this repository.
2.  Ensure you have all the required libraries installed (see [Installation](#installation)).
3.  Open and run the `code.ipynb` Jupyter Notebook.

## Results

The detailed results and outputs for each task are available within the `code.ipynb` notebook. This includes the printed answers to the initial questions, summaries of the data cleaning process, visualizations, and the performance metrics of the classification and regression models.
