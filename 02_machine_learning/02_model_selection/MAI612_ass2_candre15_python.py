# part A

def preprocess_adult_data(train_file_path, test_file_path):
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

    adult_train_data = pd.read_csv(train_file_path)
    adult_test_data = pd.read_csv(test_file_path)

    # Drop irrelevant column
    adult_train_data_cleaned = adult_train_data.drop(columns=['Unnamed: 0'])
    adult_test_data_cleaned = adult_test_data.drop(columns=['Unnamed: 0'])

    # Fill missing values with mode for categorical columns
    for col in ['Work Class', 'Occupation', 'Native Country']:
        adult_train_data_cleaned[col] = adult_train_data_cleaned[col].fillna(adult_train_data_cleaned[col].mode()[0])
        adult_test_data_cleaned[col] = adult_test_data_cleaned[col].fillna(adult_test_data_cleaned[col].mode()[0])

    # One-hot encoding for "Native Country" column, done on both datasets to avoid mismatch errors 
    native_country_encoder = OneHotEncoder(sparse_output=False, drop='first')
    native_country_combined = pd.concat([adult_train_data_cleaned[['Native Country']], adult_test_data_cleaned[['Native Country']]])
    native_country_encoded = native_country_encoder.fit_transform(native_country_combined)

    # Convert the encoded data to a DataFrame
    native_country_encoded_df = pd.DataFrame(native_country_encoded, columns=native_country_encoder.get_feature_names_out(['Native Country']))

    # Split the encoded data back into train and test sets
    train_encoded_native_country = native_country_encoded_df.iloc[:len(adult_train_data_cleaned)]
    test_encoded_native_country = native_country_encoded_df.iloc[len(adult_train_data_cleaned):]

    # Reset index to merge back with original datasets
    train_encoded_native_country.reset_index(drop=True, inplace=True)
    test_encoded_native_country.reset_index(drop=True, inplace=True)

    # Ordinal encoding for "Education" column
    education_categories = [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
        "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
        "Masters", "Doctorate", "Prof-school"
    ]
    ordinal_encoder = OrdinalEncoder(categories=[education_categories])
    adult_train_data_cleaned['education_ordinal'] = ordinal_encoder.fit_transform(adult_train_data_cleaned[['Education']])
    adult_test_data_cleaned['education_ordinal'] = ordinal_encoder.transform(adult_test_data_cleaned[['Education']])

    # Drop original categorical columns
    categorical_columns = ['Work Class', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native Country', 'Education']
    adult_train_data_cleaned.drop(columns=categorical_columns, inplace=True)
    adult_test_data_cleaned.drop(columns=categorical_columns, inplace=True)

    # Concatenate the one-hot encoded columns with the cleaned datasets
    adult_train_final = pd.concat([adult_train_data_cleaned, train_encoded_native_country], axis=1)
    adult_test_final = pd.concat([adult_test_data_cleaned, test_encoded_native_country], axis=1)

    # Scaling numerical features
    scaler = StandardScaler()
    adult_train_final[['age_scaled', 'hours_per_week_scaled']] = scaler.fit_transform(adult_train_data_cleaned[['Age', 'Hours Per Week']])
    adult_test_final[['age_scaled', 'hours_per_week_scaled']] = scaler.transform(adult_test_data_cleaned[['Age', 'Hours Per Week']])

    # Drop the original numerical columns
    adult_train_final.drop(columns=['Age', 'Hours Per Week'], inplace=True)
    adult_test_final.drop(columns=['Age', 'Hours Per Week'], inplace=True)

    # Mapping salary classes
    salary_class = {'<=50K': 0, '>50K': 1}
    adult_train_final['salary_class'] = adult_train_data_cleaned['Salary'].map(salary_class)
    adult_test_final['salary_class'] = adult_test_data_cleaned['Salary'].map(salary_class)

    # Drop the original Salary column
    adult_train_final.drop(columns=['Salary'], inplace=True)
    adult_test_final.drop(columns=['Salary'], inplace=True)

    # The final datasets are now ready for use
    return adult_train_final, adult_test_final

# Load and preprocess the data
adult_train_path = 'adults.csv'
adult_test_path = 'adults_test.csv'

# Load and preprocess the data
adult_train_processed, adult_test_processed = preprocess_adult_data(adult_train_path, adult_test_path)

# Split the data into features and target
X_train = adult_train_processed.drop(columns=['salary_class'])
y_train = adult_train_processed['salary_class']
X_test = adult_test_processed.drop(columns=['salary_class'])
y_test = adult_test_processed['salary_class']

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
import numpy as np

def finetune(clf, grid_param, rand_param, X, Y):
    # Grid Search
    grid_search = GridSearchCV(estimator=clf, param_grid=grid_param, 
                               scoring='roc_auc', cv=2, n_jobs=-1)
    grid_search.fit(X, Y)
    grid_best = grid_search.best_params_
    print("Best parameters from GridSearchCV:", grid_best)

    # Randomized Search
    rand_search = RandomizedSearchCV(estimator=clf, param_distributions=rand_param, 
                                     n_iter=50, scoring='roc_auc', cv=2, 
                                     random_state=0, n_jobs=-1)
    rand_search.fit(X, Y)
    rand_best = rand_search.best_params_
    print("Best parameters from RandomizedSearchCV:", rand_best)

    return grid_best, rand_best

def fit_and_evaluate(clf, X_train, y_train, X_test, y_test):
    # Fit the classifier
    clf.fit(X_train, y_train)

    # Predict probabilities
    y_probs = clf.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

# Define a classifier
rf_clf = RandomForestClassifier(random_state=0)
bdt = BaggingClassifier(random_state=0)
xgb = XGBClassifier(random_state=0, eval_metric='logloss')

# Define parameter grids, for grid search define a smaller range of parameters 
# because it is more computationally expensive and combinations in the range
rf_grid_param = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 10, 20]
}
# for random search define a bigger range of parameters
rf_rand_param = {
    'n_estimators': [10, 25, 50, 75, 100],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15, 20]
}

# Define parameter grids for BDT
bdt_grid_param = {
    'n_estimators': [10, 100, 200],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.1, 0.5, 1.0],
    'bootstrap': [True, False]
}
bdt_rand_param = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_samples': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'bootstrap': [True, False]
}

# Define parameter grids for XGB
xgb_grid_param = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 7, 11],
    'learning_rate': [0.01, 0.1, 0.3],  
    'min_child_weight': [1, 5, 9]
}
xgb_rand_param = {
    'n_estimators': [50, 75, 100, 150, 200],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5, 7, 9]
}


# part B.1
# Explanation of parameter choice
print("B1.1.B_These parameters are chosen to balance the trade-off between model complexity and computational efficiency, allowing the model to learn effectively from the data while avoiding overfitting. The grid search uses a smaller, more focused range to thoroughly explore specific values, while the randomized search uses a broader range to quickly explore a wider space.")
print("B1.1.C_Grid Search is expected to find the best hyperparameter combinations due to its exhaustive evaluation of all specified options, but it trades off efficiency and computational resources compared to Randomized Search.")
# Run finetune for random forest
print("Tuning Random Forest:")
rf_grid_best, rf_rand_best = finetune(rf_clf, rf_grid_param, rf_rand_param, X_train, y_train)

# Run finetune for BDT
print("Tuning Bagging Decision Tree:")
bdt_grid_best, bdt_rand_best = finetune(bdt, bdt_grid_param, bdt_rand_param, X_train, y_train)

# Run finetune for XGB
print("Tuning XGBoost:")
xgb_grid_best, xgb_rand_best = finetune(xgb, xgb_grid_param, xgb_rand_param, X_train, y_train)

print("B1.2_FIT AND EVALUATE")

# Evaluate and store results for RF
rf_clf_default = RandomForestClassifier(random_state=0)
fpr_rf_default, tpr_rf_default, auc_rf_default = fit_and_evaluate(rf_clf_default, X_train, y_train, X_test, y_test)

rf_clf_grid_best = RandomForestClassifier(random_state=0, **rf_grid_best)
fpr_rf_grid, tpr_rf_grid, auc_rf_grid = fit_and_evaluate(rf_clf_grid_best, X_train, y_train, X_test, y_test)

rf_clf_rand_best = RandomForestClassifier(random_state=0, **rf_rand_best)
fpr_rf_rand, tpr_rf_rand, auc_rf_rand = fit_and_evaluate(rf_clf_rand_best, X_train, y_train, X_test, y_test)

# Evaluate and store results for BDT
bdt_default = BaggingClassifier(random_state=0)
fpr_bdt_default, tpr_bdt_default, auc_bdt_default = fit_and_evaluate(bdt_default, X_train, y_train, X_test, y_test)

clf_bdt_grid_best = BaggingClassifier(random_state=0, **bdt_grid_best)
fpr_bdt_grid, tpr_bdt_grid, auc_bdt_grid = fit_and_evaluate(clf_bdt_grid_best, X_train, y_train, X_test, y_test)

clf_bdt_rand_best = BaggingClassifier(random_state=0, **bdt_rand_best)
fpr_bdt_rand, tpr_bdt_rand, auc_bdt_rand = fit_and_evaluate(clf_bdt_rand_best, X_train, y_train, X_test, y_test)

# Evaluate and store results for XGB
xgb_default = XGBClassifier(random_state=0, eval_metric='logloss')
fpr_xgb_default, tpr_xgb_default, auc_xgb_default = fit_and_evaluate(xgb_default, X_train, y_train, X_test, y_test)

clf_xgb_grid_best = XGBClassifier(random_state=0, eval_metric='logloss', **xgb_grid_best)
fpr_xgb_grid, tpr_xgb_grid, auc_xgb_grid = fit_and_evaluate(clf_xgb_grid_best, X_train, y_train, X_test, y_test)

clf_xgb_rand_best = XGBClassifier(random_state=0, eval_metric='logloss', **xgb_rand_best)
fpr_xgb_rand, tpr_xgb_rand, auc_xgb_rand = fit_and_evaluate(clf_xgb_rand_best, X_train, y_train, X_test, y_test)

# Add comparison comments for each model
print("\nComparison of different settings for each model:")
print(f"Random Forest: Default AUC: {auc_rf_default:.4f}, Grid Search AUC: {auc_rf_grid:.4f}, Random Search AUC: {auc_rf_rand:.4f}")
print(f"Random Forest: Grid Search improved performance by {(auc_rf_grid - auc_rf_default) / auc_rf_default * 100:.2f}% over default.")
print(f"Random Forest: Random Search improved performance by {(auc_rf_rand - auc_rf_default) / auc_rf_default * 100:.2f}% over default.")

print(f"\nBagging Decision Tree: Default AUC: {auc_bdt_default:.4f}, Grid Search AUC: {auc_bdt_grid:.4f}, Random Search AUC: {auc_bdt_rand:.4f}")
print(f"Bagging Decision Tree: Grid Search improved performance by {(auc_bdt_grid - auc_bdt_default) / auc_bdt_default * 100:.2f}% over default.")
print(f"Bagging Decision Tree: Random Search improved performance by {(auc_bdt_rand - auc_bdt_default) / auc_bdt_default * 100:.2f}% over default.")

print(f"\nXGBoost: Default AUC: {auc_xgb_default:.4f}, Grid Search AUC: {auc_xgb_grid:.4f}, Random Search AUC: {auc_xgb_rand:.4f}")
print(f"XGBoost: Grid Search improved performance by {(auc_xgb_grid - auc_xgb_default) / auc_xgb_default * 100:.2f}% over default.")
print(f"XGBoost: Random Search improved performance by {(auc_xgb_rand - auc_xgb_default) / auc_xgb_default * 100:.2f}% over default.")

# Compare the best version of each model
best_rf_auc = max(auc_rf_default, auc_rf_grid, auc_rf_rand)
best_bdt_auc = max(auc_bdt_default, auc_bdt_grid, auc_bdt_rand)
best_xgb_auc = max(auc_xgb_default, auc_xgb_grid, auc_xgb_rand)

print("\nComparison of the best version of each model:")
print(f"Best Random Forest AUC: {best_rf_auc:.4f}")
print(f"Best Bagging Decision Tree AUC: {best_bdt_auc:.4f}")
print(f"Best XGBoost AUC: {best_xgb_auc:.4f}")

best_model = max(best_rf_auc, best_bdt_auc, best_xgb_auc)
if best_model == best_rf_auc:
    print("Random Forest performed best overall.")
elif best_model == best_bdt_auc:
    print("Bagging Decision Tree performed best overall.")
else:
    print("XGBoost performed best overall.")

print(f"The best model outperformed the second-best by {(best_model - sorted([best_rf_auc, best_bdt_auc, best_xgb_auc])[-2]) / sorted([best_rf_auc, best_bdt_auc, best_xgb_auc])[-2] * 100:.2f}%")

import matplotlib.pyplot as plt

def plot_auc_comparison(fpr_default, tpr_default, auc_default, 
                        fpr_grid, tpr_grid, auc_grid, 
                        fpr_rand, tpr_rand, auc_rand, model_name):
    """
    Plots the ROC curves and AUC for a model with default, grid search, and random search settings.

    Parameters:
    - fpr_default, tpr_default, auc_default: False positive rate, true positive rate, and AUC for default settings.
    - fpr_grid, tpr_grid, auc_grid: False positive rate, true positive rate, and AUC for grid search settings.
    - fpr_rand, tpr_rand, auc_rand: False positive rate, true positive rate, and AUC for random search settings.
    - model_name: Name of the model being evaluated.
    """
    plt.figure(figsize=(15, 5))

    # Plot for default settings
    plt.subplot(1, 3, 1)
    plt.plot(fpr_default, tpr_default, color='blue', label=f'AUC = {auc_default:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.title(f'{model_name} - Default')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # Plot for grid search settings
    plt.subplot(1, 3, 2)
    plt.plot(fpr_grid, tpr_grid, color='green', label=f'AUC = {auc_grid:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.title(f'{model_name} - Grid Search')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # Plot for random search settings
    plt.subplot(1, 3, 3)
    plt.plot(fpr_rand, tpr_rand, color='red', label=f'AUC = {auc_rand:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.title(f'{model_name} - Random Search')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

# plot_auc_comparison 
plot_auc_comparison(fpr_rf_default, tpr_rf_default, auc_rf_default, 
                    fpr_rf_grid, tpr_rf_grid, auc_rf_grid, 
                    fpr_rf_rand, tpr_rf_rand, auc_rf_rand, 
                    "Random Forest")

plot_auc_comparison(fpr_bdt_default, tpr_bdt_default, auc_bdt_default, 
                    fpr_bdt_grid, tpr_bdt_grid, auc_bdt_grid, 
                    fpr_bdt_rand, tpr_bdt_rand, auc_bdt_rand, 
                    "Bagging Classifier")

plot_auc_comparison(fpr_xgb_default, tpr_xgb_default, auc_xgb_default, 
                    fpr_xgb_grid, tpr_xgb_grid, auc_xgb_grid, 
                    fpr_xgb_rand, tpr_xgb_rand, auc_xgb_rand, 
                    "XGBoost Classifier")

# part B.2
import time
from sklearn.svm import SVC

print("\nB.2 - Measuring training and evaluation time")

def measure_time(clf, X_train, y_train, X_test, y_test):
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    if hasattr(clf, "predict_proba"):
        y_probs = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    else:
        y_probs = clf.decision_function(X_test)  # For SVC without probability estimation
    eval_time = time.time() - start_time
    
    return train_time, eval_time

# Use the best hyperparameters found in B.1
best_rf = RandomForestClassifier(random_state=0, **rf_grid_best)
best_bdt = BaggingClassifier(random_state=0, **bdt_rand_best)
best_xgb = XGBClassifier(random_state=0, eval_metric='logloss', **xgb_grid_best)
svc = SVC(random_state=0, probability=True)  # Enable probability estimation

classifiers = [
    ("Random Forest", best_rf),
    ("Bagging Decision Tree", best_bdt),
    ("XGBoost", best_xgb),
    ("SVC", svc)
]

train_times = []
eval_times = []

for name, clf in classifiers:
    train_time, eval_time = measure_time(clf, X_train, y_train, X_test, y_test)
    train_times.append(train_time)
    eval_times.append(eval_time)
    print(f"{name} - Training time: {train_time:.2f}s, Evaluation time: {eval_time:.2f}s")

# Plotting
plt.figure(figsize=(12, 6))

x = range(len(classifiers))
width = 0.35

plt.bar([i - width/2 for i in x], train_times, width, label='Training Time', color='blue', alpha=0.7)
plt.bar([i + width/2 for i in x], eval_times, width, label='Evaluation Time', color='green', alpha=0.7)

plt.xlabel('Classifiers')
plt.ylabel('Time (seconds)')
plt.title('Training and Evaluation Time Comparison')
plt.xticks(x, [name for name, _ in classifiers])
plt.legend()

plt.tight_layout()
plt.show()

print("\nExplanation of findings regarding execution time:")
print(f"""
1. XGBoost: Fastest overall (train: {train_times[2]:.2f}s, eval: {eval_times[2]:.2f}s)
   - Highly optimized, leveraging gradient boosting and parallel processing

2. Bagging Decision Tree: Second fastest (train: {train_times[1]:.2f}s, eval: {eval_times[1]:.2f}s)
   - Efficient due to parallel training of base estimators

3. Random Forest: Close third (train: {train_times[0]:.2f}s, eval: {eval_times[0]:.2f}s)
   - Slightly slower in training, faster in evaluation than BDT

4. SVC: Significantly slower (train: {train_times[3]:.2f}s, eval: {eval_times[3]:.2f}s)
   - About {train_times[3]/train_times[2]:.0f}x slower in training and {eval_times[3]/eval_times[2]:.0f}x slower in evaluation than XGBoost
   - Higher time complexity, especially for larger datasets

Key Takeaway: Tree-based ensemble methods (XGBoost, RF, BDT) are significantly faster than SVC for this dataset, with XGBoost being the most time-efficient. Consider this trade-off between performance and speed when selecting classifiers for large-scale or time-sensitive applications.
""")