# Task 5: Decision Trees and Random Forests (Heart Disease Prediction)

## Overview
This repository contains the solution for Task 5 of the AIML Internship at Elevate Labs. The objective of this task is to train and evaluate tree-based models (Decision Trees and Random Forests) for binary classification, analyze feature importances, control overfitting, and perform cross-validation.

## Files Included
- `task5.py`: The complete Python script for data preprocessing, model training, tree plotting, feature importance extraction, and cross-validation.
- `heart.csv`: The Heart Disease dataset used for classification.
- `decision_tree.png`: Visualization of the trained Decision Tree model (limited to max depth 4).
- `feature_importances.png`: Horizontal bar chart displaying the top feature importances derived from the Random Forest model.

## Workflow & Steps Performed
1. **Data Loading & Preprocessing:** Loaded the heart disease dataset (`heart.csv`), separated the features (`X`) from the binary target variable (`target`), and performed an 80-20 train-test split.
2. **Decision Tree Training & Visualization:** Trained a `DecisionTreeClassifier` with a restricted max depth to prevent overfitting, evaluated its test accuracy, and plotted the tree structure (`decision_tree.png`).
3. **Random Forest Training & Comparison:** Trained an ensemble `RandomForestClassifier` (100 estimators) and compared its test accuracy against the single decision tree.
4. **Feature Importance Analysis:** Extracted and visualized the most influential predictors determining heart disease risk (`feature_importances.png`).
5. **Cross-Validation:** Evaluated the Random Forest model using 5-fold cross-validation to ensure model robustness and generalizability across different data splits.
