# Credit-Card-Fraud-Detection
## Overview
This project utilizes machine learning techniques to detect fraudulent credit card transactions.
It addresses the class imbalance problem, trains multiple models, and visualizes results using ROC curves, precision-recall curves, PCA, and t-SNE.
Objective: Identify fraudulent transactions accurately to reduce financial losses.

## Dataset
Source: Credit Card Fraud Detection Dataset (Kaggle)
Size: 284,807 transactions
Fraud cases: 492 (highly imbalanced)
Features: PCA-transformed components (V1–V28), Amount, Time, Class

## Features
Data preprocessing: Scaling Amount and Time
Handling imbalanced data: SMOTE oversampling and Random Undersampling

## Machine Learning Models:
Logistic Regression
Random Forest
XGBoost
Evaluation metrics: Confusion matrix, classification report, ROC-AUC, precision-recall curves
Visualizations: ROC curves, precision-recall curves, PCA and t-SNE scatterplots

## Technologies
Programming Language: Python 3.x
Libraries: pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn

## Future Improvements
Hyperparameter tuning for better model performance
Deploy as a web application using Flask or FastAPI

## License
This project is open source and available for personal and educational use.
