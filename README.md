# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
Fraud Detection System 

Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

1. Introduction

Fraud detection is a critical challenge in both e-commerce platforms and banking systems due to highly imbalanced data and evolving fraud patterns.
This project focuses on building machine learning models to detect fraudulent transactions using two datasets:

E-commerce Fraud Dataset

Credit Card Transaction Dataset

Both datasets were preprocessed, feature-engineered, balanced, and split into training and testing sets prior to modeling.

The main objectives were:

Handle class imbalance effectively

Train robust classification models

Compare model performance across datasets

Visualize results

Save trained models for deployment

2. Datasets
2.1 E-Commerce Fraud Dataset

Contains transaction and user behavior features

Target column: class (0 = Normal, 1 = Fraud)

Dataset was:

Cleaned

Feature engineered

Balanced

Split into train/test sets

Saved files:

fraud_train_fe.csv

fraud_test_fe.csv

2.2 Credit Card Dataset

Contains PCA-transformed transaction features (V1–V28), plus:

Time

Amount

Engineered features (user_proxy, hour)

Target column: Class

Saved files:

creditcard_train_fe.csv

creditcard_test_fe.csv

Both datasets were already balanced before modeling.

3. Methodology
3.1 Preprocessing

For each dataset:

Features separated from target labels

Standardization applied using StandardScaler

No categorical encoding required (datasets already numeric)

3.2 Models Used

Two supervised classifiers were trained on each dataset:

✅ Logistic Regression

Linear baseline model

Fast and interpretable

✅ Random Forest

Ensemble-based model

Captures non-linear relationships

Robust to noise

3.3 Evaluation Metrics

Because fraud detection is an imbalanced classification problem, the following metrics were used:

F1-score

AUC–PR (Area Under Precision–Recall Curve)

Confusion Matrix

Precision–Recall Curves

These metrics emphasize minority (fraud) class performance.

4. Results

For each dataset:

Logistic Regression and Random Forest were trained

Confusion matrices were plotted

Precision–Recall curves were generated

Final metrics were collected

Bar charts were also generated to visually compare:

F1 Scores

AUC-PR Scores

5. Model Saving

For reproducibility and deployment, trained artifacts were saved using joblib:

Saved Files
models/
│
├── fraud_logreg.pkl
├── fraud_rf.pkl
├── fraud_scaler.pkl
│
├── creditcard_logreg.pkl
├── creditcard_rf.pkl
├── creditcard_scaler.pkl


These files allow inference without retraining.

6. Key Observations

Random Forest consistently outperformed Logistic Regression in F1 and AUC-PR.

Balancing significantly improved fraud recall.

Credit Card data showed stronger separability due to PCA features.

Logistic Regression provided a strong baseline but struggled with complex patterns.

Precision–Recall curves highlighted Random Forest’s superior fraud detection capability.

7. Conclusion

This project successfully developed a modular fraud detection pipeline for two transaction datasets. Through preprocessing, balancing, and ensemble modeling:

High fraud detection performance was achieved

Models were evaluated using appropriate imbalanced metrics

Trained models were saved for deployment

Random Forest emerged as the best-performing model across both datasets.

8. Future Work

Hyperparameter tuning (GridSearch / Bayesian Optimization)

XGBoost / LightGBM experiments

Threshold optimization

Explainability (SHAP)

Real-time inference pipeline

Model monitoring for concept drift