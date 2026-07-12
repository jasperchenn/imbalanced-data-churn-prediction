# E-Commerce Customer Churn Prediction

Predicting customer churn for an e-commerce company on an
imbalanced dataset, comparing five supervised classification models.

## Goal

Identify high-risk customers early so the company can deploy
targeted retention strategies, using the likelihood of churn
produced by the model.

## Models Compared

Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost —
five supervised approaches (linear, nonlinear, ensemble) trained
with consistent procedures for a fair comparison.

## Results

- **SVM** — best at minimizing false negatives (missed churners)
- **Random Forest** — best overall accuracy

Both are recommended for deployment depending on whether the
business prioritizes recall (SVM) or accuracy (Random Forest).

## Contents

- `Final_project.ipynb` — full analysis workflow
- `app.py` — model serving application
- `E Commerce Dataset.csv` — source data
